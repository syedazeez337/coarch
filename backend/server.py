"""Production-ready FastAPI server for Coarch search."""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt
import uvicorn

from .logging_config import setup_logging, get_logger
from .security import (
    validate_path,
    sanitize_search_query,
    sanitize_language,
    get_client_id,
    GLOBAL_RATE_LIMITER,
    ThreadSafeRateLimiter,
    GLOBAL_JOB_MANAGER,
)
from .config import CoarchConfig

logger = get_logger(__name__)

security = HTTPBearer()


class AppState:
    """Application state container."""

    def __init__(self):
        self.indexer = None
        self.embedder = None
        self.faiss = None
        self.config = None
        self.rate_limiter = ThreadSafeRateLimiter(requests_per_minute=60)
        self.startup_complete = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with graceful shutdown."""
    logger.info("Starting Coarch server...")
    app.state.startup_complete = True
    yield
    logger.info("Shutting down Coarch server...")
    if hasattr(app.state, "shutdown_event"):
        app.state.shutdown_event.set()


app = FastAPI(
    title="Coarch API",
    description="Local-first semantic code search engine",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.state = AppState()


ALLOWED_ORIGINS = os.environ.get(
    "COARCH_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


def get_config():
    """Get or create config."""
    if not hasattr(app.state, "config") or app.state.config is None:
        from .config import CoarchConfig

        app.state.config = CoarchConfig.load(CoarchConfig.get_default_config_path())
    return app.state.config


def get_components():
    """Get or create global components with proper initialization."""
    state = app.state

    if state.indexer is None:
        from .indexer import RepositoryIndexer

        state.indexer = RepositoryIndexer()
        logger.info("Initialized RepositoryIndexer")

    if state.embedder is None:
        from .embeddings import CodeEmbedder

        state.embedder = CodeEmbedder()
        logger.info(f"Initialized CodeEmbedder: {state.embedder}")

    if state.faiss is None:
        from .faiss_index import FaissIndex

        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        dim = state.embedder.get_dimension()
        state.faiss = FaissIndex(dim=dim, index_path=index_path)
        logger.info(f"Initialized FaissIndex with dim={dim}")

    return state.indexer, state.embedder, state.faiss


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    config: CoarchConfig = Depends(get_config),
) -> str:
    """Verify JWT token.

    Args:
        credentials: HTTP Bearer credentials
        config: Application config

    Returns:
        Token subject (user ID)
    """
    if not config.enable_auth:
        return "anonymous"

    token = credentials.credentials
    try:
        payload = jwt.decode(token, config.jwt_secret, algorithms=["HS256"])
        return payload.get("sub", "unknown")
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=401, detail="Invalid or expired authentication token"
        )


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    config: CoarchConfig = Depends(get_config),
) -> bool:
    """Verify API key for simple authentication.

    Args:
        x_api_key: API key header
        config: Application config

    Returns:
        True if valid
    """
    if not config.enable_auth:
        return True

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    if not config._verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for the request."""
    client_id = get_client_id(request)

    if not GLOBAL_RATE_LIMITER.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=1000)
    language: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    repo_ids: Optional[List[int]] = None


class MultiSearchRequest(BaseModel):
    """Multi-repository search request model."""

    query: str = Field(..., min_length=1, max_length=1000)
    repo_ids: Optional[List[int]] = None
    language: Optional[str] = None
    limit_per_repo: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    """Search result model."""

    file_path: str
    lines: str
    code: str
    score: float
    language: str


class IndexRequest(BaseModel):
    """Index repository request model."""

    path: str = Field(..., min_length=1, max_length=4096)
    name: Optional[str] = None


class IndexResponse(BaseModel):
    """Index response model."""

    status: str
    job_id: Optional[str] = None
    stats: Dict[str, Any]


class JobStatusResponse(BaseModel):
    """Job status response model."""

    job_id: str
    status: str
    progress: float
    chunks_indexed: int
    error: Optional[str] = None
    started_at: float
    completed_at: Optional[float] = None


class StatusResponse(BaseModel):
    """Status response model."""

    status: str
    stats: Dict[str, Any]
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.exception(f"Unhandled exception: {exc}")
    request_id = request.headers.get("X-Request-ID", "unknown")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if logger.isEnabledFor(10) else None,
            request_id=request_id,
        ).model_dump(),
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all incoming requests."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    logger.info(
        f"Request {request_id}: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"Response {request_id}: {response.status_code} " f"({process_time:.3f}s)",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response
    except Exception as e:
        logger.exception(f"Request {request_id} failed: {e}")
        raise


@app.post(
    "/search",
    response_model=List[SearchResult],
    dependencies=[Depends(check_rate_limit)],
)
async def search(request: SearchRequest, req: Request):
    """Search for similar code."""
    indexer, embedder, faiss = get_components()

    sanitized_query = sanitize_search_query(request.query)
    if not sanitized_query:
        raise HTTPException(status_code=400, detail="Invalid search query")

    logger.info(f"Searching for: {sanitized_query[:100]}...")

    query_embedding = embedder.embed_query(sanitized_query)

    limit_multiplier = 3
    results = faiss.search(query_embedding, k=request.limit * limit_multiplier)

    filtered: List[SearchResult] = []
    lang_filter = sanitize_language(request.language)

    for r in results:
        if lang_filter and r.language != lang_filter:
            continue

        code_preview = r.code[:200] + "..." if len(r.code) > 200 else r.code

        filtered.append(
            SearchResult(
                file_path=r.file_path,
                lines=f"{r.start_line}-{r.end_line}",
                code=code_preview,
                score=round(r.score, 4),
                language=r.language,
            )
        )

        if len(filtered) >= request.limit:
            break

    logger.info(f"Found {len(filtered)} results for query")
    return filtered


@app.post(
    "/search/multi",
    response_model=Dict[str, List[SearchResult]],
    dependencies=[Depends(check_rate_limit)],
)
async def multi_search(request: MultiSearchRequest):
    """Search across multiple repositories."""
    indexer, embedder, faiss = get_components()

    sanitized_query = sanitize_search_query(request.query)

    query_embedding = embedder.embed_query(sanitized_query)

    limit_multiplier = 10
    results = faiss.search(query_embedding, k=request.limit_per_repo * limit_multiplier)

    repo_results: Dict[str, List[SearchResult]] = {}
    lang_filter = sanitize_language(request.language)

    for r in results:
        if lang_filter and r.language != lang_filter:
            continue

        path_parts = r.file_path.split(os.sep)
        repo_path = path_parts[0] if path_parts else r.file_path

        if repo_path not in repo_results:
            repo_results[repo_path] = []

        if len(repo_results[repo_path]) < request.limit_per_repo:
            code_preview = r.code[:200] + "..." if len(r.code) > 200 else r.code

            repo_results[repo_path].append(
                SearchResult(
                    file_path=r.file_path,
                    lines=f"{r.start_line}-{r.end_line}",
                    code=code_preview,
                    score=round(r.score, 4),
                    language=r.language,
                )
            )

    return repo_results


@app.get("/repos", response_model=List[Dict])
async def list_repos():
    """List all indexed repositories."""
    indexer, _, _ = get_components()
    stats = indexer.get_stats()
    return stats.get("repos", [])


@app.post(
    "/index/repo",
    response_model=IndexResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def index_repo(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index a repository with job tracking."""
    indexer, embedder, faiss = get_components()

    validated_path = validate_path(request.path)

    job_id = GLOBAL_JOB_MANAGER.create_job(validated_path)

    logger.info(f"Indexing repository: {validated_path} (job_id={job_id})")

    stats = indexer.index_repository(validated_path, request.name)

    def process_embeddings():
        """Process embeddings in background with job tracking."""
        job = GLOBAL_JOB_MANAGER.get_job(job_id)
        if job:
            job.status = "running"

        try:
            chunks = indexer.get_chunks_for_embedding()

            if not chunks:
                logger.warning("No chunks to embed")
                GLOBAL_JOB_MANAGER.complete_job(job_id, True)
                return

            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            total_chunks = len(chunks)
            processed = 0

            code_texts = [chunk.code for chunk in chunks]
            embeddings = embedder.embed(code_texts)

            for i, chunk in enumerate(chunks):
                if job:
                    job.progress = (i + 1) / total_chunks
                    job.chunks_indexed = i + 1

            metadata = [
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "code": chunk.code,
                    "language": chunk.language,
                }
                for chunk in chunks
            ]

            faiss.add(embeddings, metadata)
            faiss.save()

            GLOBAL_JOB_MANAGER.complete_job(job_id, True)
            logger.info(f"Indexed {len(chunks)} chunks")

        except Exception as e:
            logger.exception(f"Failed to process embeddings: {e}")
            GLOBAL_JOB_MANAGER.complete_job(job_id, False, str(e))

    background_tasks.add_task(process_embeddings)

    return IndexResponse(status="indexing", job_id=job_id, stats=stats)


@app.get("/index/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an indexing job."""
    job = GLOBAL_JOB_MANAGER.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        chunks_indexed=job.chunks_indexed,
        error=job.error,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.get("/index/status", response_model=StatusResponse)
async def status():
    """Get index status."""
    indexer, _, faiss = get_components()

    vector_count = faiss.count() if faiss else 0

    return StatusResponse(
        status="ready" if vector_count > 0 else "empty",
        stats={
            "vectors_indexed": vector_count,
            "db_stats": indexer.get_stats() if indexer else {},
            "rate_limiter_stats": GLOBAL_RATE_LIMITER.get_stats(),
        },
        version="2.0.0",
    )


@app.delete("/index/repo/{repo_id}")
async def delete_repo(repo_id: int):
    """Delete a repository from the index."""
    indexer, _, faiss = get_components()

    if not indexer:
        raise HTTPException(status_code=404, detail="Indexer not initialized")

    indexer.delete_repo(repo_id)

    logger.info(f"Deleted repository {repo_id}")

    return {"status": "deleted"}


@app.post("/admin/vacuum")
async def vacuum_database(credentials: str = Depends(verify_token)):
    """Vacuum the database to reclaim space."""
    if credentials != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    indexer, _, _ = get_components()
    indexer.vacuum()

    return {"status": "vacuum_complete"}


@app.post("/admin/cleanup")
async def cleanup_orphaned(credentials: str = Depends(verify_token)):
    """Clean up orphaned chunks."""
    if credentials != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    indexer, _, _ = get_components()
    deleted = indexer.cleanup_orphaned_chunks()

    return {"status": "complete", "deleted_chunks": deleted}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        indexer, embedder, faiss = get_components()

        faiss_count = faiss.count() if faiss else 0

        return {
            "status": "healthy",
            "components": {
                "indexer": "ok" if indexer else "error",
                "embedder": "ok" if embedder else "error",
                "faiss": "ok" if faiss else "error",
            },
            "vectors": faiss_count,
            "timestamp": time.time(),
            "version": "2.0.0",
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            },
        )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    try:
        _, _, faiss = get_components()

        return {
            "coarch_vectors_indexed": faiss.count() if faiss else 0,
            "coarch_up": 1,
            "coarch_rate_limiter_clients": GLOBAL_RATE_LIMITER.get_stats().get(
                "active_clients", 0
            ),
        }
    except Exception as e:
        return {
            "coarch_vectors_indexed": 0,
            "coarch_up": 0,
            "coarch_error": str(e),
        }


@app.get("/rate-limit-stats")
async def rate_limit_stats():
    """Get rate limiter statistics."""
    return GLOBAL_RATE_LIMITER.get_stats()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "INFO",
    reload: bool = False,
) -> None:
    """Run the Coarch server."""
    log_file = os.environ.get("COARCH_LOG_FILE")

    setup_logging(
        level=log_level,
        log_file=log_file,
        json_format=os.environ.get("COARCH_LOG_JSON", "false").lower() == "true",
    )

    logger.info(f"Starting Coarch server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        timeout_graceful_shutdown=30,
    )
