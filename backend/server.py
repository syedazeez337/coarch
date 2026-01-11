"""FastAPI server for Coarch search."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os

from .faiss_index import FaissIndex
from .embeddings import CodeEmbedder
from .indexer import RepositoryIndexer


app = FastAPI(
    title="Coarch API",
    description="Local-first code search engine",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    language: Optional[str] = None
    limit: int = 10
    repo_id: Optional[int] = None


class SearchResult(BaseModel):
    file_path: str
    lines: str
    code: str
    score: float
    language: str


class IndexRequest(BaseModel):
    path: str
    name: Optional[str] = None


class IndexResponse(BaseModel):
    status: str
    stats: Dict


class StatusResponse(BaseModel):
    status: str
    stats: Dict


def get_components():
    """Get or create global components."""
    if not hasattr(app.state, "indexer"):
        app.state.indexer = RepositoryIndexer()
    if not hasattr(app.state, "embedder"):
        app.state.embedder = CodeEmbedder()
    if not hasattr(app.state, "faiss"):
        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        app.state.faiss = FaissIndex(
            dim=app.state.embedder.get_dimension(),
            index_path=index_path
        )
    return app.state.indexer, app.state.embedder, app.state.faiss


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Search for similar code."""
    indexer, embedder, faiss = get_components()

    # Get query embedding
    query_embedding = embedder.embed_query(request.query)

    # Search FAISS
    results = faiss.search(query_embedding, k=request.limit)

    # Filter by language if specified
    filtered = []
    for r in results:
        if request.language and r.language != request.language:
            continue
        filtered.append(SearchResult(
            file_path=r.file_path,
            lines=f"{r.start_line}-{r.end_line}",
            code=r.code[:200] + "..." if len(r.code) > 200 else r.code,
            score=r.score,
            language=r.language
        ))

    return filtered


@app.post("/index/repo", response_model=IndexResponse)
async def index_repo(request: IndexRequest):
    """Index a repository."""
    indexer, embedder, faiss = get_components()

    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Path not found")

    # Index the repository
    stats = indexer.index_repository(request.path, request.name)

    # Get chunks for embedding
    chunks = indexer.get_chunks_for_embedding()

    # Generate embeddings
    code_texts = [chunk.code for chunk in chunks]
    embeddings = embedder.embed(code_texts)

    # Add to FAISS
    metadata = [{
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "code": chunk.code,
        "language": chunk.language
    } for chunk in chunks]

    faiss.add(embeddings, metadata)
    faiss.save()

    return IndexResponse(status="success", stats=stats)


@app.get("/index/status", response_model=StatusResponse)
async def status():
    """Get index status."""
    indexer, _, faiss = get_components()

    return StatusResponse(
        status="ready" if faiss.count() > 0 else "empty",
        stats={
            "vectors_indexed": faiss.count(),
            "db_stats": indexer.get_stats()
        }
    )


@app.delete("/index/repo/{repo_id}")
async def delete_repo(repo_id: int):
    """Delete a repository from the index."""
    indexer, _, faiss = get_components()
    indexer.delete_repo(repo_id)
    return {"status": "deleted"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the Coarch server."""
    uvicorn.run(app, host=host, port=port)
