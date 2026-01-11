"""Security utilities for Coarch with production-ready features."""

import os
import re
import time
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Set, Dict, Any
from dataclasses import dataclass, field
from fastapi import HTTPException, Request, status
import threading


ALLOWED_FILE_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".css",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".txt",
    ".r",
    ".toml",
    ".ini",
    ".cfg",
}

MAX_PATH_LENGTH = 4096
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024  # 10MB
MAX_QUERY_LENGTH = 1000


class ThreadSafeRateLimiter:
    """Thread-safe rate limiter with memory leak protection."""

    def __init__(self, requests_per_minute: int = 60, max_clients: int = 10000):
        self.requests_per_minute = requests_per_minute
        self.max_clients = max_clients
        self.requests: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup every 60 seconds

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit. Thread-safe."""
        now = time.time()
        minute_ago = now - 60

        with self._lock:
            self._cleanup_if_needed(now)

            if client_id not in self.requests:
                if len(self.requests) >= self.max_clients:
                    self._evict_oldest_clients(now, minute_ago)
                self.requests[client_id] = []

            client_requests = self.requests[client_id]
            client_requests[:] = [t for t in client_requests if t > minute_ago]

            if len(client_requests) >= self.requests_per_minute:
                return False

            client_requests.append(now)
            return True

    def _cleanup_if_needed(self, now: float):
        """Cleanup old entries periodically."""
        if now - self._last_cleanup > self._cleanup_interval:
            with self._cleanup_lock:
                if now - self._last_cleanup > self._cleanup_interval:
                    self._last_cleanup = now
                    cutoff = now - 120  # 2 minute buffer
                    dead_clients = [
                        cid
                        for cid, reqs in self.requests.items()
                        if reqs and reqs[0] < cutoff
                    ]
                    for cid in dead_clients:
                        del self.requests[cid]

    def _evict_oldest_clients(self, now: float, cutoff: float):
        """Evict oldest clients when at capacity."""
        sorted_clients = sorted(
            self.requests.items(), key=lambda x: x[1][0] if x[1] else now
        )
        to_remove = len(self.requests) - self.max_clients + 100
        for cid, _ in sorted_clients[:to_remove]:
            del self.requests[cid]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "active_clients": len(self.requests),
                "requests_per_minute": self.requests_per_minute,
                "max_clients": self.max_clients,
            }


GLOBAL_RATE_LIMITER = ThreadSafeRateLimiter()


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


def validate_path(path: str, allowed_base_paths: Optional[Set[str]] = None) -> str:
    """Validate and sanitize a file system path.

    Args:
        path: The path to validate
        allowed_base_paths: Optional set of allowed base paths

    Returns:
        The validated absolute path

    Raises:
        HTTPException: If path is invalid or outside allowed scope
    """
    if not path or len(path) > MAX_PATH_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path length"
        )

    if not path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty path not allowed"
        )

    normalized = os.path.normpath(path)

    if ".." in normalized.split(os.sep):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Path traversal not allowed"
        )

    if normalized.startswith("/"):
        absolute = normalized
    else:
        absolute = os.path.abspath(normalized)

    if allowed_base_paths:
        allowed = {os.path.abspath(p) for p in allowed_base_paths}
        if not any(absolute.startswith(p) for p in allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Path outside allowed scope",
            )

    if not os.path.exists(absolute):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Path not found"
        )

    if not os.path.isdir(absolute) and not os.path.isfile(absolute):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is not a valid file or directory",
        )

    return absolute


def validate_file_extension(file_path: str) -> bool:
    """Check if file extension is allowed."""
    ext = Path(file_path).suffix.lower()
    return ext in ALLOWED_FILE_EXTENSIONS


def sanitize_search_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize a search query.

    Args:
        query: The search query to sanitize
        max_length: Maximum allowed query length

    Returns:
        Sanitized query

    Raises:
        HTTPException: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid query"
        )

    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty query not allowed"
        )

    if len(query) > max_length:
        query = query[:max_length]

    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

    return sanitized.strip()


def sanitize_language(language: Optional[str]) -> Optional[str]:
    """Validate and normalize language parameter."""
    if language is None:
        return None

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", language)

    valid_languages = {
        "python",
        "javascript",
        "typescript",
        "java",
        "cpp",
        "c",
        "go",
        "rust",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "markdown",
        "json",
        "yaml",
        "xml",
        "html",
        "css",
        "sql",
        "bash",
        "shell",
    }

    if sanitized.lower() not in valid_languages:
        return None

    return sanitized.lower()


def validate_repo_path(request: Request, path: str) -> str:
    """Validate repository path from request."""
    config = request.state.config if hasattr(request.state, "config") else None

    if config and hasattr(config, "indexed_repos"):
        allowed_paths = {r.get("path", "") for r in config.indexed_repos}
        allowed_paths.add(os.path.expanduser("~"))
    else:
        allowed_paths = {os.path.expanduser("~")}

    return validate_path(path, allowed_paths)


def validate_request_size(
    content_length: Optional[int], max_size: int = MAX_REQUEST_BODY_SIZE
) -> None:
    """Validate request size against limit."""
    if content_length and content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request body exceeds maximum size of {max_size} bytes",
        )


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    salt = hashlib.sha256(api_key.encode()).hexdigest()[:32]
    return hashlib.scrypt(api_key.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its hash."""
    return secrets.compare_digest(hash_api_key(api_key), stored_hash)


def generate_api_key() -> str:
    """Generate a new secure API key."""
    return secrets.token_urlsafe(32)


@dataclass
class IndexingJob:
    """Represents an indexing job status."""

    job_id: str
    repo_path: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    chunks_indexed: int = 0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class JobManager:
    """Manages background indexing jobs."""

    def __init__(self, max_jobs: int = 100):
        self.jobs: Dict[str, IndexingJob] = {}
        self._lock = threading.Lock()
        self.max_jobs = max_jobs

    def create_job(self, repo_path: str) -> str:
        """Create a new indexing job."""
        job_id = secrets.token_urlsafe(8)

        with self._lock:
            if len(self.jobs) >= self.max_jobs:
                old_jobs = [
                    jid
                    for jid, job in self.jobs.items()
                    if job.status in ("completed", "failed")
                ]
                for jid in old_jobs[:10]:
                    del self.jobs[jid]

            self.jobs[job_id] = IndexingJob(
                job_id=job_id, repo_path=repo_path, status="pending"
            )

        return job_id

    def get_job(self, job_id: str) -> Optional[IndexingJob]:
        """Get job status."""
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        if job_id in self.jobs:
            for key, value in kwargs.items():
                if hasattr(self.jobs[job_id], key):
                    setattr(self.jobs[job_id], key, value)

    def complete_job(self, job_id: str, success: bool, error: Optional[str] = None):
        """Mark job as completed or failed."""
        self.update_job(
            job_id,
            status="completed" if success else "failed",
            completed_at=time.time(),
            error=error,
        )


GLOBAL_JOB_MANAGER = JobManager()
