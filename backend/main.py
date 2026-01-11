"""Main Coarch module."""

from .faiss_index import FaissIndex, SearchResult
from .embeddings import CodeEmbedder
from .indexer import RepositoryIndexer, CodeChunk
from .server import app, run_server

__version__ = "0.1.0"
__all__ = [
    "FaissIndex",
    "SearchResult",
    "CodeEmbedder",
    "RepositoryIndexer",
    "CodeChunk",
    "app",
    "run_server",
]
