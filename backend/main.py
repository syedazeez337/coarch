"""Main Coarch module."""

from .faiss_index import FaissIndex, SearchResult
from .embeddings import CodeEmbedder
from .hybrid_indexer import HybridIndexer
from .server import app, run_server

__version__ = "0.1.0"
__all__ = [
    "FaissIndex",
    "SearchResult",
    "CodeEmbedder",
    "HybridIndexer",
    "app",
    "run_server",
]
