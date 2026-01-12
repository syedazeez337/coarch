"""BM25 text search indexer with production-ready features."""

import os
import json
import sqlite3
import math
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from contextlib import contextmanager
import re

from .logging_config import get_logger

try:
    from .progress_tracker import (
        track_progress,
        create_faiss_progress_bar,
        should_show_progress,
        ProgressCallback,
        ETAEstimator,
    )
except ImportError:
    # Fallback if progress tracker is not available
    def track_progress(operation_name, total_items=None, unit="items"):
        class DummyProgress:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_description(self, desc):
                pass
            def set_postfix(self, **kwargs):
                pass
        return DummyProgress()

    def create_faiss_progress_bar(total_vectors):
        class DummyBar:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_description(self, desc):
                pass
            def set_postfix(self, **kwargs):
                pass
        return DummyBar()

    def should_show_progress(total_items=None):
        return total_items is not None and total_items > 100

    class ProgressCallback:
        def __init__(self, progress_bar, operation_name="Processing"):
            self.progress_bar = progress_bar
            self.operation_name = operation_name
        def __call__(self, chunks_processed, total_chunks=None):
            pass
        def set_description(self, description):
            if hasattr(self.progress_bar, 'set_description'):
                self.progress_bar.set_description(description)
        def set_postfix(self, **kwargs):
            if hasattr(self.progress_bar, 'set_postfix'):
                self.progress_bar.set_postfix(kwargs)
        def close(self):
            if hasattr(self.progress_bar, 'close'):
                self.progress_bar.close()

    class ETAEstimator:
        def __init__(self, total_items):
            self.total_items = total_items
        def update(self, processed_items):
            pass
        def get_eta(self):
            return None
        def get_eta_string(self):
            return "N/A"
        def get_rate_string(self):
            return "0 items/s"

logger = get_logger(__name__)


@dataclass
class BM25Document:
    """A document for BM25 indexing."""

    id: int
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str
    content: str
    length: int


@dataclass
class BM25SearchResult:
    """A search result from BM25 index."""

    id: int
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str
    score: float
    terms: List[str]


class BM25Indexer:
    """BM25 text search indexer with thread safety."""

    # BM25 parameters
    K1 = 1.5  # Controls term frequency saturation
    B = 0.75  # Controls length normalization

    def __init__(self, db_path: str = "coarch.db", index_path: Optional[str] = None):
        """Initialize the BM25 indexer."""
        self.db_path = db_path
        self.index_path = index_path or f"{db_path}.bm25"
        self._lock = threading.Lock()
        
        # BM25 statistics
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_lengths: Dict[int, int] = {}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.df: Dict[str, int] = {}  # Document frequency
        self.documents: Dict[int, BM25Document] = {}
        
        # Language-specific settings
        self.language_stopwords = self._load_stopwords()
        
        if os.path.exists(f"{self.index_path}.json"):
            self.load()

    def _load_stopwords(self) -> Dict[str, List[str]]:
        """Load stopwords for different languages."""
        return {
            "python": ["def", "class", "if", "else", "elif", "for", "while", "try", "except", "import", "from", "as", "return", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple", "True", "False", "None", "and", "or", "not", "in", "is"],
            "javascript": ["function", "const", "let", "var", "if", "else", "for", "while", "try", "catch", "return", "console", "log", "new", "class", "extends", "import", "from", "export", "default", "true", "false", "null", "undefined"],
            "typescript": ["function", "const", "let", "var", "if", "else", "for", "while", "try", "catch", "return", "console", "log", "new", "class", "extends", "import", "from", "export", "default", "true", "false", "null", "undefined", "interface", "type", "enum"],
            "java": ["public", "private", "protected", "static", "final", "class", "interface", "enum", "if", "else", "for", "while", "do", "try", "catch", "finally", "return", "void", "int", "String", "boolean", "true", "false", "null"],
            "cpp": ["int", "void", "char", "float", "double", "bool", "if", "else", "for", "while", "do", "try", "catch", "return", "class", "struct", "template", "namespace", "using", "true", "false", "nullptr"],
            "go": ["func", "package", "import", "const", "var", "type", "struct", "interface", "if", "else", "for", "switch", "case", "default", "return", "true", "false", "nil"],
            "rust": ["fn", "let", "mut", "const", "static", "struct", "enum", "trait", "impl", "if", "else", "for", "while", "loop", "match", "return", "true", "false"],
            "ruby": ["def", "class", "module", "if", "else", "elsif", "case", "when", "while", "until", "for", "begin", "rescue", "ensure", "return", "puts", "print", "puts", "true", "false", "nil"],
            "php": ["function", "class", "interface", "trait", "if", "else", "elseif", "for", "foreach", "while", "do", "switch", "case", "default", "return", "echo", "print", "var", "true", "false", "null"],
        }

    def _tokenize(self, text: str, language: str = "code") -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove comments and strings for code
        if language != "text":
            # Remove single-line comments
            comment_patterns = {
                "python": r'#.*$',
                "javascript": r'//.*$|/\*.*?\*/',
                "typescript": r'//.*$|/\*.*?\*/',
                "java": r'//.*$|/\*.*?\*/',
                "cpp": r'//.*$|/\*.*?\*/',
                "go": r'//.*$|/\*.*?\*/',
                "rust": r'//.*$|/\*.*?\*/',
                "ruby": r'#.*$',
                "php": r'//.*$|/\*.*?\*/|#.*$',
            }
            
            pattern = comment_patterns.get(language, r'//.*$')
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
        
        # Extract identifiers and keywords
        # Pattern matches: identifiers, numbers, operators
        if language == "text":
            # For text, use word boundaries
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        else:
            # For code, extract meaningful tokens
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\d+\.?\d*|[+\-*/=<>!&|]+', text)
        
        # Filter out stopwords and very short tokens
        stopwords = self.language_stopwords.get(language, [])
        tokens = [
            token for token in tokens 
            if len(token) >= 2 and token not in stopwords
        ]
        
        return tokens

    def _compute_doc_length(self, tokens: List[str]) -> int:
        """Compute document length for BM25."""
        return len(tokens)

    @contextmanager
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def build_index(self, chunk_limit: Optional[int] = None) -> Dict[str, Any]:
        """Build BM25 index from database."""
        with self._lock:
            logger.info("Building BM25 index...")
            
            # Clear existing index
            self._clear_index()
            
            # Get chunks from database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if chunk_limit:
                    cursor.execute(
                        """
                        SELECT id, file_path, start_line, end_line, code, language
                        FROM chunks 
                        ORDER BY id 
                        LIMIT ?
                    """,
                        (chunk_limit,)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, file_path, start_line, end_line, code, language
                        FROM chunks 
                        ORDER BY id
                    """
                    )
                
                chunks = cursor.fetchall()
            
            logger.info(f"Processing {len(chunks)} chunks for BM25 indexing")
            
            # Build inverted index
            total_length = 0
            processed = 0
            
            show_progress = should_show_progress(len(chunks))
            
            if show_progress:
                with track_progress("Building BM25 index", len(chunks), "chunks") as progress:
                    for chunk in chunks:
                        doc = BM25Document(
                            id=chunk["id"],
                            file_path=chunk["file_path"],
                            start_line=chunk["start_line"],
                            end_line=chunk["end_line"],
                            code=chunk["code"],
                            language=chunk["language"],
                            content=chunk["code"],
                            length=0  # Will be computed
                        )
                        
                        tokens = self._tokenize(doc.content, doc.language)
                        doc.length = self._compute_doc_length(tokens)
                        
                        # Add to documents
                        self.documents[doc.id] = doc
                        self.doc_lengths[doc.id] = doc.length
                        total_length += doc.length
                        
                        # Build inverted index
                        term_counts = Counter(tokens)
                        for term, count in term_counts.items():
                            if term not in self.inverted_index:
                                self.inverted_index[term] = {}
                                self.df[term] = 0
                            
                            self.inverted_index[term][doc.id] = count
                        
                        # Update document frequency
                        for term in term_counts:
                            self.df[term] += 1
                        
                        processed += 1
                        if progress and processed % 1000 == 0:
                            progress.update(1000)
                            progress.set_postfix(
                                {"terms": len(self.inverted_index), "docs": len(self.documents)}
                            )
            else:
                for chunk in chunks:
                    doc = BM25Document(
                        id=chunk["id"],
                        file_path=chunk["file_path"],
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        code=chunk["code"],
                        language=chunk["language"],
                        content=chunk["code"],
                        length=0
                    )
                    
                    tokens = self._tokenize(doc.content, doc.language)
                    doc.length = self._compute_doc_length(tokens)
                    
                    self.documents[doc.id] = doc
                    self.doc_lengths[doc.id] = doc.length
                    total_length += doc.length
                    
                    term_counts = Counter(tokens)
                    for term, count in term_counts.items():
                        if term not in self.inverted_index:
                            self.inverted_index[term] = {}
                            self.df[term] = 0
                        
                        self.inverted_index[term][doc.id] = count
                    
                    for term in term_counts:
                        self.df[term] += 1
                    
                    processed += 1
            
            self.doc_count = len(self.documents)
            self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
            
            logger.info(f"BM25 index built: {self.doc_count} documents, {len(self.inverted_index)} terms")
            
            return {
                "documents": self.doc_count,
                "terms": len(self.inverted_index),
                "avg_length": round(self.avg_doc_length, 2),
                "total_length": total_length
            }

    def _clear_index(self):
        """Clear the current index."""
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.df.clear()
        self.documents.clear()

    def search(self, query: str, language: Optional[str] = None, limit: int = 10) -> List[BM25SearchResult]:
        """Search using BM25 algorithm."""
        if self.doc_count == 0:
            logger.warning("BM25 index is empty")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query.lower(), language or "code")
        if not query_tokens:
            return []
        
        logger.debug(f"BM25 search for tokens: {query_tokens[:5]}...")
        
        # Calculate BM25 scores
        scores: Dict[int, float] = defaultdict(float)
        matched_terms = defaultdict(set)
        
        for term in query_tokens:
            if term in self.inverted_index:
                # Get document frequency for this term
                df = self.df[term]
                
                # Calculate IDF
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
                
                # Get documents containing this term
                term_docs = self.inverted_index[term]
                
                for doc_id, tf in term_docs.items():
                    # Apply BM25 formula
                    doc_length = self.doc_lengths[doc_id]
                    
                    # BM25 score calculation
                    numerator = tf * (self.K1 + 1)
                    denominator = tf + self.K1 * (1 - self.B + self.B * doc_length / self.avg_doc_length)
                    
                    score = idf * (numerator / denominator)
                    scores[doc_id] += score
                    matched_terms[doc_id].add(term)
        
        # Filter by language if specified
        if language:
            filtered_scores = {}
            for doc_id, score in scores.items():
                if doc_id in self.documents and self.documents[doc_id].language == language:
                    filtered_scores[doc_id] = score
            scores = filtered_scores
        
        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results: List[BM25SearchResult] = []
        for doc_id, score in sorted_results[:limit]:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append(
                    BM25SearchResult(
                        id=doc_id,
                        file_path=doc.file_path,
                        start_line=doc.start_line,
                        end_line=doc.end_line,
                        code=doc.code,
                        language=doc.language,
                        score=round(score, 4),
                        terms=list(matched_terms[doc_id])
                    )
                )
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "documents": self.doc_count,
            "terms": len(self.inverted_index),
            "avg_length": round(self.avg_doc_length, 2),
            "index_size_mb": self._estimate_size_mb()
        }

    def _estimate_size_mb(self) -> float:
        """Estimate index size in MB."""
        size_bytes = 0
        
        # Estimate document storage
        for doc in self.documents.values():
            size_bytes += len(doc.file_path) + len(doc.code) + 100  # Approximate
        
        # Estimate inverted index storage
        for term, docs in self.inverted_index.items():
            size_bytes += len(term.encode()) * len(docs)
            size_bytes += len(docs) * 8  # Approximate per-doc overhead
        
        return size_bytes / (1024 * 1024)

    def save(self) -> str:
        """Save index to disk."""
        with self._lock:
            index_data = {
                "doc_count": self.doc_count,
                "avg_doc_length": self.avg_doc_length,
                "doc_lengths": self.doc_lengths,
                "inverted_index": {term: dict(docs) for term, docs in self.inverted_index.items()},
                "df": self.df,
                "documents": {
                    doc_id: {
                        "id": doc.id,
                        "file_path": doc.file_path,
                        "start_line": doc.start_line,
                        "end_line": doc.end_line,
                        "code": doc.code,
                        "language": doc.language,
                        "content": doc.content,
                        "length": doc.length
                    }
                    for doc_id, doc in self.documents.items()
                }
            }
            
            try:
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                with open(f"{self.index_path}.json", "w") as f:
                    json.dump(index_data, f, separators=(',', ':'))
                
                logger.info(f"BM25 index saved to {self.index_path}")
                return self.index_path
            except Exception as e:
                logger.exception(f"Failed to save BM25 index: {e}")
                raise

    def load(self) -> bool:
        """Load index from disk."""
        try:
            with open(f"{self.index_path}.json", "r") as f:
                data = json.load(f)
            
            self.doc_count = data["doc_count"]
            self.avg_doc_length = data["avg_doc_length"]
            self.doc_lengths = data["doc_lengths"]
            self.inverted_index = {term: docs for term, docs in data["inverted_index"].items()}
            self.df = data["df"]
            
            # Reconstruct documents
            self.documents = {}
            for doc_id, doc_data in data["documents"].items():
                self.documents[int(doc_id)] = BM25Document(
                    id=doc_data["id"],
                    file_path=doc_data["file_path"],
                    start_line=doc_data["start_line"],
                    end_line=doc_data["end_line"],
                    code=doc_data["code"],
                    language=doc_data["language"],
                    content=doc_data["content"],
                    length=doc_data["length"]
                )
            
            logger.info(f"BM25 index loaded: {self.doc_count} documents, {len(self.inverted_index)} terms")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load BM25 index: {e}")
            return False

    def clear(self):
        """Clear the index."""
        with self._lock:
            self._clear_index()
            
            # Remove saved index
            try:
                if os.path.exists(f"{self.index_path}.json"):
                    os.remove(f"{self.index_path}.json")
            except Exception as e:
                logger.warning(f"Failed to remove index file: {e}")
            
            logger.info("BM25 index cleared")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._lock = None
        except Exception:
            pass