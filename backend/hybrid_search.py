"""Hybrid search combining BM25 text search with semantic search."""

import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from .logging_config import get_logger
from .bm25_index import BM25Indexer, BM25SearchResult
from .faiss_index import FaissIndex, SearchResult

try:
    from .progress_tracker import track_progress
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

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    """A hybrid search result combining BM25 and semantic scores."""

    id: int
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str
    score: float
    bm25_score: float
    semantic_score: float
    matched_terms: List[str]
    result_type: str  # "bm25_only", "semantic_only", "hybrid"


class HybridSearch:
    """Hybrid search combining BM25 and semantic search with configurable weighting."""

    def __init__(
        self,
        bm25_indexer: BM25Indexer,
        faiss_index: Optional[FaissIndex],
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        min_score_threshold: float = 0.01,
        enable_explanation: bool = False
    ):
        """Initialize hybrid search.
        
        Args:
            bm25_indexer: BM25 text search indexer
            faiss_index: FAISS semantic search indexer
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            semantic_weight: Weight for semantic scores (0.0 to 1.0)
            min_score_threshold: Minimum combined score threshold
            enable_explanation: Whether to include detailed explanations
        """
        self.bm25_indexer = bm25_indexer
        self.faiss_index = faiss_index
        
        # Normalize weights
        total_weight = bm25_weight + semantic_weight
        if total_weight == 0:
            self.bm25_weight = 0.5
            self.semantic_weight = 0.5
        else:
            self.bm25_weight = bm25_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight
        
        self.min_score_threshold = min_score_threshold
        self.enable_explanation = enable_explanation
        
        # Thread pool for parallel search
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 10,
        use_bm25: bool = True,
        use_semantic: bool = True,
        return_raw_scores: bool = False
    ) -> List[HybridSearchResult]:
        """Perform hybrid search combining BM25 and semantic search.
        
        Args:
            query: Search query
            language: Optional language filter
            limit: Maximum number of results
            use_bm25: Whether to use BM25 search
            use_semantic: Whether to use semantic search
            return_raw_scores: Whether to include raw BM25 and semantic scores
            
        Returns:
            List of hybrid search results
        """
        if not use_bm25 and not use_semantic:
            logger.warning("Both BM25 and semantic search disabled")
            return []
        
        if not query.strip():
            logger.warning("Empty search query")
            return []
        
        logger.info(f"Hybrid search: '{query[:50]}...' (BM25: {self.bm25_weight:.2f}, Semantic: {self.semantic_weight:.2f})")
        
        # Perform searches in parallel
        bm25_results = []
        semantic_results = []
        
        if use_bm25 and self.bm25_indexer.doc_count > 0:
            try:
                bm25_results = self.bm25_indexer.search(query, language, limit * 2)
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
                bm25_results = []
        
        if use_semantic and self.faiss_index and self.faiss_index.count() > 0:
            try:
                # This will be called from the embedding system
                semantic_results = self._semantic_search_wrapper(query, language, limit * 2)
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                semantic_results = []
        
        # Combine results
        combined_results = self._combine_results(
            bm25_results, semantic_results, query, language, limit
        )
        
        if return_raw_scores:
            # Add raw scores to results for debugging
            for result in combined_results:
                result.bm25_score = self._get_bm25_score_for_doc(result.id, bm25_results)
                result.semantic_score = self._get_semantic_score_for_doc(result.id, semantic_results)
        
        logger.info(f"Hybrid search returned {len(combined_results)} results")
        return combined_results

    def _semantic_search_wrapper(
        self, query: str, language: Optional[str], limit: int
    ) -> List[SearchResult]:
        """Wrapper for semantic search that can be called from thread pool."""
        # This would typically be called from the embeddings system
        # For now, return empty list - this will be integrated properly
        logger.debug("Semantic search called (wrapper)")
        return []

    def _get_bm25_score_for_doc(self, doc_id: int, bm25_results: List[BM25SearchResult]) -> float:
        """Get BM25 score for a specific document."""
        for result in bm25_results:
            if result.id == doc_id:
                return result.score
        return 0.0

    def _get_semantic_score_for_doc(self, doc_id: int, semantic_results: List[SearchResult]) -> float:
        """Get semantic score for a specific document."""
        for result in semantic_results:
            if result.id == doc_id:
                return result.score
        return 0.0

    def _combine_results(
        self,
        bm25_results: List[BM25SearchResult],
        semantic_results: List[SearchResult],
        query: str,
        language: Optional[str],
        limit: int
    ) -> List[HybridSearchResult]:
        """Combine BM25 and semantic search results using weighted scoring."""
        
        # Create result maps
        bm25_scores: Dict[int, Tuple[float, List[str]]] = {}
        semantic_scores: Dict[int, float] = {}
        
        # Process BM25 results
        for result in bm25_results:
            bm25_scores[result.id] = (result.score, result.terms)
        
        # Process semantic results
        for result in semantic_results:
            semantic_scores[result.id] = result.score
        
        # Get all document IDs
        all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Calculate combined scores
        combined_scores: Dict[int, HybridSearchResult] = {}
        
        for doc_id in all_doc_ids:
            bm25_score, matched_terms = bm25_scores.get(doc_id, (0.0, []))
            semantic_score = semantic_scores.get(doc_id, 0.0)
            
            # Apply weights
            combined_score = (
                self.bm25_weight * bm25_score +
                self.semantic_weight * semantic_score
            )
            
            # Skip if below threshold
            if combined_score < self.min_score_threshold:
                continue
            
            # Determine result type
            result_type = "hybrid"
            if bm25_score > 0 and semantic_score == 0:
                result_type = "bm25_only"
            elif semantic_score > 0 and bm25_score == 0:
                result_type = "semantic_only"
            
            # Get document info (prefer BM25 results as they have more info)
            if doc_id in bm25_results:
                bm25_result = bm25_results[doc_id]
                combined_scores[doc_id] = HybridSearchResult(
                    id=doc_id,
                    file_path=bm25_result.file_path,
                    start_line=bm25_result.start_line,
                    end_line=bm25_result.end_line,
                    code=bm25_result.code,
                    language=bm25_result.language,
                    score=combined_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    matched_terms=matched_terms,
                    result_type=result_type
                )
            elif doc_id in semantic_results:
                semantic_result = semantic_results[doc_id]
                combined_scores[doc_id] = HybridSearchResult(
                    id=doc_id,
                    file_path=semantic_result.file_path,
                    start_line=semantic_result.start_line,
                    end_line=semantic_result.end_line,
                    code=semantic_result.code,
                    language=semantic_result.language,
                    score=combined_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    matched_terms=[],
                    result_type=result_type
                )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results[:limit]

    def get_search_explanation(
        self, query: str, results: List[HybridSearchResult]
    ) -> Dict[str, Any]:
        """Generate explanation for search results (for debugging)."""
        if not self.enable_explanation:
            return {}
        
        query_tokens = self.bm25_indexer._tokenize(query.lower())
        
        explanation = {
            "query": query,
            "query_tokens": query_tokens,
            "weights": {
                "bm25_weight": self.bm25_weight,
                "semantic_weight": self.semantic_weight
            },
            "results": []
        }
        
        for i, result in enumerate(results[:10]):  # Top 10 results
            result_explanation = {
                "rank": i + 1,
                "file_path": result.file_path,
                "combined_score": round(result.score, 4),
                "bm25_score": round(result.bm25_score, 4),
                "semantic_score": round(result.semantic_score, 4),
                "result_type": result.result_type,
                "matched_terms": result.matched_terms
            }
            
            if self.enable_explanation:
                # Add detailed breakdown
                bm25_contribution = self.bm25_weight * result.bm25_score
                semantic_contribution = self.semantic_weight * result.semantic_score
                
                result_explanation["score_breakdown"] = {
                    "bm25_contribution": round(bm25_contribution, 4),
                    "semantic_contribution": round(semantic_contribution, 4),
                    "bm25_percentage": round(bm25_contribution / result.score * 100, 1) if result.score > 0 else 0,
                    "semantic_percentage": round(semantic_contribution / result.score * 100, 1) if result.score > 0 else 0
                }
            
            explanation["results"].append(result_explanation)
        
        return explanation

    def set_weights(self, bm25_weight: float, semantic_weight: float) -> None:
        """Update search weights."""
        with self._lock:
            total_weight = bm25_weight + semantic_weight
            if total_weight == 0:
                logger.warning("Both weights are zero, keeping current weights")
                return
            
            self.bm25_weight = bm25_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight
            
            logger.info(f"Updated weights - BM25: {self.bm25_weight:.3f}, Semantic: {self.semantic_weight:.3f}")

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics."""
        bm25_stats = self.bm25_indexer.get_stats()
        faiss_stats = self.faiss_index.get_status() if self.faiss_index else {"count": 0}
        
        return {
            "bm25": {
                "enabled": self.bm25_indexer.doc_count > 0,
                **bm25_stats
            },
            "semantic": {
                "enabled": self.faiss_index and self.faiss_index.count() > 0,
                **faiss_stats
            },
            "weights": {
                "bm25_weight": self.bm25_weight,
                "semantic_weight": self.semantic_weight
            },
            "min_score_threshold": self.min_score_threshold
        }


class HybridSearchManager:
    """Manager for hybrid search with configuration and lifecycle management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid search manager."""
        self.config = config
        self.bm25_indexer: Optional[BM25Indexer] = None
        self.faiss_index: Optional[FaissIndex] = None
        self.hybrid_search: Optional[HybridSearch] = None
        
        self._lock = threading.Lock()

    def initialize(
        self,
        db_path: str,
        index_path: str,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> bool:
        """Initialize hybrid search components."""
        with self._lock:
            try:
                # Initialize BM25 indexer
                self.bm25_indexer = BM25Indexer(db_path, f"{index_path}.bm25")
                logger.info("BM25 indexer initialized")
                
                # Initialize FAISS index (will be set externally)
                self.faiss_index = None
                
                # Initialize hybrid search (FAISS index will be set later)
                self.hybrid_search = HybridSearch(
                    self.bm25_indexer,
                    None,  # Will be set via set_faiss_index
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight
                )
                
                logger.info("Hybrid search manager initialized")
                return True
                
            except Exception as e:
                logger.exception(f"Failed to initialize hybrid search: {e}")
                return False

    def set_faiss_index(self, faiss_index: FaissIndex) -> None:
        """Set the FAISS index after initialization."""
        with self._lock:
            self.faiss_index = faiss_index
            if self.hybrid_search:
                self.hybrid_search.faiss_index = faiss_index

    def build_bm25_index(self, chunk_limit: Optional[int] = None) -> Dict[str, Any]:
        """Build BM25 index from database."""
        if not self.bm25_indexer:
            raise ValueError("BM25 indexer not initialized")
        
        logger.info("Building BM25 index for hybrid search")
        return self.bm25_indexer.build_index(chunk_limit)

    def search(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 10,
        use_bm25: bool = True,
        use_semantic: bool = True,
        return_explanation: bool = False
    ) -> Tuple[List[HybridSearchResult], Optional[Dict[str, Any]]]:
        """Perform hybrid search with optional explanation."""
        if not self.hybrid_search:
            raise ValueError("Hybrid search not initialized")
        
        results = self.hybrid_search.search(
            query, language, limit, use_bm25, use_semantic
        )
        
        explanation = None
        if return_explanation:
            explanation = self.hybrid_search.get_search_explanation(query, results)
        
        return results, explanation

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics."""
        if not self.hybrid_search:
            return {"error": "Hybrid search not initialized"}
        
        return self.hybrid_search.get_stats()

    def set_weights(self, bm25_weight: float, semantic_weight: float) -> None:
        """Update search weights."""
        if self.hybrid_search:
            self.hybrid_search.set_weights(bm25_weight, semantic_weight)

    def clear(self) -> None:
        """Clear all indices."""
        with self._lock:
            if self.bm25_indexer:
                self.bm25_indexer.clear()
            
            if self.hybrid_search:
                self.hybrid_search = None
            
            self.bm25_indexer = None
            self.faiss_index = None
            
            logger.info("Hybrid search cleared")