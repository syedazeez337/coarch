"""Optimized embedding service with ONNX Runtime and quantization support."""

import os
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    batch_size: int = 32
    use_quantization: bool = True
    num_threads: int = 0  # 0 = auto-detect
    use_gpu: bool = False


class OptimizedEmbedder:
    """High-performance embedding generator with ONNX + quantization."""

    def __init__(
        self, config: Optional[EmbeddingConfig] = None, onnx_path: Optional[str] = None
    ):
        """Initialize the optimized embedder.

        Args:
            config: Embedding configuration
            onnx_path: Path to ONNX model (auto-converts if not exists)
        """
        self.config = config or EmbeddingConfig()
        self.onnx_path = onnx_path or f"{self.config.model_name.replace('/', '_')}.onnx"
        self.session = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        """Initialize tokenizer and ONNX session."""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError:
            self._fallback_to_pytorch()
            return

        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        print(f"Converting to ONNX: {self.onnx_path}")
        if os.path.exists(self.onnx_path):
            print("Using cached ONNX model")
        else:
            model = ORTModelForSequenceClassification.from_pretrained(
                self.config.model_name, export=True, opset=13
            )
            model.save_pretrained(self.onnx_path)
            self.tokenizer.save_pretrained(self.onnx_path)

        self._init_onnx_session()

    def _init_onnx_session(self):
        """Initialize ONNX Runtime session with optimizations."""
        try:
            import onnxruntime as ort
        except ImportError:
            self._fallback_to_pytorch()
            return

        providers = (
            ["CUDAExecutionProvider"]
            if self.config.use_gpu
            else ["CPUExecutionProvider"]
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        if self.config.num_threads > 0:
            session_options.intra_op_num_threads = self.config.num_threads
            session_options.inter_op_num_threads = self.config.num_threads

        print(f"Loading ONNX model from: {self.onnx_path}")
        self.session = ort.InferenceSession(
            self.onnx_path, session_options, providers=providers
        )

        self.providers = self.session.get_providers()
        print(f"Execution providers: {self.providers}")

    def _fallback_to_pytorch(self):
        """Fallback to PyTorch if ONNX is not available."""
        print("ONNX not available, using PyTorch (slower)")
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.pytorch_model = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.eval()

        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        print(f"Using device: {device}")

    def quantize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert FP32 embeddings to int8 for faster storage/search.

        Args:
            embeddings: FP32 numpy array

        Returns:
            Quantized int8 array
        """
        if not self.config.use_quantization:
            return embeddings

        if embeddings.dtype == np.int8:
            return embeddings

        qmin, qmax = 0, 255
        input_range = np.abs(embeddings).max()
        if input_range == 0:
            input_range = 1.0

        scale = input_range / (qmax - qmin)
        zero_point = int((qmin + qmax) / 2)

        quantized = ((embeddings / scale) + zero_point).astype(np.int8)

        return quantized

    def dequantize_embeddings(
        self, quantized: np.ndarray, scale: float, zero_point: int
    ) -> np.ndarray:
        """Convert int8 embeddings back to FP32.

        Args:
            quantized: Quantized int8 array
            scale: Quantization scale
            zero_point: Zero point

        Returns:
            FP32 numpy array
        """
        return ((quantized.astype(np.float32) - zero_point) * scale).astype(np.float32)

    def _tokenize_batch(self, texts: List[str]) -> dict:
        """Tokenize a batch of texts efficiently.

        Args:
            texts: List of code strings

        Returns:
            Tokenized inputs dict
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="np",
        )

    def embed_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of code strings

        Returns:
            Tuple of (embeddings, quantization_scales)
        """
        if not texts:
            return np.array([]), np.array([])

        batch_size = len(texts)
        start_time = time.time()

        inputs = self._tokenize_batch(texts)

        if hasattr(self, "session") and self.session is not None:
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            ort_outputs = self.session.run(None, ort_inputs)
            embeddings = ort_outputs[0][:, 0, :]

        else:
            import torch

            with torch.no_grad():
                input_ids = torch.tensor(inputs["input_ids"]).to(self.device)
                attention_mask = torch.tensor(inputs["attention_mask"]).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        elapsed = time.time() - start_time
        print(
            f"   Batch ({batch_size}): {elapsed*1000:.1f}ms ({elapsed*1000/batch_size:.1f}ms per item)"
        )

        if self.config.use_quantization:
            scales = np.zeros(len(texts))
            for i in range(len(texts)):
                scales[i] = (
                    np.abs(embeddings[i]).max()
                    if np.abs(embeddings[i]).max() > 0
                    else 1.0
                )
            embeddings = self.quantize_embeddings(embeddings)
        else:
            scales = np.array([])

        return embeddings, scales

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of code strings
            show_progress: Show progress bar

        Returns:
            FP32 numpy array of embeddings
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        all_scales = []

        batch_size = self.config.batch_size
        n_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"Processing {len(texts)} texts in {n_batches} batches...")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            embeddings, scales = self.embed_batch(batch)
            all_embeddings.append(embeddings)

            if len(scales) > 0:
                all_scales.append(scales)

        embeddings = np.vstack(all_embeddings)

        if len(all_scales) > 0:
            scales = np.concatenate(all_scales)
            return embeddings, scales

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Code string

        Returns:
            Embedding vector
        """
        embedding, _ = self.embed_batch([text])
        return embedding[0] if len(embedding) > 0 else np.array([])

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return 768

    def get_memory_usage(self) -> dict:
        """Get memory usage information."""
        import psutil

        process = psutil.Process()
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
        }


class FastFaissIndex:
    """Optimized FAISS index with quantization support."""

    def __init__(self, dim: int = 768, use_gpu: bool = False, use_ivf: bool = False):
        """Initialize the FAISS index.

        Args:
            dim: Embedding dimension
            use_gpu: Use GPU acceleration
            use_ivf: Use IVF index for large datasets
        """
        import faiss

        self.dim = dim
        self.use_gpu = use_gpu
        self.use_ivf = use_ivf

        if use_ivf and dim <= 1024:
            nlist = 100
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

        self.quantized_data = None
        self.quantization_scales = None
        self.is_trained = False

    def train(self, embeddings: np.ndarray):
        """Train the index (required for IVF)."""
        if hasattr(self.index, "train"):
            self.index.train(embeddings.astype(np.float32))
            self.is_trained = True

    def add(self, embeddings: np.ndarray, scales: Optional[np.ndarray] = None):
        """Add embeddings to the index.

        Args:
            embeddings: FP32 or int8 embeddings
            scales: Quantization scales (if embeddings are quantized)
        """
        import faiss

        if embeddings.dtype == np.int8:
            self.quantized_data = embeddings
            self.quantization_scales = scales
            embeddings = self._dequantize_all(embeddings, scales)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))

    def _dequantize_all(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize all embeddings."""
        dequantized = np.zeros((len(quantized), quantized.shape[1]), dtype=np.float32)
        for i in range(len(quantized)):
            if scales is not None and len(scales) > i:
                scale = scales[i] if scales[i] > 0 else 1.0
            else:
                scale = 1.0
            dequantized[i] = quantized[i].astype(np.float32) * scale
        return dequantized

    def search(
        self, query: np.ndarray, k: int = 10, nprobe: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings.

        Args:
            query: Query embedding
            k: Number of results
            nprobe: Number of probes (for IVF)

        Args:
            Tuple of (scores, ids)
        """
        import faiss

        if self.use_ivf and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        if query.dtype == np.int8:
            query = query.astype(np.float32)

        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        faiss.normalize_L2(query)
        return self.index.search(query.astype(np.float32), k)

    def save(self, path: str):
        """Save the index to disk."""
        import faiss
        import pickle

        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}.meta", "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "use_gpu": self.use_gpu,
                    "use_ivf": self.use_ivf,
                    "quantized_data": self.quantized_data,
                    "quantization_scales": self.quantization_scales,
                    "is_trained": self.is_trained,
                },
                f,
            )

    def load(self, path: str):
        """Load the index from disk."""
        import faiss
        import pickle

        self.index = faiss.read_index(f"{path}.faiss")

        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)
            self.dim = meta["dim"]
            self.use_gpu = meta["use_gpu"]
            self.quantized_data = meta.get("quantized_data")
            self.quantization_scales = meta.get("quantization_scales")
            self.is_trained = meta.get("is_trained", True)

    def count(self) -> int:
        """Return the number of embeddings."""
        return self.index.ntotal


class ParallelIndexer:
    """Parallel chunking and indexing with multiprocessing."""

    def __init__(self, n_workers: int = 0):
        """Initialize the parallel indexer.

        Args:
            n_workers: Number of workers (0 = auto)
        """
        import multiprocessing

        self.n_workers = n_workers or multiprocessing.cpu_count()

    def process_files(self, files: List[str]) -> List[dict]:
        """Process multiple files in parallel.

        Args:
            files: List of file paths

        Returns:
            List of chunk dictionaries
        """
        from tqdm import tqdm
        from multiprocessing import Pool, cpu_count

        n_workers = min(self.n_workers, cpu_count())

        if n_workers <= 1 or len(files) < 10:
            return self._process_files_sequential(files)

        print(f"Processing {len(files)} files with {n_workers} workers...")

        with Pool(n_workers) as pool:
            chunks = list(
                tqdm(
                    pool.imap(self._process_single_file, files),
                    total=len(files),
                    desc="Indexing",
                )
            )

        return [c for sublist in chunks for c in sublist]

    def _process_single_file(self, file_path: str) -> List[dict]:
        """Process a single file."""
        from backend.indexer import RepositoryIndexer
        from backend.ast_analyzer import TreeSitterAnalyzer

        indexer = RepositoryIndexer(db_path=":memory:")
        analyzer = TreeSitterAnalyzer()

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except Exception:
            return []

        language = indexer.get_language(file_path)
        if not language:
            return []

        chunks = indexer.extract_code_chunks(file_path, code)

        result = []
        for chunk in chunks:
            result.append(
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "code": chunk.code,
                    "language": chunk.language,
                    "symbols": chunk.symbols,
                }
            )

        return result

    def _process_files_sequential(self, files: List[str]) -> List[dict]:
        """Process files sequentially (fallback)."""
        from tqdm import tqdm

        all_chunks = []
        for file_path in tqdm(files, desc="Indexing"):
            chunks = self._process_single_file(file_path)
            all_chunks.extend(chunks)

        return all_chunks
