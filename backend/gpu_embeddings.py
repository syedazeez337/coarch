"""GPU embedding support for Coarch."""

import os
from typing import Optional
import numpy as np


def check_gpu_availability() -> dict:
    """Check GPU availability and return info."""
    info = {
        "cuda_available": False,
        "rocm_available": False,
        "device_count": 0,
        "device_name": None,
        "memory_total_mb": 0,
        "driver_version": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            info["driver_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            info["rocm_available"] = True
            info["device_count"] = torch.hip.device_count()
    except:
        pass

    return info


class GPUEmbedder:
    """GPU-accelerated embedding generation."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str = "cuda",
        batch_size: int = 64
    ):
        """Initialize GPU embedder.

        Args:
            model_name: HuggingFace model name
            device: "cuda" or "cpu"
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        gpu_info = check_gpu_availability()
        if device == "cuda" and not gpu_info["cuda_available"]:
            print("[WARN] CUDA not available, falling back to CPU")
            self.device = "cpu"

        self._init_model()

    def _init_model(self):
        """Initialize model on GPU."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            print(f"[ERROR] PyTorch required for GPU support: {e}")
            raise

        print(f"Loading model {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            self.model = torch.compile(self.model)
            print("[OK] Model compiled with torch.compile()")

    def embed(self, texts: list, show_progress: bool = False) -> np.ndarray:
        """Generate embeddings on GPU.

        Args:
            texts: List of code strings
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        import torch
        from tqdm import tqdm

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

            if show_progress:
                print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings)

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name)
        return config.hidden_size

    def optimize_for_inference(self):
        """Apply optimization techniques for faster inference."""
        import torch

        self.model.eval()

        torch.backends.cudnn.benchmark = True

        if hasattr(self.model, 'encoder'):
            self.model.encoder.layer = torch.nn.ModuleList(
                [layer.half() for layer in self.model.encoder.layer]
            )

        print("[OK] Model optimized for inference")

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        import torch

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "device_name": torch.cuda.get_device_name(0)
        }


class GPUIndex:
    """GPU-accelerated FAISS index."""

    def __init__(self, dim: int = 768):
        """Initialize GPU index.

        Args:
            dim: Embedding dimension
        """
        self.dim = dim
        self.index = None
        self._init_index()

    def _init_index(self):
        """Initialize GPU index."""
        try:
            import faiss
            res = faiss.StandardGpuResources()
            self.index = faiss.IndexFlatIP(self.dim)
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("[OK] GPU index initialized")
        except Exception as e:
            print(f"[WARN] GPU index not available: {e}, using CPU")
            import faiss
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray):
        """Add embeddings to GPU index.

        Args:
            embeddings: numpy array of shape (n, dim)
        """
        import faiss
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: np.ndarray, k: int = 10) -> tuple:
        """Search for similar embeddings.

        Args:
            query: Query embedding
            k: Number of results

        Returns:
            Tuple of (scores, ids)
        """
        import faiss
        faiss.normalize_L2(query.reshape(1, -1))
        return self.index.search(query.reshape(1, -1).astype(np.float32), k)

    def count(self) -> int:
        """Return number of embeddings."""
        return self.index.ntotal

    def save(self, path: str):
        """Save index to disk."""
        import faiss
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load index from disk."""
        import faiss
        self.index = faiss.read_index(path)
