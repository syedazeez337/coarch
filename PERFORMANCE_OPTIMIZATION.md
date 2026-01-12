# Coarch Performance Optimization Guide

## Current Performance Baseline

| Operation | Time | Bottleneck |
|-----------|------|------------|
| File scanning | ~1ms/file | I/O |
| Chunk creation | ~0.1ms/file | Python |
| Symbol extraction | ~1ms/file | Tree-sitter |
| **Embedding generation** | **~50-100ms/chunk** | **Transformer model** |
| FAISS index update | ~0.01ms/vector | C++ |

**Total: ~100ms/file → To achieve 1ms, need 100x speedup**

## Optimization Strategies

### 1. Smaller Embedding Models (Easiest)

Replace `microsoft/codebert-base` (768D, 120M params) with faster alternatives:

```python
# Current (slow)
model_name = "microsoft/codebert-base"  # 120M params

# Faster alternatives
model_name = "microsoft/graphcodebert-base"  # 125M params
model_name = "neulab/codebert-small"  # 30M params
model_name = "gaussiage/TinyBERT-for-code"  # 4.5M params
```

**Expected speedup: 3-10x**

### 2. Quantization (Easy)

Use 8-bit or 4-bit quantization:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# Load quantized model
model = ORTModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    export=True,
    load_format="pt",  # or "dq" for dynamic quantization
)
```

**Expected speedup: 2-4x**

### 3. ONNX Runtime (Medium)

Already in codebase (`backend/optimized_embeddings.py`). Benefits:

```python
# Current: PyTorch (~100ms/chunk)
from transformers import AutoModel

# With ONNX (~20ms/chunk)
from optimum.onnxruntime import ORTModelForSequenceClassification
```

**Expected speedup: 4-5x**

### 4. Incremental Indexing (Medium)

Only re-index changed files:

```python
# Track file mtimes
indexed_files = {
    "src/auth.py": 1704067200.0,  # mtime
    "src/parser.py": 1704067200.0,
}

def incremental_index(repo_path):
    for file_path in scan_files(repo_path):
        mtime = os.path.getmtime(file_path)
        if mtime > indexed_files.get(file_path, 0):
            index_file(file_path)
            indexed_files[file_path] = mtime
```

**Speedup: Depends on change rate (10-100x for small changes)**

### 5. Multi-threaded Indexing (Medium)

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_index(files, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(index_file, files))
```

**Expected speedup: 4-8x on 8-core CPU**

### 6. Cached Embeddings (Easy)

```python
# Store embeddings in SQLite
def get_embedding(code_hash):
    return db.query("SELECT embedding FROM embeddings WHERE hash=?", code_hash)

def index_file(file_path):
    code_hash = hash(code)
    if cached := get_embedding(code_hash):
        return cached
    
    embedding = generate_embedding(code)
    db.execute("INSERT INTO embeddings VALUES (?, ?)", code_hash, embedding)
    return embedding
```

**Speedup: Near-instant for duplicate/similar code**

### 7. GPU Acceleration (Harder)

```python
embedder = CodeEmbedder(
    model_name="microsoft/codebert-base",
    device="cuda"  # Use GPU
)
```

**Expected speedup: 10-50x for embedding generation**

### 8. Rust/C++ Core (Hardest)

Rewrite indexing pipeline in Rust for 10-100x speedup:

```rust
// rust-indexer/src/main.rs
fn main() {
    let files = scan_directory("./src");
    let chunks = files.par_iter()
        .map(|f| chunk_file(f))
        .flat_map(|c| c)
        .collect::<Vec<_>>();
    
    let embeddings = generate_embeddings_parallel(&chunks);
    faiss.add(&embeddings);
}
```

**Expected speedup: 20-100x overall**

## Recommended Approach

### Phase 1: Quick Wins (1 day)
- [ ] Use smaller model (`neulab/codebert-small`)
- [ ] Add caching for embeddings
- [ ] Enable incremental indexing

### Phase 2: Medium Effort (1 week)
- [ ] Implement ONNX runtime fully
- [ ] Add multi-threaded indexing
- [ ] Quantize model to 8-bit

### Phase 3: Long-term (1 month)
- [ ] Rewrite core in Rust
- [ ] Add GPU support
- [ ] Implement streaming embeddings

## Performance Targets

| Target | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Embedding/chunk | 100ms | 20ms | 5ms | 1ms |
| 10K file index | 1000s | 200s | 50s | 10s |
| Incremental (100 files) | 10s | 2s | 0.5s | 0.1s |

## Language Comparison

| Language | Indexing Speed | Ecosystem | Difficulty |
|----------|---------------|-----------|------------|
| Python | Baseline | Excellent | Easy |
| PyPy | 1.5x faster | Good | Easy |
| Cython | 5-10x faster | Medium | Medium |
| Rust | 20-100x faster | Growing | Hard |
| Go | 10-30x faster | Excellent | Medium |
| C++ | 20-100x fastest | Complex | Hard |

**Recommendation**: Start with Python + Cython for hot paths, then Rust for core indexing.
