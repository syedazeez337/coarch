# Coarch Critical Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs and performance issues to make coarch production-ready.

**Architecture:** Fix the Rust indexer to return ALL chunks (not just first), optimize embeddings with batching/caching/ONNX, fix security vulnerabilities, and make async handlers non-blocking.

**Tech Stack:** Rust/PyO3, Python/FastAPI, FAISS, ONNX Runtime, CodeBERT

---

## Priority Summary

| Priority | Issue | Impact |
|----------|-------|--------|
| P0 | Rust indexer only returns first chunk | 90% of code unsearchable |
| P0 | Embeddings slow (5min/1000 chunks) | Unusable for large repos |
| P0 | JWT secret in plaintext config | Security breach |
| P0 | Salt derived from key | Rainbow table vulnerable |
| P1 | Async handlers block event loop | All requests slow down |
| P1 | No file size limits in embeddings | OOM crashes |

---

## Task 1: Fix Rust Indexer Chunking Bug

**Files:**
- Modify: `rust-indexer/src/lib.rs:130-151`
- Test: Manual test with multi-chunk file

**Step 1: Fix the index_file function to return Vec<CodeChunk>**

Replace the `index_file` function (lines 130-151) with:

```rust
fn index_file(&self, file_path: &Path, repo_path: &Path) -> Vec<CodeChunk> {
    // Check file size
    let metadata = match file_path.metadata() {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    if metadata.len() > MAX_FILE_SIZE as u64 {
        return Vec::new();
    }

    let rel_path = file_path.strip_prefix(repo_path)
        .unwrap_or(file_path)
        .to_string_lossy()
        .into_owned();

    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let language = self.detect_language(file_path);
    let symbols = self.extract_symbols(&content);
    let ast_hash = self.hash_code(&content);
    let lines: Vec<&str> = content.lines().collect();

    let mut chunks = Vec::new();
    let mut chunk_start = 0;

    while chunk_start < lines.len() {
        let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, lines.len());
        let chunk_lines = &lines[chunk_start..chunk_end];
        let chunk_code: String = chunk_lines.join("\n");

        if chunk_code.len() >= MIN_CHUNK_SIZE {
            chunks.push(CodeChunk {
                file_path: rel_path.clone(),
                start_line: chunk_start + 1,
                end_line: chunk_end,
                code: chunk_code,
                language: language.clone(),
                symbols: symbols.clone(),
                ast_hash: ast_hash.clone(),
            });
        }

        // Move forward with overlap
        chunk_start += CHUNK_SIZE - OVERLAP;
    }

    chunks
}
```

**Step 2: Update index_directory to handle Vec<CodeChunk>**

Change line 55 from:
```rust
let chunks: Vec<CodeChunk> = files.par_iter().filter_map(|f| self.index_file(f, &repo_path)).collect();
```

To:
```rust
let chunks: Vec<CodeChunk> = files.par_iter()
    .flat_map(|f| self.index_file(f, &repo_path))
    .collect();
```

**Step 3: Rebuild the Rust extension**

```bash
cd rust-indexer
cargo build --release
cp target/release/coarch_rust.dll ../backend/  # Windows
# OR: cp target/release/libcoarch_rust.so ../backend/coarch_rust.so  # Linux
```

**Step 4: Test with a large file**

Create test file and verify multiple chunks are created:
```bash
cd coarch
python -c "
from backend.hybrid_indexer import HybridIndexer
import tempfile, os
# Create a 200-line test file
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    for i in range(200):
        f.write(f'def function_{i}(): pass\n')
    path = f.name
indexer = HybridIndexer(':memory:')
# Should create 4+ chunks (200 lines / 50 lines per chunk)
"
```

**Step 5: Commit**

```bash
git add rust-indexer/src/lib.rs backend/coarch_rust.dll
git commit -m "fix(rust): return all chunks instead of only first chunk per file

BREAKING: index_file now returns Vec<CodeChunk> instead of Option<CodeChunk>
This fixes critical bug where 90%+ of code was not being indexed."
```

---

## Task 2: Optimize Embedding Generation

**Files:**
- Modify: `backend/embeddings.py`
- Create: `backend/embedding_cache.py`

**Step 1: Add embedding cache**

Create `backend/embedding_cache.py`:

```python
"""LRU cache for code embeddings to avoid recomputation."""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Disk-backed LRU cache for embeddings."""

    def __init__(self, cache_dir: str = "~/.coarch/embedding_cache", max_size_mb: int = 500):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "index.pkl"
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.index = pickle.load(f)
            except Exception:
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)

    def _hash_code(self, code: str) -> str:
        """Generate hash for code snippet."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def get(self, code: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_code(code)
        if key in self.index:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    return np.load(cache_file)
                except Exception:
                    del self.index[key]
        return None

    def put(self, code: str, embedding: np.ndarray):
        """Cache an embedding."""
        key = self._hash_code(code)
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)
        self.index[key] = cache_file.stat().st_size
        self._save_index()
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict old entries if cache exceeds max size."""
        total_size = sum(self.index.values())
        if total_size > self.max_size_bytes:
            # Remove oldest 20% of entries
            to_remove = list(self.index.keys())[:len(self.index) // 5]
            for key in to_remove:
                cache_file = self.cache_dir / f"{key}.npy"
                cache_file.unlink(missing_ok=True)
                del self.index[key]
            self._save_index()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self.index),
            "size_mb": sum(self.index.values()) / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
        }
```

**Step 2: Update embeddings.py with caching and larger batches**

Modify `backend/embeddings.py` - update the `embed` method:

```python
def embed(
    self,
    code_snippets: List[str],
    batch_size: int = 64,  # Increased from 32
    show_progress: bool = False,
    use_cache: bool = True,
) -> np.ndarray:
    """Generate embeddings for code snippets with caching."""
    if not code_snippets:
        logger.warning("Empty code snippets provided to embed()")
        return np.array([])

    total = len(code_snippets)
    logger.info(f"Generating embeddings for {total} snippets")

    # Check cache first
    if use_cache and self.cache:
        cached_embeddings = {}
        uncached_indices = []
        uncached_snippets = []

        for i, snippet in enumerate(code_snippets):
            cached = self.cache.get(snippet)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                uncached_indices.append(i)
                uncached_snippets.append(snippet)

        logger.info(f"Cache hit: {len(cached_embeddings)}, miss: {len(uncached_snippets)}")

        if not uncached_snippets:
            # All cached
            return np.array([cached_embeddings[i] for i in range(total)])

        # Compute uncached embeddings
        new_embeddings = self._embed_uncached(uncached_snippets, batch_size)

        # Store in cache
        for i, embedding in enumerate(new_embeddings):
            self.cache.put(uncached_snippets[i], embedding)

        # Reconstruct full result
        result = np.zeros((total, self.get_dimension()), dtype=np.float32)
        for i, embedding in cached_embeddings.items():
            result[i] = embedding
        for i, idx in enumerate(uncached_indices):
            result[idx] = new_embeddings[i]

        return result
    else:
        return self._embed_uncached(code_snippets, batch_size)

def _embed_uncached(self, code_snippets: List[str], batch_size: int) -> np.ndarray:
    """Embed snippets without cache lookup."""
    embeddings: List[np.ndarray] = []

    for i in range(0, len(code_snippets), batch_size):
        batch = code_snippets[i : i + batch_size]
        try:
            batch_embeddings = self._embed_batch(batch)
            embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch starting at {i}: {e}")
            raise

    result = np.vstack(embeddings)
    logger.info(f"Generated embeddings with shape: {result.shape}")
    return result
```

**Step 3: Update __init__ to initialize cache**

Add to `CodeEmbedder.__init__`:

```python
from .embedding_cache import EmbeddingCache

# In __init__, after self._init_model():
self.cache = EmbeddingCache()
```

**Step 4: Commit**

```bash
git add backend/embeddings.py backend/embedding_cache.py
git commit -m "perf(embeddings): add LRU disk cache and larger batch sizes

- Cache embeddings to disk to avoid recomputation
- Increase default batch size from 32 to 64
- Add cache hit/miss logging"
```

---

## Task 3: Fix Security - JWT Secret Storage

**Files:**
- Modify: `backend/config.py:116-117`

**Step 1: Remove JWT secret from config file persistence**

In `backend/config.py`, modify the `save` method to exclude sensitive fields:

Find this code (around line 116):
```python
config_data = {
    ...
    "jwt_secret": self.jwt_secret,
    ...
}
```

Replace the save method with:

```python
def save(self, path: Optional[str] = None) -> None:
    """Save config to file, excluding sensitive values."""
    path = path or self.get_default_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # SECURITY: Never persist secrets to disk
    config_data = {
        "version": self.version,
        "index_path": self.index_path,
        "db_path": self.db_path,
        "model_name": self.model_name,
        "server_host": self.server_host,
        "server_port": self.server_port,
        "rate_limit_per_minute": self.rate_limit_per_minute,
        "enable_auth": self.enable_auth,
        "cors_origins": self.cors_origins,
        "log_level": self.log_level,
        "log_json": self.log_json,
        # NOTE: api_key_hash and jwt_secret are NOT saved
        # They must be provided via environment variables
    }

    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)

    # Set restrictive permissions (Unix only)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows doesn't support chmod

    logger.info(f"Config saved to {path} (secrets excluded)")
```

**Step 2: Update load to get secrets from env only**

Update the config loading to require env vars for secrets:

```python
# In load() or __init__:
self.jwt_secret = os.environ.get("COARCH_JWT_SECRET")
if self.enable_auth and not self.jwt_secret:
    logger.warning("COARCH_JWT_SECRET not set - generating random secret (will change on restart)")
    import secrets
    self.jwt_secret = secrets.token_hex(32)
```

**Step 3: Commit**

```bash
git add backend/config.py
git commit -m "security(config): never persist JWT secrets to disk

BREAKING: JWT secret must now be provided via COARCH_JWT_SECRET env var
Secrets are no longer saved to config.json to prevent exposure"
```

---

## Task 4: Fix Security - Salt Generation

**Files:**
- Modify: `backend/security.py:292-295`

**Step 1: Fix hash_api_key with proper random salt**

Replace the `hash_api_key` function:

```python
import os
import base64

def hash_api_key(api_key: str) -> str:
    """Hash API key with random salt using scrypt.

    Returns: base64-encoded string containing salt + hash
    """
    # Generate random salt (16 bytes)
    salt = os.urandom(16)

    # Hash with scrypt
    key_hash = hashlib.scrypt(
        api_key.encode(),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=32
    )

    # Return salt + hash as base64
    return base64.b64encode(salt + key_hash).decode()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify API key against stored hash."""
    try:
        decoded = base64.b64decode(stored_hash.encode())
        salt = decoded[:16]
        stored_key_hash = decoded[16:]

        # Recompute hash with same salt
        computed_hash = hashlib.scrypt(
            api_key.encode(),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=32
        )

        # Constant-time comparison
        return hmac.compare_digest(computed_hash, stored_key_hash)
    except Exception:
        return False
```

**Step 2: Update any code that calls hash_api_key**

Search for usages and update verification logic to use `verify_api_key`.

**Step 3: Commit**

```bash
git add backend/security.py
git commit -m "security(auth): use random salt for API key hashing

BREAKING: Old hashed API keys will no longer verify
- Use os.urandom() for salt instead of deriving from key
- Add verify_api_key() for constant-time comparison
- Store salt alongside hash in base64 format"
```

---

## Task 5: Add File Size Limits to Embeddings

**Files:**
- Modify: `backend/embeddings.py:135-151`

**Step 1: Add size check to embed_file**

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def embed_file(self, file_path: str) -> np.ndarray:
    """Generate embedding for a file.

    Args:
        file_path: Path to file

    Returns:
        numpy array of shape (dim,)
    """
    try:
        # Check file size BEFORE reading
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large ({file_size} bytes): {file_path}")
            return np.zeros(self.get_dimension(), dtype=np.float32)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return np.zeros(self.get_dimension(), dtype=np.float32)

    return self.embed([code])[0]
```

**Step 2: Add import for os at top of file**

```python
import os
```

**Step 3: Commit**

```bash
git add backend/embeddings.py
git commit -m "fix(embeddings): add file size check before reading

Prevents OOM when indexing repos with large files"
```

---

## Task 6: Fix Async Blocking Handlers

**Files:**
- Modify: `backend/server.py:304-314`

**Step 1: Wrap CPU-bound operations in run_in_executor**

Update the search endpoint:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# At module level
_executor = ThreadPoolExecutor(max_workers=4)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, req: Request):
    """Search for code - runs CPU-bound ops in thread pool."""
    indexer, embedder, faiss_index = get_components()

    sanitized_query = sanitize_query(request.query)

    # Run embedding generation in thread pool
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(
        _executor,
        embedder.embed_query,
        sanitized_query
    )

    # Run FAISS search in thread pool
    limit_multiplier = 2 if request.language else 1
    results = await loop.run_in_executor(
        _executor,
        lambda: faiss_index.search(query_embedding, k=request.limit * limit_multiplier)
    )

    # Filter by language if specified
    if request.language:
        results = [r for r in results if r.language == request.language][:request.limit]

    return SearchResponse(results=results, query=sanitized_query)
```

**Step 2: Apply same pattern to index endpoint**

Wrap indexing operations similarly.

**Step 3: Commit**

```bash
git add backend/server.py
git commit -m "perf(server): run CPU-bound ops in thread pool

- Embeddings and FAISS search no longer block event loop
- Health checks remain responsive during heavy load
- Use ThreadPoolExecutor with 4 workers"
```

---

## Task 7: Rename TreeSitterAnalyzer to RegexSymbolExtractor

**Files:**
- Rename: `backend/ast_analyzer.py` to `backend/regex_symbol_extractor.py`
- Update: All imports

**Step 1: Rename class and file**

```python
# backend/regex_symbol_extractor.py
"""Regex-based symbol extraction for code chunks.

NOTE: This uses regex patterns, not actual AST parsing.
For true AST analysis, use tree-sitter bindings directly.
"""

class RegexSymbolExtractor:
    """Extract symbols from code using regex patterns.

    Limitations:
    - Cannot handle nested structures correctly
    - May match patterns inside strings/comments
    - Multi-line patterns may not match
    """
    # ... rest of class
```

**Step 2: Update imports in all files**

```bash
# Find and replace in all Python files
grep -r "TreeSitterAnalyzer" --include="*.py"
# Update each import
```

**Step 3: Commit**

```bash
git mv backend/ast_analyzer.py backend/regex_symbol_extractor.py
git add -A
git commit -m "refactor: rename TreeSitterAnalyzer to RegexSymbolExtractor

Honest naming - this uses regex patterns, not actual tree-sitter.
Documented limitations in class docstring."
```

---

## Verification Checklist

After implementing all tasks, verify:

- [ ] `python -m cli.main index /path/to/repo` creates multiple chunks per file
- [ ] Second index run is faster due to embedding cache
- [ ] `~/.coarch/config.json` does not contain jwt_secret
- [ ] API keys hash differently each time (random salt)
- [ ] Server health endpoint responds during search operations
- [ ] Files >10MB are skipped with warning

---

## Estimated Performance After Fixes

| Metric | Before | After |
|--------|--------|-------|
| Code indexed per file | ~50 lines (first chunk only) | 100% of file |
| Re-indexing time | 5 min/1000 chunks | <1 min (cached) |
| Concurrent request handling | Blocked | Non-blocking |
| Security audit findings | 3 critical | 0 critical |
