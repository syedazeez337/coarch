# Coarch Technical Architecture

## Overview
Local-first, hybrid semantic + structural code search engine with real-time indexing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Editor Plugins                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ VS Code  │  │  Neovim  │  │  Emacs   │  │ CLI (Terminal)  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬────────┘  │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │             │             │                 │
        └─────────────┴─────┬───────┴─────────────────┘
                            │
                    ┌───────▼───────┐
                    │   MCP/REST    │
                    │   Protocol    │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
    │  FastAPI      │ │  Auth     │ │   MCP Server  │
    │  Search API   │ │  (Local)  │ │   (Plugins)   │
    └───────┬───────┘ └─────┬─────┘ └───────────────┘
            │               │
    ┌───────┴───────┬───────┴───────┐
    │               │               │
┌───▼───┐    ┌──────▼──────┐    ┌───▼───┐
│ FAISS │    │   SQLite    │    │ Tree- │
│ Index │    │  Metadata   │    │ sitter│
└───────┘    └─────────────┘    └───────┘
    │
┌───▼───┐
│CodeBERT│
│ Embed- │
│ dings  │
└───────┘
```

## Components

### 1. Embedding Service
- **Model**: `microsoft/codebert-base` (MLM + fine-tuned for code)
- **Output**: 768-dim vectors per code snippet
- **Optimization**: ONNX runtime for fast inference

### 2. FAISS Index (HNSW)
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Metric**: Inner product (cosine similarity)
- **Speed**: Sub-millisecond search for millions of vectors
- **GPU**: Optional CUDA acceleration

### 3. Structural Analysis (Tree-sitter)
- **AST Parsing**: Extract function signatures, imports, calls
- **Type Info**: Variable types, return types
- **Relationships**: Call graphs, dependency maps

### 4. Search Pipeline

```
User Query → Parse (Tree-sitter) → Hybrid Search
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
            │ Semantic      │ │ Keyword   │ │ Structural    │
            │ (FAISS)       │ │ (BM25)    │ │ (AST match)   │
            └───────┬───────┘ └─────┬─────┘ └───────┬───────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                            ┌───────▼───────┐
                            │ Re-ranking    │
                            │ (Cross-encoder│
                            │  or weighted) │
                            └───────┬───────┘
                                    │
                            ┌───────▼───────┐
                            │ Results +     │
                            │ Context       │
                            └───────────────┘
```

## Data Flow

### Indexing Flow
```
Repo Scan → File Parser → Chunking (functions/classes)
            → CodeBERT Embedding
            → Store in FAISS
            → Store metadata in SQLite
            → Update index
```

### Search Flow
```
Query Input → Pre-process → Hybrid Search
                         → Re-rank
                         → Return + context
```

## Storage

### FAISS Index
- `code_index.bin` - Main vector index
- `code_index_meta.bin` - Index metadata

### SQLite Schema
```sql
-- Code chunks table
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    code_text TEXT,
    language TEXT,
    ast_hash TEXT,
    symbols TEXT,  -- JSON: extracted symbols
    embedding_id INTEGER
);

-- Repositories table
CREATE TABLE repos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    name TEXT,
    last_indexed TIMESTAMP
);
```

## API Endpoints

### Search
```
POST /search
{
    "query": "function that parses JSON",
    "language": "python",
    "limit": 10,
    "filters": {"repo": null}
}

Response:
{
    "results": [
        {
            "file": "src/parser.py",
            "lines": "10-25",
            "code": "def parse_json(data): ...",
            "score": 0.92,
            "context": "imports, calls"
        }
    ]
}
```

### Index Management
```
POST /index/repo - Index a repository
DELETE /index/repo/{id} - Remove from index
GET /index/status - Index statistics
```

## Performance Targets

- **Indexing**: 1000 lines/second (CPU), 5000 lines/second (GPU)
- **Search**: <10ms for 1M vectors
- **Memory**: 1GB for 10M code chunks
- **Real-time**: Index updates within 100ms of file change

## Unique Selling Points

1. **Local-first** - Zero cloud dependency
2. **Hybrid Search** - Semantic + Structural > Pure embeddings
3. **Real-time** - Indexes as you type
4. **Editor-native** - Built for workflow, not web UI
5. **Cross-repo** - Index all your repos, search across them
