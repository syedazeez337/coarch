# Coarch - Local-First Code Search Engine

A semantic code search engine with editor plugins. Find similar code, discover patterns, and navigate codebases intelligently.

## Why Coarch?

| Feature | Coarch | Sourcegraph | GitHub Code Search |
|---------|--------|-------------|-------------------|
| Local-first (privacy) | ✅ | ❌ | ❌ |
| Real-time indexing | ✅ | ❌ | ❌ |
| Semantic + Structural | ✅ | ❌ | ❌ |
| Editor-native | ✅ | Web only | Web only |
| Free forever | ✅ | Paid tiers | Limited |

## Features

- **Semantic Search** - Find code by meaning, not just keywords
- **Hybrid Analysis** - Combines FAISS embeddings + Tree-sitter AST analysis
- **Local-First** - 100% private, no data leaves your machine
- **Real-Time** - Indexes as you type
- **Editor Plugins** - VS Code, Neovim, Emacs support
- **Cross-Repo** - Search all your repos at once

## Quick Start

```bash
# Install
pip install coarch

# Index a repository
coarch index /path/to/your/repo

# Start the server
coarch serve

# Search from CLI
coarch search "function that parses JSON"

# Or use the API
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "function that parses JSON", "limit": 10}'
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Editor Plugins                       │
│         (VS Code ←→ Neovim ←→ Emacs ←→ CLI)                │
└────────────────────────┬───────────────────────────────────┘
                         │ REST/MCP
                         ▼
┌────────────────────────────────────────────────────────────┐
│                    Coarch Server (FastAPI)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Search    │  │   Index     │  │  WebSocket (real-   │ │
│  │   API       │  │   Manager   │  │  time updates)      │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘ │
└─────────┼────────────────┼─────────────────────────────────┘
          │                │
    ┌─────▼─────┐    ┌─────▼─────┐
    │   FAISS   │    │  SQLite   │
    │  (HNSW)   │    │ (metadata)│
    └─────┬─────┘    └───────────┘
          │
    ┌─────▼─────┐
    │ CodeBERT  │
    │ Embeddings│
    └───────────┘
```

## API Endpoints

### Search
```bash
POST /search
{
  "query": "authenticate user with JWT",
  "language": "python",
  "limit": 10
}
```

### Index Repository
```bash
POST /index/repo
{
  "path": "/path/to/repo",
  "name": "my-project"
}
```

### Status
```bash
GET /index/status
```

## Editor Plugins

### VS Code
```bash
# Install from VS Code marketplace (coming soon)
ext install coarch.vscode
```

### Neovim
```lua
-- Using lazy.nvim
{ "syedazeez337/coarch.nvim" }
```

## Configuration

```yaml
# ~/.coarch/config.yaml
index_path: ~/.coarch/index
db_path: ~/.coarch/coarch.db
model: microsoft/codebert-base
batch_size: 32
max_sequence_length: 512
```

## Tech Stack

- **FAISS** - Billion-scale vector search (HNSW index)
- **CodeBERT** - Code understanding embeddings
- **Tree-sitter** - AST parsing for structural analysis
- **FastAPI** - High-performance async API
- **SQLite** - Lightweight metadata storage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a PR

## License

MIT License - see LICENSE file

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [CodeBERT](https://huggingface.co/microsoft/codebert-base) - Microsoft Research
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Parsing tool
