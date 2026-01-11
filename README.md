# Coarch - Code Search Engine

A code search engine that can be used as a plugin in any editor. Find similar code, discover patterns, and navigate codebases semantically.

## Features

- Semantic code search using embeddings
- Editor plugin support (VS Code, Neovim, Emacs, etc.)
- Support for multiple programming languages
- Fast similarity search using FAISS
- Self-hosted or cloud deployment options

## Architecture

```
coarch/
├── backend/           # Python FAISS-based search service
├── plugins/           # Editor integrations
│   ├── vscode/
│   ├── neovim/
│   └── emacs/
├── embeddings/        # Code embedding models
├── cli/               # Command-line interface
└── docs/              # Documentation
```

## Getting Started

```bash
# Clone and install
git clone https://github.com/syedazeez337/coarch.git
cd coarch
pip install -e .
```

## License

MIT License - see LICENSE file
