# Coarch Development Plan

## Phase 1: Core Search Engine
- [ ] Set up FAISS indexing pipeline
- [ ] Implement code embedding extraction (use CodeBERT/StarCoder)
- [ ] Build indexing script for repositories
- [ ] Create search API endpoint

## Phase 2: Editor Plugins
- [ ] VS Code extension
- [ ] Neovim plugin
- [ ] Emacs package

## Phase 3: Features
- [ ] Multi-language support
- [ ] Filter by file type/language
- [ ] Highlight matching code segments
- [ ] Integration with LSP for context

## Tech Stack
- **Backend**: Python, FAISS, FastAPI
- **Embeddings**: CodeBERT, StarCoder, or Sentence-Transformers
- **Plugins**: TypeScript (VS Code), Lua (Neovim), Elisp (Emacs)
- **Storage**: FAISS for vectors, SQLite for metadata
