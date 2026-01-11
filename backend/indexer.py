"""Repository indexing service."""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import sqlite3
from tqdm import tqdm


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str
    symbols: List[str]
    ast_hash: str


class RepositoryIndexer:
    """Index a repository for code search."""

    # File extensions to language mapping
    EXT_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
    }

    # Directories to ignore
    IGNORE_DIRS = {
        ".git", ".svn", ".hg",
        "__pycache__", ".pytest_cache", "node_modules",
        "venv", ".venv", "env", ".env",
        "build", "dist", ".next", "out",
        "target", "Cargo.lock", "package-lock.json",
        ".idea", ".vscode", ".vs",
    }

    # Files to ignore
    IGNORE_FILES = {
        ".gitignore", ".dockerignore", ".editorconfig",
        "LICENSE", "README", "package.json", "Cargo.toml",
    }

    def __init__(self, db_path: str = "coarch.db"):
        """Initialize the indexer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                code TEXT,
                language TEXT,
                symbols TEXT,
                ast_hash TEXT,
                embedding_id INTEGER,
                repo_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                name TEXT,
                last_indexed TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id)
        """)

        conn.commit()
        conn.close()

    def get_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXT_TO_LANG.get(ext)

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        if path.name.startswith("."):
            return True
        if path.name in self.IGNORE_FILES:
            return True
        if path.parent.name in self.IGNORE_DIRS:
            return True
        return False

    def extract_code_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        """Extract meaningful chunks from code.

        This is a simple chunker - in production, use Tree-sitter
        to extract functions/classes.
        """
        chunks = []
        language = self.get_language(file_path)

        if not language:
            return []

        lines = content.split("\n")
        chunk_size = 50  # lines per chunk
        overlap = 10

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = "\n".join(chunk_lines)

            if len(chunk_text.strip()) < 10:  # Skip tiny chunks
                continue

            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                code=chunk_text,
                language=language,
                symbols=self._extract_symbols(chunk_text, language),
                ast_hash=self._simple_hash(chunk_text)
            ))

        return chunks

    def _extract_symbols(self, code: str, language: str) -> List[str]:
        """Extract symbol names from code."""
        import re
        symbols = []

        # Common patterns for different languages
        patterns = {
            "python": [
                r"def\s+(\w+)",
                r"class\s+(\w+)",
                r"(\w+)\s*=",
            ],
            "javascript": [
                r"function\s+(\w+)",
                r"const\s+(\w+)",
                r"let\s+(\w+)",
                r"var\s+(\w+)",
                r"class\s+(\w+)",
            ],
        }

        lang_patterns = patterns.get(language, patterns["python"])

        for pattern in lang_patterns:
            matches = re.findall(pattern, code)
            symbols.extend(matches)

        return list(set(symbols))

    def _simple_hash(self, text: str) -> str:
        """Create a simple hash of text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def index_file(self, file_path: str, conn: sqlite3.Connection) -> List[CodeChunk]:
        """Index a single file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        chunks = self.extract_code_chunks(file_path, content)

        cursor = conn.cursor()
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks (file_path, start_line, end_line, code, language, symbols, ast_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
                chunk.code,
                chunk.language,
                json.dumps(chunk.symbols),
                chunk.ast_hash
            ))

        return chunks

    def index_repository(self, repo_path: str, name: Optional[str] = None) -> Dict:
        """Index an entire repository.

        Args:
            repo_path: Path to repository
            name: Optional repository name

        Returns:
            Indexing statistics
        """
        repo_path = os.path.abspath(repo_path)
        name = name or os.path.basename(repo_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get or create repo
        cursor.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
        result = cursor.fetchone()
        if result:
            repo_id = result[0]
            cursor.execute("UPDATE repos SET last_indexed = CURRENT_TIMESTAMP WHERE id = ?", (repo_id,))
        else:
            cursor.execute("INSERT INTO repos (path, name) VALUES (?, ?)", (repo_path, name))
            repo_id = cursor.lastrowid

        # Count before
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,))
        count_before = cursor.fetchone()[0]

        # Find and index files
        stats = {"files_indexed": 0, "chunks_created": 0}
        repo_path_obj = Path(repo_path)

        for path_obj in repo_path_obj.rglob("*"):
            if path_obj.is_file() and not self.should_ignore(path_obj):
                file_path = str(path_obj)
                chunks = self.index_file(file_path, conn)
                if chunks:
                    stats["files_indexed"] += 1
                    stats["chunks_created"] += len(chunks)

        # Update chunk repo_ids
        cursor.execute("""
            UPDATE chunks SET repo_id = ?
            WHERE repo_id IS NULL AND file_path LIKE ?
        """, (repo_id, f"{repo_path}%"))

        conn.commit()
        conn.close()

        return stats

    def get_stats(self) -> Dict:
        """Get index statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM repos")
        total_repos = cursor.fetchone()[0]

        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM chunks
            GROUP BY language
            ORDER BY count DESC
        """)
        by_language = dict(cursor.fetchall())

        conn.close()

        return {
            "total_chunks": total_chunks,
            "total_repos": total_repos,
            "by_language": by_language
        }

    def get_chunks_for_embedding(self, repo_id: Optional[int] = None) -> List[CodeChunk]:
        """Get all chunks for embedding generation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if repo_id:
            cursor.execute("""
                SELECT file_path, start_line, end_line, code, language, symbols, ast_hash
                FROM chunks
                WHERE repo_id = ?
            """, (repo_id,))
        else:
            cursor.execute("""
                SELECT file_path, start_line, end_line, code, language, symbols, ast_hash
                FROM chunks
            """)

        rows = cursor.fetchall()
        conn.close()

        chunks = []
        for row in rows:
            chunks.append(CodeChunk(
                file_path=row[0],
                start_line=row[1],
                end_line=row[2],
                code=row[3],
                language=row[4],
                symbols=json.loads(row[5]) if row[5] else [],
                ast_hash=row[6]
            ))

        return chunks

    def delete_repo(self, repo_id: int):
        """Delete a repository from the index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
        cursor.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

        conn.commit()
        conn.close()
