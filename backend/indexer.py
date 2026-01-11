"""Repository indexing service with production-ready features."""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 50
OVERLAP = 10
MIN_CHUNK_SIZE = 10
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size


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
    """Index a repository for code search with proper error handling."""

    EXT_TO_LANG: Dict[str, str] = {
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
        ".r": "r",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "cfg",
        ".txt": "text",
    }

    IGNORE_DIRS: Set[str] = {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        "build",
        "dist",
        ".next",
        "out",
        "target",
        "Cargo.lock",
        "package-lock.json",
        ".idea",
        ".vscode",
        ".vs",
        ".gradle",
        "*.egg-info",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.o",
        "*.a",
        "*.lib",
        "*.dll",
        "*.exe",
        "*.dylib",
        ".git",
    }

    IGNORE_FILES: Set[str] = {
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        ".gitmodules",
        "LICENSE",
        "README",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "go.sum",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "package-lock.json",
        "yarn.lock",
        "Pipfile.lock",
    }

    def __init__(self, db_path: str = "coarch.db"):
        """Initialize the indexer."""
        self.db_path = db_path
        self._init_db()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for symbol extraction."""
        self.SYMBOL_PATTERNS = {
            "python": [
                (r"def\s+(\w+)", "function"),
                (r"class\s+(\w+)", "class"),
                (r"async\s+def\s+(\w+)", "function"),
            ],
            "javascript": [
                (r"function\s+(\w+)", "function"),
                (r"const\s+(\w+)\s*=", "const"),
                (r"let\s+(\w+)\s*=", "let"),
                (r"var\s+(\w+)\s*=", "var"),
                (r"class\s+(\w+)", "class"),
            ],
            "typescript": [
                (r"function\s+(\w+)", "function"),
                (r"const\s+(\w+)\s*=", "const"),
                (r"class\s+(\w+)", "class"),
                (r"interface\s+(\w+)", "interface"),
                (r"type\s+(\w+)", "type"),
            ],
            "java": [
                (
                    r"(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?"
                    r"[\w<>[\]]+\s+(\w+)\s*\(",
                    "method",
                ),
                (r"class\s+(\w+)", "class"),
                (r"interface\s+(\w+)", "interface"),
            ],
            "cpp": [
                (r"(?:void|int|bool|auto|const\s+auto)\s+(\w+)\s*\(", "function"),
                (r"(?:class|struct)\s+(\w+)", "class"),
            ],
            "go": [
                (r"func\s+(?:\([^)]+\)\s*)?(\w+)\s*\(", "function"),
                (r"type\s+(\w+)\s+(?:struct|interface)", "type"),
            ],
            "rust": [
                (r"fn\s+(\w+)", "function"),
                (r"struct\s+(\w+)", "struct"),
                (r"impl\s+(?:<[^>]+>\s+)?(\w+)", "impl"),
            ],
        }

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper cleanup and isolation level."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the SQLite database with proper schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.execute("PRAGMA temp_store=MEMORY")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    name TEXT,
                    last_indexed TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_file
                ON chunks(file_path)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_language
                ON chunks(language)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_repo
                ON chunks(repo_id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_ast_hash
                ON chunks(ast_hash)
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_name TEXT UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute("BEGIN")
            conn.execute("COMMIT")

        logger.info(f"Database initialized at {self.db_path}")

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
        if path.name in self.IGNORE_DIRS:
            return True
        return False

    def _compute_ast_hash(self, text: str) -> str:
        """Compute a hash for AST-based deduplication."""
        return hashlib.md5(text.encode()).hexdigest()

    def _extract_symbols(self, code: str, language: str) -> List[str]:
        """Extract symbol names from code."""
        symbols: Set[str] = set()
        patterns = self.SYMBOL_PATTERNS.get(language, [])

        for pattern, _ in patterns:
            try:
                matches = re.findall(pattern, code)
                symbols.update(matches)
            except re.error as e:
                logger.warning(f"Regex error for {language}: {e}")

        return list(symbols)

    def extract_code_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        """Extract meaningful chunks from code."""
        language = self.get_language(file_path)
        if not language:
            logger.debug(f"No language detected for {file_path}")
            return []

        lines = content.split("\n")
        chunks: List[CodeChunk] = []

        for i in range(0, len(lines), CHUNK_SIZE - OVERLAP):
            chunk_lines = lines[i : i + CHUNK_SIZE]
            chunk_text = "\n".join(chunk_lines)

            if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
                continue

            ast_hash = self._compute_ast_hash(chunk_text)
            symbols = self._extract_symbols(chunk_text, language)

            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + CHUNK_SIZE, len(lines)),
                    code=chunk_text,
                    language=language,
                    symbols=symbols,
                    ast_hash=ast_hash,
                )
            )

        return chunks

    def index_file(self, file_path: str) -> List[CodeChunk]:
        """Index a single file with proper error handling."""
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping large file {file_path}: {file_size} bytes")
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        chunks = self.extract_code_chunks(file_path, content)

        if not chunks:
            return []

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        file_path, start_line, end_line, code,
                        language, symbols, ast_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.file_path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.code,
                        chunk.language,
                        json.dumps(chunk.symbols),
                        chunk.ast_hash,
                    ),
                )

            conn.execute("BEGIN")
            conn.execute("COMMIT")

        logger.debug(f"Indexed {len(chunks)} chunks from {file_path}")
        return chunks

    def index_repository(
        self, repo_path: str, name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Index an entire repository."""
        abs_path = os.path.abspath(repo_path)

        if not os.path.isdir(abs_path):
            raise ValueError(f"Path is not a directory: {abs_path}")

        repo_name = name or os.path.basename(abs_path)

        logger.info(f"Indexing repository: {abs_path}")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM repos WHERE path = ?", (abs_path,))
            result = cursor.fetchone()

            if result:
                repo_id = result[0]
                cursor.execute(
                    "UPDATE repos SET last_indexed = CURRENT_TIMESTAMP WHERE id = ?",
                    (repo_id,),
                )
            else:
                cursor.execute(
                    "INSERT INTO repos (path, name) VALUES (?, ?)",
                    (abs_path, repo_name),
                )
                repo_id = cursor.lastrowid

        stats: Dict[str, Any] = {
            "files_indexed": 0,
            "chunks_created": 0,
            "repo_id": repo_id,
            "repo_path": abs_path,
        }

        repo_path_obj = Path(abs_path)
        indexed_files = 0
        created_chunks = 0

        try:
            for path_obj in repo_path_obj.rglob("*"):
                if not path_obj.is_file():
                    continue
                if self.should_ignore(path_obj):
                    continue

                file_path = str(path_obj)
                chunks = self.index_file(file_path)

                if chunks:
                    indexed_files += 1
                    created_chunks += len(chunks)

        except Exception as e:
            logger.exception(f"Error during indexing: {e}")
            raise

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE chunks SET repo_id = ?
                WHERE repo_id IS NULL AND file_path LIKE ?
            """,
                (repo_id, f"{abs_path}%"),
            )

            conn.execute("BEGIN")
            conn.execute("COMMIT")

        stats["files_indexed"] = indexed_files
        stats["chunks_created"] = created_chunks

        logger.info(
            f"Indexed {indexed_files} files, "
            f"{created_chunks} chunks from {repo_name}"
        )

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM repos")
            total_repos = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT language, COUNT(*) as count
                FROM chunks
                GROUP BY language
                ORDER BY count DESC
            """
            )
            by_language = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT SUM(LENGTH(code)) FROM chunks")
            total_size = cursor.fetchone()[0] or 0

        return {
            "total_chunks": total_chunks,
            "total_repos": total_repos,
            "by_language": by_language,
            "total_size_bytes": total_size,
        }

    def get_chunks_for_embedding(
        self, repo_id: Optional[int] = None
    ) -> List[CodeChunk]:
        """Get all chunks for embedding generation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if repo_id:
                cursor.execute(
                    """
                    SELECT file_path, start_line, end_line, code,
                           language, symbols, ast_hash
                    FROM chunks
                    WHERE repo_id = ?
                    ORDER BY file_path, start_line
                """,
                    (repo_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT file_path, start_line, end_line, code,
                           language, symbols, ast_hash
                    FROM chunks
                    ORDER BY file_path, start_line
                """
                )

            rows = cursor.fetchall()

        chunks: List[CodeChunk] = []
        for row in rows:
            symbols = json.loads(row[5]) if row[5] else []

            chunks.append(
                CodeChunk(
                    file_path=row[0],
                    start_line=row[1],
                    end_line=row[2],
                    code=row[3],
                    language=row[4],
                    symbols=symbols,
                    ast_hash=row[6],
                )
            )

        return chunks

    def delete_repo(self, repo_id: int) -> int:
        """Delete a repository from the index."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            chunks_deleted = cursor.rowcount

            cursor.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

            conn.execute("BEGIN")
            conn.execute("COMMIT")

        logger.info(f"Deleted repo {repo_id}: {chunks_deleted} chunks")
        return chunks_deleted

    def vacuum(self) -> None:
        """Perform database vacuum to reclaim space."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuum completed")

    def cleanup_orphaned_chunks(self) -> int:
        """Remove chunks without valid repo references."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM chunks
                WHERE repo_id IS NULL OR repo_id NOT IN (SELECT id FROM repos)
            """
            )
            deleted = cursor.rowcount

            conn.execute("BEGIN")
            conn.execute("COMMIT")

        if deleted:
            logger.info(f"Cleaned up {deleted} orphaned chunks")

        return deleted

    def update_chunk(self, chunk_id: int, code: str, ast_hash: str) -> bool:
        """Update a chunk's code and hash."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE chunks
                SET code = ?, ast_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (code, ast_hash, chunk_id),
            )

            conn.execute("BEGIN")
            conn.execute("COMMIT")

            return cursor.rowcount > 0

    def find_duplicate_chunks(self, ast_hash: str) -> List[Dict]:
        """Find chunks with the same AST hash."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, file_path, start_line, end_line
                FROM chunks
                WHERE ast_hash = ?
                ORDER BY file_path
            """,
                (ast_hash,),
            )

            return [
                {
                    "id": row[0],
                    "file_path": row[1],
                    "start_line": row[2],
                    "end_line": row[3],
                }
                for row in cursor.fetchall()
            ]
