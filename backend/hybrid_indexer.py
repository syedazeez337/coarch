"""Hybrid indexer - uses Rust for fast scanning, Python for storage."""

import os
import sqlite3
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from datetime import datetime

try:
    import coarch_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust module not available, falling back to pure Python")

from .logging_config import get_logger

logger = get_logger(__name__)


class HybridIndexer:
    """Hybrid indexer using Rust for scanning, Python for storage."""

    def __init__(self, db_path: str = "coarch.db"):
        self.db_path = db_path
        self._init_db()

        if RUST_AVAILABLE:
            self.rust_indexer = coarch_rust.RustIndexer()
        else:
            self.rust_indexer = None

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id)")
            logger.info(f"Database initialized at {self.db_path}")

    def index_repository(self, repo_path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Index a repository using Rust for fast scanning."""
        abs_path = os.path.abspath(repo_path)
        repo_name = name or os.path.basename(abs_path)

        logger.info(f"Indexing repository: {abs_path}")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM repos WHERE path = ?", (abs_path,))
            result = cursor.fetchone()

            if result:
                repo_id = result["id"]
                cursor.execute(
                    "UPDATE repos SET last_indexed = ? WHERE id = ?",
                    (datetime.now().isoformat(), repo_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO repos (path, name, last_indexed) VALUES (?, ?, ?)",
                    (abs_path, repo_name, datetime.now().isoformat()),
                )
                repo_id = cursor.lastrowid

        chunks_data = []
        files_indexed = 0
        time_ms = 0

        if RUST_AVAILABLE and self.rust_indexer:
            logger.info("Using Rust for fast file scanning...")
            result = self.rust_indexer.index_directory(abs_path)
            stats = dict(result)

            chunks_list = list(stats.get("chunks_data", []))
            for chunk_str in chunks_list:
                parts = chunk_str.split("|")
                if len(parts) >= 7:
                    symbols = parts[6] if parts[6] else ""
                    chunks_data.append(
                        {
                            "file_path": parts[0],
                            "start_line": int(parts[1]),
                            "end_line": int(parts[2]),
                            "language": parts[3],
                            "ast_hash": parts[4],
                            "code": parts[5].replace("\n", "\n").replace("|", "|"),
                            "symbols": symbols,
                        }
                    )

            files_indexed = stats.get("files_indexed", 0)
            time_ms = stats.get("time_ms", 0)
        else:
            logger.info("Using pure Python (slow mode)...")
            from .indexer import RepositoryIndexer as PythonIndexer

            python_indexer = PythonIndexer(self.db_path)
            result = python_indexer.index_repository(abs_path, repo_name)

            if isinstance(result, dict):
                files_indexed = result.get("files_indexed", 0)
                time_ms = result.get("time_ms", 0)

        if chunks_data:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for chunk in chunks_data:
                    symbols_json = chunk.get("symbols", "")
                    cursor.execute(
                        """
                        INSERT INTO chunks (file_path, start_line, end_line, code, language, symbols, ast_hash, repo_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk["file_path"],
                            chunk["start_line"],
                            chunk["end_line"],
                            chunk["code"],
                            chunk["language"],
                            symbols_json,
                            chunk["ast_hash"],
                            repo_id,
                        ),
                    )

        return {
            "files_indexed": files_indexed,
            "chunks_created": len(chunks_data),
            "time_ms": time_ms,
            "repo_id": repo_id,
            "repo_name": repo_name,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            total_chunks = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM repos")
            total_repos = cursor.fetchone()["count"]

            cursor.execute("SELECT language, COUNT(*) as count FROM chunks GROUP BY language")
            by_language = {row["language"]: row["count"] for row in cursor.fetchall()}

            by_symbol_type: dict[str, int] = {}
            cursor.execute("SELECT symbols FROM chunks")
            for row in cursor.fetchall():
                symbols = row["symbols"] or ""
                for sym in symbols.split(","):
                    if ":" in sym:
                        sym_type = sym.split(":")[-1]
                        by_symbol_type[sym_type] = by_symbol_type.get(sym_type, 0) + 1

        return {
            "total_chunks": total_chunks,
            "total_repos": total_repos,
            "by_language": by_language,
            "by_symbol_type": by_symbol_type,
        }

    def get_chunks_for_embedding(self, limit: int = 1000) -> List[Dict]:
        """Get chunks that need embeddings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_path, code, language, symbols
                FROM chunks 
                WHERE embedding_id IS NULL 
                LIMIT ?
            """,
                (limit,),
            )

            return [
                {
                    "id": row["id"],
                    "file_path": row["file_path"],
                    "code": row["code"],
                    "language": row["language"],
                    "symbols": row["symbols"] or "",
                }
                for row in cursor.fetchall()
            ]

    def update_chunk_embedding(self, chunk_id: int, embedding_id: int) -> bool:
        """Update chunk with embedding ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chunks SET embedding_id = ? WHERE id = ?",
                (embedding_id, chunk_id),
            )
            return cursor.rowcount > 0

    def delete_repo(self, repo_id: int) -> int:
        """Delete a repository and its chunks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            chunks_deleted = cursor.rowcount
            cursor.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
            repos_deleted = cursor.rowcount
            return repos_deleted

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search chunks by query."""
        query_lower = query.lower()
        query_words = query_lower.split()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path, start_line, end_line, code, language, symbols FROM chunks"
            )

            results = []
            for row in cursor.fetchall():
                code_lower = row["code"].lower()
                score = 0

                for word in query_words:
                    if word in code_lower:
                        score += 1
                    if word in (row["symbols"] or "").lower():
                        score += 2

                if score > 0:
                    results.append(
                        {
                            "file": row["file_path"],
                            "score": score,
                            "code": row["code"],
                            "language": row["language"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "symbols": row["symbols"] or "",
                        }
                    )

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
