"""Real-time file watching for incremental indexing."""

import os
import time
from pathlib import Path
from typing import Set, Callable, Optional, Dict, List
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty


@dataclass
class FileEvent:
    """A file system event."""
    event_type: str  # created, modified, deleted, moved
    file_path: str
    timestamp: float = field(default_factory=time.time)


class FileWatcher:
    """Watch a directory for file changes."""

    def __init__(
        self,
        paths: Set[str],
        ignore_patterns: Optional[Set[str]] = None,
        debounce_ms: int = 100
    ):
        """Initialize the file watcher.

        Args:
            paths: Paths to watch
            ignore_patterns: Patterns to ignore
            debounce_ms: Debounce time in milliseconds
        """
        self.paths = set(os.path.abspath(p) for p in paths)
        self.ignore_patterns = ignore_patterns or {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "*.pyc", "*.pyo", "*.so", "*.o", "*.a", "*.lib",
            "build", "dist", ".next", "out", "target"
        }
        self.debounce_ms = debounce_ms
        self._callbacks: List[Callable[[FileEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue: Queue = Queue()

    def start(self):
        """Start watching for changes."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop watching for changes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def add_callback(self, callback: Callable[[FileEvent], None]):
        """Add a callback for file events."""
        self._callbacks.append(callback)

    def _watch_loop(self):
        """Main watching loop."""
        import time

        last_events: Dict[str, float] = {}

        while self._running:
            try:
                events = self._scan_changes()
                for event in events:
                    now = time.time()
                    key = f"{event.event_type}:{event.file_path}"

                    # Debounce
                    if key in last_events and (now - last_events[key]) < (self.debounce_ms / 1000):
                        continue
                    last_events[key] = now

                    for callback in self._callbacks:
                        callback(event)

                time.sleep(0.1)  # Poll interval
            except Exception:
                pass

    def _scan_changes(self):
        """Scan for file changes (cross-platform)."""
        events = []

        for path in self.paths:
            if not os.path.exists(path):
                continue

            for root, dirs, files in os.walk(path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]

                for filename in files:
                    if self._should_ignore(filename):
                        continue

                    file_path = os.path.join(root, filename)
                    if not self._should_ignore(file_path):
                        mtime = os.path.getmtime(file_path)
                        events.append(FileEvent(
                            event_type="modified",
                            file_path=file_path
                        ))

        return events

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        path_obj = Path(path)
        name = path_obj.name

        # Check patterns
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in path:
                return True

        # Check hidden files/directories
        if name.startswith("."):
            return True

        return False


class IncrementalIndexer:
    """Incrementally index files as they change."""

    def __init__(
        self,
        indexer,
        embedder,
        faiss_index,
        watch_paths: Set[str],
        debounce_ms: int = 500
    ):
        """Initialize the incremental indexer.

        Args:
            indexer: RepositoryIndexer instance
            embedder: CodeEmbedder instance
            faiss_index: FaissIndex instance
            watch_paths: Paths to watch for changes
            debounce_ms: Debounce time for file changes
        """
        self.indexer = indexer
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.watcher = FileWatcher(watch_paths, debounce_ms=debounce_ms)
        self.pending_files: Set[str] = set()
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start incremental indexing."""
        self._running = True

        def on_change(event: FileEvent):
            if self._running:
                with self._lock:
                    self.pending_files.add(event.file_path)

        self.watcher.add_callback(on_change)
        self.watcher.start()

        # Start processing thread
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop(self):
        """Stop incremental indexing."""
        self._running = False
        self.watcher.stop()

    def _process_loop(self):
        """Process pending file changes."""
        import time

        while self._running:
            with self._lock:
                files = list(self.pending_files)
                self.pending_files.clear()

            for file_path in files:
                try:
                    self._reindex_file(file_path)
                except Exception:
                    pass

            time.sleep(0.5)  # Process interval

    def _reindex_file(self, file_path: str):
        """Reindex a single file."""
        # Check if file should be indexed
        language = self.indexer.get_language(file_path)
        if not language:
            return

        # Read file
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except Exception:
            return

        # Extract chunks
        chunks = self.indexer.extract_code_chunks(file_path, code)

        # TODO: Update FAISS index with new embeddings
        # For now, this is a placeholder
        print(f"增量索引: {file_path} ({len(chunks)} chunks)")

    def get_pending_count(self) -> int:
        """Get the number of pending files."""
        with self._lock:
            return len(self.pending_files)
