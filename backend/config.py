"""Configuration management for Coarch."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class CoarchConfig:
    """Coarch configuration."""

    index_path: str = "~/.coarch/index"
    db_path: str = "~/.coarch/coarch.db"
    model_name: str = "microsoft/codebert-base"
    max_sequence_length: int = 512
    batch_size: int = 32
    use_quantization: bool = True
    use_gpu: bool = False
    num_threads: int = 0
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    watch_debounce_ms: int = 500
    ignore_patterns: List[str] = field(default_factory=lambda: [
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "build",
        "dist",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.o",
        "*.a",
    ])
    indexed_repos: List[Dict[str, str]] = field(default_factory=list)
    theme: str = "dark"
    max_results: int = 20
    show_line_numbers: bool = True
    auto_index: bool = True

    def save(self, path: str):
        """Save config to file."""
        config_dir = os.path.dirname(path)
        os.makedirs(config_dir, exist_ok=True)

        config_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "index_path": self.index_path,
            "db_path": self.db_path,
            "model_name": self.model_name,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "use_quantization": self.use_quantization,
            "use_gpu": self.use_gpu,
            "num_threads": self.num_threads,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "watch_debounce_ms": self.watch_debounce_ms,
            "ignore_patterns": self.ignore_patterns,
            "indexed_repos": self.indexed_repos,
            "theme": self.theme,
            "max_results": self.max_results,
            "show_line_numbers": self.show_line_numbers,
            "auto_index": self.auto_index,
        }

        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CoarchConfig":
        """Load config from file."""
        if not os.path.exists(path):
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            index_path=data.get("index_path", "~/.coarch/index"),
            db_path=data.get("db_path", "~/.coarch/coarch.db"),
            model_name=data.get("model_name", "microsoft/codebert-base"),
            max_sequence_length=data.get("max_sequence_length", 512),
            batch_size=data.get("batch_size", 32),
            use_quantization=data.get("use_quantization", True),
            use_gpu=data.get("use_gpu", False),
            num_threads=data.get("num_threads", 0),
            server_host=data.get("server_host", "0.0.0.0"),
            server_port=data.get("server_port", 8000),
            watch_debounce_ms=data.get("watch_debounce_ms", 500),
            ignore_patterns=data.get("ignore_patterns", []),
            indexed_repos=data.get("indexed_repos", []),
            theme=data.get("theme", "dark"),
            max_results=data.get("max_results", 20),
            show_line_numbers=data.get("show_line_numbers", True),
            auto_index=data.get("auto_index", True),
        )

    @classmethod
    def get_default_config_path(cls) -> str:
        """Get default config path."""
        return os.path.expanduser("~/.coarch/config.json")

    @classmethod
    def get_default(cls) -> "CoarchConfig":
        """Get default configuration."""
        return cls.load(cls.get_default_config_path())

    def add_indexed_repo(self, path: str, name: str):
        """Add a repository to the indexed list."""
        repo_entry = {
            "path": path,
            "name": name,
            "added_at": datetime.now().isoformat(),
        }
        self.indexed_repos.append(repo_entry)
        self.save(self.get_default_config_path())

    def remove_indexed_repo(self, path: str):
        """Remove a repository from the indexed list."""
        self.indexed_repos = [r for r in self.indexed_repos if r.get("path") != path]
        self.save(self.get_default_config_path())

    def get_expanded_paths(self) -> Dict[str, str]:
        """Get expanded paths for index and database."""
        return {
            "index_path": os.path.expanduser(self.index_path),
            "db_path": os.path.expanduser(self.db_path),
        }


class ConfigManager:
    """Configuration manager with environment variable support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager."""
        self.config_path = config_path or CoarchConfig.get_default_config_path()
        self.config = CoarchConfig.load(self.config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return default

    def set(self, key: str, value: Any):
        """Set config value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save()

    def save(self):
        """Save current config."""
        self.config.save(self.config_path)

    def update_from_env(self, prefix: str = "COARCH_"):
        """Update config from environment variables."""
        env_mappings = {
            f"{prefix}INDEX_PATH": "index_path",
            f"{prefix}DB_PATH": "db_path",
            f"{prefix}MODEL_NAME": "model_name",
            f"{prefix}BATCH_SIZE": "batch_size",
            f"{prefix}SERVER_HOST": "server_host",
            f"{prefix}SERVER_PORT": "server_port",
            f"{prefix}USE_GPU": "use_gpu",
            f"{prefix}USE_QUANTIZATION": "use_quantization",
        }

        for env_var, config_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(getattr(self.config, config_key), bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(getattr(self.config, config_key), int):
                    value = int(value)
                setattr(self.config, config_key, value)

    def reset(self):
        """Reset config to defaults."""
        self.config = CoarchConfig()
        self.save()


def init_config():
    """Initialize default configuration."""
    config = CoarchConfig.get_default()
    print(f"Config initialized at: {CoarchConfig.get_default_config_path()}")
    print(f"  Index path: {config.index_path}")
    print(f"  DB path: {config.db_path}")
    print(f"  Model: {config.model_name}")
    print(f"  Server: {config.server_host}:{config.server_port}")
    return config
