"""Configuration management for Coarch with security and environment support."""

import os
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)

CONFIG_VERSION = "2.0"


@dataclass
class CoarchConfig:
    """Coarch configuration with validation and security."""

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
    ignore_patterns: List[str] = field(
        default_factory=lambda: [
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
        ]
    )
    indexed_repos: List[Dict[str, str]] = field(default_factory=list)
    theme: str = "dark"
    max_results: int = 20
    show_line_numbers: bool = True
    auto_index: bool = True
    log_level: str = "INFO"
    cors_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )
    rate_limit_per_minute: int = 60
    enable_auth: bool = False
    api_key_hash: Optional[str] = None
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        salt = hashlib.scrypt(
            self.jwt_secret.encode(), salt=b"coarch_salt", n=16384, r=8, p=1
        ).hex()[:32]
        return hashlib.scrypt(
            api_key.encode(), salt=salt.encode(), n=16384, r=8, p=1
        ).hex()

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify an API key against its hash."""
        if not self.api_key_hash:
            return False
        return secrets.compare_digest(self._hash_api_key(api_key), self.api_key_hash)

    def set_api_key(self, api_key: str) -> None:
        """Set a new API key (will be hashed)."""
        self.api_key_hash = self._hash_api_key(api_key)

    def save(self, path: str) -> None:
        """Save config to file."""
        config_dir = os.path.dirname(path)
        if config_dir:
            Path(config_dir).mkdir(parents=True, exist_ok=True)

        config_data = {
            "version": CONFIG_VERSION,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
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
            "log_level": self.log_level,
            "cors_origins": self.cors_origins,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "enable_auth": self.enable_auth,
            "api_key_hash": self.api_key_hash,
            "jwt_secret": self.jwt_secret,
            "max_request_size": self.max_request_size,
            "max_file_size": self.max_file_size,
        }

        try:
            with open(path, "w") as f:
                json.dump(config_data, f, indent=2)

            try:
                os.chmod(path, 0o600)
            except (OSError, NotImplementedError):
                pass  # chmod not supported on Windows

            logger.info(f"Config saved to {path}")
        except Exception as e:
            logger.exception(f"Failed to save config: {e}")
            raise

    @classmethod
    def load(cls, path: str) -> "CoarchConfig":
        """Load config from file."""
        if not os.path.exists(path):
            logger.info(f"Config not found at {path}, using defaults")
            return cls()

        try:
            with open(path, "r") as f:
                data = json.load(f)

            logger.info(f"Config loaded from {path}")

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
                log_level=data.get("log_level", "INFO"),
                cors_origins=data.get("cors_origins", []),
                rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
                enable_auth=data.get("enable_auth", False),
                api_key_hash=data.get("api_key_hash"),
                jwt_secret=data.get("jwt_secret", secrets.token_urlsafe(32)),
                max_request_size=data.get("max_request_size", 10 * 1024 * 1024),
                max_file_size=data.get("max_file_size", 10 * 1024 * 1024),
            )
        except Exception:
            logger.exception("Failed to load config")
            logger.warning("Using default config due to load error")
            return cls()

    @staticmethod
    def get_default_config_path() -> str:
        """Get default config path."""
        return os.path.expanduser("~/.coarch/config.json")

    @classmethod
    def get_default(cls) -> "CoarchConfig":
        """Get default configuration."""
        return cls.load(cls.get_default_config_path())

    def add_indexed_repo(self, path: str, name: str) -> None:
        """Add a repository to the indexed list."""
        abs_path = os.path.abspath(path)
        repo_entry = {
            "path": abs_path,
            "name": name,
            "added_at": datetime.now().isoformat(),
        }

        self.indexed_repos = [
            r for r in self.indexed_repos if r.get("path") != abs_path
        ]

        self.indexed_repos.append(repo_entry)
        self.save(self.get_default_config_path())

        logger.info(f"Added indexed repo: {name} ({abs_path})")

    def remove_indexed_repo(self, path: str) -> bool:
        """Remove a repository from the indexed list."""
        abs_path = os.path.abspath(path)
        original_count = len(self.indexed_repos)
        self.indexed_repos = [
            r for r in self.indexed_repos if r.get("path") != abs_path
        ]

        if len(self.indexed_repos) < original_count:
            self.save(self.get_default_config_path())
            logger.info(f"Removed indexed repo: {abs_path}")
            return True

        return False

    def get_expanded_paths(self) -> Dict[str, str]:
        """Get expanded paths for index and database."""
        return {
            "index_path": os.path.expanduser(self.index_path),
            "db_path": os.path.expanduser(self.db_path),
        }

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors: List[str] = []

        if self.server_port < 1 or self.server_port > 65535:
            errors.append(f"Invalid server port: {self.server_port}")

        if self.max_results < 1 or self.max_results > 1000:
            errors.append(f"Invalid max_results: {self.max_results}")

        if self.batch_size < 1:
            errors.append(f"Invalid batch_size: {self.batch_size}")

        if self.rate_limit_per_minute < 1:
            errors.append(
                f"Invalid rate_limit_per_minute: {self.rate_limit_per_minute}"
            )

        if self.enable_auth and not self.api_key_hash:
            errors.append("API key required when auth is enabled")

        if self.max_request_size < 1024:
            errors.append(f"max_request_size too small: {self.max_request_size}")

        if self.max_file_size < 1024:
            errors.append(f"max_file_size too small: {self.max_file_size}")

        return errors

    def generate_api_key(self) -> str:
        """Generate a new secure API key."""
        api_key = secrets.token_urlsafe(32)
        self.set_api_key(api_key)
        return api_key


class ConfigManager:
    """Configuration manager with environment variable support."""

    ENV_PREFIX = "COARCH_"

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager."""
        self.config_path = config_path or CoarchConfig.get_default_config_path()
        self.config = CoarchConfig.load(self.config_path)

        self._update_from_env()

    def _update_from_env(self) -> None:
        """Update config from environment variables."""
        mappings = {
            f"{self.ENV_PREFIX}INDEX_PATH": "index_path",
            f"{self.ENV_PREFIX}DB_PATH": "db_path",
            f"{self.ENV_PREFIX}MODEL_NAME": "model_name",
            f"{self.ENV_PREFIX}BATCH_SIZE": "batch_size",
            f"{self.ENV_PREFIX}SERVER_HOST": "server_host",
            f"{self.ENV_PREFIX}SERVER_PORT": "server_port",
            f"{self.ENV_PREFIX}USE_GPU": "use_gpu",
            f"{self.ENV_PREFIX}USE_QUANTIZATION": "use_quantization",
            f"{self.ENV_PREFIX}LOG_LEVEL": "log_level",
            f"{self.ENV_PREFIX}RATE_LIMIT": "rate_limit_per_minute",
            f"{self.ENV_PREFIX}ENABLE_AUTH": "enable_auth",
            f"{self.ENV_PREFIX}MAX_REQUEST_SIZE": "max_request_size",
            f"{self.ENV_PREFIX}MAX_FILE_SIZE": "max_file_size",
        }

        updated = False

        for env_var, config_key in mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                current = getattr(self.config, config_key)

                if isinstance(current, bool):
                    new_value: Any = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    try:
                        new_value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid int value for {env_var}")
                        continue
                else:
                    new_value = value

                setattr(self.config, config_key, new_value)
                updated = True
                logger.debug(f"Updated {config_key} from env: {env_var}")

        if updated:
            self.save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set config value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save()
            logger.debug(f"Set config {key}")

    def save(self) -> None:
        """Save current config."""
        self.config.save(self.config_path)

    def reset(self) -> None:
        """Reset config to defaults."""
        self.config = CoarchConfig()
        self.save()
        logger.info("Config reset to defaults")

    def validate(self) -> List[str]:
        """Validate current configuration."""
        return self.config.validate()


def init_config() -> CoarchConfig:
    """Initialize default configuration."""
    config = CoarchConfig.get_default()
    config_path = CoarchConfig.get_default_config_path()

    logger.info(f"Config initialized at: {config_path}")
    logger.info(f"  Index path: {config.index_path}")
    logger.info(f"  DB path: {config.db_path}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Server: {config.server_host}:{config.server_port}")

    return config
