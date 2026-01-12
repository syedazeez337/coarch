"""Unified Configuration Management for Coarch with clear priority order.

Priority order (highest to lowest):
1. CLI options
2. Environment variables
3. Configuration files
4. Default values (lowest priority)

Features:
- Config validation and conflict detection
- Config templates for quick setup
- --print-config to show resolved configuration
- Config migration for version upgrades
- Config history and rollback
- Environment variable validation on startup
"""

import os
import json
import copy
import secrets
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
import shutil

from .logging_config import get_logger
from .validation import (
    CoarchValidationError,
    CoarchConfigError,
    validate_limit,
    validate_port,
    validate_host,
    sanitize_path,
)

logger = get_logger(__name__)

CONFIG_VERSION = "3.0"
CONFIG_DIR = "~/.coarch"
CONFIG_FILENAME = "config.json"
CONFIG_HISTORY_DIR = "config_history"
CONFIG_TEMPLATES_DIR = "config_templates"


class ConfigSource(Enum):
    """Configuration source enum for tracking origin."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    CLI = "cli"


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata about its source."""
    value: Any
    source: ConfigSource
    source_name: Optional[str] = None
    is_sensitive: bool = False

    def __repr__(self) -> str:
        if self.is_sensitive:
            return f"ConfigValue(***, source={self.source.value})"
        return f"ConfigValue({self.value}, source={self.source.value})"


@dataclass
class ConfigConflict:
    """Represents a configuration conflict between sources."""
    key: str
    values: List[Tuple[Any, ConfigSource]]
    resolved_value: Any
    resolution_strategy: str

    def __repr__(self) -> str:
        return f"ConfigConflict({self.key}: {len(self.values)} sources)"


@dataclass
class ConfigTemplate:
    """Configuration template for quick setup."""
    name: str
    description: str
    values: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ConfigTemplate({self.name}: {self.description})"


@dataclass
class ConfigHistoryEntry:
    """Configuration history entry for rollback."""
    timestamp: str
    config_snapshot: Dict[str, Any]
    config_version: str
    change_description: Optional[str] = None

    def __repr__(self) -> str:
        return f"ConfigHistoryEntry({self.timestamp}: v{self.config_version})"


class UnifiedConfigManager:
    """Unified configuration manager with clear priority order and conflict detection."""

    ENV_PREFIX = "COARCH_"
    CLI_OVERRIDE_PREFIX = "COARCH_CLI_"

    def __init__(
        self,
        config_path: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        enable_history: bool = True
    ):
        """Initialize the unified config manager.
        
        Args:
            config_path: Path to config file (default: ~/.coarch/config.json)
            cli_overrides: CLI option overrides (highest priority)
            enable_history: Whether to track config history
        """
        self.config_dir = os.path.expanduser(CONFIG_DIR)
        self.config_path = config_path or os.path.join(self.config_dir, CONFIG_FILENAME)
        self.history_dir = os.path.join(self.config_dir, CONFIG_HISTORY_DIR)
        self.templates_dir = os.path.join(self.config_dir, CONFIG_TEMPLATES_DIR)
        self.cli_overrides = cli_overrides or {}
        self.enable_history = enable_history
        self._config_cache: Optional[Dict[str, ConfigValue]] = None
        self._resolved_config: Optional[Any] = None
        self._conflicts: List[ConfigConflict] = []
        self._config_version = CONFIG_VERSION

        self._init_directories()
        self._validate_env_vars()

    def _init_directories(self) -> None:
        """Initialize configuration directories."""
        for directory in [self.config_dir, self.history_dir, self.templates_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, ConfigValue]:
        """Get default configuration values."""
        defaults = {
            "index_path": ConfigValue("~/.coarch/index", ConfigSource.DEFAULT),
            "db_path": ConfigValue("~/.coarch/coarch.db", ConfigSource.DEFAULT),
            "model_name": ConfigValue("microsoft/codebert-base", ConfigSource.DEFAULT),
            "max_sequence_length": ConfigValue(512, ConfigSource.DEFAULT),
            "batch_size": ConfigValue(32, ConfigSource.DEFAULT),
            "use_quantization": ConfigValue(True, ConfigSource.DEFAULT),
            "use_gpu": ConfigValue(False, ConfigSource.DEFAULT),
            "num_threads": ConfigValue(0, ConfigSource.DEFAULT),
            "server_host": ConfigValue("0.0.0.0", ConfigSource.DEFAULT),
            "server_port": ConfigValue(8000, ConfigSource.DEFAULT),
            "watch_debounce_ms": ConfigValue(500, ConfigSource.DEFAULT),
            "enable_bm25": ConfigValue(True, ConfigSource.DEFAULT),
            "bm25_weight": ConfigValue(0.3, ConfigSource.DEFAULT),
            "semantic_weight": ConfigValue(0.7, ConfigSource.DEFAULT),
            "min_bm25_score_threshold": ConfigValue(0.01, ConfigSource.DEFAULT),
            "bm25_chunk_limit": ConfigValue(None, ConfigSource.DEFAULT),
            "ignore_patterns": ConfigValue([
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                "build", "dist", "*.pyc", "*.pyo", "*.so", "*.o", "*.a"
            ], ConfigSource.DEFAULT),
            "indexed_repos": ConfigValue([], ConfigSource.DEFAULT),
            "theme": ConfigValue("dark", ConfigSource.DEFAULT),
            "max_results": ConfigValue(20, ConfigSource.DEFAULT),
            "show_line_numbers": ConfigValue(True, ConfigSource.DEFAULT),
            "auto_index": ConfigValue(True, ConfigSource.DEFAULT),
            "log_level": ConfigValue("INFO", ConfigSource.DEFAULT),
            "cors_origins": ConfigValue([
                "http://localhost:3000", "http://localhost:8000"
            ], ConfigSource.DEFAULT),
            "rate_limit_per_minute": ConfigValue(60, ConfigSource.DEFAULT),
            "enable_auth": ConfigValue(False, ConfigSource.DEFAULT),
            "api_key_hash": ConfigValue(None, ConfigSource.DEFAULT, is_sensitive=True),
            "jwt_secret": ConfigValue(None, ConfigSource.DEFAULT, is_sensitive=True),
            "max_request_size": ConfigValue(10 * 1024 * 1024, ConfigSource.DEFAULT),
            "max_file_size": ConfigValue(10 * 1024 * 1024, ConfigSource.DEFAULT),
            "config_version": ConfigValue(CONFIG_VERSION, ConfigSource.DEFAULT),
        }
        return defaults

    def _load_from_file(self) -> Dict[str, ConfigValue]:
        """Load configuration from file."""
        config_values: Dict[str, ConfigValue] = {}

        if not os.path.exists(self.config_path):
            logger.debug(f"Config file not found at {self.config_path}")
            return config_values

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            version = data.get("config_version", "1.0")
            if version != CONFIG_VERSION:
                logger.info(f"Config version mismatch: {version} -> {CONFIG_VERSION}")
                self._migrate_config(version)

            for key, value in data.items():
                if key == "config_version":
                    continue
                config_values[key] = ConfigValue(value, ConfigSource.FILE, self.config_path)

            logger.info(f"Config loaded from {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            logger.warning(f"Using defaults due to corrupted config file: {self.config_path}")
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            logger.warning(f"Using defaults due to config load error")

        return config_values

    def _load_from_env(self) -> Dict[str, ConfigValue]:
        """Load configuration from environment variables."""
        config_values: Dict[str, ConfigValue] = {}

        env_mappings = {
            f"{self.ENV_PREFIX}INDEX_PATH": ("index_path", str),
            f"{self.ENV_PREFIX}DB_PATH": ("db_path", str),
            f"{self.ENV_PREFIX}MODEL_NAME": ("model_name", str),
            f"{self.ENV_PREFIX}MAX_SEQUENCE_LENGTH": ("max_sequence_length", int),
            f"{self.ENV_PREFIX}BATCH_SIZE": ("batch_size", int),
            f"{self.ENV_PREFIX}USE_QUANTIZATION": ("use_quantization", bool),
            f"{self.ENV_PREFIX}USE_GPU": ("use_gpu", bool),
            f"{self.ENV_PREFIX}NUM_THREADS": ("num_threads", int),
            f"{self.ENV_PREFIX}SERVER_HOST": ("server_host", str),
            f"{self.ENV_PREFIX}SERVER_PORT": ("server_port", int),
            f"{self.ENV_PREFIX}WATCH_DEBOUNCE_MS": ("watch_debounce_ms", int),
            f"{self.ENV_PREFIX}ENABLE_BM25": ("enable_bm25", bool),
            f"{self.ENV_PREFIX}BM25_WEIGHT": ("bm25_weight", float),
            f"{self.ENV_PREFIX}SEMANTIC_WEIGHT": ("semantic_weight", float),
            f"{self.ENV_PREFIX}MIN_BM25_SCORE_THRESHOLD": ("min_bm25_score_threshold", float),
            f"{self.ENV_PREFIX}BM25_CHUNK_LIMIT": ("bm25_chunk_limit", int),
            f"{self.ENV_PREFIX}IGNORE_PATTERNS": ("ignore_patterns", list),
            f"{self.ENV_PREFIX}THEME": ("theme", str),
            f"{self.ENV_PREFIX}MAX_RESULTS": ("max_results", int),
            f"{self.ENV_PREFIX}SHOW_LINE_NUMBERS": ("show_line_numbers", bool),
            f"{self.ENV_PREFIX}AUTO_INDEX": ("auto_index", bool),
            f"{self.ENV_PREFIX}LOG_LEVEL": ("log_level", str),
            f"{self.ENV_PREFIX}CORS_ORIGINS": ("cors_origins", list),
            f"{self.ENV_PREFIX}RATE_LIMIT": ("rate_limit_per_minute", int),
            f"{self.ENV_PREFIX}ENABLE_AUTH": ("enable_auth", bool),
            f"{self.ENV_PREFIX}MAX_REQUEST_SIZE": ("max_request_size", int),
            f"{self.ENV_PREFIX}MAX_FILE_SIZE": ("max_file_size", int),
        }

        sensitive_vars = {
            f"{self.ENV_PREFIX}JWT_SECRET": "jwt_secret",
            f"{self.ENV_PREFIX}API_KEY": "api_key_hash",
        }

        for env_var, (config_key, value_type) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    if value_type == bool:
                        parsed_value = value.lower() in ("true", "1", "yes")
                    elif value_type == int:
                        parsed_value = int(value)
                    elif value_type == float:
                        parsed_value = float(value)
                    elif value_type == list:
                        parsed_value = [x.strip() for x in value.split(",")]
                    else:
                        parsed_value = value

                    is_sensitive = env_var in sensitive_vars
                    config_values[config_key] = ConfigValue(
                        parsed_value, ConfigSource.ENVIRONMENT, env_var, is_sensitive
                    )
                    logger.debug(f"Loaded {config_key} from env: {env_var}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {e}")
                    continue

        for env_var, config_key in sensitive_vars.items():
            value = os.environ.get(env_var)
            if value is not None:
                if config_key == "api_key_hash":
                    import hashlib
                    import base64
                    salt = os.urandom(16)
                    key_hash = hashlib.scrypt(
                        value.encode(), salt=salt, n=16384, r=8, p=1, dklen=32
                    )
                    parsed_value = base64.b64encode(salt + key_hash).decode()
                else:
                    parsed_value = value

                config_values[config_key] = ConfigValue(
                    parsed_value, ConfigSource.ENVIRONMENT, env_var, True
                )
                logger.debug(f"Loaded sensitive {config_key} from env: {env_var}")

        return config_values

    def _load_cli_overrides(self) -> Dict[str, ConfigValue]:
        """Load CLI option overrides from constructor parameter."""
        config_values: Dict[str, ConfigValue] = {}

        cli_mappings = {
            "index_path": ("index_path", str),
            "db_path": ("db_path", str),
            "model_name": ("model_name", str),
            "server_host": ("server_host", str),
            "server_port": ("server_port", int),
            "log_level": ("log_level", str),
        }

        for cli_key, (config_key, value_type) in cli_mappings.items():
            if cli_key in self.cli_overrides:
                value = self.cli_overrides[cli_key]
                try:
                    if value_type == int:
                        parsed_value = int(value)
                    else:
                        parsed_value = value
                    config_values[config_key] = ConfigValue(
                        parsed_value, ConfigSource.CLI, f"CLI:{cli_key}"
                    )
                    logger.debug(f"Loaded CLI override {config_key} from: {cli_key}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid CLI override for {cli_key}: {e}")
                    continue

        return config_values

    def _validate_env_vars(self) -> None:
        """Validate environment variables on startup."""
        valid_env_vars = {
            "COARCH_INDEX_PATH", "COARCH_DB_PATH", "COARCH_MODEL_NAME",
            "COARCH_MAX_SEQUENCE_LENGTH", "COARCH_BATCH_SIZE", "COARCH_USE_QUANTIZATION",
            "COARCH_USE_GPU", "COARCH_NUM_THREADS", "COARCH_SERVER_HOST",
            "COARCH_SERVER_PORT", "COARCH_WATCH_DEBounce_MS", "COARCH_ENABLE_BM25",
            "COARCH_BM25_WEIGHT", "COARCH_SEMANTIC_WEIGHT", "COARCH_MIN_BM25_SCORE_THRESHOLD",
            "COARCH_BM25_CHUNK_LIMIT", "COARCH_IGNORE_PATTERNS", "COARCH_THEME",
            "COARCH_MAX_RESULTS", "COARCH_SHOW_LINE_NUMBERS", "COARCH_AUTO_INDEX",
            "COARCH_LOG_LEVEL", "COARCH_CORS_ORIGINS", "COARCH_RATE_LIMIT",
            "COARCH_ENABLE_AUTH", "COARCH_MAX_REQUEST_SIZE", "COARCH_MAX_FILE_SIZE",
            "COARCH_JWT_SECRET", "COARCH_API_KEY", "COARCH_LOG_JSON",
            "COARCH_INDEX_PATH", "COARCH_TEST_MODE",
        }

        for key, value in os.environ.items():
            if key.startswith("COARCH_") and key not in valid_env_vars:
                logger.warning(f"Unknown COARCH environment variable: {key}")

    def _merge_configs(
        self,
        defaults: Dict[str, ConfigValue],
        file_config: Dict[str, ConfigValue],
        env_config: Dict[str, ConfigValue],
        cli_config: Dict[str, ConfigValue]
    ) -> Tuple[Dict[str, ConfigValue], List[ConfigConflict]]:
        """Merge configurations from all sources with conflict detection.

        Priority order: CLI > Environment > File > Default
        """
        merged: Dict[str, ConfigValue] = copy.deepcopy(defaults)
        conflicts: List[ConfigConflict] = []

        all_sources = [
            (file_config, ConfigSource.FILE),
            (env_config, ConfigSource.ENVIRONMENT),
            (cli_config, ConfigSource.CLI),
        ]

        for source_config, source_type in all_sources:
            for key, value in source_config.items():
                if key in merged:
                    existing = merged[key]
                    if existing.value != value.value:
                        conflicts.append(ConfigConflict(
                            key=key,
                            values=[(existing.value, existing.source), (value.value, source_type)],
                            resolved_value=value.value,
                            resolution_strategy="highest_priority_wins"
                        ))
                merged[key] = value

        return merged, conflicts

    def load(self) -> "UnifiedConfigManager":
        """Load and merge configuration from all sources."""
        defaults = self._get_default_config()
        file_config = self._load_from_file()
        env_config = self._load_from_env()
        cli_config = self._load_cli_overrides()

        self._config_cache, self._conflicts = self._merge_configs(
            defaults, file_config, env_config, cli_config
        )

        if self._conflicts:
            for conflict in self._conflicts:
                logger.info(
                    f"Config conflict detected for '{conflict.key}': "
                    f"resolved to {conflict.resolved_value} "
                    f"({conflict.resolution_strategy})"
                )

        self._resolved_config = self._build_config_object()

        return self

    def _build_config_object(self) -> Any:
        """Build a CoarchConfig object from resolved values."""
        from .config import CoarchConfig

        kwargs = {}
        cache = self._config_cache or {}
        for key, config_value in cache.items():
            if key == "config_version":
                continue
            if hasattr(CoarchConfig, key):
                kwargs[key] = config_value.value

        return CoarchConfig(**kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if self._config_cache is None:
            self.load()

        cache = self._config_cache
        assert cache is not None
        if key in cache:
            return cache[key].value
        return default

    def get_with_source(self, key: str) -> Optional[ConfigValue]:
        """Get a configuration value with its source information."""
        if self._config_cache is None:
            self.load()
        cache = self._config_cache
        assert cache is not None
        return cache.get(key)

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.FILE) -> None:
        """Set a configuration value."""
        if self._config_cache is None:
            self.load()

        cache = self._config_cache
        assert cache is not None
        cache[key] = ConfigValue(value, source)
        self._save_to_file()
        self._resolved_config = self._build_config_object()

    def _save_to_file(self) -> None:
        """Save current configuration to file."""
        if self._config_cache is None:
            return

        config_data = {}
        for key, config_value in self._config_cache.items():
            if config_value.is_sensitive:
                continue
            if key == "config_version":
                continue
            config_data[key] = config_value.value

        config_data["config_version"] = CONFIG_VERSION

        Path(self.config_path).mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=2)
        except PermissionError:
            logger.warning(f"Permission denied saving config to {self.config_path}")
            return

        try:
            os.chmod(self.config_path, 0o600)
        except (OSError, NotImplementedError):
            pass

        logger.info(f"Config saved to {self.config_path}")

    def _migrate_config(self, from_version: str) -> None:
        """Migrate configuration from an older version."""
        logger.info(f"Migrating config from version {from_version} to {CONFIG_VERSION}")

        migration_methods = {
            "1.0": self._migrate_v1_to_v2,
            "2.0": self._migrate_v2_to_v3,
        }

        current_version = from_version
        while current_version != CONFIG_VERSION:
            if current_version in migration_methods:
                migration_methods[current_version]()
                current_version = self._get_next_version(current_version)
            else:
                logger.warning(f"Unknown config version {current_version}, using defaults")
                break

    def _get_next_version(self, version: str) -> str:
        """Get the next version in the migration chain."""
        version_map = {"1.0": "2.0", "2.0": "3.0"}
        return version_map.get(version, CONFIG_VERSION)

    def _migrate_v1_to_v2(self) -> None:
        """Migrate from v1.0 to v2.0."""
        if self._config_cache is None:
            return

        if "jwt_secret" in self._config_cache:
            del self._config_cache["jwt_secret"]
        if "api_key_hash" in self._config_cache:
            del self._config_cache["api_key_hash"]

        self._save_to_file()
        logger.info("Migrated config to v2.0 (secrets removed from file)")

    def _migrate_v2_to_v3(self) -> None:
        """Migrate from v2.0 to v3.0."""
        if self._config_cache is None:
            return

        if "ignore_patterns" in self._config_cache:
            default_patterns = self._get_default_config()["ignore_patterns"].value
            current = self._config_cache["ignore_patterns"].value
            if isinstance(current, str):
                self._config_cache["ignore_patterns"] = ConfigValue(
                    [x.strip() for x in current.split(",")],
                    ConfigSource.FILE
                )

        self._config_cache["config_version"] = ConfigValue(
            "3.0", ConfigSource.DEFAULT
        )
        self._save_to_file()
        logger.info("Migrated config to v3.0")

    def save_history(self, description: Optional[str] = None) -> str:
        """Save current configuration to history for rollback."""
        if not self.enable_history or self._config_cache is None:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(
            self.history_dir, f"config_{timestamp}.json"
        )

        config_data = {}
        for key, config_value in self._config_cache.items():
            if config_value.is_sensitive:
                continue
            config_data[key] = config_value.value

        entry = ConfigHistoryEntry(
            timestamp=timestamp,
            config_snapshot=config_data,
            config_version=self._config_version,
            change_description=description
        )

        with open(history_file, "w") as f:
            json.dump({
                "timestamp": entry.timestamp,
                "config_version": entry.config_version,
                "change_description": entry.change_description,
                "config": entry.config_snapshot
            }, f, indent=2)

        logger.info(f"Saved config history to {history_file}")
        return timestamp

    def list_history(self) -> List[ConfigHistoryEntry]:
        """List all configuration history entries."""
        entries = []

        if not os.path.exists(self.history_dir):
            return entries

        for filename in os.listdir(self.history_dir):
            if filename.startswith("config_") and filename.endswith(".json"):
                filepath = os.path.join(self.history_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    entries.append(ConfigHistoryEntry(
                        timestamp=data["timestamp"],
                        config_snapshot=data.get("config", {}),
                        config_version=data.get("config_version", "unknown"),
                        change_description=data.get("change_description")
                    ))
                except Exception as e:
                    logger.warning(f"Failed to read config history {filename}: {e}")

        return sorted(entries, key=lambda x: x.timestamp, reverse=True)

    def rollback(self, timestamp: str) -> bool:
        """Rollback configuration to a specific history entry.
        
        Args:
            timestamp: The timestamp of the history entry to restore
            
        Returns:
            True if rollback was successful, False otherwise
        """
        history_file = os.path.join(self.history_dir, f"config_{timestamp}.json")

        if not os.path.exists(history_file):
            logger.error(f"Config history not found: {timestamp}")
            return False

        try:
            with open(history_file, "r") as f:
                data = json.load(f)

            config_data = data.get("config", {})
            self._config_cache = {}

            for key, value in config_data.items():
                self._config_cache[key] = ConfigValue(
                    value, ConfigSource.FILE, self.config_path
                )

            self._save_to_file()
            self._resolved_config = self._build_config_object()

            logger.info(f"Rolled back config to {timestamp}")
            return True
        except Exception as e:
            logger.exception(f"Failed to rollback config: {e}")
            return False

    def get_resolved_config(self) -> Any:
        """Get the resolved configuration object."""
        if self._resolved_config is None:
            self.load()
        return self._resolved_config

    def print_config(self, format: str = "text") -> str:
        """Print the resolved configuration.
        
        Args:
            format: Output format ('text', 'json', 'env')
            
        Returns:
            String representation of the configuration
        """
        if self._config_cache is None:
            self.load()

        cache = self._config_cache
        assert cache is not None

        if format == "json":
            config_data = {
                k: ("***" if v.is_sensitive else v.value)
                for k, v in cache.items()
            }
            return json.dumps(config_data, indent=2)

        elif format == "env":
            lines = ["# Coarch Configuration Environment Variables"]
            lines.append(f"# Generated: {datetime.now().isoformat()}")
            lines.append("")

            for key, config_value in sorted(cache.items()):
                if config_value.is_sensitive:
                    continue
                env_var = f"COARCH_{key.upper()}"
                value = config_value.value
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, list):
                    value = ",".join(value)
                lines.append(f"{env_var}={value}")

            return "\n".join(lines)

        else:
            lines = ["Coarch Configuration", "=" * 40, ""]

            for key, config_value in sorted(cache.items()):
                if config_value.is_sensitive:
                    display_value = "***"
                else:
                    display_value = config_value.value

                source_indicator = {
                    ConfigSource.DEFAULT: "[default]",
                    ConfigSource.FILE: "[file]",
                    ConfigSource.ENVIRONMENT: "[env]",
                    ConfigSource.CLI: "[CLI]",
                }.get(config_value.source, "")

                lines.append(f"{key}: {display_value} {source_indicator}")

            return "\n".join(lines)

    def validate(self) -> List[str]:
        """Validate the current configuration."""
        errors = []

        port = self.get("server_port")
        if port is not None:
            try:
                validate_port(port, must_be_available=False)
            except CoarchValidationError as e:
                errors.append(f"server_port: {e}")

        max_results = self.get("max_results")
        if max_results is not None:
            try:
                validate_limit(max_results, min_value=1, max_value=1000)
            except CoarchValidationError as e:
                errors.append(f"max_results: {e}")

        host = self.get("server_host")
        if host is not None:
            try:
                validate_host(host)
            except CoarchValidationError as e:
                errors.append(f"server_host: {e}")

        batch_size = self.get("batch_size")
        if batch_size is not None and batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {batch_size}")

        rate_limit = self.get("rate_limit_per_minute")
        if rate_limit is not None and rate_limit < 1:
            errors.append(f"rate_limit_per_minute must be >= 1, got {rate_limit}")

        return errors

    def get_conflicts(self) -> List[ConfigConflict]:
        """Get list of configuration conflicts detected during loading."""
        if self._config_cache is None:
            self.load()
        return self._conflicts


class ConfigTemplateManager:
    """Manager for configuration templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize template manager."""
        self.templates_dir = templates_dir or os.path.join(
            os.path.expanduser(CONFIG_DIR), CONFIG_TEMPLATES_DIR
        )
        Path(self.templates_dir).mkdir(parents=True, exist_ok=True)
        self._builtin_templates = self._get_builtin_templates()

    def _get_builtin_templates(self) -> Dict[str, ConfigTemplate]:
        """Get built-in configuration templates."""
        return {
            "development": ConfigTemplate(
                name="development",
                description="Development configuration with debugging enabled",
                values={
                    "log_level": "DEBUG",
                    "batch_size": 16,
                    "use_quantization": False,
                    "auto_index": True,
                    "max_results": 50,
                },
                tags=["dev", "debug"]
            ),
            "production": ConfigTemplate(
                name="production",
                description="Production configuration optimized for performance",
                values={
                    "log_level": "WARNING",
                    "batch_size": 64,
                    "use_quantization": True,
                    "auto_index": False,
                    "max_results": 100,
                    "rate_limit_per_minute": 120,
                },
                tags=["prod", "performance"]
            ),
            "research": ConfigTemplate(
                name="research",
                description="Configuration for research/analysis with high recall",
                values={
                    "log_level": "INFO",
                    "batch_size": 32,
                    "bm25_weight": 0.5,
                    "semantic_weight": 0.5,
                    "max_results": 100,
                    "min_bm25_score_threshold": 0.001,
                },
                tags=["research", "analysis"]
            ),
            "minimal": ConfigTemplate(
                name="minimal",
                description="Minimal configuration for resource-constrained environments",
                values={
                    "log_level": "ERROR",
                    "batch_size": 8,
                    "use_quantization": True,
                    "use_gpu": False,
                    "max_results": 10,
                    "enable_bm25": False,
                },
                tags=["minimal", "resource-constrained"]
            ),
            "gpu-optimized": ConfigTemplate(
                name="gpu-optimized",
                description="Configuration optimized for GPU acceleration",
                values={
                    "use_gpu": True,
                    "batch_size": 128,
                    "use_quantization": False,
                    "num_threads": 0,
                    "log_level": "INFO",
                },
                tags=["gpu", "performance", "cuda"]
            ),
        }

    def list_templates(self) -> List[ConfigTemplate]:
        """List all available templates."""
        templates = list(self._builtin_templates.values())

        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.templates_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    templates.append(ConfigTemplate(
                        name=data.get("name", filename),
                        description=data.get("description", "Custom template"),
                        values=data.get("values", {}),
                        tags=data.get("tags", [])
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load template {filename}: {e}")

        return templates

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a specific template by name."""
        if name in self._builtin_templates:
            return self._builtin_templates[name]

        filepath = os.path.join(self.templates_dir, f"{name}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                return ConfigTemplate(
                    name=data.get("name", name),
                    description=data.get("description", "Custom template"),
                    values=data.get("values", {}),
                    tags=data.get("tags", [])
                )
            except Exception as e:
                logger.warning(f"Failed to load template {name}: {e}")

        return None

    def apply_template(
        self,
        template_name: str,
        config_manager: UnifiedConfigManager,
        save: bool = True
    ) -> bool:
        """Apply a template to the configuration.
        
        Args:
            template_name: Name of the template to apply
            config_manager: Config manager to update
            save: Whether to save the config after applying
            
        Returns:
            True if template was applied successfully
        """
        template = self.get_template(template_name)
        if template is None:
            logger.error(f"Template not found: {template_name}")
            return False

        config_manager.load()
        for key, value in template.values.items():
            config_manager.set(key, value, ConfigSource.FILE)

        if save:
            config_manager.save_history(f"Applied template: {template_name}")

        logger.info(f"Applied template: {template_name}")
        return True

    def save_template(
        self,
        name: str,
        values: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Save a custom configuration template.
        
        Args:
            name: Template name
            values: Configuration values to save
            description: Template description
            tags: Template tags
            
        Returns:
            True if template was saved successfully
        """
        filepath = os.path.join(self.templates_dir, f"{name}.json")

        template_data = {
            "name": name,
            "description": description,
            "values": values,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }

        try:
            with open(filepath, "w") as f:
                json.dump(template_data, f, indent=2)
            logger.info(f"Saved template to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            return False


def get_unified_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> UnifiedConfigManager:
    """Get a unified configuration manager instance.
    
    This is the main entry point for configuration access.
    
    Args:
        config_path: Optional custom config path
        cli_overrides: Optional CLI option overrides
        
    Returns:
        Configured UnifiedConfigManager instance
    """
    manager = UnifiedConfigManager(config_path, cli_overrides)
    manager.load()
    return manager


def init_config(
    config_path: Optional[str] = None,
    template: Optional[str] = None
) -> UnifiedConfigManager:
    """Initialize configuration with optional template.
    
    Args:
        config_path: Optional custom config path
        template: Optional template to apply
        
    Returns:
        Initialized UnifiedConfigManager instance
    """
    manager = UnifiedConfigManager(config_path)
    manager.load()

    if template:
        template_manager = ConfigTemplateManager()
        template_manager.apply_template(template, manager)

    manager.save_history("Initial configuration")
    return manager
