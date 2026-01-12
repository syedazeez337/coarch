"""Tests for unified configuration management."""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigSource:
    """Test ConfigSource enum."""

    def test_config_source_values(self):
        """Test ConfigSource enum has expected values."""
        from backend.unified_config import ConfigSource

        assert ConfigSource.DEFAULT.value == "default"
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.ENVIRONMENT.value == "environment"
        assert ConfigSource.CLI.value == "cli"


class TestConfigValue:
    """Test ConfigValue dataclass."""

    def test_config_value_creation(self):
        """Test creating a ConfigValue."""
        from backend.unified_config import ConfigValue, ConfigSource

        cv = ConfigValue("test_value", ConfigSource.FILE, "/path/to/config.json")
        assert cv.value == "test_value"
        assert cv.source == ConfigSource.FILE
        assert cv.source_name == "/path/to/config.json"
        assert cv.is_sensitive is False

    def test_config_value_sensitive(self):
        """Test creating a sensitive ConfigValue."""
        from backend.unified_config import ConfigValue, ConfigSource

        cv = ConfigValue("secret", ConfigSource.ENVIRONMENT, "COARCH_API_KEY", is_sensitive=True)
        assert cv.is_sensitive is True
        assert "***" in repr(cv)


class TestConfigConflict:
    """Test ConfigConflict dataclass."""

    def test_config_conflict_creation(self):
        """Test creating a ConfigConflict."""
        from backend.unified_config import ConfigConflict, ConfigSource

        conflict = ConfigConflict(
            key="server_port",
            values=[(8000, ConfigSource.FILE), (9000, ConfigSource.ENVIRONMENT)],
            resolved_value=9000,
            resolution_strategy="highest_priority_wins"
        )
        assert conflict.key == "server_port"
        assert len(conflict.values) == 2
        assert conflict.resolved_value == 9000


class TestConfigTemplate:
    """Test ConfigTemplate dataclass."""

    def test_config_template_creation(self):
        """Test creating a ConfigTemplate."""
        from backend.unified_config import ConfigTemplate

        template = ConfigTemplate(
            name="development",
            description="Dev config",
            values={"log_level": "DEBUG", "batch_size": 16},
            tags=["dev", "debug"]
        )
        assert template.name == "development"
        assert template.values["log_level"] == "DEBUG"
        assert "dev" in template.tags


class TestConfigHistoryEntry:
    """Test ConfigHistoryEntry dataclass."""

    def test_config_history_entry_creation(self):
        """Test creating a ConfigHistoryEntry."""
        from backend.unified_config import ConfigHistoryEntry

        entry = ConfigHistoryEntry(
            timestamp="20240112_120000",
            config_snapshot={"server_port": 8000},
            config_version="3.0",
            change_description="Initial config"
        )
        assert entry.timestamp == "20240112_120000"
        assert entry.config_version == "3.0"


class TestUnifiedConfigManager:
    """Test UnifiedConfigManager functionality."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_default_config_values(self, temp_config_dir):
        """Test that default configuration values are correct."""
        from backend.unified_config import UnifiedConfigManager, CONFIG_VERSION

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 8000
        assert manager.get("server_host") == "0.0.0.0"
        assert manager.get("log_level") == "INFO"
        assert manager.get("batch_size") == 32
        assert manager.get("config_version") == CONFIG_VERSION

    def test_environment_variable_override(self, temp_config_dir, monkeypatch):
        """Test that environment variables override defaults."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        monkeypatch.setenv("COARCH_SERVER_PORT", "9000")
        monkeypatch.setenv("COARCH_LOG_LEVEL", "DEBUG")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 9000
        assert manager.get("log_level") == "DEBUG"

        port_source = manager.get_with_source("server_port")
        assert port_source.source == ConfigSource.ENVIRONMENT

    def test_file_config_override(self, temp_config_dir):
        """Test that file configuration overrides defaults."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "server_port": 7000,
                "batch_size": 64,
                "config_version": "3.0"
            }, f)

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 7000
        assert manager.get("batch_size") == 64

        port_source = manager.get_with_source("server_port")
        assert port_source.source == ConfigSource.FILE

    def test_cli_override_has_highest_priority(self, temp_config_dir, monkeypatch):
        """Test that CLI overrides have highest priority."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        monkeypatch.setenv("COARCH_SERVER_PORT", "9000")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            cli_overrides={"server_port": 9500},
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 9500

        port_source = manager.get_with_source("server_port")
        assert port_source.source == ConfigSource.CLI

    def test_conflict_detection(self, temp_config_dir, monkeypatch):
        """Test that configuration conflicts are detected."""
        from backend.unified_config import UnifiedConfigManager

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"server_port": 7000, "config_version": "3.0"}, f)

        monkeypatch.setenv("COARCH_SERVER_PORT", "9000")

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        conflicts = manager.get_conflicts()
        server_port_conflicts = [c for c in conflicts if c.key == "server_port"]
        assert len(server_port_conflicts) >= 1
        assert server_port_conflicts[-1].resolved_value == 9000

    def test_get_with_source(self, temp_config_dir):
        """Test getting configuration with source information."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"server_port": 7000, "config_version": "3.0"}, f)

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        port_value = manager.get_with_source("server_port")
        assert port_value is not None
        assert port_value.value == 7000
        assert port_value.source == ConfigSource.FILE

    def test_set_config_value(self, temp_config_dir):
        """Test setting a configuration value."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        config_path = os.path.join(temp_config_dir, "config.json")

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        manager.set("log_level", "DEBUG")
        assert manager.get("log_level") == "DEBUG"

    def test_config_validation(self, temp_config_dir):
        """Test configuration validation."""
        from backend.unified_config import UnifiedConfigManager

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"server_port": 7000, "batch_size": -5, "config_version": "3.0"}, f)

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        errors = manager.validate()
        assert len(errors) > 0
        assert any("batch_size" in e for e in errors)

    def test_print_config_text_format(self, temp_config_dir):
        """Test printing configuration in text format."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        output = manager.print_config(format="text")
        assert "Coarch Configuration" in output
        assert "server_port" in output

    def test_print_config_json_format(self, temp_config_dir):
        """Test printing configuration in JSON format."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        output = manager.print_config(format="json")
        data = json.loads(output)
        assert "server_port" in data

    def test_print_config_env_format(self, temp_config_dir):
        """Test printing configuration in environment format."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        output = manager.print_config(format="env")
        assert "COARCH_SERVER_PORT" in output
        assert "=" in output

    def test_env_var_validation(self, temp_config_dir, monkeypatch):
        """Test environment variable validation on startup."""
        from backend.unified_config import UnifiedConfigManager

        monkeypatch.setenv("COARCH_UNKNOWN_VAR", "value")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )

        assert manager is not None


class TestConfigHistory:
    """Test configuration history and rollback functionality."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory with history."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        history_dir = config_dir / "config_history"
        history_dir.mkdir()
        return str(config_dir)

    def test_save_history(self, temp_config_dir):
        """Test saving configuration to history."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=True
        )
        manager.load()

        timestamp = manager.save_history("Test save")
        assert timestamp != ""

        history = manager.list_history()
        assert len(history) >= 1
        assert any(h.change_description == "Test save" for h in history)

    def test_rollback(self, temp_config_dir):
        """Test rolling back configuration."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=True
        )
        manager.load()

        manager.set("server_port", 5000)
        timestamp = manager.save_history("First save")

        manager.set("server_port", 6000)
        assert manager.get("server_port") == 6000

        success = manager.rollback(timestamp)
        assert success is True
        assert manager.get("server_port") == 5000

    def test_rollback_to_nonexistent(self, temp_config_dir):
        """Test rollback to a non-existent timestamp."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=True
        )
        manager.load()

        success = manager.rollback("nonexistent_timestamp")
        assert success is False


class TestConfigMigration:
    """Test configuration migration."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_migrate_v2_to_v3(self, temp_config_dir):
        """Test migrating from v2.0 to v3.0."""
        from backend.unified_config import UnifiedConfigManager, CONFIG_VERSION

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "ignore_patterns": "*.pyc,*.pyo,*.so",
                "config_version": "2.0"
            }, f)

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        assert manager.get("config_version") == CONFIG_VERSION

    def test_unknown_version_uses_defaults(self, temp_config_dir):
        """Test that unknown versions use defaults."""
        from backend.unified_config import UnifiedConfigManager

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "server_port": 7000,
                "config_version": "99.0"
            }, f)

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 7000


class TestConfigTemplateManager:
    """Test configuration template manager."""

    @pytest.fixture
    def temp_templates_dir(self, tmp_path):
        """Create a temporary templates directory."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        return str(templates_dir)

    def test_list_builtin_templates(self):
        """Test listing built-in templates."""
        from backend.unified_config import ConfigTemplateManager

        manager = ConfigTemplateManager()
        templates = manager.list_templates()

        template_names = [t.name for t in templates]
        assert "development" in template_names
        assert "production" in template_names
        assert "research" in template_names

    def test_get_builtin_template(self):
        """Test getting a specific built-in template."""
        from backend.unified_config import ConfigTemplateManager

        manager = ConfigTemplateManager()
        template = manager.get_template("development")

        assert template is not None
        assert template.values["log_level"] == "DEBUG"

    def test_get_nonexistent_template(self):
        """Test getting a non-existent template."""
        from backend.unified_config import ConfigTemplateManager

        manager = ConfigTemplateManager()
        template = manager.get_template("nonexistent")

        assert template is None

    def test_apply_template(self):
        """Test applying a template."""
        from backend.unified_config import (
            UnifiedConfigManager,
            ConfigTemplateManager,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UnifiedConfigManager(
                config_path=os.path.join(temp_dir, "config.json"),
                enable_history=False
            )
            manager.load()

            template_manager = ConfigTemplateManager()
            success = template_manager.apply_template("development", manager, save=False)

            assert success is True
            assert manager.get("log_level") == "DEBUG"

    def test_save_and_load_custom_template(self, temp_templates_dir):
        """Test saving and loading a custom template."""
        from backend.unified_config import ConfigTemplateManager

        manager = ConfigTemplateManager(temp_templates_dir)
        success = manager.save_template(
            "custom",
            {"batch_size": 128, "use_gpu": True},
            description="Custom GPU config",
            tags=["gpu", "custom"]
        )

        assert success is True

        template = manager.get_template("custom")
        assert template is not None
        assert template.values["batch_size"] == 128
        assert "gpu" in template.tags


class TestPriorityOrder:
    """Test configuration priority order."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_priority_cli_over_env_over_file_over_default(self, temp_config_dir, monkeypatch):
        """Test that CLI > Environment > File > Default priority order works."""
        from backend.unified_config import UnifiedConfigManager, ConfigSource

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"server_port": 1000, "config_version": "3.0"}, f)

        monkeypatch.setenv("COARCH_SERVER_PORT", "2000")

        manager = UnifiedConfigManager(
            config_path=config_path,
            cli_overrides={"server_port": 3000},
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 3000

        source = manager.get_with_source("server_port")
        assert source.source == ConfigSource.CLI


class TestSensitiveValues:
    """Test handling of sensitive configuration values."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_sensitive_values_not_in_output(self, temp_config_dir, monkeypatch):
        """Test that sensitive values are masked in output."""
        from backend.unified_config import UnifiedConfigManager

        monkeypatch.setenv("COARCH_JWT_SECRET", "super_secret")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        text_output = manager.print_config(format="text")
        assert "super_secret" not in text_output
        assert "***" in text_output

        json_output = manager.print_config(format="json")
        assert "super_secret" not in json_output

    def test_sensitive_values_not_saved_to_file(self, temp_config_dir, monkeypatch):
        """Test that sensitive values are not saved to config file."""
        from backend.unified_config import UnifiedConfigManager

        monkeypatch.setenv("COARCH_JWT_SECRET", "super_secret")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        manager._save_to_file()

        assert manager.get("jwt_secret") is not None


class TestIntegration:
    """Integration tests for configuration workflow."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_full_config_workflow(self, temp_config_dir, monkeypatch):
        """Test a complete configuration workflow."""
        from backend.unified_config import (
            UnifiedConfigManager,
            ConfigTemplateManager,
            init_config
        )

        monkeypatch.setenv("COARCH_LOG_LEVEL", "DEBUG")

        config = init_config(
            config_path=os.path.join(temp_config_dir, "config.json"),
            template="development"
        )

        assert config.get("log_level") == "DEBUG"
        assert config.get("batch_size") == 16

        template_manager = ConfigTemplateManager()
        templates = template_manager.list_templates()
        assert len(templates) > 0

        history = config.list_history()
        assert len(history) >= 1

    def test_get_unified_config_function(self, temp_config_dir):
        """Test the get_unified_config convenience function."""
        from backend.unified_config import get_unified_config, UnifiedConfigManager

        config = get_unified_config(
            config_path=os.path.join(temp_config_dir, "config.json")
        )

        assert config.get("server_port") == 8000
        assert isinstance(config, UnifiedConfigManager)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        config_dir = tmp_path / ".coarch"
        config_dir.mkdir()
        return str(config_dir)

    def test_nonexistent_config_file(self, temp_config_dir):
        """Test loading when config file doesn't exist."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "nonexistent.json"),
            enable_history=False
        )
        manager.load()

        assert manager.get("server_port") == 8000

    def test_corrupted_config_file(self, temp_config_dir):
        """Test handling of corrupted config file."""
        from backend.unified_config import UnifiedConfigManager

        config_path = os.path.join(temp_config_dir, "config.json")
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        manager = UnifiedConfigManager(
            config_path=config_path,
            enable_history=False
        )

        try:
            manager.load()
        except Exception:
            pass

        assert manager.get("server_port") == 8000

    def test_get_nonexistent_key(self, temp_config_dir):
        """Test getting a non-existent key returns default."""
        from backend.unified_config import UnifiedConfigManager

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        value = manager.get("nonexistent_key", "default_value")
        assert value == "default_value"

    def test_empty_env_list_value(self, temp_config_dir, monkeypatch):
        """Test handling of empty list from environment."""
        from backend.unified_config import UnifiedConfigManager

        monkeypatch.setenv("COARCH_CORS_ORIGINS", "")

        manager = UnifiedConfigManager(
            config_path=os.path.join(temp_config_dir, "config.json"),
            enable_history=False
        )
        manager.load()

        assert manager.get("cors_origins") == [""]
