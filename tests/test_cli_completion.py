#!/usr/bin/env python3
"""Tests for CLI-6: Command Aliases and Tab Completion."""

import os
import sys
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
import pytest

# Add parent directory to path to import CLI module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.main import (
    SUPPORTED_LANGUAGES, 
    COMMAND_ALIASES,
    get_all_commands_with_aliases,
    complete_command_names,
    complete_languages,
    complete_hostnames,
    complete_ports,
    get_completion_script,
    install_completion
)


class TestCommandAliases:
    """Test command aliases functionality."""

    def test_supported_languages(self):
        """Test that supported languages are properly defined."""
        expected_languages = [
            "python", "javascript", "typescript", "java", "go", "rust", "ruby", "php",
            "csharp", "cpp", "c", "scala", "kotlin", "swift", "shell", "bash", "zsh",
            "powershell", "sql", "html", "css", "scss", "less", "json", "yaml", "toml",
            "markdown", "text", "dockerfile", "makefile", "r", "matlab", "perl", "lua"
        ]
        
        assert SUPPORTED_LANGUAGES == expected_languages
        assert len(SUPPORTED_LANGUAGES) > 0

    def test_command_aliases(self):
        """Test that command aliases are properly defined."""
        expected_aliases = {
            'find': 'search',
            'stats': 'status', 
            'server': 'serve',
            'remove': 'delete',
            'ping': 'health',
            'idx': 'index',
            'setup': 'init'
        }
        
        assert COMMAND_ALIASES == expected_aliases

    def test_get_all_commands_with_aliases(self):
        """Test that all commands including aliases are returned."""
        commands = get_all_commands_with_aliases()
        
        # Should include main commands
        assert 'index' in commands
        assert 'search' in commands
        assert 'serve' in commands
        assert 'status' in commands
        assert 'delete' in commands
        assert 'health' in commands
        assert 'init' in commands
        assert 'completion' in commands
        
        # Should include aliases
        assert 'find' in commands  # alias for search
        assert 'stats' in commands  # alias for status
        assert 'server' in commands  # alias for serve
        assert 'remove' in commands  # alias for delete
        assert 'ping' in commands  # alias for health
        assert 'idx' in commands  # alias for index
        assert 'setup' in commands  # alias for init


class TestCompletionCallbacks:
    """Test completion callback functions."""

    def test_complete_command_names(self):
        """Test command name completion."""
        # Test with empty string
        result = complete_command_names(None, [], "")
        expected = get_all_commands_with_aliases()
        assert sorted(result) == sorted(expected)
        
        # Test with partial match
        result = complete_command_names(None, [], "s")
        assert all(cmd.startswith("s") for cmd in result)
        assert "search" in result
        assert "serve" in result
        assert "status" in result
        
        # Test with exact match
        result = complete_command_names(None, [], "search")
        assert "search" in result

    def test_complete_languages(self):
        """Test language completion."""
        # Test with empty string
        result = complete_languages(None, [], "")
        assert sorted(result) == sorted(SUPPORTED_LANGUAGES)
        
        # Test with partial match
        result = complete_languages(None, [], "p")
        assert all(lang.startswith("p") for lang in result)
        assert "python" in result
        assert "perl" in result
        
        # Test with exact match
        result = complete_languages(None, [], "python")
        assert "python" in result
        assert len(result) == 1

    def test_complete_hostnames(self):
        """Test hostname completion."""
        # Test with empty string
        result = complete_hostnames(None, [], "")
        expected = ['localhost', '127.0.0.1', '0.0.0.0']
        assert sorted(result) == sorted(expected)
        
        # Test with partial match
        result = complete_hostnames(None, [], "127")
        assert "127.0.0.1" in result
        assert len(result) == 1

    def test_complete_ports(self):
        """Test port number completion."""
        # Test with empty string
        result = complete_ports(None, [], "")
        expected = ['8000', '8080', '3000', '5000', '9000', '5432', '3306', '6379']
        assert sorted(result) == sorted(expected)
        
        # Test with partial match
        result = complete_ports(None, [], "80")
        assert "8000" in result
        assert "8080" in result
        
        # Test with number prefix
        result = complete_ports(None, [], "8")
        assert "8000" in result
        assert "8080" in result


class TestCompletionScripts:
    """Test shell completion script generation."""

    def test_get_completion_script_bash(self):
        """Test bash completion script generation."""
        script = get_completion_script("bash")
        
        # Should contain bash-specific elements
        assert "#!/bin/bash" in script
        assert "_coarch_completion" in script
        assert "complete -F _coarch_completion coarch" in script
        
        # Should include all commands and aliases
        assert "index" in script
        assert "search" in script
        assert "find" in script  # alias
        assert "serve" in script
        assert "server" in script  # alias

    def test_get_completion_script_zsh(self):
        """Test zsh completion script generation."""
        script = get_completion_script("zsh")
        
        # Should contain zsh-specific elements
        assert "#compdef coarch" in script
        assert "_coarch_cli" in script
        assert "_coarch_commands" in script
        
        # Should include all commands and aliases
        assert "index" in script
        assert "search" in script
        assert "find" in script  # alias
        assert "serve" in script
        assert "server" in script  # alias

    def test_get_completion_script_fish(self):
        """Test fish completion script generation."""
        script = get_completion_script("fish")
        
        # Should contain fish-specific elements
        assert "#!/usr/bin/env fish" in script
        assert "__fish_coarch_complete" in script
        assert "complete -f -c coarch" in script
        
        # Should include all commands and aliases
        assert "index" in script
        assert "search" in script
        assert "find" in script  # alias
        assert "serve" in script
        assert "server" in script  # alias

    def test_get_completion_script_unsupported(self):
        """Test unsupported shell type."""
        script = get_completion_script("unsupported")
        assert script == ""


class TestCompletionInstallation:
    """Test completion installation functionality."""

    @patch('os.path.expanduser')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    def test_install_bash_completion(self, mock_exists, mock_open, mock_makedirs, mock_expanduser):
        """Test bash completion installation."""
        # Mock setup
        mock_expanduser.side_effect = lambda path: path.replace('~', '/home/user')
        mock_exists.return_value = True
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test installation
        success = install_completion("bash", "/home/user/.bashrc")
        
        assert success
        # Should write completion script
        mock_file.write.assert_called()

    @patch('os.path.expanduser')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    def test_install_fish_completion(self, mock_open, mock_makedirs, mock_expanduser):
        """Test fish completion installation."""
        # Mock setup
        mock_expanduser.side_effect = lambda path: path.replace('~', '/home/user')
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test installation
        success = install_completion("fish")
        
        assert success
        # Should create completions directory and write file
        mock_makedirs.assert_called_once()
        mock_file.write.assert_called()

    @patch('os.path.expanduser')
    def test_install_unsupported_shell(self, mock_expanduser):
        """Test installation with unsupported shell."""
        mock_expanduser.side_effect = lambda path: path.replace('~', '/home/user')
        
        success = install_completion("unsupported")
        assert not success

    @patch('os.makedirs')
    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_install_bash_completion_permission_error(self, mock_open, mock_makedirs):
        """Test bash completion installation with permission error."""
        success = install_completion("bash")
        assert not success


class TestCLIIntegration:
    """Test CLI integration with completion."""

    def test_completion_command_exists(self):
        """Test that completion command is available."""
        # This would require actually importing and testing the CLI
        # For now, we'll just verify the function exists
        assert callable(get_completion_script)
        assert callable(install_completion)

    def test_aliases_dont_conflict(self):
        """Test that aliases don't conflict with main commands."""
        commands = get_all_commands_with_aliases()
        aliases = set(COMMAND_ALIASES.keys())
        main_commands = set(['index', 'search', 'serve', 'status', 'delete', 'health', 'init'])
        
        # No overlap between aliases and main commands
        assert not aliases.intersection(main_commands)
        
        # All aliases are valid
        for alias, command in COMMAND_ALIASES.items():
            assert alias in commands
            assert command in commands
        
        # Verify specific new aliases
        assert 'idx' in aliases
        assert COMMAND_ALIASES['idx'] == 'index'
        assert 'setup' in aliases
        assert COMMAND_ALIASES['setup'] == 'init'


def test_completion_help():
    """Test that completion command shows help properly."""
    # Test the help functionality by checking if completion scripts contain proper documentation
    for shell in ["bash", "zsh", "fish"]:
        script = get_completion_script(shell)
        assert len(script) > 0
        assert shell in script.lower()


def test_new_aliases_in_completion_scripts():
    """Test that new aliases (idx, setup) are included in all completion scripts."""
    # Test bash completion
    bash_script = get_completion_script("bash")
    assert "idx" in bash_script
    assert "setup" in bash_script
    
    # Test zsh completion
    zsh_script = get_completion_script("zsh")
    assert "idx" in zsh_script
    assert "setup" in zsh_script
    
    # Test fish completion
    fish_script = get_completion_script("fish")
    assert "idx" in fish_script
    assert "setup" in fish_script


def test_context_aware_completion():
    """Test that context-aware completion works correctly."""
    from cli.main import complete_file_paths, complete_languages
    
    # Test language completion with partial input
    result = complete_languages(None, [], "pyt")
    assert "python" in result
    
    # Test file path completion - should handle various path formats
    # (Note: actual file path completion depends on the filesystem)
    result = complete_file_paths(None, [], "./")
    assert len(result) >= 0  # Just verify it doesn't crash


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])