#!/usr/bin/env python3
"""Smoke tests for CI/CD pipeline.

This script provides quick smoke tests that can be run in CI/CD to verify
basic functionality without running the full test suite.
"""

import subprocess
import sys
import os
import time

COARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COARCH_DIR)


def run_command(cmd: list, timeout: int = 60) -> tuple:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=COARCH_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONPATH=COARCH_DIR)
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def test_version():
    """Test that coarch --version works."""
    print("Testing version command...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "--version"])
    if code == 0 and "Coarch" in stdout:
        print("  [PASS] Version command works")
        return True
    else:
        print(f"  [FAIL] Version command failed: {stderr}")
        return False


def test_help():
    """Test that coarch --help works."""
    print("Testing help command...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "--help"])
    if code == 0 and "Commands" in stdout:
        print("  [PASS] Help command works")
        return True
    else:
        print(f"  [FAIL] Help command failed: {stderr}")
        return False


def test_init():
    """Test that coarch init works."""
    print("Testing init command...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "init"])
    if code == 0 and "Configuration initialized" in stdout:
        print("  [PASS] Init command works")
        return True
    else:
        print(f"  [FAIL] Init command failed: {stderr}")
        return False


def test_status():
    """Test that coarch status works."""
    print("Testing status command...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "status"])
    if code == 0 and "Index Statistics" in stdout:
        print("  [PASS] Status command works")
        return True
    else:
        print(f"  [FAIL] Status command failed: {stderr}")
        return False


def test_config():
    """Test that coarch config works."""
    print("Testing config command...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "config"])
    if code == 0:
        print("  [PASS] Config command works")
        return True
    else:
        print(f"  [FAIL] Config command failed: {stderr}")
        return False


def test_completion():
    """Test that coarch completion works."""
    print("Testing completion command...")
    code, stdout, stderr = run_command([
        sys.executable, "-m", "cli.main", "completion", "--shell", "bash", "--no-install"
    ])
    if code == 0 and "bash" in stdout.lower():
        print("  [PASS] Completion command works")
        return True
    else:
        print(f"  [FAIL] Completion command failed: {stderr}")
        return False


def test_index_with_test_repo():
    """Test indexing the test_repo."""
    print("Testing index command with test_repo...")
    test_repo = os.path.join(COARCH_DIR, "test_repo")
    if not os.path.exists(test_repo):
        print(f"  [SKIP] test_repo not found at {test_repo}")
        return True

    code, stdout, stderr = run_command([
        sys.executable, "-m", "cli.main", "index", test_repo
    ], timeout=120)

    if code == 0 and "Files indexed" in stdout:
        print("  [PASS] Index command works")
        return True
    else:
        print(f"  [FAIL] Index command failed: {stderr}")
        return False


def test_search():
    """Test search command."""
    print("Testing search command...")
    code, stdout, stderr = run_command([
        sys.executable, "-m", "cli.main", "search", "hello"
    ])

    if code == 0 and ("Searching for" in stdout or "Found" in stdout or "No results" in stdout):
        print("  [PASS] Search command works")
        return True
    else:
        print(f"  [FAIL] Search command failed: {stderr}")
        return False


def test_invalid_index_path():
    """Test that invalid index path is handled."""
    print("Testing invalid index path handling...")
    code, stdout, stderr = run_command([
        sys.executable, "-m", "cli.main", "index", "/nonexistent/path"
    ])
    if code != 0:
        print("  [PASS] Invalid path handled correctly")
        return True
    else:
        print(f"  [FAIL] Should have failed for invalid path")
        return False


def test_invalid_port():
    """Test that invalid port is handled."""
    print("Testing invalid port handling...")
    code, stdout, stderr = run_command([
        sys.executable, "-m", "cli.main", "serve", "--port", "70000"
    ])
    if code != 0:
        print("  [PASS] Invalid port handled correctly")
        return True
    else:
        print(f"  [FAIL] Should have failed for invalid port")
        return False


def test_health_no_server():
    """Test health check when server is down."""
    print("Testing health check without server...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "health"])
    if code == 1 and "unhealthy" in stdout.lower():
        print("  [PASS] Health check correctly reports unhealthy server")
        return True
    else:
        print(f"  [FAIL] Health check should report unhealthy: {stderr}")
        return False


def test_delete_nonexistent():
    """Test deleting non-existent repo."""
    print("Testing delete non-existent repo...")
    code, stdout, stderr = run_command([sys.executable, "-m", "cli.main", "delete", "999"])
    if code == 0:
        print("  [PASS] Delete handles non-existent repo")
        return True
    else:
        print(f"  [FAIL] Delete failed: {stderr}")
        return False


def run_smoke_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("COARCH SMOKE TESTS")
    print("=" * 60)
    print()

    tests = [
        test_version,
        test_help,
        test_init,
        test_status,
        test_config,
        test_completion,
        test_index_with_test_repo,
        test_search,
        test_invalid_index_path,
        test_invalid_port,
        test_health_no_server,
        test_delete_nonexistent,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            results.append(False)
        print()

    print("=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
