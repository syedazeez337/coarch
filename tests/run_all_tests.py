#!/usr/bin/env python3
"""Test runner for Coarch CLI integration tests.

This script runs all CLI integration tests and generates a comprehensive report.
Run with: python tests/run_all_tests.py
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

COARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd: list, timeout: int = 300, description: str = "") -> dict:
    """Run a command and return detailed results."""
    print(f"\n{'=' * 60}")
    if description:
        print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=COARCH_DIR,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed = time.time() - start_time

        output = {
            'command': ' '.join(cmd),
            'return_code': result.returncode,
            'elapsed': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }

        if result.stdout:
            print(result.stdout[:2000])
        if result.stderr:
            print(result.stderr[:500])

        print(f"\nCompleted in {elapsed:.2f}s with code {result.returncode}")

        return output

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'command': ' '.join(cmd),
            'return_code': -1,
            'elapsed': elapsed,
            'stdout': '',
            'stderr': 'Command timed out',
            'success': False
        }


def run_integration_tests():
    """Run CLI integration tests."""
    print("\n" + "=" * 70)
    print("RUNNING CLI INTEGRATION TESTS")
    print("=" * 70)

    return run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_cli_integration.py', '-v', '--tb=short'],
        timeout=300,
        description="Integration Tests"
    )


def run_smoke_tests():
    """Run smoke tests."""
    print("\n" + "=" * 70)
    print("RUNNING SMOKE TESTS")
    print("=" * 70)

    return run_command(
        [sys.executable, 'tests/smoke_tests.py'],
        timeout=120,
        description="Smoke Tests"
    )


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 70)

    return run_command(
        [sys.executable, 'tests/perf_benchmarks.py'],
        timeout=120,
        description="Performance Benchmarks"
    )


def run_unit_tests():
    """Run unit tests."""
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)

    return run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_coarch.py', 'tests/test_bm25.py',
         'tests/test_hybrid_search.py', '-v', '--tb=short'],
        timeout=180,
        description="Unit Tests"
    )


def generate_report(results: dict) -> str:
    """Generate a comprehensive test report."""
    report = []
    report.append("=" * 70)
    report.append("COARCH TEST REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        report.append(f"\n{test_name}:")
        report.append(f"  Status: {status}")
        report.append(f"  Return Code: {result['return_code']}")
        report.append(f"  Elapsed: {result['elapsed']:.2f}s")

    report.append("\n" + "=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)

    report.append(f"Total test suites: {total}")
    report.append(f"Passed: {passed}")
    report.append(f"Failed: {total - passed}")
    report.append(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        report.append("\nAll tests PASSED!")
    else:
        report.append("\nSome tests FAILED!")

    report.append("=" * 70)

    return '\n'.join(report)


def main():
    """Main test runner."""
    print("=" * 70)
    print("COARCH TEST SUITE")
    print("=" * 70)
    print(f"\nWorking directory: {COARCH_DIR}")
    print(f"Python: {sys.executable}")

    results = {}

    test_suites = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("Smoke Tests", run_smoke_tests),
        ("Performance Benchmarks", run_performance_benchmarks),
    ]

    for name, test_func in test_suites:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            results[name] = {
                'success': False,
                'return_code': -1,
                'elapsed': 0,
                'stderr': str(e)
            }

    report = generate_report(results)
    print(report)

    with open(os.path.join(COARCH_DIR, 'test_report.txt'), 'w') as f:
        f.write(report)

    with open(os.path.join(COARCH_DIR, 'test_report.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nReports saved to:")
    print(f"  - {os.path.join(COARCH_DIR, 'test_report.txt')}")
    print(f"  - {os.path.join(COARCH_DIR, 'test_report.json')}")

    all_passed = all(r['success'] for r in results.values())
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
