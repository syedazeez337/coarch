"""CLI test configuration and runner."""

import os
import sys
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_cli_tests():
    """Run CLI tests using pytest."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run CLI tests
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_cli/',
        '-v',
        '--tb=short',
        '--cov=cli',
        '--cov-report=term-missing'
    ]
    
    print("Running Coarch CLI tests...")
    print("=" * 50)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    print("=" * 50)
    if result.returncode == 0:
        print("✅ All CLI tests passed!")
    else:
        print("❌ Some CLI tests failed!")
        
    return result.returncode


def run_specific_cli_tests(pattern):
    """Run specific CLI tests matching pattern."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_cli/',
        '-v',
        '-k', pattern
    ]
    
    print(f"Running CLI tests matching: {pattern}")
    print("=" * 50)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    print("=" * 50)
    if result.returncode == 0:
        print(f"✅ CLI tests matching '{pattern}' passed!")
    else:
        print(f"❌ CLI tests matching '{pattern}' failed!")
        
    return result.returncode


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Coarch CLI tests')
    parser.add_argument('--pattern', '-p', help='Run tests matching pattern')
    parser.add_argument('--integration', '-i', action='store_true', help='Run only integration tests')
    parser.add_argument('--commands', '-c', action='store_true', help='Run only command tests')
    parser.add_argument('--errors', '-e', action='store_true', help='Run only error handling tests')
    
    args = parser.parse_args()
    
    if args.integration:
        sys.exit(run_specific_cli_tests('integration'))
    elif args.commands:
        sys.exit(run_specific_cli_tests('command'))
    elif args.errors:
        sys.exit(run_specific_cli_tests('error'))
    elif args.pattern:
        sys.exit(run_specific_cli_tests(args.pattern))
    else:
        sys.exit(run_cli_tests())