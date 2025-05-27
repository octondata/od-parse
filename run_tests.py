#!/usr/bin/env python3
"""
Test runner for the od-parse library.
"""

import argparse
import sys
import unittest
from pathlib import Path


def run_tests(test_pattern=None, verbose=False):
    """
    Run tests matching the specified pattern.

    Args:
        test_pattern: Pattern to match test files (default: all tests)
        verbose: Whether to run tests in verbose mode
    """
    # Discover and run tests
    test_dir = Path(__file__).parent / "tests"

    if test_pattern:
        pattern = f"test_{test_pattern}.py"
    else:
        pattern = "test_*.py"

    verbosity = 2 if verbose else 1

    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


def main():
    """
    Parse command-line arguments and run tests.
    """
    parser = argparse.ArgumentParser(description="Run tests for the od-parse library.")
    parser.add_argument(
        "--module",
        "-m",
        help="Run tests for a specific module (e.g., parser, ocr, converter)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )

    args = parser.parse_args()

    # Run tests
    sys.exit(run_tests(args.module, args.verbose))


if __name__ == "__main__":
    main()
