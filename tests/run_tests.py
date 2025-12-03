#!/usr/bin/env python3
"""
Test Runner for od-parse

This script runs all unit tests with proper environment setup
and provides detailed reporting.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import subprocess
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_test_environment():
    """Set up a clean test environment with mock API keys."""
    # Set mock API keys for testing (these are fake keys for testing only)
    test_env = {
        'OPENAI_API_KEY': 'sk-test1234567890abcdef1234567890abcdef1234567890abcdef',
        'GOOGLE_API_KEY': 'AIzaSyTest1234567890abcdef1234567890abcdef1234567890',
        'ANTHROPIC_API_KEY': 'claude-test1234567890abcdef1234567890abcdef1234567890',
        'AZURE_OPENAI_API_KEY': 'azure-test1234567890abcdef1234567890abcdef1234567890',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_VERSION': '2024-02-15-preview',
        'OD_PARSE_LOG_LEVEL': 'WARNING',  # Reduce log noise during tests
        'ENABLE_DEEP_LEARNING': 'false',  # Disable heavy features for faster tests
        'ENABLE_ADVANCED_OCR': 'false',
        'ENABLE_MULTILINGUAL': 'false'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("✅ Test environment configured with mock API keys")


def run_unit_tests():
    """Run all unit tests."""
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"🏁 Test Summary")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"   Time: {end_time - start_time:.2f}s")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n❌ Some tests failed")
    
    return success


def run_pytest():
    """Run tests using pytest for better reporting."""
    print("🧪 Running Tests with pytest")
    print("=" * 50)
    
    try:
        # Run pytest with coverage if available
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-v',
            '--tb=short',
            '--durations=10'
        ]
        
        # Add coverage if available
        try:
            import coverage
            cmd.extend(['--cov=od_parse', '--cov-report=term-missing'])
        except ImportError:
            print("📊 Coverage not available (install with: pip install coverage pytest-cov)")
        
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False


def run_security_checks():
    """Run basic security checks."""
    print("\n🔒 Security Checks")
    print("=" * 30)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: No real API keys in code
    total_checks += 1
    print("🔍 Checking for hardcoded API keys...")
    
    suspicious_patterns = ['sk-', 'AIza', 'claude-', 'anthropic']
    found_keys = []
    
    for py_file in project_root.rglob('*.py'):
        if 'venv' in str(py_file) or '.git' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in suspicious_patterns:
                    if pattern in content and 'test' not in content.lower():
                        # Check if it's not a test/example pattern
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line and not any(test_word in line.lower()
                                for test_word in ['test', 'example', 'your-api-key-here', 'mock',
                                                 'placeholder', 'template', 'sk-test', 'aiza.*test',
                                                 'claude-test', 'azure-test', 'fake', 'dummy']):
                                found_keys.append(f"{py_file}:{i}")
        except Exception:
            continue
    
    if not found_keys:
        print("   ✅ No hardcoded API keys found")
        checks_passed += 1
    else:
        print("   ❌ Potential API keys found:")
        for location in found_keys:
            print(f"      {location}")
    
    # Check 2: .env file not tracked
    total_checks += 1
    print("🔍 Checking .env file tracking...")
    
    env_file = project_root / '.env'
    if env_file.exists():
        # Check if .env is in .gitignore
        gitignore = project_root / '.gitignore'
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                gitignore_content = f.read()
                if '.env' in gitignore_content:
                    print("   ✅ .env file properly ignored")
                    checks_passed += 1
                else:
                    print("   ❌ .env file exists but not in .gitignore")
        else:
            print("   ❌ .env file exists but no .gitignore found")
    else:
        print("   ✅ No .env file in repository")
        checks_passed += 1
    
    # Check 3: Test files properly ignored
    total_checks += 1
    print("🔍 Checking for debug/test files...")
    
    debug_files = list(project_root.glob('test_*.py')) + list(project_root.glob('debug_*.py'))
    debug_files = [f for f in debug_files if 'tests/' not in str(f)]
    
    if not debug_files:
        print("   ✅ No debug files in repository root")
        checks_passed += 1
    else:
        print("   ❌ Debug files found:")
        for f in debug_files:
            print(f"      {f}")
    
    print(f"\n🔒 Security Score: {checks_passed}/{total_checks}")
    return checks_passed == total_checks


def main():
    """Main test runner."""
    print("🚀 od-parse Test Suite")
    print("=" * 60)
    
    # Setup test environment
    setup_test_environment()
    
    # Run security checks first
    security_ok = run_security_checks()
    
    # Choose test runner
    use_pytest = '--pytest' in sys.argv or '-p' in sys.argv
    
    if use_pytest:
        tests_ok = run_pytest()
    else:
        tests_ok = run_unit_tests()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏆 Final Results")
    print(f"   Security Checks: {'✅ PASS' if security_ok else '❌ FAIL'}")
    print(f"   Unit Tests: {'✅ PASS' if tests_ok else '❌ FAIL'}")
    
    if security_ok and tests_ok:
        print("\n🎉 All checks passed! Ready for production.")
        return 0
    else:
        print("\n❌ Some checks failed. Please review and fix.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
