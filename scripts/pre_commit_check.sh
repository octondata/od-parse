#!/bin/bash
# Pre-commit check script - runs all CI checks locally
# Usage: ./scripts/pre_commit_check.sh

set -e  # Exit on error

echo "🔍 Running pre-commit checks..."
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if any checks fail
FAILED=0

# Function to run a check and track failures
run_check() {
    local name=$1
    shift
    echo -e "${YELLOW}Running: $name${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ $name passed${NC}\n"
    else
        echo -e "${RED}✗ $name failed${NC}\n"
        FAILED=1
    fi
}

# 1. Format check with black
run_check "Black format check" black --check od_parse examples

# 2. Lint with flake8 (critical errors)
run_check "Flake8 critical checks" flake8 od_parse --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Lint with flake8 (warnings)
run_check "Flake8 style checks" flake8 od_parse --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 4. Type check with mypy (non-blocking)
echo -e "${YELLOW}Running: Type check (mypy)${NC}"
if mypy od_parse --ignore-missing-imports --no-strict-optional 2>/dev/null; then
    echo -e "${GREEN}✓ Type check passed${NC}\n"
else
    echo -e "${YELLOW}⚠ Type check has warnings (non-blocking)${NC}\n"
fi

# 5. Security check with bandit (non-blocking)
echo -e "${YELLOW}Running: Security check (bandit)${NC}"
if bandit -r od_parse -ll 2>/dev/null; then
    echo -e "${GREEN}✓ Security check passed${NC}\n"
else
    echo -e "${YELLOW}⚠ Security check has warnings (non-blocking)${NC}\n"
fi

# 6. Run tests
echo -e "${YELLOW}Running: Tests${NC}"
if python -c "import pytest_cov" 2>/dev/null; then
    if pytest tests/ -v --cov=od_parse --cov-report=term-missing; then
        echo -e "${GREEN}✓ Tests passed${NC}\n"
    else
        echo -e "${RED}✗ Tests failed${NC}\n"
        FAILED=1
    fi
else
    echo -e "${YELLOW}⚠ pytest-cov not installed, running tests without coverage${NC}"
    if pytest tests/ -v; then
        echo -e "${GREEN}✓ Tests passed${NC}\n"
    else
        echo -e "${RED}✗ Tests failed${NC}\n"
        FAILED=1
    fi
fi

# Summary
echo "=================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to commit.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Please fix before committing.${NC}"
    exit 1
fi

