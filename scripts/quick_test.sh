#!/bin/bash
# Quick test script - runs only tests (faster)
# Usage: ./scripts/quick_test.sh

set -e

echo "🧪 Running quick tests..."
echo "========================="
echo ""

pytest tests/ -v --tb=short

echo ""
echo "✅ Tests completed!"

