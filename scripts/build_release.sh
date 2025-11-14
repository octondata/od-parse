#!/bin/bash
# Build and prepare release for od-parse
# Usage: ./scripts/build_release.sh [version]
# Example: ./scripts/build_release.sh 0.2.0

set -e

VERSION=${1:-"0.2.0"}
TAG="v${VERSION}"

echo "🚀 Building od-parse release ${VERSION}"
echo "=================================="

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build wheel
echo "📦 Building wheel..."
python3 -m build --wheel

# Check package
echo "✅ Checking package..."
python3 -m twine check dist/*

# Show what was built
echo ""
echo "📦 Built files:"
ls -lh dist/*.whl

echo ""
echo "✅ Build complete!"
echo ""
echo "📋 Next steps:"
echo "1. Test the wheel: pip install dist/od_parse-${VERSION}-py3-none-any.whl"
echo "2. Create git tag: git tag ${TAG}"
echo "3. Push tag: git push origin ${TAG}"
echo "4. Or manually upload to GitHub Releases:"
echo "   - Go to: https://github.com/octondata/od-parse/releases/new"
echo "   - Tag: ${TAG}"
echo "   - Title: Release ${VERSION}"
echo "   - Upload: dist/od_parse-${VERSION}-py3-none-any.whl"
echo ""
echo "💡 To automate releases, push the tag and GitHub Actions will create the release automatically."

