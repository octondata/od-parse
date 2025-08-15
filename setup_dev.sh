#!/bin/bash
# Development setup script for od-parse
# This script helps users install od-parse from source

set -e  # Exit on any error

echo "🚀 od-parse Development Setup"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: This doesn't appear to be the od-parse directory"
    echo "Please run this script from the od-parse root directory"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
}

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first
echo "📚 Installing core dependencies..."
pip install pdfminer.six tabula-py opencv-python pillow pytesseract pandas numpy

# Install od-parse in development mode
echo "🔨 Installing od-parse in development mode..."
pip install -e .

# Test installation
echo "🧪 Testing installation..."
python -c "import od_parse; print('✅ od-parse imported successfully')" || {
    echo "❌ Error: od-parse import failed"
    exit 1
}

python -c "from od_parse.intelligence import DocumentType; print('✅ Smart classification available')" || {
    echo "❌ Error: Smart classification import failed"
    exit 1
}

python -c "from od_parse.main import parse_pdf; print('✅ Main parser available')" || {
    echo "❌ Error: Main parser import failed"
    exit 1
}

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Test with a PDF: python -c \"from od_parse.main import parse_pdf; print('Ready to parse!')\""
echo "3. See README.md for usage examples"
echo ""
echo "💡 To deactivate the virtual environment later, run: deactivate"
echo ""
echo "🚀 Happy parsing!"
