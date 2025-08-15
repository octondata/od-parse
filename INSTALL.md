# Installation Guide for od-parse

## ⚠️ Important Notice

**od-parse is currently in development and NOT available on PyPI yet.**

If you're getting this error:
```
ERROR: Could not find a version that satisfies the requirement od-parse
```

This is expected! Follow the development installation instructions below.

## 🚀 Development Installation

### Method 1: Git Clone (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/od-parse.git
cd od-parse

# 2. Create virtual environment (IMPORTANT!)
python3 -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install in development mode
pip install -e .

# 5. Test installation
python -c "import od_parse; print('✅ od-parse installed successfully!')"
```

### Method 2: Direct Download

If you don't have git:

```bash
# 1. Download the source code
wget https://github.com/your-username/od-parse/archive/main.zip
# Or download manually from GitHub

# 2. Extract and navigate
unzip main.zip
cd od-parse-main

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install
pip install -e .
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "externally-managed-environment"

**Error:**
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution:** Always use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue 2: "pip: command not found"

**Solution:** Use python3 -m pip:
```bash
python3 -m pip install -e .
```

### Issue 3: "Could not find a version that satisfies the requirement od-parse"

**Solution:** This is expected! od-parse is not on PyPI yet. Use development installation above.

### Issue 4: Missing Dependencies

**Solution:** Install core dependencies manually:
```bash
pip install pdfminer.six tabula-py opencv-python pillow pytesseract pandas numpy
```

## 📦 Installation Options

### Basic Installation
```bash
pip install -e .
```

### With Advanced Features
```bash
pip install -e .[advanced]
```

### With All Features
```bash
pip install -e .[all]
```

## ✅ Verify Installation

```bash
# Test basic import
python -c "import od_parse; print('✅ Basic import works')"

# Test smart classification
python -c "from od_parse.intelligence import DocumentType; print('✅ Smart classification available')"

# Test main functionality
python -c "from od_parse.main import parse_pdf; print('✅ Main parser available')"
```

## 🎯 Quick Test

Create a test file `test_install.py`:

```python
#!/usr/bin/env python3
"""Test od-parse installation."""

try:
    from od_parse.main import parse_pdf
    from od_parse.intelligence import DocumentType, DocumentClassifier
    print("✅ od-parse installed successfully!")
    print(f"✅ {len(list(DocumentType))} document types supported")
    print("✅ Smart document classification available")
    print("🎉 Ready to parse PDFs!")
except ImportError as e:
    print(f"❌ Installation issue: {e}")
    print("Please follow the installation guide above.")
```

Run it:
```bash
python test_install.py
```

## 🔮 Future PyPI Installation

Once od-parse is published to PyPI, you'll be able to install with:

```bash
pip install od-parse
pip install "od-parse[all]"
```

But for now, use the development installation method above.

## 💡 Need Help?

1. **Check you're in a virtual environment**: You should see `(venv)` in your terminal prompt
2. **Verify Python version**: `python3 --version` (should be 3.8+)
3. **Check installation**: Run the verification commands above
4. **Still having issues?** Create an issue on GitHub with your error message

## 🚀 Ready to Use

Once installed, you can start using od-parse:

```python
from od_parse.main import parse_pdf

# Parse a PDF with smart document classification
result = parse_pdf("your_document.pdf", use_deep_learning=True)
classification = result['parsed_data']['document_classification']

print(f"Document Type: {classification['document_type']}")
print(f"Confidence: {classification['confidence']:.2f}")
```

Happy parsing! 🎉
