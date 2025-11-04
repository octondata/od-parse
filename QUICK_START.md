# od-parse v0.2.0 - Quick Start Guide

**Get started in 3 minutes!**

---

## 🚀 Installation (3 Steps)

### Step 1: Install Tesseract (System Dependency)

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 2: Install od-parse

```bash
pip install od_parse-0.2.0-py3-none-any.whl
```

### Step 3: Verify

```bash
tesseract --version
python -c "from od_parse import parse_pdf_optimized; print('✅ Ready!')"
```

---

## 💡 Usage

### Basic Usage

```python
from od_parse import parse_pdf_optimized

# Parse any PDF (automatic detection)
result = parse_pdf_optimized("document.pdf")

print(f"Text: {result['text'][:200]}")
print(f"Length: {len(result['text'])} characters")
```

### That's it! The parser automatically:
- ✅ Detects scanned PDFs → Uses OCR
- ✅ Detects CID codes → Uses PyMuPDF
- ✅ Uses fastest method for normal PDFs

---

## 📊 What Works

| PDF Type | Works? | Speed | Method |
|----------|--------|-------|--------|
| Normal PDFs | ✅ Yes | 0.5s | pdfminer |
| Scanned PDFs | ✅ Yes | 15-30s | OCR |
| CID-encoded PDFs | ✅ Yes | 1-2s | PyMuPDF |
| CamScanner docs | ✅ Yes | 15-30s | OCR |
| Government forms | ✅ Yes | 1-2s | PyMuPDF |

---

## ⚠️ Common Issues

### Issue: "tesseract is not installed"

**Solution:**
```bash
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

### Issue: Scanned PDFs returning empty content

**Solution:** Install Tesseract (see above)

### Issue: CID codes in output

**Solution:** Upgrade to v0.2.0 (includes PyMuPDF)

---

## 📚 Documentation

- **INSTALLATION_REQUIREMENTS.md** - Complete installation guide
- **CID_ENCODING_FIX.md** - CID encoding details
- **OCR_FIX_FOR_SCANNED_PDFS.md** - Scanned PDF handling
- **FINAL_SUMMARY_v0.2.0.md** - Complete summary

---

## 🎯 Examples

### Example 1: Scanned PDF
```python
result = parse_pdf_optimized("CamScanner_doc.pdf")
print(f"✅ Extracted {len(result['text'])} characters")
```

### Example 2: CID-Encoded PDF
```python
result = parse_pdf_optimized("government_form.pdf")
cid_count = result['text'].count('(cid:')
print(f"✅ CID codes: {cid_count}")
```

### Example 3: Batch Processing
```python
import glob

for pdf_file in glob.glob("documents/*.pdf"):
    result = parse_pdf_optimized(pdf_file)
    print(f"✅ {pdf_file}: {len(result['text'])} chars")
```

---

## ✅ Quick Checklist

- [ ] Install Tesseract: `brew install tesseract`
- [ ] Install Poppler: `brew install poppler`
- [ ] Install od-parse: `pip install od_parse-0.2.0-py3-none-any.whl`
- [ ] Verify: `tesseract --version`
- [ ] Test: `python -c "from od_parse import parse_pdf_optimized; print('✅')"`

**Done! You're ready to parse any PDF!** 🎉


