# Development Guidelines

## 🔒 Security Requirements

### **NEVER Commit These Files:**

- ❌ API keys or secrets in any form
- ❌ `.env` files with real credentials  
- ❌ Debug/test scripts with hardcoded keys
- ❌ Processing results with sensitive data
- ❌ Temporary files with API responses

### **Always Use:**

- ✅ `.env.example` templates (no real keys)
- ✅ Environment variables or `.env` files
- ✅ Secure configuration helpers
- ✅ Placeholder values in documentation
- ✅ `.gitignore` patterns for sensitive files

## 📁 File Organization

### **Commit to Repository:**
```
od_parse/                 # Core library code
├── config/
│   ├── env_config.py    # Secure configuration management
│   └── advanced_config.py
├── parser/              # Document parsing logic
├── llm/                 # LLM integration
└── ...

tests/                   # Unit tests (no real API keys)
docs/                    # Documentation
.env.example            # Template (safe to commit)
.gitignore              # Security patterns
README.md               # Public documentation
```

### **Never Commit:**
```
.env                     # Real API keys
test_*.py               # Debug scripts
debug_*.py              # Temporary debugging
demo_*.py               # Demo scripts with keys
*_results/              # Processing outputs
*.json                  # Result files (except config templates)
```

## 🧪 Testing & Development

### **Creating Test Scripts:**

1. **Use secure configuration:**
```python
from od_parse.config.env_config import setup_secure_environment

# This loads from .env automatically
setup_secure_environment()

# No hardcoded keys needed
result = parse_pdf("test.pdf")
```

2. **Name files appropriately:**
```bash
# These will be ignored by git:
test_new_feature.py
debug_parsing_issue.py
demo_llm_integration.py
```

3. **Use temporary directories:**
```python
import tempfile
import os

# Create temp directory for results
with tempfile.TemporaryDirectory() as temp_dir:
    output_file = os.path.join(temp_dir, "results.json")
    # Process and save to temp location
```

### **Environment Setup for Development:**

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Add your development API keys to `.env`:**
```bash
# .env (never commit this file!)
OPENAI_API_KEY=sk-your-actual-key-here
GOOGLE_API_KEY=your-actual-google-key-here
```

3. **Verify security:**
```bash
# Check that .env is ignored
git status
# Should NOT show .env file

# Check gitignore is working
git check-ignore .env
# Should return: .env
```

## 🚀 Release Process

### **Before Committing:**

1. **Remove all debug files:**
```bash
rm test_*.py debug_*.py demo_*.py
rm -rf *_results/ *_analysis/
```

2. **Check for API keys:**
```bash
# Search for potential API keys
grep -r "sk-" . --exclude-dir=.git
grep -r "AIza" . --exclude-dir=.git
grep -r "anthropic" . --exclude-dir=.git
```

3. **Verify .gitignore:**
```bash
git status
# Should not show any sensitive files
```

### **Safe Documentation:**

- ✅ Use `"your-api-key-here"` in examples
- ✅ Reference `.env.example` for setup
- ✅ Include security warnings
- ❌ Never include real API keys
- ❌ Never include actual processing results with sensitive data

## 🔧 Configuration Management

### **Adding New API Providers:**

1. **Update `env_config.py`:**
```python
key_mappings = {
    'openai': ['OPENAI_API_KEY', 'OPENAI_KEY'],
    'google': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
    'new_provider': ['NEW_PROVIDER_API_KEY', 'NEW_PROVIDER_KEY']  # Add here
}
```

2. **Update `.env.example`:**
```bash
# New Provider Configuration
NEW_PROVIDER_API_KEY=your-new-provider-key-here
NEW_PROVIDER_MODEL=default-model-name
```

3. **Update documentation with secure examples**

## 📋 Code Review Checklist

Before submitting any code:

- [ ] No hardcoded API keys anywhere
- [ ] No `.env` files committed
- [ ] No debug/test scripts committed
- [ ] No processing results with sensitive data
- [ ] Updated `.gitignore` if needed
- [ ] Used secure configuration helpers
- [ ] Documentation uses placeholder keys only
- [ ] All examples are safe to share publicly

## 🆘 If You Accidentally Commit Secrets

1. **Immediately revoke the API key**
2. **Remove from git history:**
```bash
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch path/to/file' \
--prune-empty --tag-name-filter cat -- --all
```
3. **Generate new API key**
4. **Update `.env` with new key**
5. **Force push to remote (if necessary)**

## 📞 Questions?

If you're unsure about security practices:
1. Check this document
2. Look at existing secure code patterns
3. Ask before committing anything with potential secrets
