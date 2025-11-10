# Example Code Review Summary

This document summarizes the review and fixes made to all example files in the `examples/` directory.

## Files Reviewed

1. ✅ `basic_usage.py` - Fixed
2. ✅ `advanced_pdf_parsing.py` - Fixed
3. ✅ `pdf_with_images_google.py` - Verified (already correct)
4. ⚠️ `advanced_features_demo.py` - Needs review (uses optional features)

## Issues Found and Fixed

### 1. basic_usage.py

**Issues:**
- ❌ Missing API keys (library now requires LLM API keys)
- ❌ Wrong result structure access (should use `result['parsed_data']['text']` not `result['text']`)
- ❌ No error handling for missing API keys

**Fixes:**
- ✅ Added `--api-key` command-line argument
- ✅ Added support for `GOOGLE_API_KEY` environment variable
- ✅ Fixed result structure access to use `result['parsed_data']`
- ✅ Added error handling for missing API keys
- ✅ Added helpful error messages with link to get API key
- ✅ Updated to use `llm_model` and `api_keys` parameters

### 2. advanced_pdf_parsing.py

**Issues:**
- ❌ Wrong import path (`from od_parse.main import parse_pdf` should be `from od_parse import parse_pdf`)
- ❌ Missing API keys in all functions
- ❌ Used non-existent pipeline classes (`PDFPipeline`, `LoadDocumentStage`, etc.)
- ❌ Wrong result structure access
- ❌ No error handling

**Fixes:**
- ✅ Fixed import to use `from od_parse import parse_pdf`
- ✅ Removed non-existent pipeline imports
- ✅ Added API key parameter to all functions
- ✅ Added support for `GOOGLE_API_KEY` environment variable
- ✅ Fixed result structure access to use `result['parsed_data']`
- ✅ Updated form extraction to handle new form structure
- ✅ Updated table extraction to handle new table structure
- ✅ Added error handling and helpful messages
- ✅ Updated all functions to use `llm_model` and `api_keys` parameters

### 3. pdf_with_images_google.py

**Status:** ✅ Already correct
- ✅ Uses correct imports
- ✅ Includes API keys
- ✅ Uses correct result structure access
- ✅ Has good documentation
- ✅ Includes multiple examples

### 4. advanced_features_demo.py

**Status:** ⚠️ Uses optional features
- Uses `od_parse.config` which exists
- Uses optional features that may not be available
- Has proper error handling for missing dependencies
- May need updates if features change

## Common Patterns Fixed

### API Key Handling
All examples now:
- Accept API keys as parameters
- Support `GOOGLE_API_KEY` environment variable
- Provide helpful error messages if API key is missing
- Include link to get API key

### Result Structure Access
All examples now:
- Use `result['parsed_data']` to access parsed data
- Use `result['metadata']` to access metadata
- Use `result['summary']` to access summary
- Handle missing keys gracefully with `.get()`

### Function Signatures
All examples now:
- Use `llm_model` parameter to specify model
- Use `api_keys` parameter to pass API keys
- Use correct output formats
- Include proper error handling

## Best Practices Applied

1. **Error Handling**: All examples now have proper error handling
2. **Documentation**: All examples have clear docstrings and comments
3. **User Experience**: Helpful error messages with actionable guidance
4. **Flexibility**: Support both environment variables and parameters
5. **Consistency**: All examples follow the same patterns

## Testing Recommendations

1. Test all examples with valid API keys
2. Test error handling with missing API keys
3. Test with different PDF types
4. Verify result structure access works correctly
5. Test with environment variables vs parameters

## Notes

- All examples now require LLM API keys (this is a library requirement)
- The library structure has changed - results are now nested under `parsed_data`
- Pipeline classes mentioned in old examples don't exist - use `parse_pdf` directly
- All examples use Google Gemini models by default (can be changed)

## Next Steps

1. Update `advanced_features_demo.py` if needed
2. Add more examples for specific use cases
3. Create examples for other LLM providers (OpenAI, Anthropic)
4. Add examples for vLLM usage
5. Add examples for batch processing

