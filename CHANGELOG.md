# Changelog

All notable changes to od-parse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub-based distribution (no PyPI required)
- Wheel file releases via GitHub Releases
- Automated release workflow
- Comprehensive installation documentation

## [0.2.0] - 2025-11-09

### Added
- LLM-powered document understanding
- Support for multiple LLM providers (OpenAI, Anthropic, Google, Azure OpenAI)
- Vision-Language Model (VLM) processing
- vLLM support for local inference
- API key configuration via parameters
- Multi-form PDF extraction
- Page-by-page processing for large PDFs
- Adaptive DPI for OCR
- Document segmentation
- Helper functions for model selection

### Changed
- Library now requires LLM API keys
- Default model changed to `gemini-2.0-flash`
- Improved form extraction using PyPDF2
- Enhanced error handling and logging

### Fixed
- Fixed duplicate class definition in `document_segmenter.py`
- Fixed `core_parse_pdf` import issue
- Improved result structure access in examples
- Fixed API key handling across all providers

## [0.1.0] - Initial Release

### Added
- Basic PDF parsing
- Text extraction
- Table extraction
- Form extraction
- Image extraction
- OCR support
- Markdown conversion

---

[Unreleased]: https://github.com/octondata/od-parse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/octondata/od-parse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/octondata/od-parse/releases/tag/v0.1.0

