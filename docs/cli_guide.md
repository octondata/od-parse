# Command Line Interface Guide

The OctonData Parse library includes a command-line interface (CLI) for quick document processing without writing code. This guide shows you how to use the CLI to leverage all the advanced PDF parsing capabilities.

## Basic Usage

The CLI is available through the `od_parse.main` module. Here's the basic syntax:

```bash
python -m od_parse.main [document.pdf] [options]
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--output-format` | Format for output: json, markdown, or summary |
| `--output-file` | Path to output file (if not specified, outputs to console) |
| `--pipeline` | Type of pipeline to use: default, full, fast, tables, forms, or structure |
| `--deep-learning` | Use deep learning models for enhanced extraction |
| `--debug` | Enable debug logging |

## Examples

### Basic Document Processing

To process a document with the default settings and output to the console:

```bash
python -m od_parse.main /path/to/document.pdf
```

### Generating Markdown Output

To convert a PDF to markdown format:

```bash
python -m od_parse.main /path/to/document.pdf --output-format markdown --output-file document.md
```

### Fast Processing for Large Documents

For large documents where speed is more important than accuracy:

```bash
python -m od_parse.main /path/to/document.pdf --pipeline fast --output-format summary
```

### Extracting Only Tables

To focus only on table extraction:

```bash
python -m od_parse.main /path/to/document.pdf --pipeline tables --output-format json --output-file tables.json
```

### Using Deep Learning for Enhanced Accuracy

For the highest quality extraction using deep learning models:

```bash
python -m od_parse.main /path/to/document.pdf --deep-learning --output-format json --output-file result.json
```

### Form Field Extraction

To focus specifically on form fields:

```bash
python -m od_parse.main /path/to/document.pdf --pipeline forms --output-format json --output-file forms.json
```

## Pipeline Types

The `--pipeline` option allows you to select different processing pipelines:

- **default**: Standard processing with configurable options
- **full**: Complete processing with all advanced features
- **fast**: Speed-optimized processing without deep learning
- **tables**: Focused on table extraction
- **forms**: Focused on form element extraction
- **structure**: Focused on document structure extraction

## Output Formats

The `--output-format` option controls how the results are presented:

- **json**: Complete structured output in JSON format
- **markdown**: Human-readable markdown document
- **summary**: Brief summary of the extraction results

## Integration with Other Tools

The CLI can be easily integrated into shell scripts or automation workflows:

```bash
#!/bin/bash
# Process all PDFs in a directory
for pdf in /path/to/documents/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  python -m od_parse.main "$pdf" --output-format json --output-file "/path/to/output/$filename.json"
done
```

## Advanced Usage: Custom Pipeline Configuration

For the most control, you can use the Python API to create a custom pipeline and then process documents. See the Python API documentation for more details.

This CLI provides a convenient way to quickly test and utilize the advanced PDF parsing capabilities without writing code, making it ideal for quick document processing tasks and evaluations.
