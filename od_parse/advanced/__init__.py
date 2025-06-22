"""
Advanced document parsing components for od-parse library.
This module provides cutting-edge parsing capabilities that go beyond
basic PDF extraction.
"""

# Import advanced components with optional dependencies
try:
    from od_parse.advanced.table_transformer import (
        TableTransformerEngine,
        extract_tables_with_transformer
    )
    __all__ = ["TableTransformerEngine", "extract_tables_with_transformer"]
except ImportError:
    __all__ = []

try:
    from od_parse.advanced.llava_next import (
        LLaVANextEngine,
        understand_document_with_llava
    )
    __all__.extend(["LLaVANextEngine", "understand_document_with_llava"])
except ImportError:
    pass
