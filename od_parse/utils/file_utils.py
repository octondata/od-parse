"""
File utility functions for the od-parse library.
"""

import os
from typing import Union, Optional
from pathlib import Path

class FileValidationError(Exception):
    """Exception raised for file validation errors."""
    pass

def validate_file(file_path: Union[str, Path], extension: Optional[str] = None) -> Path:
    """
    Validate that a file exists and has the correct extension.
    
    Args:
        file_path: Path to the file
        extension: Expected file extension (e.g., '.pdf')
    
    Returns:
        Path object for the validated file
    
    Raises:
        FileValidationError: If the file does not exist or has an incorrect extension
    """
    # Convert to Path object if string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileValidationError(f"File does not exist: {file_path}")
    
    # Check if it's a file (not a directory)
    if not file_path.is_file():
        raise FileValidationError(f"Path is not a file: {file_path}")
    
    # Check file extension if provided
    if extension and file_path.suffix.lower() != extension.lower():
        raise FileValidationError(
            f"File has incorrect extension: {file_path.suffix} (expected {extension})"
        )
    
    return file_path

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    
    Returns:
        Path object for the directory
    """
    # Convert to Path object if string
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)
    
    # Create directory if it doesn't exist
    directory_path.mkdir(parents=True, exist_ok=True)
    
    return directory_path

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Size of the file in bytes
    
    Raises:
        FileValidationError: If the file does not exist
    """
    file_path = validate_file(file_path)
    return file_path.stat().st_size

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File extension (e.g., '.pdf')
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    return file_path.suffix.lower()
