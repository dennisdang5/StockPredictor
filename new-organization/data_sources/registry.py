"""
Data source registry for convenient data source creation.

Provides factory functions to create data sources by name.
"""

from typing import Optional
from .base import DataSource
from .yfinance_source import YFinanceDataSource
from .static_file_source import StaticFileDataSource


def get_data_source(
    source_type: str,
    **kwargs
) -> DataSource:
    """
    Create a data source instance by name.
    
    Args:
        source_type: Type of data source - "yfinance", "static", or "file"
        **kwargs: Additional arguments passed to the data source constructor
                  - For "yfinance": no additional args needed
                  - For "static" or "file": requires "file_path" keyword argument
                    Optional: "file_format" (auto-detected if not provided)
    
    Returns:
        DataSource instance
    
    Examples:
        >>> # Create yfinance source
        >>> source = get_data_source("yfinance")
        
        >>> # Create static file source
        >>> source = get_data_source("static", file_path="data.csv")
        >>> source = get_data_source("file", file_path="data.parquet", file_format="parquet")
    """
    source_type_lower = source_type.lower()
    
    if source_type_lower in ["yfinance", "yf"]:
        return YFinanceDataSource()
    
    elif source_type_lower in ["static", "file", "static_file"]:
        if "file_path" not in kwargs:
            raise ValueError(f"file_path is required for {source_type} data source")
        file_path = kwargs.pop("file_path")
        file_format = kwargs.pop("file_format", None)
        if kwargs:
            raise ValueError(f"Unknown arguments for {source_type} data source: {kwargs.keys()}")
        return StaticFileDataSource(file_path=file_path, file_format=file_format)
    
    else:
        raise ValueError(
            f"Unknown data source type: {source_type}. "
            f"Supported types: 'yfinance', 'static', 'file'"
        )

