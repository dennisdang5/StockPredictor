"""
Data sources module for stock data fetching.

This module provides a pluggable data source system using the strategy pattern.
All data sources implement the same interface, allowing easy switching between
different data providers (yfinance, static files, APIs, etc.).
"""

from .base import DataSource
from .yfinance_source import YFinanceDataSource
from .static_file_source import StaticFileDataSource
from .api_source import APIDataSource

__all__ = [
    'DataSource',
    'YFinanceDataSource',
    'StaticFileDataSource',
    'APIDataSource',  # Abstract base class for API implementations
]

