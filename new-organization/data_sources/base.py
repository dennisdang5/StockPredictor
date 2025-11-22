"""
Base data source abstraction.

All data sources must implement the DataSource interface to ensure
consistent behavior across different data providers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    All data sources must implement three main methods:
    1. fetch_stock_data - Fetch Open/Close data for multiple stocks
    2. fetch_sp500_data - Fetch S&P 500 returns for given dates
    3. validate_stocks - Check which stocks are available/valid
    """
    
    @abstractmethod
    def fetch_stock_data(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 3
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Fetch stock data (Open/Close prices) for multiple stocks.
        
        Args:
            stocks: List of stock tickers
            args: Date range arguments - either:
                  - [period_string] e.g., ["3y"]
                  - [start_date, end_date] e.g., ["1990-01-01", "2015-12-31"]
            max_retries: Maximum number of retry attempts per stock
        
        Returns:
            tuple: (open_close_dataframe, failed_stocks)
                - open_close_dataframe: DataFrame with MultiIndex columns (Open/Close, stock_symbols)
                  Format: columns are (Level1: "Open"/"Close", Level2: stock_symbol)
                  Index: DatetimeIndex of trading days
                  Returns None if no stocks were successfully downloaded
                - failed_stocks: Dictionary mapping error types to lists of failed stocks
                  Format: {'ErrorType': [(stock, error_type, error_msg), ...], ...}
        """
        pass
    
    @abstractmethod
    def fetch_sp500_data(
        self,
        test_dates: List[Any]
    ) -> np.ndarray:
        """
        Fetch S&P 500 returns aligned with test dates.
        
        Args:
            test_dates: List of test dates (datetime objects, dates, or timestamps)
        
        Returns:
            numpy array of S&P 500 daily returns aligned with test_dates
            Returns zeros array if data cannot be fetched
        """
        pass
    
    @abstractmethod
    def validate_stocks(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Identify which stocks from the input list are problematic (cannot be downloaded).
        This is a lightweight check that only validates availability without processing data.
        
        Args:
            stocks: List of stock tickers
            args: Date range arguments - either:
                  - [period_string] e.g., ["3y"]
                  - [start_date, end_date] e.g., ["1990-01-01", "2015-12-31"]
            max_retries: Maximum number of retry attempts per stock (default: 1 for speed)
        
        Returns:
            tuple: (valid_stocks, problematic_stocks)
                - valid_stocks: List of stocks that can be downloaded
                - problematic_stocks: List of stocks that failed to download
        """
        pass

