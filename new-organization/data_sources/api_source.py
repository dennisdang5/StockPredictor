"""
API data source base class.

Provides structure for future API implementations (Alpha Vantage, Polygon.io, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
import time

from .base import DataSource


class APIDataSource(DataSource):
    """
    Abstract base class for API-based data sources.
    
    Provides common patterns for API implementations:
    - Rate limiting
    - Authentication
    - Error handling
    - Retry logic
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.0):
        """
        Initialize API data source.
        
        Args:
            api_key: Optional API key for authentication
            rate_limit_delay: Delay between API calls in seconds (for rate limiting)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0
    
    def _rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.
        """
        if self.rate_limit_delay > 0:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            if time_since_last_call < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last_call)
            self._last_call_time = time.time()
    
    def _handle_api_error(self, error: Exception, stock: str) -> Tuple[str, str]:
        """
        Handle API-specific errors and categorize them.
        
        Args:
            error: Exception that occurred
            stock: Stock symbol that failed
        
        Returns:
            tuple: (error_category, error_message)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Categorize common API errors
        if 'rate limit' in error_msg.lower() or '429' in error_msg:
            return ('RateLimit', f'Rate limit exceeded: {error_msg}')
        elif 'authentication' in error_msg.lower() or '401' in error_msg or '403' in error_msg:
            return ('Authentication', f'Authentication failed: {error_msg}')
        elif 'not found' in error_msg.lower() or '404' in error_msg:
            return ('YFPricesMissingError', f'Stock not found: {error_msg}')
        else:
            return ('Other', f'{error_type}: {error_msg}')
    
    @abstractmethod
    def _fetch_single_stock(self, stock: str, start_date: Optional[str], end_date: Optional[str], period: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single stock from the API.
        
        Args:
            stock: Stock ticker symbol
            start_date: Start date string (if date range)
            end_date: End date string (if date range)
            period: Period string (if period-based)
        
        Returns:
            DataFrame with Open/Close columns, or None if failed
        """
        pass
    
    def fetch_stock_data(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 3
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Fetch stock data from API with rate limiting and error handling.
        
        Subclasses should override _fetch_single_stock() to implement API-specific logic.
        """
        failed_stocks = {'YFPricesMissingError': [], 'YFTzMissingError': [], 'Other': []}
        valid_stocks_data = []
        valid_stocks = []
        
        # Parse args
        if len(args) == 1:
            period = args[0]
            start_date = None
            end_date = None
        elif len(args) == 2:
            period = None
            start_date = args[0]
            end_date = args[1]
        else:
            failed_stocks['Other'].append(('', 'InvalidArgs', f'Invalid args: {args}'))
            return None, failed_stocks
        
        print(f"Fetching data for {len(stocks)} stocks from API...")
        for i, stock in enumerate(stocks, 1):
            if (i % 25 == 0) or (i == 1) or (i == len(stocks)):
                print(f"  Progress: {i}/{len(stocks)} stocks processed...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    # Rate limiting
                    self._rate_limit()
                    
                    # Fetch data
                    stock_data = self._fetch_single_stock(stock, start_date, end_date, period)
                    
                    if stock_data is not None and not stock_data.empty:
                        if "Open" in stock_data.columns and "Close" in stock_data.columns:
                            if not stock_data[["Open", "Close"]].isna().all().all():
                                valid_stocks_data.append(stock_data[["Open", "Close"]])
                                valid_stocks.append(stock)
                                success = True
                                break
                            else:
                                failed_stocks['YFPricesMissingError'].append((stock, 'AllNaN', 'All data is NaN'))
                                break
                        else:
                            failed_stocks['Other'].append((stock, 'MissingColumns', 'Open or Close columns not found'))
                            break
                    else:
                        failed_stocks['YFPricesMissingError'].append((stock, 'EmptyData', 'No data returned'))
                        break
                        
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        error_category, error_msg = self._handle_api_error(api_error, stock)
                        failed_stocks[error_category].append((stock, type(api_error).__name__, error_msg))
                    else:
                        # Wait before retrying
                        time.sleep(0.5 * (attempt + 1))
        
        # Print summary
        total_failed = sum(len(failed_stocks[key]) for key in failed_stocks)
        if total_failed > 0:
            print(f"\n{total_failed} Failed downloads:")
            for error_type, failures in failed_stocks.items():
                if failures:
                    symbols = [f[0] for f in failures if isinstance(f, tuple)]
                    if len(symbols) <= 50:
                        print(f"{symbols}: {error_type}")
                    else:
                        print(f"{len(symbols)} stocks: {error_type}")
        else:
            print("All downloads successful!")
        
        # Combine into MultiIndex DataFrame
        if valid_stocks_data:
            combined = pd.concat(valid_stocks_data, axis=1, keys=valid_stocks, names=['Stock', 'PriceType'])
            combined = combined.swaplevel(axis=1)
            open_close = combined[["Open", "Close"]]
            
            print(f"Successfully fetched data for {len(valid_stocks)} stocks: {valid_stocks}")
            return open_close, failed_stocks
        else:
            print("ERROR: No stocks were successfully downloaded!")
            return None, failed_stocks
    
    @abstractmethod
    def fetch_sp500_data(
        self,
        test_dates: List[Any]
    ) -> np.ndarray:
        """
        Fetch S&P 500 returns from API.
        
        Subclasses must implement this method.
        """
        pass
    
    def validate_stocks(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Validate which stocks are available via the API.
        """
        valid_stocks = []
        problematic_stocks = []
        
        # Parse args
        if len(args) == 1:
            period = args[0]
            start_date = None
            end_date = None
        elif len(args) == 2:
            period = None
            start_date = args[0]
            end_date = args[1]
        else:
            return [], stocks
        
        print(f"[check] Validating {len(stocks)} stocks via API...")
        for i, stock in enumerate(stocks, 1):
            if (i % 50 == 0) or (i == 1) or (i == len(stocks)):
                print(f"  Progress: {i}/{len(stocks)} stocks checked...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    self._rate_limit()
                    stock_data = self._fetch_single_stock(stock, start_date, end_date, period)
                    
                    if stock_data is not None and not stock_data.empty:
                        if "Open" in stock_data.columns and "Close" in stock_data.columns:
                            if not stock_data[["Open", "Close"]].isna().all().all():
                                valid_stocks.append(stock)
                                success = True
                                break
                    
                    problematic_stocks.append(stock)
                    break
                    
                except Exception:
                    if attempt == max_retries - 1:
                        problematic_stocks.append(stock)
                    else:
                        time.sleep(0.5 * (attempt + 1))
        
        print(f"[check] {len(valid_stocks)} valid stocks identified")
        return valid_stocks, problematic_stocks

