"""
Static file data source implementation.

Fetches stock data from local static files (CSV, Parquet, JSON, etc.).
The file format is flexible and will be auto-detected or specified.
"""

import os
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np

from .base import DataSource


class StaticFileDataSource(DataSource):
    """
    Data source implementation using static files.
    
    Supports single file containing all stocks. File format detection
    will be added later - for now, expects a specific structure.
    """
    
    def __init__(self, file_path: str, file_format: Optional[str] = None):
        """
        Initialize static file data source.
        
        Args:
            file_path: Path to the file containing stock data
            file_format: Optional file format hint ("csv", "parquet", "json", etc.)
                        If None, will attempt to auto-detect from file extension
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.file_path = file_path
        self.file_format = file_format or self._detect_format(file_path)
        self._cached_data: Optional[pd.DataFrame] = None
    
    def _detect_format(self, file_path: str) -> str:
        """
        Detect file format from extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            Detected format string
        """
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
        }
        return format_map.get(ext, 'csv')  # Default to CSV
    
    def _load_file(self) -> pd.DataFrame:
        """
        Load the data file into a DataFrame.
        
        Returns:
            DataFrame containing stock data
        
        Raises:
            ValueError: If file format is not supported or data structure is invalid
        """
        if self._cached_data is not None:
            return self._cached_data
        
        print(f"[static] Loading data from {self.file_path} (format: {self.file_format})...")
        
        try:
            if self.file_format == 'csv':
                df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
            elif self.file_format == 'parquet':
                df = pd.read_parquet(self.file_path)
            elif self.file_format == 'json':
                df = pd.read_json(self.file_path, orient='index')
                df.index = pd.to_datetime(df.index)
            elif self.file_format == 'pickle':
                df = pd.read_pickle(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            # Validate that index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            self._cached_data = df
            print(f"[static] Loaded {len(df)} rows from file")
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading file {self.file_path}: {e}")
    
    def _parse_dataframe(self, df: pd.DataFrame, stocks: List[str], args: List[str]) -> pd.DataFrame:
        """
        Parse loaded DataFrame into expected MultiIndex format.
        
        Expected input format (flexible - will be clarified later):
        - Option 1: MultiIndex columns (PriceType, Stock) with DatetimeIndex
        - Option 2: Columns like "AAPL_Open", "AAPL_Close", "MSFT_Open", etc.
        - Option 3: Long format with columns: Date, Stock, Open, Close
        
        For now, we'll handle MultiIndex format (most common).
        
        Args:
            df: Loaded DataFrame
            stocks: List of stock symbols to extract
            args: Date range arguments
        
        Returns:
            DataFrame in MultiIndex format: (Open/Close, StockSymbol)
        """
        # Filter by date range if args provided
        if len(args) == 2:
            start_date = pd.to_datetime(args[0])
            end_date = pd.to_datetime(args[1])
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        elif len(args) == 1:
            # Period string - calculate date range
            end_date = df.index.max()
            period = args[0]
            if period.endswith('y'):
                years = int(period[:-1])
                start_date = end_date - pd.DateOffset(years=years)
            elif period.endswith('mo'):
                months = int(period[:-2])
                start_date = end_date - pd.DateOffset(months=months)
            elif period.endswith('d'):
                days = int(period[:-1])
                start_date = end_date - pd.Timedelta(days=days)
            else:
                # Default to max period
                start_date = df.index.min()
            df = df[df.index >= start_date]
        
        # Check if already in MultiIndex format
        if isinstance(df.columns, pd.MultiIndex):
            # Already in correct format, just filter stocks
            available_stocks = [s for s in stocks if s in df.columns.get_level_values(1)]
            if not available_stocks:
                raise ValueError(f"None of the requested stocks found in file: {stocks}")
            return df[["Open", "Close"]].loc[:, (slice(None), available_stocks)]
        else:
            # Need to convert to MultiIndex format
            # This is a placeholder - actual conversion logic will depend on file format
            raise NotImplementedError(
                "File format conversion not yet implemented. "
                "Please provide file in MultiIndex format: columns (Open/Close, StockSymbol)"
            )
    
    def fetch_stock_data(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 3
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Fetch stock data from static file.
        
        Args:
            stocks: List of stock tickers
            args: Date range arguments
            max_retries: Not used for static files, but kept for interface consistency
        
        Returns:
            tuple: (open_close_dataframe, failed_stocks)
        """
        failed_stocks = {'YFPricesMissingError': [], 'YFTzMissingError': [], 'Other': []}
        
        try:
            # Load and parse file
            df = self._load_file()
            open_close = self._parse_dataframe(df, stocks, args)
            
            # Check which stocks were successfully loaded
            available_stocks = open_close["Open"].columns.tolist()
            missing_stocks = [s for s in stocks if s not in available_stocks]
            
            if missing_stocks:
                for stock in missing_stocks:
                    failed_stocks['YFPricesMissingError'].append((stock, 'NotFound', f'Stock not found in file'))
            
            if not available_stocks:
                print("ERROR: No stocks were successfully loaded from file!")
                return None, failed_stocks
            
            print(f"Successfully loaded data for {len(available_stocks)} stocks from file: {available_stocks}")
            return open_close, failed_stocks
            
        except Exception as e:
            error_msg = str(e)
            for stock in stocks:
                failed_stocks['Other'].append((stock, type(e).__name__, error_msg))
            print(f"ERROR: Failed to load data from file: {e}")
            return None, failed_stocks
    
    def fetch_sp500_data(
        self,
        test_dates: List[Any]
    ) -> np.ndarray:
        """
        Fetch S&P 500 returns from static file.
        
        Note: This assumes the file contains S&P 500 data with ticker "^GSPC" or "SP500".
        If not available, returns zeros.
        
        Args:
            test_dates: List of test dates
        
        Returns:
            numpy array of S&P 500 returns
        """
        if not test_dates or len(test_dates) == 0:
            print("[static] No test dates available for S&P 500")
            return np.array([])
        
        try:
            df = self._load_file()
            
            # Try to find S&P 500 data
            sp500_tickers = ["^GSPC", "SP500", "SPY"]  # Common S&P 500 tickers
            sp500_data = None
            
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in sp500_tickers:
                    if ticker in df.columns.get_level_values(1):
                        sp500_data = df["Close"][ticker]
                        break
            else:
                # Try to find Close column for S&P 500
                for ticker in sp500_tickers:
                    close_col = f"{ticker}_Close"
                    if close_col in df.columns:
                        sp500_data = df[close_col]
                        break
            
            if sp500_data is None:
                print("[static] Warning: S&P 500 data not found in file, using zeros")
                return np.zeros(len(test_dates))
            
            # Calculate returns
            sp500_returns = sp500_data.pct_change().dropna()
            
            # Align with test dates (same logic as yfinance)
            test_dates_pd = pd.to_datetime(test_dates)
            if isinstance(test_dates_pd, pd.DatetimeIndex) and test_dates_pd.tz is not None:
                test_dates_pd = test_dates_pd.tz_localize(None)
            
            sp500_index_normalized = pd.DatetimeIndex([pd.Timestamp(idx).normalize() for idx in sp500_returns.index])
            
            aligned_returns = []
            for test_date in test_dates_pd:
                try:
                    if isinstance(test_date, pd.Timestamp):
                        date_val_normalized = test_date.normalize()
                    else:
                        date_val_normalized = pd.Timestamp(test_date).normalize()
                    
                    exact_match = sp500_index_normalized == date_val_normalized
                    if exact_match.any():
                        return_val = sp500_returns.iloc[exact_match.argmax()]
                    else:
                        before_mask = sp500_index_normalized <= date_val_normalized
                        if before_mask.any():
                            before_indices = np.where(before_mask)[0]
                            return_val = sp500_returns.iloc[before_indices[-1]]
                        else:
                            return_val = 0.0
                    
                    if pd.isna(return_val):
                        aligned_returns.append(0.0)
                    else:
                        aligned_returns.append(float(return_val))
                except Exception as e:
                    print(f"[static] Warning: Could not get S&P 500 return for {test_date}: {e}")
                    aligned_returns.append(0.0)
            
            aligned_returns = np.array(aligned_returns, dtype=float)
            print(f"[static] S&P 500 returns loaded: {len(aligned_returns)} values")
            return aligned_returns
            
        except Exception as e:
            print(f"[static] Warning: Error loading S&P 500 data: {e}, using zeros")
            return np.zeros(len(test_dates))
    
    def validate_stocks(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Validate which stocks are available in the static file.
        """
        valid_stocks = []
        problematic_stocks = []
        
        try:
            df = self._load_file()
            
            # Get available stocks from file
            if isinstance(df.columns, pd.MultiIndex):
                available_stocks = df.columns.get_level_values(1).unique().tolist()
            else:
                # Try to infer from column names (e.g., "AAPL_Open" -> "AAPL")
                # This is a placeholder - actual logic depends on file format
                available_stocks = []
            
            for stock in stocks:
                if stock in available_stocks:
                    valid_stocks.append(stock)
                else:
                    problematic_stocks.append(stock)
            
            print(f"[check] Found {len(valid_stocks)} valid stocks in file")
            if problematic_stocks:
                print(f"[check] {len(problematic_stocks)} stocks not found in file")
            
            return valid_stocks, problematic_stocks
            
        except Exception as e:
            # If file can't be loaded, all stocks are problematic
            print(f"[check] Error loading file: {e}")
            return [], stocks

