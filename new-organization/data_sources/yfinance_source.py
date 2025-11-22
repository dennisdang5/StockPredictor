"""
YFinance data source implementation.

Fetches stock data using the yfinance library.
"""

import time
import datetime
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf

from .base import DataSource


class YFinanceDataSource(DataSource):
    """
    Data source implementation using yfinance API.
    """
    
    def fetch_stock_data(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 3
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Fetch stock data using yfinance API.
        
        This function attempts to fetch stock data with error handling for:
        - YFPricesMissingError: Stock may be delisted or has no data for the date range
        - YFTzMissingError: Missing timezone information
        - Other yfinance errors
        """
        failed_stocks = {'YFPricesMissingError': [], 'YFTzMissingError': [], 'Other': []}
        valid_stocks_data = []
        valid_stocks = []
        
        # Try downloading each stock individually to handle errors gracefully
        print(f"Fetching data for {len(stocks)} stocks...")
        for i, stock in enumerate(stocks, 1):
            if (i % 25 == 0) or (i == 1) or (i == len(stocks)):  # Print every 25 stocks, first, and last
                print(f"  Progress: {i}/{len(stocks)} stocks processed...")
            success = False
            for attempt in range(max_retries):
                try:
                    # Create a Ticker object for individual download
                    ticker = yf.Ticker(stock)
                    
                    if len(args) == 1:
                        stock_data = ticker.history(period=args[0], repair=True)
                    elif len(args) == 2:
                        stock_data = ticker.history(period=None, start=args[0], end=args[1], interval="1d", repair=True)
                    else:
                        failed_stocks['Other'].append((stock, 'InvalidArgs', f'Invalid args: {args}'))
                        break
                    
                    # Check if we have valid data with required columns
                    if stock_data is not None and not stock_data.empty:
                        if "Open" in stock_data.columns and "Close" in stock_data.columns:
                            # Check that we have some valid data (not all NaN)
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
                        
                except Exception as stock_error:
                    error_type = type(stock_error).__name__
                    error_msg = str(stock_error)
                    
                    # Last attempt, log the error
                    if attempt == max_retries - 1:
                        if 'YFPricesMissingError' in error_type or 'no price data' in error_msg.lower() or 'delisted' in error_msg.lower():
                            failed_stocks['YFPricesMissingError'].append((stock, error_type, error_msg))
                        elif 'YFTzMissingError' in error_type or 'no timezone' in error_msg.lower():
                            failed_stocks['YFTzMissingError'].append((stock, error_type, error_msg))
                        else:
                            failed_stocks['Other'].append((stock, error_type, error_msg))
                    else:
                        # Wait before retrying
                        time.sleep(0.5 * (attempt + 1))
        
        # Print summary of failures
        total_failed = sum(len(failed_stocks[key]) for key in failed_stocks)
        if total_failed > 0:
            print(f"\n{total_failed} Failed downloads:")
            for error_type, failures in failed_stocks.items():
                if failures:
                    symbols = [f[0] for f in failures if isinstance(f, tuple)]
                    if len(symbols) <= 50:  # Only print if not too many
                        print(f"{symbols}: {error_type}")
                    else:
                        print(f"{len(symbols)} stocks: {error_type}")
        else:
            print("All downloads successful!")
        
        # Combine successful downloads into MultiIndex DataFrame
        if valid_stocks_data:
            # Combine all successful stock data
            # yfinance Tickers.history() creates MultiIndex with (Open/Close, StockSymbol)
            # So we need to replicate that structure
            combined = pd.concat(valid_stocks_data, axis=1, keys=valid_stocks, names=['Stock', 'PriceType'])
            # Swap levels to match yfinance format: (PriceType, Stock)
            combined = combined.swaplevel(axis=1)
            # Reorganize to match expected format: columns are (Level1: Open/Close, Level2: StockSymbol)
            open_close = combined[["Open", "Close"]]
            
            print(f"Successfully fetched data for {len(valid_stocks)} stocks: {valid_stocks}")
            return open_close, failed_stocks
        else:
            print("ERROR: No stocks were successfully downloaded!")
            return None, failed_stocks
    
    def fetch_sp500_data(
        self,
        test_dates: List[Any]
    ) -> np.ndarray:
        """
        Fetch S&P 500 returns aligned with test dates using yfinance.
        """
        if not test_dates or len(test_dates) == 0:
            print("[util] No test dates available for S&P 500")
            return np.array([])
        
        try:
            print("[util] Fetching S&P 500 data for metrics...")
            
            # Get date range from test dates
            min_date = min(test_dates)
            max_date = max(test_dates)
            
            # Add a small buffer to ensure we get data (yfinance sometimes needs end date + 1 day)
            if isinstance(max_date, datetime.datetime):
                max_date_buffered = max_date + datetime.timedelta(days=1)
            elif isinstance(max_date, datetime.date):
                max_date_buffered = datetime.datetime.combine(max_date, datetime.time()) + datetime.timedelta(days=1)
            else:
                max_date_buffered = pd.Timestamp(max_date) + pd.Timedelta(days=1)
            
            # Fetch S&P 500 data (^GSPC is the ticker for S&P 500)
            try:
                sp500 = yf.Ticker("^GSPC")
                sp500_data = sp500.history(start=min_date, end=max_date_buffered, repair=True)
            except Exception as yf_error:
                print(f"[util] Warning: Error fetching S&P 500 data from yfinance: {yf_error}")
                print("[util] Using zeros for S&P 500 returns")
                return np.zeros(len(test_dates))
            
            if sp500_data is None or sp500_data.empty:
                print("[util] Warning: Could not fetch S&P 500 data, using zeros")
                return np.zeros(len(test_dates))
            
            # Calculate daily returns from S&P 500 close prices
            sp500_close = sp500_data['Close']
            sp500_daily_returns = sp500_close.pct_change().dropna()
            
            if isinstance(sp500_daily_returns.index, pd.DatetimeIndex) and sp500_daily_returns.index.tz is not None:
                sp500_daily_returns.index = sp500_daily_returns.index.tz_localize(None)
            
            # Normalize sp500 index once for efficient comparison (remove time component)
            sp500_index_normalized = pd.DatetimeIndex([pd.Timestamp(idx).normalize() for idx in sp500_daily_returns.index])
            
            # Convert test dates to pandas Timestamp for alignment
            test_dates_pd = pd.to_datetime(test_dates)
            if isinstance(test_dates_pd, pd.DatetimeIndex) and test_dates_pd.tz is not None:
                test_dates_pd = test_dates_pd.tz_localize(None)
            
            # Align S&P 500 returns with test dates
            # Use forward fill to match each test date with the most recent S&P 500 return
            sp500_returns = []
            for test_date in test_dates_pd:
                try:
                    # Normalize test_date for comparison (remove time component and timezone)
                    if isinstance(test_date, pd.Timestamp):
                        date_val_normalized = test_date.normalize()
                    else:
                        date_val_normalized = pd.Timestamp(test_date).normalize()
                    
                    # Get the most recent return up to this date (forward fill behavior)
                    # Check for exact match first
                    exact_match = sp500_index_normalized == date_val_normalized
                    if exact_match.any():
                        return_val = sp500_daily_returns.iloc[exact_match.argmax()]
                    else:
                        # Find the most recent value before or at this date (forward fill)
                        before_mask = sp500_index_normalized <= date_val_normalized
                        if before_mask.any():
                            # Get the last index where condition is True
                            before_indices = np.where(before_mask)[0]
                            return_val = sp500_daily_returns.iloc[before_indices[-1]]
                        else:
                            # No data before this date, use 0
                            return_val = 0.0
                    
                    if pd.isna(return_val):
                        sp500_returns.append(0.0)
                    else:
                        sp500_returns.append(float(return_val))
                except Exception as e:
                    # If anything fails, use 0
                    print(f"[util] Warning: Could not get S&P 500 return for {test_date}: {e}")
                    sp500_returns.append(0.0)
            
            sp500_returns = np.array(sp500_returns, dtype=float)
            
            print(f"[util] S&P 500 returns fetched: {len(sp500_returns)} values, mean={np.mean(sp500_returns):.6f}, std={np.std(sp500_returns):.6f}")
            return sp500_returns
            
        except Exception as e:
            print(f"[util] Warning: Error fetching S&P 500 data: {e}, using zeros")
            return np.zeros(len(test_dates))
    
    def validate_stocks(
        self,
        stocks: List[str],
        args: List[str],
        max_retries: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Identify which stocks from the input list are problematic (cannot be downloaded).
        This is a lightweight check that only validates downloadability without processing data.
        """
        valid_stocks = []
        problematic_stocks = []
        
        print(f"[check] Identifying problematic stocks from {len(stocks)} stocks...")
        for i, stock in enumerate(stocks, 1):
            if (i % 50 == 0) or (i == 1) or (i == len(stocks)):
                print(f"  Progress: {i}/{len(stocks)} stocks checked...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(stock)
                    
                    if len(args) == 1:
                        stock_data = ticker.history(period=args[0], repair=True)
                    elif len(args) == 2:
                        stock_data = ticker.history(period=None, start=args[0], end=args[1], interval="1d", repair=True)
                    else:
                        problematic_stocks.append(stock)
                        break
                    
                    # Check if we have valid data with required columns
                    if stock_data is not None and not stock_data.empty:
                        if "Open" in stock_data.columns and "Close" in stock_data.columns:
                            # Check that we have some valid data (not all NaN)
                            if not stock_data[["Open", "Close"]].isna().all().all():
                                valid_stocks.append(stock)
                                success = True
                                break
                            else:
                                problematic_stocks.append(stock)
                                break
                        else:
                            problematic_stocks.append(stock)
                            break
                    else:
                        problematic_stocks.append(stock)
                        break
                        
                except Exception:
                    # Last attempt, mark as problematic
                    if attempt == max_retries - 1:
                        problematic_stocks.append(stock)
                    else:
                        time.sleep(0.5 * (attempt + 1))
        
        if problematic_stocks:
            print(f"[check] Found {len(problematic_stocks)} problematic stocks (will be excluded)")
            if len(problematic_stocks) <= 50:
                print(f"  Problematic: {problematic_stocks}")
        else:
            print(f"[check] All stocks are valid!")
        
        print(f"[check] {len(valid_stocks)} valid stocks identified")
        return valid_stocks, problematic_stocks

