"""
Utility script to update metrics (S&P 500 returns, Returns) without redownloading
training/validation/test data.

METRICS DATA STRUCTURE:
=======================
The metrics file ({data_id}_metrics.npz) contains three numpy arrays, all aligned with TEST samples:

1. Rev (Revenues):
   - Shape: (N_test,) - one value per test sample
   - Type: float64
   - Meaning: Intraday price difference for each stock on each test date
   - Calculation: Close[t] - Open[t] (in dollars, not percentage)
   - Alignment: Each element corresponds to one test sample (one stock-date combination)
   - Example: If stock XYZ opened at $100 and closed at $105, Rev = $5.00

2. Returns:
   - Shape: (N_test,) - one value per test sample
   - Type: float64
   - Meaning: Same as Rev - intraday price difference (Close - Open)
   - Calculation: Close[t] - Open[t] (currently identical to Rev)
   - Alignment: Each element corresponds to one test sample
   - Note: Currently same as Rev, but kept separate for potential future use
   - Can be updated by redownloading stock Open/Close prices (requires stock indices in test data)

3. Sp500 (S&P 500 Returns):
   - Shape: (N_test,) - one value per test sample
   - Type: float64
   - Meaning: Daily percentage return of S&P 500 index aligned with test dates
   - Calculation: (Close[t] - Close[t-1]) / Close[t-1] (close-to-close percentage return)
   - Alignment: Each element corresponds to one test sample's date
   - Note: Multiple test samples can share the same date, so same Sp500 value may repeat
   - Source: Fetched from yfinance using ticker ^SP500TR
   - Can be updated independently without redownloading stock data

IMPORTANT NOTES:
- All three arrays have the SAME length (N_test = number of test samples)
- Each array element corresponds to the same test sample index
- Test samples are created from (stock, date) combinations
- Multiple samples can share the same date (different stocks on same day)
- Rev and Returns require stock data to calculate
- Returns can be updated by redownloading stock prices (requires stock indices in test data)
- Sp500 can be updated independently using this script

Usage:
    # Option 1: Set variables below and run directly
    python update_metrics.py
    
    # Option 2: Use command-line arguments (overrides variables)
    python update_metrics.py --data-id <data_id> [--update-sp500] [--update-returns]
    python update_metrics.py --data-ids "id1,id2,id3" [--update-sp500] [--update-returns]
    python update_metrics.py --list-datasets  # List all available datasets
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# Configuration Variables - Set these directly or use command-line args
# ============================================================================
DATA_ID = None  # Set to your data_id string (single ID), or None to use command-line arg
DATA_IDS = ["d742ba71a0", "9f985021ad", "47d2ec3267", "f2b33bcace"]  # Set to a list of data_id strings (e.g., ["id1", "id2", "id3"]), or None to use command-line arg
UPDATE_SP500 = True  # Set to True to update S&P 500 returns
UPDATE_RETURNS = True  # Set to True to update Returns (requires stock indices in test data)
LIST_DATASETS = False  # Set to True to list all available datasets
# ============================================================================

# Add parent directory to path to import util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util import (
    DATA_DIR, 
    _load_npz_progress, 
    _save_npz_progress,
    _load_id_mapping,
    _get_sp500_returns_for_dates
)
from data_sources.yfinance_source import YFinanceDataSource
import yfinance as yf


def validate_ticker(ticker_symbol, timeout=5):
    """
    Quick validation to check if a ticker is valid for yfinance.
    
    Args:
        ticker_symbol: Stock ticker symbol to validate
        timeout: Timeout in seconds for the validation check
    
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Try to get basic info - this is a lightweight check
        info = ticker.info
        # If info is empty or has error, ticker is invalid
        if not info or len(info) == 0:
            return False
        # Check if it's a valid stock (has symbol)
        if 'symbol' not in info and 'longName' not in info:
            return False
        return True
    except Exception as e:
        # Any exception means ticker is likely invalid
        return False


def list_available_datasets():
    """List all available cached datasets."""
    mapping = _load_id_mapping()
    if not mapping:
        print("No cached datasets found.")
        return
    
    print(f"\nFound {len(mapping)} cached dataset(s):\n")
    for data_id, info in mapping.items():
        metrics_path = os.path.join(DATA_DIR, f"{data_id}_metrics.npz")
        exists = os.path.exists(metrics_path)
        status = "✓" if exists else "✗"
        
        print(f"  {status} Data ID: {data_id}")
        print(f"    Stocks: {len(info.get('stocks', []))} stocks")
        print(f"    Args: {info.get('args', [])}")
        print(f"    Prediction Type: {info.get('prediction_type', 'classification')}")
        print(f"    NLP: {info.get('use_nlp', False)} ({info.get('nlp_method', 'N/A')})")
        print(f"    Metrics file exists: {exists}")
        print()


def update_metrics(data_id, update_sp500=False, update_returns=False):
    """
    Update metrics file for a given data_id.
    
    Args:
        data_id: The data ID to update
        update_sp500: Whether to update S&P 500 returns
        update_returns: Whether to update Returns (currently not implemented)
    """
    metrics_path = os.path.join(DATA_DIR, f"{data_id}_metrics.npz")
    test_path = os.path.join(DATA_DIR, f"{data_id}_test.npz")
    
    # Check if files exist
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found: {metrics_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"Error: Test data file not found: {test_path}")
        print("Cannot update metrics without test dates.")
        return False
    
    # Load existing metrics
    print(f"Loading existing metrics from {metrics_path}...")
    try:
        metrics_data = _load_npz_progress(
            metrics_path, 
            ["Rev", "Returns", "Sp500"], 
            desc="Loading metrics",
            optional_names=["Sp500"]
        )
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return False
    
    # Load test dates (and stock indices if needed for Returns update)
    # Optimize: load both D and S together if Returns update is needed
    if update_returns:
        print(f"Loading test data (dates and stock indices) from {test_path}...")
        try:
            test_data = _load_npz_progress(
                test_path,
                ["D"],
                desc="Loading test data",
                optional_names=["S"]
            )
            test_dates = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
            stock_indices = test_data.get("S")
        except Exception as e:
            print(f"Error loading test data: {e}")
            return False
    else:
        print(f"Loading test dates from {test_path}...")
        try:
            test_data = _load_npz_progress(
                test_path,
                ["D"],
                desc="Loading test dates"
            )
            test_dates = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
            stock_indices = None
        except Exception as e:
            print(f"Error loading test dates: {e}")
            return False
    
    print(f"Found {len(test_dates)} test samples")
    print(f"  Date range: {min(test_dates)} to {max(test_dates)}")
    
    # Update S&P 500 returns if requested
    if update_sp500:
        print("\nUpdating S&P 500 returns...")
        data_source = YFinanceDataSource()
        try:
            new_sp500 = _get_sp500_returns_for_dates(test_dates, None, None, data_source)
            metrics_data["Sp500"] = new_sp500
            print(f"✓ Updated S&P 500 returns: {len(new_sp500)} values")
            print(f"  Mean: {new_sp500.mean():.6f}, Std: {new_sp500.std():.6f}")
        except Exception as e:
            print(f"Error fetching S&P 500 data: {e}")
            return False
    
    # Update Returns if requested
    if update_returns:
        print("\nUpdating Returns (and Rev) from stock data...")
        
        if stock_indices is None:
            print("Error: Stock indices not found in test data.")
            print("This dataset was created without stock indices. Cannot update Returns.")
            print("Tip: Regenerate the dataset with return_stock_indices=True to enable Returns updates.")
            return False
        
        stock_indices = stock_indices.astype(int)
        
        # Verify alignment
        if len(stock_indices) != len(test_dates):
            print(f"Error: Mismatch between stock indices ({len(stock_indices)}) and test dates ({len(test_dates)})")
            return False
        
        # Load data mapping to get stock list
        mapping = _load_id_mapping()
        if data_id not in mapping:
            print(f"Error: Data ID {data_id} not found in mapping.")
            return False
        
        stock_list = mapping[data_id].get('stocks', [])
        if not stock_list:
            print("Error: Stock list not found in data mapping.")
            return False
        
        print(f"Found {len(stock_list)} stocks in dataset")
        print(f"Found {len(stock_indices)} test samples with stock indices")
        
        # Get unique dates and stocks needed
        unique_dates = sorted(set(test_dates))
        unique_stock_indices = sorted(set(stock_indices))
        unique_stocks = [stock_list[i] for i in unique_stock_indices if i < len(stock_list)]
        
        print(f"Need to download data for {len(unique_stocks)} unique stocks")
        print(f"Date range: {min(unique_dates)} to {max(unique_dates)}")
        
        # Download stock data
        data_source = YFinanceDataSource()
        min_date = min(unique_dates)
        max_date = max(unique_dates)
        
        # Convert to datetime if needed
        if isinstance(min_date, pd.Timestamp):
            min_date_dt = min_date.to_pydatetime()
        elif isinstance(min_date, datetime):
            min_date_dt = min_date
        else:
            min_date_dt = pd.Timestamp(min_date).to_pydatetime()
        
        if isinstance(max_date, pd.Timestamp):
            max_date_dt = max_date.to_pydatetime()
        elif isinstance(max_date, datetime):
            max_date_dt = max_date
        else:
            max_date_dt = pd.Timestamp(max_date).to_pydatetime()
        
        # Add buffer for yfinance
        max_date_buffered = max_date_dt + timedelta(days=1)
        
        print(f"\nDownloading Open/Close prices for {len(unique_stocks)} stocks...")
        args = [min_date_dt.strftime('%Y-%m-%d'), max_date_buffered.strftime('%Y-%m-%d')]
        
        open_close_data, failed_stocks = data_source.fetch_stock_data(unique_stocks, args, max_retries=3)
        
        if open_close_data is None or open_close_data.empty:
            print("Error: Could not download stock data.")
            if failed_stocks:
                print(f"Failed stocks: {failed_stocks}")
            return False
        
        print(f"✓ Downloaded data for {len(open_close_data.columns.levels[1])} stocks")
        
        # Extract Open and Close DataFrames
        if 'Open' in open_close_data.columns.levels[0] and 'Close' in open_close_data.columns.levels[0]:
            op_data = open_close_data['Open']
            cp_data = open_close_data['Close']
        else:
            print("Error: Open or Close columns not found in downloaded data.")
            return False
        
        # Create mapping from stock symbol to index in unique_stocks
        stock_to_idx = {stock: idx for idx, stock in enumerate(unique_stocks)}
        
        # Recalculate Returns for each test sample
        print(f"\nRecalculating Returns for {len(test_dates)} test samples...")
        new_returns = np.full(len(test_dates), np.nan, dtype=float)
        new_rev = np.full(len(test_dates), np.nan, dtype=float)
        
        missing_count = 0
        success_count = 0
        
        for i, (test_date, stock_idx) in enumerate(zip(test_dates, stock_indices)):
            if stock_idx >= len(stock_list):
                print(f"Warning: Stock index {stock_idx} out of range for sample {i}")
                missing_count += 1
                continue
            
            stock_symbol = stock_list[stock_idx]
            
            # Check if we have data for this stock
            if stock_symbol not in stock_to_idx:
                # Stock might have failed to download
                missing_count += 1
                continue
            
            # Normalize date for comparison
            if isinstance(test_date, pd.Timestamp):
                date_normalized = test_date.normalize()
            elif isinstance(test_date, datetime):
                date_normalized = pd.Timestamp(test_date).normalize()
            else:
                date_normalized = pd.Timestamp(test_date).normalize()
            
            # Get Open and Close prices for this stock on this date
            try:
                if stock_symbol in op_data.columns and stock_symbol in cp_data.columns:
                    # Find matching date in the data
                    op_series = op_data[stock_symbol]
                    cp_series = cp_data[stock_symbol]
                    
                    # Normalize index
                    if isinstance(op_series.index, pd.DatetimeIndex) and op_series.index.tz is not None:
                        op_series.index = op_series.index.tz_localize(None)
                    if isinstance(cp_series.index, pd.DatetimeIndex) and cp_series.index.tz is not None:
                        cp_series.index = cp_series.index.tz_localize(None)
                    
                    # Find exact match or closest before
                    op_series_normalized = pd.DatetimeIndex([pd.Timestamp(idx).normalize() for idx in op_series.index])
                    cp_series_normalized = pd.DatetimeIndex([pd.Timestamp(idx).normalize() for idx in cp_series.index])
                    
                    # Try exact match first
                    exact_match_op = op_series_normalized == date_normalized
                    exact_match_cp = cp_series_normalized == date_normalized
                    
                    if exact_match_op.any() and exact_match_cp.any():
                        open_price = op_series.iloc[exact_match_op.argmax()]
                        close_price = cp_series.iloc[exact_match_cp.argmax()]
                    else:
                        # Forward fill: find most recent value before or at this date
                        before_mask_op = op_series_normalized <= date_normalized
                        before_mask_cp = cp_series_normalized <= date_normalized
                        
                        if before_mask_op.any() and before_mask_cp.any():
                            before_indices_op = np.where(before_mask_op)[0]
                            before_indices_cp = np.where(before_mask_cp)[0]
                            open_price = op_series.iloc[before_indices_op[-1]]
                            close_price = cp_series.iloc[before_indices_cp[-1]]
                        else:
                            # No data available
                            missing_count += 1
                            continue
                    
                    # Calculate return (Close - Open)
                    if pd.notna(open_price) and pd.notna(close_price) and open_price != 0:
                        return_val = float(close_price - open_price)
                        new_returns[i] = return_val
                        new_rev[i] = return_val  # Rev is same as Returns
                        success_count += 1
                    else:
                        missing_count += 1
                else:
                    missing_count += 1
            except Exception as e:
                print(f"Warning: Error processing sample {i} (stock={stock_symbol}, date={test_date}): {e}")
                missing_count += 1
        
        # Fill missing values with existing Returns values
        if missing_count > 0:
            print(f"Warning: Could not calculate Returns for {missing_count} samples (using existing values)")
            existing_returns = metrics_data.get("Returns", metrics_data.get("Rev", np.zeros(len(test_dates))))
            missing_mask = np.isnan(new_returns)
            new_returns[missing_mask] = existing_returns[missing_mask]
            new_rev[missing_mask] = existing_returns[missing_mask]
        
        metrics_data["Returns"] = new_returns
        metrics_data["Rev"] = new_rev
        
        print(f"✓ Updated Returns: {len(new_returns)} values")
        print(f"  Successfully recalculated: {success_count} samples")
        print(f"  Mean: {new_returns.mean():.6f}, Std: {new_returns.std():.6f}")
        print(f"  Min: {new_returns.min():.6f}, Max: {new_returns.max():.6f}")
        if missing_count > 0:
            print(f"  Note: {missing_count} samples used existing values (data unavailable)")
    
    # Save updated metrics
    print(f"\nSaving updated metrics to {metrics_path}...")
    try:
        _save_npz_progress(metrics_path, metrics_data, desc="Saving updated metrics")
        print("✓ Metrics updated successfully!")
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update metrics (S&P 500 returns, Returns) without redownloading training/validation/test data"
    )
    parser.add_argument(
        "--data-id",
        type=str,
        default=None,
        help="Data ID of a single dataset to update (default: uses DATA_ID variable)"
    )
    parser.add_argument(
        "--data-ids",
        type=str,
        default=None,
        help="Comma-separated list of data IDs to update (e.g., 'id1,id2,id3') (default: uses DATA_IDS variable)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available cached datasets (default: uses LIST_DATASETS variable)"
    )
    parser.add_argument(
        "--update-sp500",
        action="store_true",
        help="Update S&P 500 returns (default: uses UPDATE_SP500 variable)"
    )
    parser.add_argument(
        "--update-returns",
        action="store_true",
        help="Update Returns by redownloading stock prices (requires stock indices in test data) (default: uses UPDATE_RETURNS variable)"
    )
    
    args = parser.parse_args()
    
    # Use command-line args if provided, otherwise use variables
    list_datasets = args.list_datasets if args.list_datasets else LIST_DATASETS
    update_sp500 = args.update_sp500 if args.update_sp500 else UPDATE_SP500
    update_returns = args.update_returns if args.update_returns else UPDATE_RETURNS
    
    if list_datasets:
        list_available_datasets()
        return
    
    # Determine which data IDs to process
    data_ids_to_process = []
    
    # Priority: command-line args > variables
    if args.data_ids:
        # Parse comma-separated list
        data_ids_to_process = [id.strip() for id in args.data_ids.split(',') if id.strip()]
    elif args.data_id:
        # Single ID from command line
        data_ids_to_process = [args.data_id]
    elif DATA_IDS:
        # List from variable
        if isinstance(DATA_IDS, list):
            data_ids_to_process = DATA_IDS
        elif isinstance(DATA_IDS, str):
            # Handle comma-separated string
            data_ids_to_process = [id.strip() for id in DATA_IDS.split(',') if id.strip()]
    elif DATA_ID:
        # Single ID from variable
        data_ids_to_process = [DATA_ID]
    
    if not data_ids_to_process:
        print("Error: --data-id or --data-ids is required (or set DATA_ID/DATA_IDS variable, or use --list-datasets to see available IDs)")
        parser.print_help()
        return
    
    if not update_sp500 and not update_returns:
        print("Error: At least one of --update-sp500 or --update-returns must be specified (or set UPDATE_SP500/UPDATE_RETURNS variables)")
        parser.print_help()
        return
    
    # Process all data IDs
    print(f"\n{'='*70}")
    print(f"Processing {len(data_ids_to_process)} dataset(s)")
    print(f"{'='*70}\n")
    
    results = []
    for idx, data_id in enumerate(data_ids_to_process, 1):
        print(f"\n[{idx}/{len(data_ids_to_process)}] Processing dataset: {data_id}")
        print("-" * 70)
        
        try:
            success = update_metrics(
                data_id,
                update_sp500=update_sp500,
                update_returns=update_returns
            )
            results.append((data_id, success))
        except Exception as e:
            print(f"Error processing {data_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append((data_id, False))
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = [data_id for data_id, success in results if success]
    failed = [data_id for data_id, success in results if not success]
    
    print(f"Successfully processed: {len(successful)}/{len(results)}")
    if successful:
        print(f"  ✓ {', '.join(successful)}")
    if failed:
        print(f"Failed: {len(failed)}/{len(results)}")
        print(f"  ✗ {', '.join(failed)}")
    
    # Exit with error code if any failed
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
