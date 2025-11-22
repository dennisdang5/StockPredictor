"""
Check stock index alignment between op/cp DataFrames and successfully_downloaded_stocks.

This verifies that when we use op.iloc[n, t], the stock at index n matches
successfully_downloaded_stocks[n] for individual NLP feature lookup.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import fetch_stock_data
from data_sources import YFinanceDataSource


def check_stock_alignment():
    """Check if stock indices align correctly."""
    print("=" * 80)
    print("STOCK INDEX ALIGNMENT CHECK")
    print("=" * 80)
    
    # Test with a small set of stocks
    test_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    test_args = ["2020-01-01", "2020-12-31"]
    
    print(f"\nTest stocks: {test_stocks}")
    print(f"Date range: {test_args}")
    
    # Download data
    print("\nDownloading stock data...")
    data_source = YFinanceDataSource()
    open_close, failed_stocks = fetch_stock_data(test_stocks, test_args, data_source, max_retries=1)
    
    if open_close is None:
        print("✗ Failed to download data")
        return
    
    # Get successfully downloaded stocks (matching util.py logic)
    valid_stocks = test_stocks  # In real code, this would be filtered
    successfully_downloaded_stocks = [stock for stock in valid_stocks if stock in open_close["Open"].columns]
    
    print(f"\nSuccessfully downloaded stocks: {successfully_downloaded_stocks}")
    print(f"open_close['Open'].columns order: {list(open_close['Open'].columns)}")
    
    # Create op and cp (matching util.py logic)
    op = open_close["Open"].T   # (S, T)
    cp = open_close["Close"].T  # (S, T)
    
    print(f"\nop.index (stock symbols): {list(op.index)}")
    print(f"op.shape: {op.shape}")
    
    # Check alignment
    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    
    misaligned = []
    for n in range(len(successfully_downloaded_stocks)):
        stock_from_list = successfully_downloaded_stocks[n]
        stock_from_op = op.index[n]
        
        if stock_from_list != stock_from_op:
            misaligned.append((n, stock_from_list, stock_from_op))
            print(f"✗ MISALIGNMENT at index {n}:")
            print(f"    successfully_downloaded_stocks[{n}] = {stock_from_list}")
            print(f"    op.index[{n}] = {stock_from_op}")
        else:
            print(f"✓ Index {n}: {stock_from_list} == {stock_from_op}")
    
    if misaligned:
        print(f"\n⚠ WARNING: Found {len(misaligned)} misalignments!")
        print("This could cause incorrect NLP feature lookup in individual method.")
        print("\nThe issue is:")
        print("  - successfully_downloaded_stocks preserves order of valid_stocks")
        print("  - op.index preserves order of open_close['Open'].columns")
        print("  - These orders may differ!")
        print("\nFix: Use op.index[n] instead of successfully_downloaded_stocks[n]")
        return False
    else:
        print("\n✓ All stocks are correctly aligned!")
        return True


if __name__ == "__main__":
    check_stock_alignment()

