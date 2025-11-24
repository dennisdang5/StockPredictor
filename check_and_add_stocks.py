#!/usr/bin/env python3
"""
Script to ensure all stocks from train_and_evaluate.py are in FNSPID lists.
"""

import os
import pandas as pd
import re

# Paths
FNSPID_LISTS_DIR = "new-organization/data_sources/FNSPID_Financial_News_Dataset/data_scraper/lists_original"
TRAIN_EVAL_FILE = "new-organization/train_and_evaluate.py"

def extract_stocks_from_train_eval():
    """Extract stock symbols from train_and_evaluate.py"""
    with open(TRAIN_EVAL_FILE, 'r') as f:
        content = f.read()
    
    # Find the commented-out STOCKS list (the larger one in triple quotes)
    # Look for the pattern: """\nSTOCKS = [ ... ]\n"""
    match = re.search(r'"""\s*\nSTOCKS = \[(.*?)\]', content, re.DOTALL)
    if not match:
        # Fallback: try to find any STOCKS list
        match = re.search(r'STOCKS = \[(.*?)\]', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find STOCKS list in train_and_evaluate.py")
    
    stocks_section = match.group(1)
    # Extract all quoted strings
    stocks = re.findall(r'"([^"]+)"', stocks_section)
    
    # Convert to lowercase and filter out comments/empty strings
    stocks = [s.lower().strip() for s in stocks if s.strip() and not s.startswith('#')]
    
    return set(stocks)

def get_stocks_from_fnspid_lists():
    """Get all stocks currently in FNSPID lists"""
    all_stocks = set()
    
    for filename in os.listdir(FNSPID_LISTS_DIR):
        if filename.startswith('list_') and filename.endswith('.csv'):
            filepath = os.path.join(FNSPID_LISTS_DIR, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
                if 'Stock_name' in df.columns:
                    stocks = df['Stock_name'].str.lower().str.strip().dropna()
                    all_stocks.update(stocks.tolist())
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    
    return all_stocks

def get_list_file_for_stock(stock_symbol):
    """Determine which list file a stock should go in based on first letter"""
    first_letter = stock_symbol[0].lower()
    return f"list_{first_letter}.csv"

def add_missing_stocks(missing_stocks):
    """Add missing stocks to appropriate list files"""
    # Group missing stocks by first letter
    stocks_by_letter = {}
    for stock in missing_stocks:
        first_letter = stock[0].lower()
        if first_letter not in stocks_by_letter:
            stocks_by_letter[first_letter] = []
        stocks_by_letter[first_letter].append(stock)
    
    added_count = 0
    for letter, stocks in stocks_by_letter.items():
        list_file = os.path.join(FNSPID_LISTS_DIR, f"list_{letter}.csv")
        
        if not os.path.exists(list_file):
            print(f"Warning: List file {list_file} does not exist. Creating it...")
            df = pd.DataFrame({'Stock_name': stocks})
            df.to_csv(list_file, index=False, encoding='utf-8-sig')
            added_count += len(stocks)
        else:
            # Read existing list
            df = pd.read_csv(list_file, encoding='utf-8', on_bad_lines='skip')
            
            # Get existing stocks (lowercase)
            existing_stocks = set(df['Stock_name'].str.lower().str.strip().tolist())
            
            # Add missing stocks
            new_stocks = [s for s in stocks if s.lower() not in existing_stocks]
            
            if new_stocks:
                new_df = pd.DataFrame({'Stock_name': new_stocks})
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.sort_values('Stock_name')
                df.to_csv(list_file, index=False, encoding='utf-8-sig')
                added_count += len(new_stocks)
                print(f"Added {len(new_stocks)} stocks to {list_file}")
    
    return added_count

def main():
    print("=" * 80)
    print("Checking stocks from train_and_evaluate.py against FNSPID lists")
    print("=" * 80)
    
    # Extract stocks from train_and_evaluate.py
    print("\n1. Extracting stocks from train_and_evaluate.py...")
    train_stocks = extract_stocks_from_train_eval()
    print(f"   Found {len(train_stocks)} stocks")
    
    # Get stocks from FNSPID lists
    print("\n2. Reading existing FNSPID lists...")
    fnspid_stocks = get_stocks_from_fnspid_lists()
    print(f"   Found {len(fnspid_stocks)} stocks in FNSPID lists")
    
    # Find missing stocks
    print("\n3. Comparing lists...")
    missing_stocks = train_stocks - fnspid_stocks
    
    if not missing_stocks:
        print("\n✓ All stocks from train_and_evaluate.py are already in FNSPID lists!")
        return
    
    print(f"\n⚠ Found {len(missing_stocks)} missing stocks:")
    print("\nMissing stocks:")
    for stock in sorted(missing_stocks):
        print(f"  - {stock.upper()}")
    
    # Add missing stocks automatically
    print("\n" + "=" * 80)
    print("Adding missing stocks to FNSPID lists...")
    added = add_missing_stocks(missing_stocks)
    print(f"\n✓ Successfully added {added} stocks to FNSPID lists!")
    print("\nNote: You may need to run initialize_lists.py in the headline scraper folder")
    print("      to add the 'Desired_page' column to the updated lists.")

if __name__ == "__main__":
    main()

