#!/usr/bin/env python3
"""
Script to extract daily metrics from evaluation results JSON files
and save each metric as a separate CSV file with models as columns.
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. S&P 500 metrics will not be extracted.")


def extract_model_name(filename):
    """Extract model name from evaluation results filename.
    
    Example: evaluation_results_lstm_base_20251201_071852.json -> lstm_base
    """
    basename = os.path.basename(filename)
    # Remove prefix and suffix
    match = re.match(r'evaluation_results_(.+?)_\d{8}_\d{6}\.json', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def load_evaluation_results(json_file):
    """Load evaluation results JSON and extract daily metrics.
    
    Daily metrics are located at: metrics -> paper_aligned -> cross_sectional_diagnostics -> daily_metrics
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Navigate to daily_metrics
    # Path: metrics -> paper_aligned -> cross_sectional_diagnostics -> daily_metrics
    daily_metrics = []
    try:
        metrics = data.get('metrics', {})
        paper_aligned = metrics.get('paper_aligned', {})
        cross_sectional = paper_aligned.get('cross_sectional_diagnostics', {})
        daily_metrics = cross_sectional.get('daily_metrics', [])
    except (KeyError, AttributeError) as e:
        print(f"  Warning: Could not find daily_metrics in expected location: {e}")
        # Try alternative path - maybe it's at top level or different location
        daily_metrics = data.get('daily_metrics', [])
    
    model_name = extract_model_name(json_file)
    
    return model_name, daily_metrics


def normalize_date(date_str):
    """Convert date string to datetime object and format as YYYY-MM-DD."""
    try:
        # Parse ISO format date string
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        # Return as date string (YYYY-MM-DD)
        return dt.date().isoformat()
    except:
        # Fallback: try to parse various formats
        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.date().isoformat()
            except:
                continue
        return date_str  # Return as-is if can't parse


def fetch_sp500_daily_data(dates):
    """
    Fetch S&P 500 daily returns for the given dates.
    
    Args:
        dates: List of date strings in YYYY-MM-DD format
        
    Returns:
        Dictionary mapping date strings to S&P 500 return values
    """
    if not YFINANCE_AVAILABLE:
        print("  Warning: yfinance not available, cannot fetch S&P 500 data")
        return {}
    
    if not dates:
        return {}
    
    # Convert dates to datetime objects
    date_objs = [datetime.strptime(d, '%Y-%m-%d').date() if isinstance(d, str) else d for d in dates]
    min_date = min(date_objs)
    max_date = max(date_objs)
    
    print(f"  Fetching S&P 500 data from {min_date} to {max_date}...")
    
    try:
        # Fetch S&P 500 data (^GSPC or SPY)
        ticker = "^GSPC"  # S&P 500 index
        # Extend end date by a few days to ensure we get all dates
        from datetime import timedelta
        import warnings
        # Suppress FutureWarning about auto_adjust (new default is True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            end_date_extended = max_date + timedelta(days=7)
            sp500_data = yf.download(ticker, start=min_date, end=end_date_extended, 
                                     progress=False)
        
        if sp500_data.empty:
            print("  Warning: No S&P 500 data retrieved")
            return {}
        
        # Handle MultiIndex columns from yfinance (if present)
        if isinstance(sp500_data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take first level or flatten completely
            if sp500_data.columns.nlevels > 1:
                sp500_data.columns = sp500_data.columns.droplevel(1)
        
        # Get close prices for normalization
        close_col = None
        if 'Close' in sp500_data.columns:
            close_col = 'Close'
        elif 'Adj Close' in sp500_data.columns:
            close_col = 'Adj Close'
        
        if close_col is None:
            print("  Error: Close price column not found in S&P 500 data")
            print(f"  Available columns: {list(sp500_data.columns)}")
            return {}
        
        # Extract first close price as scalar value using .item() or .values[0]
        close_series = sp500_data[close_col]
        if len(close_series) == 0:
            print("  Error: No price data available")
            return {}
        
        # Get first value as scalar
        try:
            first_close = float(close_series.iloc[0])
        except (ValueError, TypeError):
            # Try using .item() if it's a single-element Series
            try:
                first_close = float(close_series.values[0])
            except:
                print("  Error: Could not extract first price value")
                return {}
        
        # Normalize prices using subtraction: subtract first price and add 1.0
        # Formula: normalized_value = (price - first_price) + 1.0
        # This ensures the first value is exactly 1.0
        if pd.notna(first_close):
            # Normalize by subtracting first price and adding 1.0
            normalized_close = (close_series - first_close) + 1.0
        else:
            print("  Warning: First price is invalid (NaN), using original prices")
            normalized_close = close_series
        
        # Calculate daily returns from normalized prices
        sp500_data['Return'] = normalized_close.pct_change()
        
        # Fill NaN values (first row will be NaN after pct_change)
        sp500_data['Return'] = sp500_data['Return'].fillna(0.0)
        
        # Create date-to-return mapping
        sp500_returns_raw = {}
        for idx, row in sp500_data.iterrows():
            if hasattr(idx, 'date'):
                date_obj = idx.date()
            elif hasattr(idx, 'to_pydatetime'):
                date_obj = idx.to_pydatetime().date()
            else:
                date_obj = pd.to_datetime(idx).date()
            date_str = date_obj.isoformat()
            return_val = float(row['Return']) if not pd.isna(row['Return']) else 0.0
            sp500_returns_raw[date_str] = return_val
        
        # Align with requested dates (forward-fill for weekends/holidays)
        sp500_returns = {}
        last_return = 0.0
        for date in sorted(dates):
            if date in sp500_returns_raw:
                last_return = sp500_returns_raw[date]
                sp500_returns[date] = last_return
            else:
                # Use last known return (forward-fill)
                sp500_returns[date] = last_return
        
        print(f"  Retrieved {len(sp500_returns_raw)} S&P 500 return values, aligned to {len(dates)} dates")
        return sp500_returns
        
    except Exception as e:
        print(f"  Error fetching S&P 500 data: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calculate_sp500_metrics(sp500_returns, dates):
    """
    Calculate S&P 500 daily metrics comparable to portfolio metrics.
    
    The total_value starts at 1.0 on the first date, then applies returns
    from subsequent dates to calculate cumulative value.
    Prices are normalized using subtraction: (price - first_price) + 1.0,
    ensuring the first value is exactly 1.0.
    
    Args:
        sp500_returns: Dictionary mapping date strings to return values (from normalized prices using subtraction)
        dates: Sorted list of date strings
        
    Returns:
        Dictionary with metrics: {
            date: {
                'sp500_return': float,
                'sp500_return_percentage': float,
                'sp500_total_value': float
            }
        }
    """
    metrics = {}
    cumulative_value = 1.0
    sorted_dates = sorted(dates)
    
    for i, date in enumerate(sorted_dates):
        return_val = sp500_returns.get(date, 0.0)
        if pd.isna(return_val):
            return_val = 0.0
        
        # First date: total_value starts exactly at 1.0
        if i == 0:
            total_value = 1.0
        else:
            # Apply return from previous date to get this date's cumulative value
            # The return on date N affects the value on date N+1
            prev_date = sorted_dates[i-1]
            prev_return = sp500_returns.get(prev_date, 0.0)
            if pd.isna(prev_return):
                prev_return = 0.0
            cumulative_value = cumulative_value * (1.0 + prev_return)
            total_value = cumulative_value
        
        metrics[date] = {
            'sp500_return': return_val,
            'sp500_return_percentage': return_val * 100.0,  # Convert to percentage
            'sp500_total_value': total_value
        }
    
    return metrics


# Metrics to extract (as they appear in the JSON)
METRICS = [
    'auc',
    'ic',
    'logloss',
    'brier',
    'topk_hit_rate',
    'long_short_spread',
    'difficulty',
    'ic_rolling_mean',
    'ic_rolling_std',
    'long_return',
    'long_return_percentage',
    'short_return',
    'short_return_percentage',
    'gross_return',
    'gross_return_percentage',
    'net_return',
    'net_return_percentage',
    'total_value'
]

# Mapping for CSV column names (some metrics have spaces in user's request)
METRIC_CSV_NAMES = {
    'auc': 'auc',
    'ic': 'ic',
    'logloss': 'logloss',
    'brier': 'brier',
    'topk_hit_rate': 'topk hit rate',
    'long_short_spread': 'long short spread',
    'difficulty': 'difficulty',
    'ic_rolling_mean': 'ic rolling mean',
    'ic_rolling_std': 'ic rolling std',
    'long_return': 'long return',
    'long_return_percentage': 'long return percentage',
    'short_return': 'short return',
    'short_return_percentage': 'short return percentage',
    'gross_return': 'gross return',
    'gross_return_percentage': 'gross return percentage',
    'net_return': 'net return',
    'net_return_percentage': 'net return percentage',
    'total_value': 'total value'
}


def extract_metrics_from_json_files(results_dir, output_dir=None):
    """Extract daily metrics from all evaluation results JSON files."""
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'daily_metrics_csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all evaluation results JSON files
    json_pattern = os.path.join(results_dir, 'evaluation_results_*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No evaluation results JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} evaluation results files:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    # Dictionary to store all metrics by date and model
    # Structure: {metric_name: {date: {model_name: value}}}
    all_metrics = defaultdict(lambda: defaultdict(dict))
    all_dates = set()
    all_models = set()
    
    # Load data from all JSON files
    for json_file in json_files:
        print(f"\nProcessing {os.path.basename(json_file)}...")
        try:
            model_name, daily_metrics = load_evaluation_results(json_file)
            all_models.add(model_name)
            
            print(f"  Model: {model_name}")
            print(f"  Found {len(daily_metrics)} daily metric entries")
            
            for entry in daily_metrics:
                date_str = entry.get('date')
                if not date_str:
                    continue
                
                date_normalized = normalize_date(date_str)
                all_dates.add(date_normalized)
                
                # Extract each metric
                for metric in METRICS:
                    value = entry.get(metric)
                    # Handle NaN values
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        value = np.nan
                    all_metrics[metric][date_normalized][model_name] = value
        
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort dates
    sorted_dates = sorted(all_dates)
    sorted_models = sorted(all_models)
    
    print(f"\nFound {len(sorted_dates)} unique dates")
    print(f"Found {len(sorted_models)} models: {', '.join(sorted_models)}")
    
    # Fetch S&P 500 data for all dates
    print("\nFetching S&P 500 daily data...")
    sp500_returns = fetch_sp500_daily_data(sorted_dates)
    sp500_metrics = calculate_sp500_metrics(sp500_returns, sorted_dates) if sp500_returns else {}
    
    # Metrics that have S&P 500 equivalents
    SP500_METRIC_MAPPING = {
        'long_return': 'sp500_return',
        'long_return_percentage': 'sp500_return_percentage',
        'gross_return': 'sp500_return',
        'gross_return_percentage': 'sp500_return_percentage',
        'net_return': 'sp500_return',  # S&P 500 has no transaction costs
        'net_return_percentage': 'sp500_return_percentage',
        'total_value': 'sp500_total_value'
    }
    
    # Create CSV file for each metric
    for metric in METRICS:
        csv_filename = f"daily_{metric}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        print(f"\nCreating {csv_filename}...")
        
        # Build DataFrame
        rows = []
        for date in sorted_dates:
            row = {'date': date}
            # Add model columns
            for model in sorted_models:
                value = all_metrics[metric][date].get(model, np.nan)
                row[model] = value
            
            # Add S&P 500 column if this metric has an S&P 500 equivalent
            if metric in SP500_METRIC_MAPPING and sp500_metrics:
                sp500_metric_key = SP500_METRIC_MAPPING[metric]
                sp500_value = sp500_metrics.get(date, {}).get(sp500_metric_key, np.nan)
                row['sp500'] = sp500_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns: date first, then models, then sp500 if present
        if 'sp500' in df.columns:
            cols = ['date'] + sorted_models + ['sp500']
            df = df[cols]
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        
        sp500_info = " (with S&P 500)" if metric in SP500_METRIC_MAPPING and sp500_metrics else ""
        print(f"  Saved {len(df)} rows to {csv_path}{sp500_info}")
    
    print(f"\n✅ All metrics saved to {output_dir}/")
    print(f"\nGenerated {len(METRICS)} CSV files:")
    for metric in METRICS:
        csv_name = f"daily_{metric}.csv"
        print(f"  - {csv_name}")


def find_project_root():
    """Find the project root directory (StockPredictor) by looking for deliverables/ directory."""
    current_dir = Path(__file__).parent.absolute()
    
    # Walk up the directory tree to find the project root
    for parent in [current_dir] + list(current_dir.parents):
        deliverables_path = parent / 'deliverables'
        if deliverables_path.exists() and deliverables_path.is_dir():
            return parent
    
    # If not found, assume we're in the project root
    return current_dir.parent.parent if 'new-organization' in str(current_dir) else current_dir.parent


if __name__ == '__main__':
    import argparse
    
    # Find project root and default results directory
    project_root = find_project_root()
    default_results_dir = os.path.join(project_root, 'deliverables', 'results')
    
    parser = argparse.ArgumentParser(
        description='Extract daily metrics from evaluation results JSON files'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=default_results_dir,
        help=f'Directory containing evaluation_results_*.json files (default: {default_results_dir})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for CSV files (default: results_dir/daily_metrics_csv)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    
    print(f"Results directory: {results_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    
    extract_metrics_from_json_files(results_dir, output_dir)

