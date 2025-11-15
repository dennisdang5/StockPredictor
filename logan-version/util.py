# https://doi.org/10.1016/j.frl.2021.102280
# "Forecasting directional movements of stock prices for intraday trading using LSTM and random forests" Ghosh, Neufeld, Sahoo 2022
# modifications for classification model

import pandas as pd
import torch
import yfinance as yf
import numpy as np
import statistics
import torchsummary
import os
import json
import hashlib
from tqdm import tqdm
from zipfile import ZipFile, ZIP_STORED
from io import BytesIO

########################################################
# helper functions
########################################################

def _save_npz_progress(path: str, arrays: dict, desc="Saving dataset"):
    with ZipFile(path, mode="w", compression=ZIP_STORED) as zf:
        bar = tqdm(total=len(arrays), desc=desc)
        for name, arr in arrays.items():
            buf = BytesIO()
            np.save(buf, np.asanyarray(arr), allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())
            bar.update(1)
        bar.close()

def _load_npz_progress(path: str, names: list, desc="Loading dataset"):
    # Use numpy.load for correctness but show progress as we realize arrays.
    with np.load(path, allow_pickle=False) as z:
        out = {}
        bar = tqdm(total=len(names), desc=desc)
        for n in names:
            out[n] = z[n]            # triggers decompression/read for that entry
            bar.update(1)
        bar.close()
        return out

########################################################
# data management
########################################################

def _get_data_id(stocks, args):
    """
    Generate a short hash-based ID for a unique combination of stocks and time args.
    
    Args:
        stocks: List of stock symbols
        args: Time arguments
        
    Returns:
        A short 10-character hash string
    """
    # Create a stable string representation
    stocks_str = ",".join(sorted(stocks))  # Sort for consistency
    args_str = ",".join(str(a) for a in args)
    combined = f"{stocks_str}|{args_str}"
    
    # Generate a short hash
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:10]

def _load_id_mapping(data_dir="data"):
    """
    Load the ID to (stocks, args) mapping from disk.
    
    Returns:
        Dictionary mapping data_id -> {'stocks': [...], 'args': [...], 'full_name': '...'}
    """
    mapping_path = os.path.join(data_dir, "_data_mapping.json")
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load data mapping: {e}")
            return {}
    return {}

def _save_id_mapping(data_id, stocks, args, data_dir="data"):
    """
    Save the ID to (stocks, args) mapping to disk.
    
    Args:
        data_id: Short hash ID
        stocks: List of stock symbols
        args: Time arguments
        data_dir: Directory for data files
    """
    os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists
    mapping_path = os.path.join(data_dir, "_data_mapping.json")
    
    try:
        mapping = _load_id_mapping(data_dir)
        mapping[data_id] = {
            'stocks': stocks,
            'args': args,
            'full_name': "_".join(stocks[:5]) + (f"_{len(stocks)-5}_more" if len(stocks) > 5 else "")
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save data mapping: {e}")

def _get_args_key(args):
    """
    Generate a stable key for time arguments to use for storing problematic stocks.
    
    Args:
        args: Time arguments (either period string or [start, end] tuple)
    
    Returns:
        A string key representing the time arguments
    """
    if len(args) == 1:
        return f"period_{args[0]}"
    elif len(args) == 2:
        return f"start_{args[0]}_end_{args[1]}"
    else:
        return f"args_{hash(tuple(args))}"

def _load_problematic_stocks(args, data_dir="data"):
    """
    Load the list of problematic stocks for given time arguments.
    
    Args:
        args: Time arguments
        data_dir: Directory for data files
    
    Returns:
        Set of problematic stock symbols, or empty set if not found
    """
    args_key = _get_args_key(args)
    problematic_path = os.path.join(data_dir, f"_problematic_stocks_{args_key}.json")
    
    if os.path.exists(problematic_path):
        try:
            with open(problematic_path, 'r') as f:
                data = json.load(f)
                return set(data.get('problematic_stocks', []))
        except Exception as e:
            print(f"Warning: Could not load problematic stocks: {e}")
            return set()
    return set()

def _save_problematic_stocks(problematic_stocks, args, data_dir="data"):
    """
    Save the list of problematic stocks for given time arguments.
    
    Args:
        problematic_stocks: List or set of problematic stock symbols
        args: Time arguments
        data_dir: Directory for data files
    """
    os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists
    args_key = _get_args_key(args)
    problematic_path = os.path.join(data_dir, f"_problematic_stocks_{args_key}.json")
    
    try:
        # Convert to list and sort for consistency
        stocks_list = sorted(list(set(problematic_stocks)))
        data = {
            'args': args,
            'problematic_stocks': stocks_list,
            'count': len(stocks_list)
        }
        
        with open(problematic_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save problematic stocks: {e}")

def _get_sp500_returns_for_dates(test_dates, date_index, study_period):
    """
    Fetch S&P 500 returns aligned with test dates.
    
    Args:
        test_dates: List of test dates (datetime objects)
        date_index: Original date index from the data
        study_period: Study period length
        
    Returns:
        numpy array of S&P 500 returns aligned with test_dates
    """
    if not test_dates or len(test_dates) == 0:
        print("[util] No test dates available for S&P 500")
        return np.array([])
    
    try:
        print("[util] Fetching S&P 500 data for metrics...")
        
        # Get date range from test dates
        min_date = min(test_dates)
        max_date = max(test_dates)
        
        # Fetch S&P 500 data (^GSPC is the ticker for S&P 500)
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(start=min_date, end=max_date, repair=True)
        
        if sp500_data is None or sp500_data.empty:
            print("[util] Warning: Could not fetch S&P 500 data, using zeros")
            return np.zeros(len(test_dates))
        
        # Calculate daily returns from S&P 500 close prices
        sp500_close = sp500_data['Close']
        sp500_daily_returns = sp500_close.pct_change().dropna()

        if isinstance(sp500_daily_returns.index, pd.DatetimeIndex) and sp500_daily_returns.index.tz is not None:
            sp500_daily_returns.index = sp500_daily_returns.index.tz_localize(None)
        
        # Convert test dates to pandas Timestamp for alignment
        test_dates_pd = pd.to_datetime(test_dates)
        if isinstance(test_dates_pd, pd.DatetimeIndex) and test_dates_pd.tz is not None:
            test_dates_pd = test_dates_pd.tz_localize(None)
        
        # Align S&P 500 returns with test dates
        # Use forward fill to match each test date with the most recent S&P 500 return
        sp500_returns = []
        for test_date in test_dates_pd:
            # Find S&P 500 return for this date or previous trading day
            # Use reindex with method='ffill' to forward fill from previous trading day
            date_returns = sp500_daily_returns.reindex([test_date], method='ffill')
            
            if len(date_returns) > 0 and not pd.isna(date_returns.iloc[0]):
                sp500_returns.append(date_returns.iloc[0])
            else:
                # If no data available, use 0 (no return)
                sp500_returns.append(0.0)
        
        sp500_returns = np.array(sp500_returns, dtype=float)
        
        print(f"[util] S&P 500 returns fetched: {len(sp500_returns)} values, mean={np.mean(sp500_returns):.6f}, std={np.std(sp500_returns):.6f}")
        return sp500_returns
        
    except Exception as e:
        print(f"[util] Warning: Error fetching S&P 500 data: {e}, using zeros")
        return np.zeros(len(test_dates))

def identify_problematic_stocks(stocks, args, max_retries=1):
    """
    Identify which stocks from the input list are problematic (cannot be downloaded).
    This is a lightweight check that only validates downloadability without processing data.
    
    Args:
        stocks: List of stock tickers
        args: Date range arguments (either period string or [start, end] tuple)
        max_retries: Maximum number of retry attempts per stock (default: 1 for speed)
    
    Returns:
        tuple: (valid_stocks, problematic_stocks)
               - valid_stocks: List of stocks that can be downloaded
               - problematic_stocks: List of stocks that failed to download
    """
    import time
    
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

def handle_yfinance_errors(stocks, args, max_retries=3):
    """
    Utility function to handle yfinance errors gracefully.
    
    This function attempts to fetch stock data with error handling for:
    - YFPricesMissingError: Stock may be delisted or has no data for the date range
    - YFTzMissingError: Missing timezone information
    - Other yfinance errors
    
    Args:
        stocks: List of stock tickers
        args: Date range arguments (either period string or [start, end] tuple)
        max_retries: Maximum number of retry attempts per stock (default: 3)
    
    Returns:
        tuple: (open_close_dataframe, failed_stocks)
               - open_close_dataframe: DataFrame with MultiIndex columns (Open/Close, stock_symbols)
               - failed_stocks: Dictionary mapping error types to lists of failed stocks with details
    """
    import time
    
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

def get_data(stocks, args, data_dir="data", lookback=240, force=False, prediction_type="classification", open_close_data=None, problematic_stocks=None, use_nlp=False, nlp_csv_paths=None, nlp_method="aggregated"):
    """
    Return 12-tuple: (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest, Rev_test, Returns_test, Sp500_test)
    Loads from .npz if present (and not force); otherwise builds from yfinance,
    saves .npz, and returns the tuple.
    
    Args:
        stocks: List of stock tickers (should already be filtered to remove problematic stocks)
        args: Date range arguments
        data_dir: Directory for data files
        lookback: Lookback period for features
        force: Force rebuild even if cache exists
        prediction_type: Type of prediction (default: "classification")
        open_close_data: Optional pre-downloaded open_close DataFrame to avoid redundant download
        problematic_stocks: Optional list of problematic stocks to save (needed when open_close_data is provided)
        use_nlp: Whether to include NLP features (default: False)
        nlp_csv_paths: Path(s) to NYT CSV file(s) for NLP features. If None and use_nlp=True, 
            tries to find CSV files in ../other_models/huggingface_nyt_articles/
        nlp_method: Method for NLP feature extraction - "aggregated" (NYT headlines, shared across stocks)
            or "individual" (yfinance news per stock ticker). Default: "aggregated"
    """
    os.makedirs(data_dir, exist_ok=True)
    
    prediction_suffix = f"_{prediction_type}"
    # Differentiate cache paths for aggregated vs individual NLP methods
    if use_nlp:
        if nlp_method == "aggregated":
            nlp_suffix = "_nlp_agg"  # Aggregated method (NYT)
        elif nlp_method == "individual":
            nlp_suffix = "_nlp_ind"  # Individual method (yfinance per stock)
        else:
            nlp_suffix = "_nlp"  # Fallback
    else:
        nlp_suffix = ""

    # Build from raw prices using error-handled download
    if len(args) not in [1, 2]:
        print("Invalid Data Input Arguments")
        return 1
    
    # If open_close_data is provided, use it (avoids redundant download)
    if open_close_data is not None:
        open_close = open_close_data
        # Get successfully downloaded stocks from the DataFrame columns
        successfully_downloaded_stocks = [stock for stock in stocks if stock in open_close["Open"].columns]
        # Use provided problematic_stocks if available, otherwise calculate from input
        if problematic_stocks is None:
            # This shouldn't happen if called correctly, but calculate as fallback
            problematic_stocks = [stock for stock in stocks if stock not in successfully_downloaded_stocks]
    else:
        # For large stock lists, use fewer retries to speed up the download
        # Most errors are permanent (delisted, no data for date range), so 1 retry is sufficient
        max_retries = 1 if len(stocks) > 50 else 3
        
        open_close, failed_stocks = handle_yfinance_errors(stocks, args, max_retries=max_retries)
        
        if open_close is None:
            print("ERROR: Failed to download any stock data. Cannot proceed.")
            return 1

        # Update stocks list to only include successfully downloaded stocks
        successfully_downloaded_stocks = [stock for stock in stocks if stock in open_close["Open"].columns]
        # Calculate problematic stocks
        problematic_stocks = [stock for stock in stocks if stock not in successfully_downloaded_stocks]
    
    # Save problematic stocks for this time period (so we can skip checking them in future runs)
    if problematic_stocks:
        _save_problematic_stocks(problematic_stocks, args, data_dir)

    # Now build op/cp/date_index from the cleaned frame
    op = open_close["Open"].T   # (S, T)
    cp = open_close["Close"].T  # (S, T)
    date_index = open_close.index

    if lookback >= op.shape[1]:
        raise ValueError("study period too short for chosen lookback")

    # Extract NLP features if requested
    nlp_features_dict = None
    if use_nlp:
        print(f"[nlp] Extracting NLP features using method: {nlp_method}...")
        try:
            from nlp_features import extract_daily_nlp_features, extract_daily_nlp_features_yfinance, align_nlp_with_trading_days, get_nlp_feature_vector
            
            # Get date range from args
            if len(args) == 2:
                start_date = args[0]
                end_date = args[1]
            elif len(args) == 1:
                start_date = None
                end_date = None
            else:
                start_date = None
                end_date = None
            
            if nlp_method == "aggregated":
                # Aggregated method: Use NYT headlines (shared across all stocks)
                # Determine NLP CSV paths
                if nlp_csv_paths is None:
                    import glob
                    from pathlib import Path
                    nyt_dir = Path("./huggingface_nyt_articles")
                    nlp_csv_paths = sorted(glob.glob(str(nyt_dir / "new_york_times_stories_*.csv")))
                    if not nlp_csv_paths:
                        import warnings
                        warnings.warn(f"No NYT CSV files found in {nyt_dir}. NLP features will be disabled.")
                        use_nlp = False
                        nlp_suffix = ""
                
                if use_nlp and nlp_csv_paths:
                    # Extract daily NLP features from NYT
                    nlp_df = extract_daily_nlp_features(
                        csv_paths=nlp_csv_paths,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=32,
                        progress=True
                    )
                    
                    if len(nlp_df) > 0:
                        # Align NLP features with trading days (date_index from yfinance)
                        nlp_aligned = align_nlp_with_trading_days(
                            nlp_df,
                            trading_days=date_index,
                            fill_method='zero'
                        )
                        
                        # Create a dictionary mapping date -> NLP feature vector (shared for all stocks)
                        nlp_features_dict = {}
                        for _, row in nlp_aligned.iterrows():
                            date_obj = row['date']
                            if isinstance(date_obj, pd.Timestamp):
                                date_obj = date_obj.date()
                            nlp_features_dict[date_obj] = row
                        
                        print(f"[nlp] NLP features aligned to {len(nlp_features_dict)} trading days (aggregated method)")
                    else:
                        import warnings
                        warnings.warn("No NLP features extracted. Continuing without NLP features.")
                        use_nlp = False
                        nlp_suffix = ""
                        nlp_features_dict = None
            
            elif nlp_method == "individual":
                # Individual method: Use yfinance news per stock ticker
                print(f"[nlp] Extracting individual NLP features for {len(successfully_downloaded_stocks)} stocks...")
                
                stock_nlp_features = extract_daily_nlp_features_yfinance(
                    stocks=successfully_downloaded_stocks,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=32,
                    progress=True
                )
                
                # Create a dictionary mapping (stock, date) -> NLP feature vector
                # Format: {(stock_ticker, date): nlp_row}
                nlp_features_dict = {}
                for stock, nlp_df in stock_nlp_features.items():
                    if len(nlp_df) > 0:
                        # Align NLP features with trading days for this stock
                        nlp_aligned = align_nlp_with_trading_days(
                            nlp_df,
                            trading_days=date_index,
                            fill_method='zero'
                        )
                        
                        # Store per stock-date
                        for _, row in nlp_aligned.iterrows():
                            date_obj = row['date']
                            if isinstance(date_obj, pd.Timestamp):
                                date_obj = date_obj.date()
                            nlp_features_dict[(stock, date_obj)] = row
                
                if len(nlp_features_dict) > 0:
                    print(f"[nlp] NLP features extracted for {len(set(k[0] for k in nlp_features_dict.keys()))} stocks, {len(set(k[1] for k in nlp_features_dict.keys()))} unique dates (individual method)")
                    # Mark that we're using simple format for individual method
                    nlp_features_dict['_method'] = 'individual_simple'
                else:
                    import warnings
                    warnings.warn("No NLP features extracted from yfinance. Continuing without NLP features.")
                    use_nlp = False
                    nlp_suffix = ""
                    nlp_features_dict = None
            
            else:
                raise ValueError(f"Unknown nlp_method: {nlp_method}. Must be 'aggregated' or 'individual'")
                
        except Exception as e:
            import warnings
            import traceback
            warnings.warn(f"Error extracting NLP features: {e}. Continuing without NLP features.")
            traceback.print_exc()
            use_nlp = False
            nlp_suffix = ""
            nlp_features_dict = None

    # Features/targets
    if prediction_type == "classification":
        xdata, ydata, dates, revenues, returns = get_feature_input_classification(
            op, cp, lookback, op.shape[1], len(successfully_downloaded_stocks), date_index, 
            nlp_features=nlp_features_dict, use_nlp=use_nlp, successfully_downloaded_stocks=successfully_downloaded_stocks
        )
    else:
        raise ValueError("Invalid prediction type - only 'classification' is supported")
    xdata = torch.from_numpy(xdata).to(torch.float32)  # (S, W, L, F)
    ydata = torch.from_numpy(ydata).to(torch.float32)  # (S, W, 1)

    # Date-based split (works even if some stocks miss certain dates)
    dates_np = np.array(dates, dtype="datetime64[ns]")
    uniq = np.unique(dates_np)
    n_total = len(uniq)
    n_train_val = int(n_total * (2 / 3))
    n_train = int(n_train_val * 0.8)
    start = uniq[0]
    train_end = uniq[n_train - 1] if n_train > 0 else start
    val_end = uniq[n_train_val - 1] if n_train_val > 0 else train_end
    last = uniq[-1]

    train_mask = (dates_np >= start) & (dates_np <= train_end)
    val_mask   = (dates_np >  train_end) & (dates_np <= val_end)
    test_mask  = (dates_np >  val_end)   & (dates_np <= last)

    def _sel(mask):
        X = xdata[mask]
        Y = ydata[mask]
        D = [pd.Timestamp(d).to_pydatetime() for d in dates_np[mask]]
        return X, Y, D

    Xtr_f, Ytr_f, Dtrain_f = _sel(train_mask)
    Xva_f, Yva_f, Dvalidation_f = _sel(val_mask)
    Xte_f, Yte_f, Dtest_f = _sel(test_mask)
    Rev_f = revenues[test_mask]
    Returns_f = returns[test_mask]

    # Fetch S&P 500 returns aligned with test dates
    Sp500_f = _get_sp500_returns_for_dates(Dtest_f, date_index, op.shape[1])

    # --- split summary ---
    n_tr, n_va, n_te = len(Dtrain_f), len(Dvalidation_f), len(Dtest_f)
    n_tot = n_tr + n_va + n_te
    print(f"[split] train/val/test sizes = {n_tr:,} / {n_va:,} / {n_te:,}  | total kept = {n_tot:,}")

    # NOW create the ID based on successfully downloaded stocks (after error handling)
    data_id = _get_data_id(successfully_downloaded_stocks, args)
    base = f"data_{data_id}"
    
    # Save datasets separately
    def _to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    
    # Create prediction-type specific separate file paths for each dataset
    # Include NLP suffix in cache filenames
    train_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_test.npz")
    metrics_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_metrics.npz")
    
    # Save training dataset
    _save_npz_progress(train_path, {
        "X": _to_np(Xtr_f),
        "Y": _to_np(Ytr_f),
        "D": np.array(Dtrain_f, dtype="datetime64[ns]")
    }, desc="Saving training dataset (.npz)")
    
    # Save validation dataset
    _save_npz_progress(val_path, {
        "X": _to_np(Xva_f),
        "Y": _to_np(Yva_f),
        "D": np.array(Dvalidation_f, dtype="datetime64[ns]")
    }, desc="Saving validation dataset (.npz)")
    
    # Save test dataset
    _save_npz_progress(test_path, {
        "X": _to_np(Xte_f),
        "Y": _to_np(Yte_f),
        "D": np.array(Dtest_f, dtype="datetime64[ns]")
    }, desc="Saving test dataset (.npz)")

    _save_npz_progress(metrics_path, {
        "Rev": _to_np(Rev_f),
        "Returns": _to_np(Returns_f),
        "Sp500": _to_np(Sp500_f)
    }, desc="Saving metrics dataset (.npz)")
    
    print(f"{prediction_type.title()} datasets saved separately:")
    # Use absolute paths for clarity and to avoid confusion with relative paths
    print(f"  Training: {os.path.abspath(train_path)}")
    print(f"  Validation: {os.path.abspath(val_path)}")
    print(f"  Test: {os.path.abspath(test_path)}")
    print(f"  Metrics: {os.path.abspath(metrics_path)}")
    
    # Save ID mapping so we can reference this data later (using successfully downloaded stocks)
    _save_id_mapping(data_id, successfully_downloaded_stocks, args, data_dir)
    
    return (Xtr_f, Xva_f, Xte_f, Ytr_f, Yva_f, Yte_f, Dtrain_f, Dvalidation_f, Dtest_f, Rev_f, Returns_f, Sp500_f)

def load_data_from_cache(stocks, args, data_dir="data", prediction_type="classification", use_nlp=False, nlp_method="aggregated"):
    """
    Load data from cache files if they exist.
    
    This function filters out problematic stocks before checking for cache,
    matching the behavior of get_data() which saves cache with filtered stocks.
    It also uses the mapping file to find cache by matching stocks and args.
    
    Args:
        stocks: List of stock tickers (may include problematic stocks - will be filtered)
        args: Date range arguments
        data_dir: Directory where cache files are stored
        prediction_type: Type of prediction (default: "classification")
        use_nlp: Whether NLP features were used (default: False)
        nlp_method: NLP method used - "aggregated" or "individual" (default: "aggregated")
    
    Returns:
        data tuple if cache exists, None otherwise.
    """
    prediction_suffix = f"_{prediction_type}"
    # Differentiate cache paths for aggregated vs individual NLP methods
    if use_nlp:
        if nlp_method == "aggregated":
            nlp_suffix = "_nlp_agg"  # Aggregated method (NYT)
        elif nlp_method == "individual":
            nlp_suffix = "_nlp_ind"  # Individual method (yfinance per stock)
        else:
            nlp_suffix = "_nlp"  # Fallback
    else:
        nlp_suffix = ""
    
    # Step 1: Load problematic stocks for this time period (if they exist)
    problematic_stocks = _load_problematic_stocks(args, data_dir)
    
    # Step 2: Filter out problematic stocks from input stocks (matching get_data behavior)
    filtered_stocks = [stock for stock in stocks if stock not in problematic_stocks]
    
    # Step 3: Try to find cache using filtered stocks
    # First, try direct match with filtered stocks
    data_id = _get_data_id(filtered_stocks, args)
    base = f"data_{data_id}"
    
    train_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_test.npz")
    metrics_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_metrics.npz")
    
    # Check if exact match cache exists with filtered stocks
    if all(os.path.exists(p) for p in [train_path, val_path, test_path, metrics_path]):
        # Exact match found - use it
        pass  # Will load below
    else:
        # Step 4: Try to find cache using the mapping file
        # This handles cases where the cache was created with a different stock list
        # but the filtered stocks match
        mapping = _load_id_mapping(data_dir)
        found_cache = False
        
        for cached_id, cached_info in mapping.items():
            cached_stocks = set(cached_info.get('stocks', []))
            cached_args = cached_info.get('args', [])
            
            # Normalize args for comparison (convert to list if needed)
            cached_args_list = list(cached_args) if cached_args else []
            args_list = list(args) if args else []
            
            # Check if args match and filtered stocks match cached stocks
            if cached_args_list == args_list and set(filtered_stocks) == cached_stocks:
                # Found matching cache - use this data_id
                data_id = cached_id
                base = f"data_{data_id}"
                
                # Use nlp_suffix when constructing paths (same as exact match above)
                train_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_train.npz")
                val_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_val.npz")
                test_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_test.npz")
                metrics_path = os.path.join(data_dir, base + prediction_suffix + nlp_suffix + "_metrics.npz")
                
                if all(os.path.exists(p) for p in [train_path, val_path, test_path, metrics_path]):
                    found_cache = True
                    break
        
        if not found_cache:
            # No cache found
            return None
    
    # At this point, we have valid cache paths (exact match found)
    
    # Load from cache
    train_data = _load_npz_progress(train_path, ["X", "Y", "D"], desc="Loading training dataset (.npz)")
    val_data = _load_npz_progress(val_path, ["X", "Y", "D"], desc="Loading validation dataset (.npz)")
    test_data = _load_npz_progress(test_path, ["X", "Y", "D"], desc="Loading test dataset (.npz)")
    
    # Validate that cached data has expected NLP features if NLP is requested
    if use_nlp:
        # Check feature dimensions in a sample
        sample_X = train_data.get("X")
        if sample_X is not None:
            if isinstance(sample_X, torch.Tensor):
                actual_features = sample_X.shape[2] if len(sample_X.shape) >= 3 else sample_X.shape[1]
            else:
                actual_features = sample_X.shape[2] if len(sample_X.shape) >= 3 else sample_X.shape[1]
            
            # Expected features: 3 (price) + NLP features
            expected_nlp_dim = 4 if nlp_method == "individual" else 10
            expected_features = 3 + expected_nlp_dim
            
            if actual_features != expected_features:
                # Cache mismatch - return None to force regeneration
                print(f"[cache] Warning: Cached data has {actual_features} features but expected {expected_features} features with NLP (method: {nlp_method}). Cache will be regenerated.")
                return None
    
    # Load metrics - handle both old format and new format (Rev + Returns + Sp500)
    try:
        metrics_data = _load_npz_progress(metrics_path, ["Rev", "Returns", "Sp500"], desc="Loading metrics dataset (.npz)")
        Sp500 = metrics_data.get("Sp500", None)
        if Sp500 is None:
            # Old format - fetch S&P 500
            print("[util] Old metrics format detected, fetching S&P 500 returns...")
            Dte_temp = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
            Sp500 = _get_sp500_returns_for_dates(Dte_temp, None, None)
        Returns = metrics_data.get("Returns", None)
        if Returns is None:
            print("[util] Warning: Returns not found in metrics cache, using Rev as fallback")
            Returns = metrics_data["Rev"]
        metrics_data["Returns"] = Returns
    except KeyError:
        # Old format - only Rev available
        metrics_data = _load_npz_progress(metrics_path, ["Rev"], desc="Loading metrics dataset (.npz)")
        print("[util] Old metrics format detected, fetching S&P 500 returns...")
        Dte_temp = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
        Sp500 = _get_sp500_returns_for_dates(Dte_temp, None, None)
        metrics_data["Returns"] = metrics_data["Rev"]
    
    # Convert back to torch tensors and Python datetime objects
    Xtr = torch.from_numpy(train_data["X"]).to(torch.float32)
    Ytr = torch.from_numpy(train_data["Y"]).to(torch.float32)
    Xva = torch.from_numpy(val_data["X"]).to(torch.float32)
    Yva = torch.from_numpy(val_data["Y"]).to(torch.float32)
    Xte = torch.from_numpy(test_data["X"]).to(torch.float32)
    Yte = torch.from_numpy(test_data["Y"]).to(torch.float32)
    
    # Convert datetime64 back to Python datetime objects
    Dtr = [pd.Timestamp(d).to_pydatetime() for d in train_data["D"]]
    Dva = [pd.Timestamp(d).to_pydatetime() for d in val_data["D"]]
    Dte = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
    
    Rev = metrics_data["Rev"]
    Returns = metrics_data.get("Returns", Rev)
    
    return (Xtr, Xva, Xte, Ytr, Yva, Yte, Dtr, Dva, Dte, Rev, Returns, Sp500)

def save_data_locally(stocks, args, data_dir="data", force=False, prediction_type="classification"):
    # Force a rebuild/save and return the 9-tuple
    return get_data(stocks, args, data_dir=data_dir, lookback=240, force=True, prediction_type=prediction_type)


########################################################
# get feature input
########################################################

# op[x] is the op vector for stock x
# op and cp has indices from time 0 to T_study-1
def get_feature_input_classification(op, cp, lookback, study_period, num_stocks, date_index, nlp_features=None, use_nlp=False, successfully_downloaded_stocks=None):

    T = study_period
    print(f"[features] Computing features for {num_stocks} stocks over {T} time periods...")
    # Precompute elementary series (may contain NaNs)
    f_t1 = np.full((num_stocks,3, T), np.nan, dtype=float)
    return_labels = np.full((num_stocks, T), np.nan, dtype=float)  # Initialize with NaN for proper alignment
    rev_t = np.full((num_stocks, T), np.nan, dtype=float)
    return_t = np.full((num_stocks, T), np.nan, dtype=float)
    
    print(f"[features] Step 1/2: Precomputing elementary series...")
    for t in tqdm(range(2, T), desc="Precomputing elementary series", unit="time step"):
        valid_stocks = []
        
        # First pass: collect valid revenues and their corresponding stock indices
        for n in range(num_stocks):
            o_t1, c_t1, c_t2, o_t = op.iloc[n, t-1], cp.iloc[n, t-1], cp.iloc[n, t-2], op.iloc[n, t]
            if pd.notna(c_t1) and pd.notna(o_t1):
                f_t1[n, 0, t] = c_t1 / o_t1 - 1.0     # ir
            if pd.notna(c_t1) and pd.notna(c_t2):
                f_t1[n, 1, t] = c_t1 / c_t2 - 1.0     # cpr
            if pd.notna(o_t) and pd.notna(o_t1):
                f_t1[n, 2, t] = o_t / o_t1 - 1.0      # opr
            
            # Calculate revenue/return for valid stocks only
            if pd.notna(cp.iloc[n, t]) and pd.notna(op.iloc[n, t]) and op.iloc[n, t] != 0:
                rev_t[n, t] = cp.iloc[n, t] - op.iloc[n, t]
                return_t[n, t] = (cp.iloc[n, t] - op.iloc[n, t]) / op.iloc[n, t]
                valid_stocks.append(n)
        
        # Calculate median return at time t
        return_t_valid = return_t[valid_stocks, t]
        if len(return_t_valid) > 0:
            median_return = np.median(return_t_valid)
            return_labels[valid_stocks, t] = np.where(return_t_valid >= median_return, 1.0, -1.0)
    
    print(f"[features] Step 1/2 complete. Step 2/2: Building feature windows (this may take several minutes)...")
    X_list, y_list, d_list, rev_list, return_list = [], [], [], [], []
    # --- counters ---
    total_candidates = 0
    kept = 0
    dropped_nan_target = 0
    dropped_feature_nan = 0
    dropped_flat_iqr = 0
    dropped_return_nan = 0

    # Add tqdm progress bar for the heavy computation loop
    total_windows = (T - lookback - 2) * num_stocks
    pbar = tqdm(total=total_windows, desc="Building feature windows", unit="window")
    
    for end_t in range(lookback + 2, T):
        for n in range(num_stocks):
            pbar.update(1)
            total_candidates += 1
            tgt = return_labels[n, end_t]
            rev = rev_t[n, end_t]
            ret = return_t[n, end_t]
            if pd.isna(tgt) or pd.isna(rev) or pd.isna(ret):
                if pd.isna(tgt):
                    dropped_nan_target += 1
                if pd.isna(rev) or pd.isna(ret):
                    dropped_return_nan += 1
                continue

            # {240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40}
            period_long = np.arange(end_t - lookback + 1, end_t - 40 + 2, step=20, dtype=int)

            # {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
            period_short = np.arange(end_t - 20 + 1, end_t + 1, step=1, dtype=int)

            period = np.concatenate([period_long, period_short])

            # Build 3-feature window; skip if any feature is invalid over the window
            window = np.empty((len(period), 3), dtype=float)
            valid = True
            for i in range(3):  # ir, cpr, opr (robust z-score)
                vec = f_t1[n, i, period]
                if np.isnan(vec).any():
                    dropped_feature_nan += 1
                    valid = False; break
                q1, q2, q3 = np.quantile(vec, [0.25, 0.5, 0.75])
                iqr = (q3 - q1)
                if iqr == 0:
                    dropped_flat_iqr += 1
                    valid = False
                    break
                window[:, i] = (vec - q2) / iqr
            if not valid:
                continue

            # Add NLP features if available
            if use_nlp and nlp_features is not None:
                from nlp_features import get_nlp_feature_vector
                
                # Determine if using aggregated (date -> row) or individual ((stock, date) -> row) method
                # Check for method marker first
                use_simple = nlp_features.get('_method') == 'individual_simple'
                if '_method' in nlp_features:
                    # Remove method marker for key checking
                    nlp_features_clean = {k: v for k, v in nlp_features.items() if k != '_method'}
                else:
                    nlp_features_clean = nlp_features
                
                sample_key = next(iter(nlp_features_clean.keys())) if nlp_features_clean else None
                is_individual_method = (sample_key is not None and isinstance(sample_key, tuple))
                
                # Determine feature dimension based on method
                nlp_feature_dim = 4 if use_simple else 10
                nlp_window = np.empty((len(period), nlp_feature_dim), dtype=float)
                
                # Get stock ticker for this sample (for individual method)
                stock_ticker = None
                if is_individual_method and successfully_downloaded_stocks is not None:
                    if n < len(successfully_downloaded_stocks):
                        stock_ticker = successfully_downloaded_stocks[n]
                
                for idx, t in enumerate(period):
                    # Get date for this time step
                    if t < len(date_index):
                        date_at_t = date_index[t]
                        if isinstance(date_at_t, pd.Timestamp):
                            date_at_t = date_at_t.date()
                        
                        # Get NLP features for this date
                        if is_individual_method:
                            # Individual method: (stock, date) -> row
                            if stock_ticker and (stock_ticker, date_at_t) in nlp_features_clean:
                                nlp_row = nlp_features_clean[(stock_ticker, date_at_t)]
                                nlp_vec = get_nlp_feature_vector(nlp_row, use_simple=use_simple)
                                nlp_window[idx, :] = nlp_vec
                            else:
                                # No NLP data for this stock-date - use zeros
                                nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                        else:
                            # Aggregated method: date -> row
                            if date_at_t in nlp_features_clean:
                                nlp_row = nlp_features_clean[date_at_t]
                                nlp_vec = get_nlp_feature_vector(nlp_row, use_simple=use_simple)
                                nlp_window[idx, :] = nlp_vec
                            else:
                                # No NLP data for this date - use zeros
                                nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                    else:
                        # Date index out of range - use zeros
                        nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                
                # Concatenate price features with NLP features
                window = np.concatenate([window, nlp_window], axis=1)  # (len(period), 3 + nlp_feature_dim)

            X_list.append(window)
            y_list.append([tgt])
            d_list.append(date_index[end_t])
            rev_list.append(rev)
            return_list.append(ret)
            kept += 1
    
    # Close progress bar
    pbar.close()

    # --- summary (before split) ---
    removed = total_candidates - kept
    pct = (kept / total_candidates * 100.0) if total_candidates else 0.0
    print(f"[windows] candidates: {total_candidates:,} | kept: {kept:,} ({pct:.1f}%) | removed: {removed:,}")
    if removed:
        print(f"[windows] removed breakdown — NaN target: {dropped_nan_target:,}, "
              f"feature NaN: {dropped_feature_nan:,}, zero-IQR: {dropped_flat_iqr:,},"
              f" return NaN: {dropped_return_nan:,}")

    X = np.array(X_list, dtype=float)   # (num_stocks, (lookback-40)/20 + 20, 3)
    y = np.array(y_list, dtype=float)   # (num_stocks, 1)
    dates = d_list                      # list of length num_stocks
    revenues = np.array(rev_list, dtype=float)
    returns = np.array(return_list, dtype=float)

    return X, y, dates, revenues, returns

    """
    trying to get prediction at time t
    looking back at previous 241 days to predict the 242nd day

    start with opening prices (op) and closing prices(cp)
    prices for both ordered from 0-t

    intraday returns (ir)= cp/op -1
    returns wrt to last cp = 

    """




########################################################
# model
########################################################

def get_model_summary(model):
    torchsummary.summary(model,(31,3),batch_size=8)
