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

def get_data(stocks, args, data_dir="data", lookback=240, force=False, prediction_type="classification"):
    """
    Return 9-tuple: (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest)
    Loads from .npz if present (and not force); otherwise builds from yfinance,
    saves .npz, and returns the tuple.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    prediction_suffix = f"_{prediction_type}"

    # No cache loading here since we need to know which stocks actually downloaded successfully
    # This would require matching the input stocks/args to stored mappings which is complex
    # So we always download first, then save with the correct ID
    # Cache reuse happens naturally when the same stocks/args are used

    # Build from raw prices using error-handled download
    if len(args) not in [1, 2]:
        print("Invalid Data Input Arguments")
        return 1
    
    # For large stock lists, use fewer retries to speed up the download
    # Most errors are permanent (delisted, no data for date range), so 1 retry is sufficient
    max_retries = 1 if len(stocks) > 50 else 3
    
    open_close, failed_stocks = handle_yfinance_errors(stocks, args, max_retries=max_retries)
    
    if open_close is None:
        print("ERROR: Failed to download any stock data. Cannot proceed.")
        return 1

    # Now build op/cp/date_index from the cleaned frame
    op = open_close["Open"].T   # (S, T)
    cp = open_close["Close"].T  # (S, T)
    date_index = open_close.index

    # Update stocks list to only include successfully downloaded stocks
    successfully_downloaded_stocks = [stock for stock in stocks if stock in open_close["Open"].columns]

    if lookback >= op.shape[1]:
        raise ValueError("study period too short for chosen lookback")

    # Features/targets
    if prediction_type == "classification":
        xdata, ydata, dates, revenues = get_feature_input_classification(op, cp, lookback, op.shape[1], len(successfully_downloaded_stocks), date_index)
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
    train_path = os.path.join(data_dir, base + prediction_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + "_test.npz")
    metrics_path = os.path.join(data_dir, base + prediction_suffix + "_metrics.npz")
    
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
        "Rev": _to_np(Rev_f)
    }, desc="Saving metrics dataset (.npz)")
    
    print(f"{prediction_type.title()} datasets saved separately:")
    # Use absolute paths for clarity and to avoid confusion with relative paths
    print(f"  Training: {os.path.abspath(train_path)}")
    print(f"  Validation: {os.path.abspath(val_path)}")
    print(f"  Test: {os.path.abspath(test_path)}")
    print(f"  Metrics: {os.path.abspath(metrics_path)}")
    
    # Save ID mapping so we can reference this data later (using successfully downloaded stocks)
    _save_id_mapping(data_id, successfully_downloaded_stocks, args, data_dir)
    
    return (Xtr_f, Xva_f, Xte_f, Ytr_f, Yva_f, Yte_f, Dtrain_f, Dvalidation_f, Dtest_f, Rev_f)

def load_data_from_cache(stocks, args, data_dir="data", prediction_type="classification"):
    """
    Load data from cache files if they exist.
    
    This function handles the case where the input stocks list may include stocks
    that failed to download. It first tries an exact match, then searches the
    mapping file to find a cache that was created from a subset of these stocks.
    
    Returns data tuple if cache exists, None otherwise.
    """
    prediction_suffix = f"_{prediction_type}"
    
    # First, try exact match with the input stocks list
    data_id = _get_data_id(stocks, args)
    base = f"data_{data_id}"
    
    train_path = os.path.join(data_dir, base + prediction_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + "_test.npz")
    metrics_path = os.path.join(data_dir, base + prediction_suffix + "_metrics.npz")
    
    # Check if exact match cache exists
    if all(os.path.exists(p) for p in [train_path, val_path, test_path, metrics_path]):
        # Exact match found - use it
        pass  # Will load below
    else:
        # No exact match - try to find a cache from mapping file
        # This handles the case where some stocks failed to download
        mapping = _load_id_mapping(data_dir)
        
        stocks_set = set(stocks)
        args_str = ",".join(str(a) for a in args)
        
        # Find a mapping entry where:
        # 1. The stored args match
        # 2. The stored stocks are a subset of the input stocks (some may have failed)
        found_data_id = None
        for cached_id, cached_info in mapping.items():
            cached_args_str = ",".join(str(a) for a in cached_info.get('args', []))
            if cached_args_str == args_str:
                cached_stocks = set(cached_info.get('stocks', []))
                # Check if cached stocks are a subset of input stocks
                # (meaning we can use this cache even if some input stocks failed)
                if cached_stocks.issubset(stocks_set):
                    cached_base = f"data_{cached_id}"
                    cached_train = os.path.join(data_dir, cached_base + prediction_suffix + "_train.npz")
                    cached_val = os.path.join(data_dir, cached_base + prediction_suffix + "_val.npz")
                    cached_test = os.path.join(data_dir, cached_base + prediction_suffix + "_test.npz")
                    cached_metrics = os.path.join(data_dir, cached_base + prediction_suffix + "_metrics.npz")
                    
                    if all(os.path.exists(p) for p in [cached_train, cached_val, cached_test, cached_metrics]):
                        found_data_id = cached_id
                        base = cached_base
                        train_path = cached_train
                        val_path = cached_val
                        test_path = cached_test
                        metrics_path = cached_metrics
                        break
        
        if found_data_id is None:
            # No matching cache found
            return None
    
    # At this point, we have valid cache paths (either exact match or found via mapping)
    
    # Load from cache
    train_data = _load_npz_progress(train_path, ["X", "Y", "D"], desc="Loading training dataset (.npz)")
    val_data = _load_npz_progress(val_path, ["X", "Y", "D"], desc="Loading validation dataset (.npz)")
    test_data = _load_npz_progress(test_path, ["X", "Y", "D"], desc="Loading test dataset (.npz)")
    metrics_data = _load_npz_progress(metrics_path, ["Rev"], desc="Loading metrics dataset (.npz)")
    
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
    
    return (Xtr, Xva, Xte, Ytr, Yva, Yte, Dtr, Dva, Dte, Rev)

def save_data_locally(stocks, args, data_dir="data", force=False, prediction_type="classification"):
    # Force a rebuild/save and return the 9-tuple
    return get_data(stocks, args, data_dir=data_dir, lookback=240, force=True, prediction_type=prediction_type)


########################################################
# get feature input
########################################################

# op[x] is the op vector for stock x
# op and cp has indices from time 0 to T_study-1
def get_feature_input_classification(op, cp, lookback, study_period, num_stocks, date_index):

    T = study_period
    # Precompute elementary series (may contain NaNs)
    f_t1 = np.full((num_stocks,3, T), np.nan, dtype=float)
    rev_labels = np.full((num_stocks, T), np.nan, dtype=float)  # Initialize with NaN for proper alignment
    rev_t = np.full((num_stocks, T), np.nan, dtype=float)
    for t in range(2, T):
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
            
            # Calculate revenue for valid stocks only
            if pd.notna(cp.iloc[n, t]) and pd.notna(op.iloc[n, t]):
                rev_t[n, t] = cp.iloc[n, t] - op.iloc[n, t]
                valid_stocks.append(n)
        
        # Calculate median revenue at time t
        rev_t_valid = rev_t[valid_stocks, t]
        if len(rev_t_valid) > 0:
            median_rev = np.median(rev_t_valid)
            rev_labels[valid_stocks, t] = np.where(rev_t_valid > median_rev, 1.0, -1.0)

    X_list, y_list, d_list, rev_list = [], [], [], []
    # --- counters ---
    total_candidates = 0
    kept = 0
    dropped_nan_target = 0
    dropped_feature_nan = 0
    dropped_flat_iqr = 0
    dropped_rev_nan = 0

    for end_t in range(lookback + 2, T):
        for n in range(num_stocks):
            total_candidates += 1
            tgt = rev_labels[n, end_t]
            rev = rev_t[n, end_t]
            if pd.isna(tgt) or pd.isna(rev):
                if pd.isna(tgt):
                    dropped_nan_target += 1
                if pd.isna(rev):
                    dropped_rev_nan += 1
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

            X_list.append(window)
            y_list.append([tgt])
            d_list.append(date_index[end_t])
            rev_list.append(rev)
            kept += 1

    # --- summary (before split) ---
    removed = total_candidates - kept
    pct = (kept / total_candidates * 100.0) if total_candidates else 0.0
    print(f"[windows] candidates: {total_candidates:,} | kept: {kept:,} ({pct:.1f}%) | removed: {removed:,}")
    if removed:
        print(f"[windows] removed breakdown — NaN target: {dropped_nan_target:,}, "
              f"feature NaN: {dropped_feature_nan:,}, zero-IQR: {dropped_flat_iqr:,}")

    X = np.array(X_list, dtype=float)   # (num_stocks, (lookback-40)/20 + 20, 3)
    y = np.array(y_list, dtype=float)   # (num_stocks, 1)
    dates = d_list                      # list of length num_stocks
    revenues = np.array(rev_list, dtype=float)

    return X, y, dates, revenues

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
