# https://doi.org/10.1016/j.frl.2021.102280
# "Forecasting directional movements of stock prices for intraday trading using LSTM and random forests" Ghosh, Neufeld, Sahoo 2022
# modifications since our model predicts price not 

import pandas as pd
import torch
import yfinance as yf
import numpy as np
import statistics
import torchsummary
import os
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

def get_data(stocks, args, data_dir="data", lookback=240, force=False, prediction_type="classification"):
    """
    Return 9-tuple: (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest)
    Loads from .npz if present (and not force); otherwise builds from yfinance,
    saves .npz, and returns the tuple.
    """
    os.makedirs(data_dir, exist_ok=True)
    base = "_".join(stocks) + "_" + "_".join(args)
    
    # Create prediction-type specific file names
    prediction_suffix = f"_{prediction_type}"
    npz_path = os.path.join(data_dir, base + prediction_suffix + ".npz")

    # Fast path: try to load prediction-type specific datasets first, then fall back to combined format
    if not force:
        # Try prediction-type specific separate format first
        separate_data = load_separate_datasets(stocks, args, data_dir, prediction_type)
        if separate_data != 1:
            return separate_data
        
        # Fall back to prediction-type specific combined format if separate files don't exist
        if os.path.exists(npz_path):
            print(f"Prediction-type specific data file {npz_path} exists. Loading .npz ...")
            return load_data_from_local(npz_path)
        
        # Fall back to old format without prediction type for backward compatibility
        old_npz_path = os.path.join(data_dir, base + ".npz")
        if os.path.exists(old_npz_path):
            print(f"Legacy data file {old_npz_path} exists. Loading .npz ...")
            return load_data_from_local(old_npz_path)

    # Build from raw prices
    dat = yf.Tickers(" ".join(stocks))
    if len(args) == 1:
        open_close = dat.history(period=args[0])[["Open", "Close"]]
    elif len(args) == 2:
        open_close = dat.history(period=None, start=args[0], end=args[1], interval="1d")[["Open", "Close"]]
    else:
        print("Invalid Data Input Arguments")
        return 1

    # Now build op/cp/date_index from the cleaned frame
    op = open_close["Open"].T   # (S, T)
    cp = open_close["Close"].T  # (S, T)
    date_index = open_close.index

    if lookback >= op.shape[1]:
        raise ValueError("study period too short for chosen lookback")

    # Features/targets
    if prediction_type == "classification":
        xdata, ydata, dates = get_feature_input_classification(op, cp, lookback, op.shape[1], len(stocks), date_index)
    elif prediction_type == "price" :
        xdata, ydata, dates = get_feature_input_price(op, cp, lookback, op.shape[1], len(stocks), date_index)
    else:
        raise ValueError("Invalid prediction type")
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

    # --- split summary ---
    n_tr, n_va, n_te = len(Dtrain_f), len(Dvalidation_f), len(Dtest_f)
    n_tot = n_tr + n_va + n_te
    print(f"[split] train/val/test sizes = {n_tr:,} / {n_va:,} / {n_te:,}  | total kept = {n_tot:,}")

    # Save datasets separately
    def _to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    
    # Create prediction-type specific separate file paths for each dataset
    train_path = os.path.join(data_dir, base + prediction_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + "_test.npz")
    
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
    
    print(f"{prediction_type.title()} datasets saved separately:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")

    return (Xtr_f, Xva_f, Xte_f, Ytr_f, Yva_f, Yte_f, Dtrain_f, Dvalidation_f, Dtest_f)

def save_data_locally(stocks, args, data_dir="data", force=False, prediction_type="classification"):
    # Force a rebuild/save and return the 9-tuple
    return get_data(stocks, args, data_dir=data_dir, lookback=240, force=True, prediction_type=prediction_type)


def load_data_from_local(filename):
    """
    Load from .npz (fast path). Handles both old combined format and new separate format.
    If a legacy .pkl is passed accidentally, refuse and return 1 so the caller can rebuild.
    """
    if filename.endswith(".npz"):
        # Check if this is the old combined format
        with np.load(filename, allow_pickle=False) as z:
            if "X_train" in z:
                # Old combined format
                names = ["X_train","X_val","X_test","Y_train","Y_val","Y_test","D_train","D_val","D_test"]
                z = _load_npz_progress(filename, names, desc="Loading dataset (.npz)")
                Xtr = torch.from_numpy(z["X_train"]).to(torch.float32)
                Xva = torch.from_numpy(z["X_val"]).to(torch.float32)
                Xte = torch.from_numpy(z["X_test"]).to(torch.float32)
                Ytr = torch.from_numpy(z["Y_train"]).to(torch.float32)
                Yva = torch.from_numpy(z["Y_val"]).to(torch.float32)
                Yte = torch.from_numpy(z["Y_test"]).to(torch.float32)
                Dtr = [pd.Timestamp(d).to_pydatetime() for d in z["D_train"]]
                Dva = [pd.Timestamp(d).to_pydatetime() for d in z["D_val"]]
                Dte = [pd.Timestamp(d).to_pydatetime() for d in z["D_test"]]
                return (Xtr, Xva, Xte, Ytr, Yva, Yte, Dtr, Dva, Dte)
            else:
                # New separate format - this shouldn't happen with the current filename
                print("Error: Trying to load separate format with combined filename")
                return 1

    if filename.endswith(".pkl"):
        print("Legacy pickle file detected. Ignoring it—rebuilding .npz instead.")
        return 1

    print(f"Unknown data file type: {filename}")
    return 1

def load_separate_datasets(stocks, args, data_dir="data", prediction_type="classification"):
    """
    Load train, validation, and test datasets from separate .npz files.
    Returns 9-tuple: (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest)
    """
    base = "_".join(stocks) + "_" + "_".join(args)
    prediction_suffix = f"_{prediction_type}"
    train_path = os.path.join(data_dir, base + prediction_suffix + "_train.npz")
    val_path = os.path.join(data_dir, base + prediction_suffix + "_val.npz")
    test_path = os.path.join(data_dir, base + prediction_suffix + "_test.npz")
    
    # Check if all separate files exist
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        print(f"Separate {prediction_type} dataset files not found. Need to rebuild datasets.")
        return 1
    
    # Load training dataset
    train_data = _load_npz_progress(train_path, ["X", "Y", "D"], desc="Loading training dataset")
    Xtr = torch.from_numpy(train_data["X"]).to(torch.float32)
    Ytr = torch.from_numpy(train_data["Y"]).to(torch.float32)
    Dtr = [pd.Timestamp(d).to_pydatetime() for d in train_data["D"]]
    
    # Load validation dataset
    val_data = _load_npz_progress(val_path, ["X", "Y", "D"], desc="Loading validation dataset")
    Xva = torch.from_numpy(val_data["X"]).to(torch.float32)
    Yva = torch.from_numpy(val_data["Y"]).to(torch.float32)
    Dva = [pd.Timestamp(d).to_pydatetime() for d in val_data["D"]]
    
    # Load test dataset
    test_data = _load_npz_progress(test_path, ["X", "Y", "D"], desc="Loading test dataset")
    Xte = torch.from_numpy(test_data["X"]).to(torch.float32)
    Yte = torch.from_numpy(test_data["Y"]).to(torch.float32)
    Dte = [pd.Timestamp(d).to_pydatetime() for d in test_data["D"]]
    
    print(f"Loaded separate datasets:")
    print(f"  Training: {len(Xtr)} samples from {train_path}")
    print(f"  Validation: {len(Xva)} samples from {val_path}")
    print(f"  Test: {len(Xte)} samples from {test_path}")
    
    return (Xtr, Xva, Xte, Ytr, Yva, Yte, Dtr, Dva, Dte)

########################################################
# get feature input
########################################################

def get_feature_input_price(op, cp, lookback, study_period, num_stocks, date_index):

    T = study_period
    # Precompute elementary series (may contain NaNs)
    f_t1 = np.full((num_stocks, 4, T), np.nan, dtype=float)
    
    for t in range(2, T):
        
        for n in range(num_stocks):
            o_t1, c_t1, c_t2, o_t = op.iloc[n, t-1], cp.iloc[n, t-1], cp.iloc[n, t-2], op.iloc[n, t]
            if pd.notna(c_t1) and pd.notna(o_t1):
                f_t1[n, 0, t] = c_t1 / o_t1 - 1.0     # ir
            if pd.notna(c_t1) and pd.notna(c_t2):
                f_t1[n, 1, t] = c_t1 / c_t2 - 1.0     # cpr
            if pd.notna(o_t) and pd.notna(o_t1):
                f_t1[n, 2, t] = o_t / o_t1 - 1.0      # opr
            if pd.notna(o_t):
                f_t1[n, 3, t] = o_t                   # op
        

    X_list, y_list, d_list = [], [], []
    # --- counters ---
    total_candidates = 0
    kept = 0
    dropped_nan_target = 0
    dropped_feature_nan = 0
    dropped_flat_iqr = 0
    dropped_op_nan = 0
    

    for end_t in range(lookback + 2, T):
        for n in range(num_stocks):
            total_candidates += 1
            tgt = cp.iloc[n, end_t]
            if pd.isna(tgt):
                dropped_nan_target += 1
                continue
            period = np.arange(end_t - lookback + 1, end_t + 1, dtype=int)

            # Build 4-feature window; skip if any feature is invalid over the window
            window = np.empty((lookback, 4), dtype=float)
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
            vec_op = f_t1[n, 3, period]
            if np.isnan(vec_op).any():
                dropped_op_nan += 1
                continue
            window[:, 3] = vec_op

            X_list.append(window)
            y_list.append([tgt])
            d_list.append(date_index[end_t])
            kept += 1

    # --- summary (before split) ---
    removed = total_candidates - kept
    pct = (kept / total_candidates * 100.0) if total_candidates else 0.0
    print(f"[windows] candidates: {total_candidates:,} | kept: {kept:,} ({pct:.1f}%) | removed: {removed:,}")
    if removed:
        print(f"[windows] removed breakdown — NaN target: {dropped_nan_target:,}, "
              f"feature NaN: {dropped_feature_nan:,}, zero-IQR: {dropped_flat_iqr:,}, NaN open: {dropped_op_nan:,}")

    X = np.array(X_list, dtype=float)   # (N, lookback, 4)
    y = np.array(y_list, dtype=float)   # (N, 1)
    dates = d_list                      # list of length N
    return X, y, dates

    """
    trying to get prediction at time t
    looking back at previous 241 days to predict the 242nd day

    start with opening prices (op) and closing prices(cp)
    prices for both ordered from 0-t

    intraday returns (ir)= cp/op -1
    returns wrt to last cp = 

    """

# op[x] is the op vector for stock x
# op and cp has indices from time 0 to T_study-1
def get_feature_input_classification(op, cp, lookback, study_period, num_stocks, date_index):

    T = study_period
    # Precompute elementary series (may contain NaNs)
    f_t1 = np.full((num_stocks, 4, T), np.nan, dtype=float)
    rev_labels = np.full((num_stocks, T), np.nan, dtype=float)  # Initialize with NaN for proper alignment
    
    for t in range(2, T):
        rev_t = []
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
            if pd.notna(o_t):
                f_t1[n, 3, t] = o_t                   # op
            
            # Calculate revenue for valid stocks only
            if pd.notna(cp.iloc[n, t]) and pd.notna(op.iloc[n, t]):
                revenue = cp.iloc[n, t] - op.iloc[n, t]
                rev_t.append(revenue)
                valid_stocks.append(n)
        
        # Calculate median revenue at time t
        if len(rev_t) > 0:
            median_rev = np.median(rev_t)
            
            # Second pass: assign binary labels based on median
            for i, n in enumerate(valid_stocks):
                if rev_t[i] > median_rev:
                    rev_labels[n, t] = 1.0  # Revenue > median
                else:
                    rev_labels[n, t] = 0.0  # Revenue <= median

    X_list, y_list, d_list = [], [], []
    # --- counters ---
    total_candidates = 0
    kept = 0
    dropped_nan_target = 0
    dropped_feature_nan = 0
    dropped_flat_iqr = 0
    dropped_op_nan = 0
    

    for end_t in range(lookback + 2, T):
        for n in range(num_stocks):
            total_candidates += 1
            tgt = rev_labels[n, end_t]
            if pd.isna(tgt):
                dropped_nan_target += 1
                continue
            period = np.arange(end_t - lookback + 1, end_t + 1, dtype=int)

            # Build 4-feature window; skip if any feature is invalid over the window
            window = np.empty((lookback, 4), dtype=float)
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
            vec_op = f_t1[n, 3, period]
            if np.isnan(vec_op).any():
                dropped_op_nan += 1
                continue
            window[:, 3] = vec_op

            X_list.append(window)
            y_list.append([tgt])
            d_list.append(date_index[end_t])
            kept += 1

    # --- summary (before split) ---
    removed = total_candidates - kept
    pct = (kept / total_candidates * 100.0) if total_candidates else 0.0
    print(f"[windows] candidates: {total_candidates:,} | kept: {kept:,} ({pct:.1f}%) | removed: {removed:,}")
    if removed:
        print(f"[windows] removed breakdown — NaN target: {dropped_nan_target:,}, "
              f"feature NaN: {dropped_feature_nan:,}, zero-IQR: {dropped_flat_iqr:,}, NaN open: {dropped_op_nan:,}")

    X = np.array(X_list, dtype=float)   # (N, lookback, 4)
    y = np.array(y_list, dtype=float)   # (N, 1)
    dates = d_list                      # list of length N
    return X, y, dates

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
    torchsummary.summary(model,(240,3),batch_size=8)
