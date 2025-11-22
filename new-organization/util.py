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
from typing import Optional
from tqdm import tqdm
from zipfile import ZipFile, ZIP_STORED
from io import BytesIO
from data_sources.base import DataSource

########################################################
# data directory configuration
########################################################

# Default data directory - all data stored here (relative to new-organization/)
DEFAULT_DATA_DIR = "data"

def get_default_data_dir():
    """
    Get the absolute path to the default data directory.
    Returns: Absolute path to new-organization/data/
    """
    # Get directory where util.py is located (new-organization/)
    util_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(util_dir, DEFAULT_DATA_DIR)

# Compute data directory once at module load time
# This is called only once when util.py is imported
DATA_DIR = get_default_data_dir()

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
# validation helper functions
########################################################

def validate_time_args(args):
    """
    Validate time arguments length.
    
    Args:
        args: Time arguments (either period string or [start, end] tuple)
    
    Returns:
        bool: True if valid, False otherwise
    """
    if len(args) not in [1, 2]:
        print("Invalid Data Input Arguments")
        return False
    return True

def validate_stocks_not_empty(stocks, context=""):
    """
    Validate that stocks list is not empty.
    
    Args:
        stocks: List of stock symbols
        context: Optional context string for error message
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        RuntimeError: If stocks list is empty
    """
    if len(stocks) == 0:
        error_msg = "No valid stocks found"
        if context:
            error_msg += f" {context}"
        error_msg += "."
        raise RuntimeError(error_msg)
    return True

def validate_dataframe_not_none(df, name="DataFrame"):
    """
    Validate that DataFrame is not None.
    
    Args:
        df: DataFrame to validate
        name: Name of DataFrame for error message
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        RuntimeError: If DataFrame is None
    """
    if df is None:
        raise RuntimeError(f"ERROR: {name} is None. Cannot proceed.")
    return True

def validate_mask_type(mask_type):
    """
    Validate mask type for period extraction.
    
    Args:
        mask_type: Mask type string ("LS" or "full")
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        ValueError: If mask_type is invalid
    """
    if mask_type not in ["LS", "full"]:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return True

########################################################
# data management
########################################################

def _serialize_data_source(data_source: DataSource) -> str:
    """
    Serialize a DataSource instance to a string representation for hashing.
    
    Args:
        data_source: DataSource instance
    
    Returns:
        String representation of the data source
    """
    source_type = type(data_source).__name__
    
    if source_type == "YFinanceDataSource":
        return "yfinance"
    elif source_type == "StaticFileDataSource":
        # Include file path in the serialization
        file_path = getattr(data_source, 'file_path', '')
        file_format = getattr(data_source, 'file_format', '')
        return f"static_file:{file_path}:{file_format}"
    elif source_type == "APIDataSource":
        # For API sources, include API name if available
        api_name = getattr(data_source, 'api_name', 'api')
        return f"api:{api_name}"
    else:
        # Fallback: use class name
        return f"source:{source_type}"

def _get_data_id(stocks, args, use_nlp=False, nlp_method="aggregated", prediction_type="classification", period_type="LS", seq_len=240, data_source_str=None):
    """
    Generate a short hash-based ID for a unique combination of all dataset parameters.
    
    Args:
        stocks: List of stock symbols
        args: Time arguments
        use_nlp: Whether NLP features are used
        nlp_method: NLP method ("aggregated" or "individual")
        prediction_type: Prediction type ("classification" or "regression")
        period_type: Period type ("LS" or "full")
        seq_len: Sequence length (lookback window size)
        data_source_str: String representation of data source (required)
        
    Returns:
        A short 10-character hash string
    """
    if data_source_str is None:
        raise ValueError("data_source_str is required for _get_data_id()")
    # Create a stable string representation
    stocks_str = ",".join(sorted(stocks))  # Sort for consistency
    args_str = ",".join(str(a) for a in args)
    nlp_str = f"nlp_{nlp_method}" if use_nlp else "no_nlp"
    combined = f"{stocks_str}|{args_str}|{nlp_str}|{prediction_type}|{period_type}|{seq_len}|{data_source_str}"
    
    # Generate a short hash
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:10]

def _load_id_mapping():
    """
    Load the ID to (stocks, args, use_nlp, nlp_method, prediction_type, period_type, seq_len, data_source) mapping from disk.
    
    Returns:
        Dictionary mapping data_id -> {'stocks': [...], 'args': [...], 'use_nlp': bool, 'nlp_method': str, 
                                      'prediction_type': str, 'period_type': str, 'seq_len': int, 'data_source': str, 'full_name': '...'}
    """
    mapping_path = os.path.join(DATA_DIR, "_data_mapping.json")
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load data mapping: {e}")
            return {}
    return {}

def _save_id_mapping(data_id, stocks, args, use_nlp=False, nlp_method="aggregated", prediction_type="classification", period_type="LS", seq_len=240, data_source_str=None):
    """
    Save the ID to (stocks, args, use_nlp, nlp_method, prediction_type, period_type, seq_len, data_source) mapping to disk.
    
    Args:
        data_id: Short hash ID
        stocks: List of stock symbols
        args: Time arguments
        use_nlp: Whether NLP features are used
        nlp_method: NLP method ("aggregated" or "individual")
        prediction_type: Prediction type ("classification" or "regression")
        period_type: Period type ("LS" or "full")
        seq_len: Sequence length (lookback window size)
        data_source_str: String representation of data source (required)
    """
    if data_source_str is None:
        raise ValueError("data_source_str is required for _save_id_mapping()")
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists
    mapping_path = os.path.join(DATA_DIR, "_data_mapping.json")
    
    try:
        mapping = _load_id_mapping()
        mapping[data_id] = {
            'stocks': stocks,
            'args': args,
            'use_nlp': use_nlp,
            'nlp_method': nlp_method,
            'prediction_type': prediction_type,
            'period_type': period_type,
            'seq_len': seq_len,
            'data_source': data_source_str,
            'full_name': "_".join(stocks[:5]) + (f"_{len(stocks)-5}_more" if len(stocks) > 5 else "")
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save data mapping: {e}")

########################################################
# model mapping functions
########################################################

# Default models directory - all models stored here (relative to new-organization/)
DEFAULT_MODELS_DIR = "trained_models"
DEFAULT_MODELS_SUBDIR = "models"

def get_default_models_dir():
    """
    Get the absolute path to the default models directory.
    Returns: Absolute path to new-organization/trained_models/models/
    """
    # Get directory where util.py is located (new-organization/)
    util_dir = os.path.dirname(os.path.abspath(__file__))
    models_base_dir = os.path.join(util_dir, DEFAULT_MODELS_DIR)
    models_subdir = os.path.join(models_base_dir, DEFAULT_MODELS_SUBDIR)
    return models_subdir

# Compute models directory once at module load time
MODELS_DIR = get_default_models_dir()

def _get_model_id(model_config):
    """
    Generate a short hash-based ID for a unique combination of model config class type and parameters.
    
    Args:
        model_config: Model configuration object (e.g., LSTMConfig, TimesNetConfig)
        
    Returns:
        A short 10-character hash string
    """
    import inspect
    
    # Get config class name
    config_class_name = model_config.__class__.__name__
    
    # Get all parameters from the config
    # Sort parameters for consistency
    if hasattr(model_config, 'parameters'):
        params = model_config.parameters
        if isinstance(params, dict):
            # Sort dictionary items for consistent hashing
            params_str = json.dumps(params, sort_keys=True)
        else:
            params_str = str(params)
    else:
        # Fallback: get all attributes that aren't private
        attrs = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
        params_str = json.dumps(attrs, sort_keys=True, default=str)
    
    # Combine class name and parameters
    combined = f"{config_class_name}|{params_str}"
    
    # Generate a short hash
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:10]

def _load_model_mapping():
    """
    Load the ID to model config mapping from disk.
    
    Returns:
        Dictionary mapping model_id -> {'config_class': str, 'parameters': dict, 'full_name': str}
    """
    mapping_path = os.path.join(MODELS_DIR, "_model_mapping.json")
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model mapping: {e}")
            return {}
    return {}

def _save_model_mapping(model_id, model_config):
    """
    Save the ID to model config mapping to disk.
    
    Args:
        model_id: Short hash ID
        model_config: Model configuration object
    """
    os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure directory exists
    mapping_path = os.path.join(MODELS_DIR, "_model_mapping.json")
    
    try:
        mapping = _load_model_mapping()
        
        # Extract config class name
        config_class_name = model_config.__class__.__name__
        
        # Extract parameters
        if hasattr(model_config, 'parameters'):
            parameters = model_config.parameters
            if not isinstance(parameters, dict):
                parameters = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
        else:
            parameters = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
        
        mapping[model_id] = {
            'config_class': config_class_name,
            'parameters': parameters,
            'full_name': config_class_name
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save model mapping: {e}")

def find_model_by_config(model_config):
    """
    Find an existing model with matching config by checking the mapping file.
    
    First checks if config class types match, then checks if all parameters are equivalent.
    
    Args:
        model_config: Model configuration object to match
        
    Returns:
        Tuple (model_id, model_path) if match found, (None, None) otherwise
    """
    mapping = _load_model_mapping()
    
    # Get config class name
    config_class_name = model_config.__class__.__name__
    
    # Extract parameters from input config
    if hasattr(model_config, 'parameters'):
        input_params = model_config.parameters
        if not isinstance(input_params, dict):
            input_params = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
    else:
        input_params = {k: v for k, v in model_config.__dict__.items() if not k.startswith('_')}
    
    # Normalize parameters for comparison (convert to JSON-serializable format)
    def normalize_value(v):
        """Convert value to JSON-serializable format for comparison."""
        if isinstance(v, (list, tuple)):
            return tuple(normalize_value(item) for item in v)
        elif isinstance(v, dict):
            return {k: normalize_value(val) for k, val in sorted(v.items())}
        elif isinstance(v, (int, float, str, bool, type(None))):
            return v
        else:
            return str(v)
    
    input_params_normalized = {k: normalize_value(v) for k, v in sorted(input_params.items())}
    
    # Search for matching config
    for model_id, cached_info in mapping.items():
        cached_config_class = cached_info.get('config_class', '')
        cached_params = cached_info.get('parameters', {})
        
        # Step 1: Check if config class types match
        if cached_config_class != config_class_name:
            continue
        
        # Step 2: Check if all parameters are equivalent
        cached_params_normalized = {k: normalize_value(v) for k, v in sorted(cached_params.items())}
        
        if input_params_normalized == cached_params_normalized:
            # Found matching config - verify model file exists
            model_path = os.path.join(MODELS_DIR, f"{model_id}.pth")
            if os.path.exists(model_path):
                return (model_id, model_path)
    
    # No matching model found
    return (None, None)

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

def _load_problematic_stocks(args):
    """
    Load the list of problematic stocks for given time arguments.
    
    Args:
        args: Time arguments
    
    Returns:
        Set of problematic stock symbols, or empty set if not found
    """
    args_key = _get_args_key(args)
    problematic_path = os.path.join(DATA_DIR, f"_problematic_stocks_{args_key}.json")
    
    if os.path.exists(problematic_path):
        try:
            with open(problematic_path, 'r') as f:
                data = json.load(f)
                return set(data.get('problematic_stocks', []))
        except Exception as e:
            print(f"Warning: Could not load problematic stocks: {e}")
            return set()
    return set()

def _save_problematic_stocks(problematic_stocks, args):
    """
    Save the list of problematic stocks for given time arguments.
    
    Args:
        problematic_stocks: List or set of problematic stock symbols
        args: Time arguments
    """
    args_key = _get_args_key(args)
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists
    problematic_path = os.path.join(DATA_DIR, f"_problematic_stocks_{args_key}.json")
    
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

def _get_sp500_returns_for_dates(test_dates, date_index, study_period, data_source: DataSource):
    """
    Fetch S&P 500 returns aligned with test dates using the provided data source.
    
    Args:
        test_dates: List of test dates (datetime objects)
        date_index: Original date index from the data (unused, kept for compatibility)
        study_period: Study period length (unused, kept for compatibility)
        data_source: DataSource instance to use for fetching S&P 500 data
        
    Returns:
        numpy array of S&P 500 returns aligned with test_dates
    """
    if not test_dates or len(test_dates) == 0:
        print("[util] No test dates available for S&P 500")
        return np.array([])
    
    return data_source.fetch_sp500_data(test_dates)

def identify_problematic_stocks(stocks, args, data_source: DataSource, max_retries=1):
    """
    Identify which stocks from the input list are problematic (cannot be downloaded).
    This is a lightweight check that only validates downloadability without processing data.
    
    Args:
        stocks: List of stock tickers
        args: Date range arguments (either period string or [start, end] tuple)
        data_source: DataSource instance to use for validation
        max_retries: Maximum number of retry attempts per stock (default: 1 for speed)
    
    Returns:
        tuple: (valid_stocks, problematic_stocks)
               - valid_stocks: List of stocks that can be downloaded
               - problematic_stocks: List of stocks that failed to download
    """
    return data_source.validate_stocks(stocks, args, max_retries)

def fetch_stock_data(stocks, args, data_source: DataSource, max_retries=3):
    """
    Generic function to fetch stock data using the provided data source.
    
    This function delegates to the data source's fetch_stock_data method,
    providing a consistent interface regardless of the underlying data provider.
    
    Args:
        stocks: List of stock tickers
        args: Date range arguments (either period string or [start, end] tuple)
        data_source: DataSource instance to use for fetching data
        max_retries: Maximum number of retry attempts per stock (default: 3)
    
    Returns:
        tuple: (open_close_dataframe, failed_stocks)
               - open_close_dataframe: DataFrame with MultiIndex columns (Open/Close, stock_symbols)
               - failed_stocks: Dictionary mapping error types to lists of failed stocks with details
    """
    return data_source.fetch_stock_data(stocks, args, max_retries)

def get_data(stocks, args, seq_len, data_source: DataSource, force=False, prediction_type="classification", open_close_data=None, problematic_stocks=None, use_nlp=False, nlp_csv_paths=None, nlp_method="aggregated", period_type="LS"):
    """
    Return 12-tuple: (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Dtrain, Dval, Dtest, Rev_test, Returns_test, Sp500_test)
    Loads from .npz if present (and not force); otherwise builds from the provided data source,
    saves .npz, and returns the tuple.
    
    Args:
        stocks: List of stock tickers (should already be filtered to remove problematic stocks)
        args: Date range arguments
        seq_len: Sequence length (lookback window size) for period window calculation.
                Must be specified explicitly. For period_type="LS", this is the lookback window
                from which ~31 timesteps are sampled. For period_type="full", this is the
                actual sequence length used.
        data_source: DataSource instance to use for fetching stock data (required)
        force: Force rebuild even if cache exists
        prediction_type: Type of prediction (default: "classification")
        open_close_data: Optional pre-downloaded open_close DataFrame to avoid redundant download
        problematic_stocks: Optional list of problematic stocks to save (needed when open_close_data is provided)
        use_nlp: Whether to include NLP features (default: False)
        nlp_csv_paths: Path(s) to NYT CSV file(s) for NLP features. If None and use_nlp=True, 
            tries to find CSV files in ./huggingface_nyt_articles/ (relative to logan-version directory)
        nlp_method: Method for NLP feature extraction - "aggregated" (NYT headlines, shared across stocks)
            or "individual" (yfinance news per stock ticker). Default: "aggregated"
        period_type: Period type for feature extraction ("LS" or "full"). Default: "LS"
                    - "LS": Creates long periods (stepped) + short period (last 20% of seq_len)
                    - "full": Uses full sequence length window
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Validate args first
    if not validate_time_args(args):
        raise RuntimeError("Invalid Data Input Arguments")
    
    # Validate nlp_method if use_nlp is True
    if use_nlp and nlp_method is None:
        print("Warning: use_nlp=True but nlp_method not provided. Defaulting to 'aggregated'.")
        nlp_method = "aggregated"
    elif use_nlp and nlp_method not in ["aggregated", "individual"]:
        raise ValueError(f"Invalid nlp_method '{nlp_method}'. Must be 'aggregated' or 'individual' when use_nlp=True.")
    
    # Step 1: Check cache first (unless force=True)
    # load_data_from_cache verifies all 7 conditions before loading:
    # 1. Cleaned stock list matches
    # 2. Time period matches
    # 3. use_nlp matches
    # 4. nlp_method matches
    # 5. prediction_type matches
    # 6. period_type matches
    # 7. seq_len matches
    if not force:
        cached_data = load_data_from_cache(
            stocks=stocks,
            args=args,
            data_source=data_source,
            prediction_type=prediction_type,
            use_nlp=use_nlp,
            nlp_method=nlp_method,
            period_type=period_type,
            seq_len=seq_len
        )
        if cached_data is not None:
            print(f"[cache] Loaded data from cache (prediction_type={prediction_type}, use_nlp={use_nlp}, nlp_method={nlp_method})")
            return cached_data
        else:
            print(f"[cache] Cache not found (or conditions don't match), will download and process data...")
    
    # Step 2: Load saved problematic stocks for this time period
    problematic_stocks_saved = _load_problematic_stocks(args)
    
    # Step 3: Remove problematic stocks from input set
    if problematic_stocks_saved:
        valid_stocks = [stock for stock in stocks if stock not in problematic_stocks_saved]
        print(f"[data] Loaded {len(problematic_stocks_saved)} previously identified problematic stocks for this time period")
        print(f"[data] Filtered input: {len(stocks)} -> {len(valid_stocks)} stocks")
    else:
        valid_stocks = stocks
    
    validate_stocks_not_empty(valid_stocks, "after filtering problematic stocks")
    
    # Step 4: Check cache again with filtered stocks (in case cache exists for filtered set)
    # This checks all 7 conditions again: cleaned stocks, time period, use_nlp, nlp_method, prediction_type, period_type, seq_len
    if not force:
        cached_data = load_data_from_cache(
            stocks=valid_stocks,
            args=args,
            data_source=data_source,
            prediction_type=prediction_type,
            use_nlp=use_nlp,
            nlp_method=nlp_method,
            period_type=period_type,
            seq_len=seq_len
        )
        if cached_data is not None:
            print(f"[cache] Loaded data from cache (filtered stocks)")
            return cached_data
    
    # Step 5: Download data if needed
    # If open_close_data is provided, use it (avoids redundant download)
    if open_close_data is not None:
        open_close = open_close_data
        # Get successfully downloaded stocks from the DataFrame columns (use valid_stocks, not original stocks)
        successfully_downloaded_stocks = [stock for stock in valid_stocks if stock in open_close["Open"].columns]
        # Use provided problematic_stocks if available, otherwise calculate from input
        if problematic_stocks is None:
            # Calculate problematic stocks from valid_stocks
            problematic_stocks = [stock for stock in valid_stocks if stock not in successfully_downloaded_stocks]
    else:
        # For large stock lists, use fewer retries to speed up the download
        # Most errors are permanent (delisted, no data for date range), so 1 retry is sufficient
        max_retries = 1 if len(valid_stocks) > 50 else 3
        
        print(f"[data] Downloading data for {len(valid_stocks)} stocks...")
        open_close, failed_stocks = fetch_stock_data(valid_stocks, args, data_source, max_retries=max_retries)
        
        validate_dataframe_not_none(open_close, "open_close DataFrame")

        # Update stocks list to only include successfully downloaded stocks
        successfully_downloaded_stocks = [stock for stock in valid_stocks if stock in open_close["Open"].columns]
        # Calculate problematic stocks from valid_stocks
        new_problematic = [stock for stock in valid_stocks if stock not in successfully_downloaded_stocks]
        # Combine with previously known problematic stocks
        problematic_stocks = list(problematic_stocks_saved) + new_problematic
    
    # Save problematic stocks for this time period (so we can skip checking them in future runs)
    if problematic_stocks:
        _save_problematic_stocks(problematic_stocks, args)

    # ========================================================================
    # Step 1: Pull all the data
    # ========================================================================
    # Now build op/cp/date_index from the cleaned frame
    op = open_close["Open"].T   # (S, T)
    cp = open_close["Close"].T  # (S, T)
    date_index = open_close.index

    # Validate sequence length vs study period
    if seq_len >= op.shape[1]:
        raise ValueError(f"study period too short for chosen sequence length (seq_len={seq_len})")

    print(f"[data] Pulled data: {op.shape[0]} stocks, {op.shape[1]} time periods")
    
    # Stage 2 NaN filtering has been removed - using all downloaded data

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
                    # Look for NYT CSV files in the default data directory
                    nyt_dir = Path(DATA_DIR) / "huggingface_nyt_articles"
                    nlp_csv_paths = sorted(glob.glob(str(nyt_dir / "new_york_times_stories_*.csv")))
                    if not nlp_csv_paths:
                        import warnings
                        warnings.warn(f"No NYT CSV files found in {nyt_dir}. NLP features will be disabled.")
                        use_nlp = False
                
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
                        # Align NLP features with trading days
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
                        nlp_features_dict = None
            
            elif nlp_method == "individual":
                # Individual method: Use yfinance news per stock ticker
                # Check if date range is historical (yfinance typically only has recent news)
                from datetime import datetime, timedelta
                if end_date is not None:
                    if isinstance(end_date, str):
                        end_date_parsed = pd.to_datetime(end_date).date()
                    elif isinstance(end_date, datetime):
                        end_date_parsed = end_date.date()
                    else:
                        end_date_parsed = end_date
                    
                    # Check if end_date is more than 1 year ago
                    one_year_ago = datetime.now().date() - timedelta(days=365)
                    if end_date_parsed < one_year_ago:
                        import warnings
                        warnings.warn(
                            f"Individual NLP method with historical date range ({start_date} to {end_date}). "
                            f"yfinance news API typically only provides recent news (last 1-3 months). "
                            f"For historical data, consider using nlp_method='aggregated' with NYT articles instead.",
                            UserWarning
                        )
                
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
                    nlp_features_dict = None
            
            else:
                raise ValueError(f"Unknown nlp_method: {nlp_method}. Must be 'aggregated' or 'individual'")
                
        except Exception as e:
            import warnings
            import traceback
            warnings.warn(f"Error extracting NLP features: {e}. Continuing without NLP features.")
            traceback.print_exc()
            use_nlp = False
            nlp_features_dict = None

    # ========================================================================
    # Step 3: Extract features from cleaned data
    # ========================================================================
    print(f"[features] Extracting features from cleaned data...")
    
    # Determine the actual nlp_method being used (needed for feature extraction)
    # This should match what was used to extract features, or the validated nlp_method parameter
    actual_nlp_method = None
    if use_nlp:
        if nlp_features_dict is not None:
            # Determine method from extracted features (in case extraction changed it)
            if nlp_features_dict.get('_method') == 'individual_simple':
                actual_nlp_method = "individual"
            else:
                # Check if keys are tuples (individual) or dates (aggregated)
                sample_key = next(iter([k for k in nlp_features_dict.keys() if k != '_method']), None)
                if sample_key is not None and isinstance(sample_key, tuple):
                    actual_nlp_method = "individual"
                else:
                    actual_nlp_method = "aggregated"
        else:
            # If use_nlp=True but no features extracted (extraction failed), use the validated nlp_method
            # Note: use_nlp might have been set to False during extraction, so this may not be reached
            actual_nlp_method = nlp_method
    
    if prediction_type == "classification":
        xdata, ydata, dates, revenues, returns = get_feature_input_classification(
            op, cp, seq_len, op.shape[1], len(successfully_downloaded_stocks), date_index, 
            nlp_features=nlp_features_dict, use_nlp=use_nlp, nlp_method=actual_nlp_method, successfully_downloaded_stocks=successfully_downloaded_stocks, period_type=period_type,
            normalize_nlp_separately=True, normalize_per_stock=False
        )
    else:
        raise ValueError("Invalid prediction type - only 'classification' is supported")
    xdata = torch.from_numpy(xdata).to(torch.float32)  # (N, W, L, F) where N is number of samples
    ydata = torch.from_numpy(ydata).to(torch.float32)  # (N, 1)
    
    print(f"[features] Extracted {len(xdata):,} samples")

    # ========================================================================
    # Step 4: Separate into train/val/test split with given proportions
    # ========================================================================
    print(f"[split] Calculating train/val/test split...")
    
    # 3.1. Calculate total number of samples
    dates_np = np.array(dates, dtype="datetime64[ns]")
    n_total_samples = len(dates_np)
    print(f"[split] Total samples: {n_total_samples:,}")
    
    # Warn if total samples are very low (insufficient for training)
    if n_total_samples < 100:
        print(f"[split] ⚠️  WARNING: Very few total samples ({n_total_samples}). With a sequence length of {seq_len} days, you need a longer date range.")
        print(f"[split]    Recommended: Use at least 3-5 years of data for adequate training samples.")
    
    # 3.2. Sort all samples by date to find split boundaries
    # Get sorted indices (samples sorted by date)
    sorted_indices = np.argsort(dates_np)
    sorted_dates = dates_np[sorted_indices]
    
    # Group samples by date for statistics
    uniq_dates = np.unique(dates_np)
    n_total_dates = len(uniq_dates)
    print(f"[split] Unique dates: {n_total_dates:,}")
    
    date_to_indices = {}
    for i, date_val in enumerate(dates_np):
        if date_val not in date_to_indices:
            date_to_indices[date_val] = []
        date_to_indices[date_val].append(i)
    
    # Print grouping statistics
    samples_per_date = {date: len(indices) for date, indices in date_to_indices.items()}
    print(f"[split] Samples per date: min={min(samples_per_date.values())}, max={max(samples_per_date.values())}, mean={np.mean(list(samples_per_date.values())):.1f}")
    
    # 3.3. Calculate split proportions and find split boundaries by UNIQUE DATES
    # This ensures no data leakage - all samples from the same date go to the same split
    train_prop = 0.64   # 64% for training
    val_prop = 0.16     # 16% for validation  
    test_prop = 0.2     # 20% for test
    
    # Get unique dates sorted chronologically
    unique_dates = sorted(uniq_dates)
    n_unique_dates = len(unique_dates)
    
    # Calculate split boundaries based on unique dates (not sample count)
    train_end_date_idx = int(n_unique_dates * train_prop)
    val_end_date_idx = int(n_unique_dates * (train_prop + val_prop))
    
    # Ensure we have at least 1 date per split
    if train_end_date_idx == 0:
        train_end_date_idx = 1
    if val_end_date_idx <= train_end_date_idx:
        val_end_date_idx = train_end_date_idx + 1
    if val_end_date_idx >= n_unique_dates:
        val_end_date_idx = n_unique_dates - 1
    
    # Get the actual date boundaries
    train_end_date = unique_dates[train_end_date_idx - 1]  # Last date in training set
    val_end_date = unique_dates[val_end_date_idx - 1]      # Last date in validation set
    
    # Convert to comparable format
    train_end_date_ts = pd.Timestamp(train_end_date)
    val_end_date_ts = pd.Timestamp(val_end_date)
    
    print(f"[split] Split boundaries (based on UNIQUE DATES to prevent data leakage):")
    print(f"  Train end date: {train_end_date} (date index: {train_end_date_idx}/{n_unique_dates})")
    print(f"  Val end date: {val_end_date} (date index: {val_end_date_idx}/{n_unique_dates})")
    print(f"  Test starts after: {val_end_date}")
    
    # 3.4. Assign all samples to splits based on date boundaries
    # All samples from same date go to same split - this prevents data leakage
    train_mask = np.zeros(n_total_samples, dtype=bool)
    val_mask = np.zeros(n_total_samples, dtype=bool)
    test_mask = np.zeros(n_total_samples, dtype=bool)
    
    # Group samples by date and assign entire date groups
    for date_val, indices in date_to_indices.items():
        date_val_ts = pd.Timestamp(date_val)
        
        # Determine split based on date boundaries (strict comparison)
        if date_val_ts <= train_end_date_ts:
            split = 'train'
        elif date_val_ts <= val_end_date_ts:
            split = 'val'
        else:
            split = 'test'
        
        # Assign all samples from this date to the same split
        for idx in indices:
            if split == 'train':
                train_mask[idx] = True
            elif split == 'val':
                val_mask[idx] = True
            else:
                test_mask[idx] = True
    
    # Validate: Ensure no date appears in multiple splits (data leakage check)
    train_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[train_mask])
    val_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[val_mask])
    test_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[test_mask])
    
    date_overlap_train_val = train_dates_set & val_dates_set
    date_overlap_train_test = train_dates_set & test_dates_set
    date_overlap_val_test = val_dates_set & test_dates_set
    
    if date_overlap_train_val or date_overlap_train_test or date_overlap_val_test:
        overlap_msg = []
        if date_overlap_train_val:
            overlap_msg.append(f"train-val: {len(date_overlap_train_val)} dates")
        if date_overlap_train_test:
            overlap_msg.append(f"train-test: {len(date_overlap_train_test)} dates")
        if date_overlap_val_test:
            overlap_msg.append(f"val-test: {len(date_overlap_val_test)} dates")
        raise RuntimeError(f"DATA LEAKAGE DETECTED: Date overlap between splits ({', '.join(overlap_msg)})")
    
    print(f"[split] ✓ Data leakage check passed: No date overlap between splits")
    
    # Calculate actual sample counts and proportions
    n_train_samples = np.sum(train_mask)
    n_val_samples = np.sum(val_mask)
    n_test_samples = np.sum(test_mask)
    
    actual_train_prop = n_train_samples / n_total_samples
    actual_val_prop = n_val_samples / n_total_samples
    actual_test_prop = n_test_samples / n_total_samples
    
    print(f"[split] Sample proportions: train={actual_train_prop:.1%} ({n_train_samples:,}), val={actual_val_prop:.1%} ({n_val_samples:,}), test={actual_test_prop:.1%} ({n_test_samples:,})")
    
    # Count unique dates per split
    train_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[train_mask])
    val_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[val_mask])
    test_dates_set = set(pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[test_mask])
    
    print(f"[split] Date assignments: {len(train_dates_set)} train dates, {len(val_dates_set)} val dates, {len(test_dates_set)} test dates")
    
    # Validation: Check temporal order
    print(f"\n[validation] Checking temporal order...")
    train_dates_list = sorted([pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[train_mask]])
    val_dates_list = sorted([pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[val_mask]])
    test_dates_list = sorted([pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, np.datetime64)) else d for d in dates_np[test_mask]])
    
    if train_dates_list and val_dates_list:
        max_train_date = max(train_dates_list)
        min_val_date = min(val_dates_list)
        if max_train_date >= min_val_date:
            raise RuntimeError(f"TEMPORAL ORDER VIOLATION: Max train date ({max_train_date}) >= min val date ({min_val_date})")
        print(f"  ✓ Train dates end before validation dates: {max_train_date} < {min_val_date}")
    
    if val_dates_list and test_dates_list:
        max_val_date = max(val_dates_list)
        min_test_date = min(test_dates_list)
        if max_val_date >= min_test_date:
            raise RuntimeError(f"TEMPORAL ORDER VIOLATION: Max val date ({max_val_date}) >= min test date ({min_test_date})")
        print(f"  ✓ Validation dates end before test dates: {max_val_date} < {min_test_date}")
    
    # Validation: Monitor feature distributions across splits
    print(f"\n[validation] Checking feature distributions across splits...")
    X_train_samples = xdata[train_mask]
    X_val_samples = xdata[val_mask]
    X_test_samples = xdata[test_mask]
    
    # Compute statistics for each split
    def compute_feature_stats(X_split, split_name):
        if len(X_split) == 0:
            return None
        # Flatten to (N*W, F) for statistics
        X_flat = X_split.reshape(-1, X_split.shape[-1])
        return {
            'mean': np.mean(X_flat, axis=0),
            'std': np.std(X_flat, axis=0),
            'min': np.min(X_flat, axis=0),
            'max': np.max(X_flat, axis=0)
        }
    
    train_stats = compute_feature_stats(X_train_samples, 'train')
    val_stats = compute_feature_stats(X_val_samples, 'val')
    test_stats = compute_feature_stats(X_test_samples, 'test')
    
    if train_stats and val_stats:
        # Check if means are similar (should be if normalized together)
        mean_diff = np.abs(train_stats['mean'] - val_stats['mean'])
        std_diff = np.abs(train_stats['std'] - val_stats['std'])
        max_mean_diff = np.max(mean_diff)
        max_std_diff = np.max(std_diff)
        
        print(f"  Train/Val mean difference (max): {max_mean_diff:.6f}")
        print(f"  Train/Val std difference (max): {max_std_diff:.6f}")
        
        if max_mean_diff > 0.5:  # Threshold for normalized features
            print(f"  ⚠️  WARNING: Large mean difference between train and val ({max_mean_diff:.6f})")
        else:
            print(f"  ✓ Mean distributions are similar")
        
        if max_std_diff > 0.5:
            print(f"  ⚠️  WARNING: Large std difference between train and val ({max_std_diff:.6f})")
        else:
            print(f"  ✓ Std distributions are similar")
    
    # Warn if training data is very limited
    if n_train_samples < 100:
        print(f"[split] ⚠️  WARNING: Very few training samples ({n_train_samples}). Consider using a longer date range for better model training.")

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
    Sp500_f = _get_sp500_returns_for_dates(Dtest_f, date_index, op.shape[1], data_source)

    # --- split summary ---
    n_tr, n_va, n_te = len(Dtrain_f), len(Dvalidation_f), len(Dtest_f)
    n_tot = n_tr + n_va + n_te
    print(f"[split] train/val/test sizes = {n_tr:,} / {n_va:,} / {n_te:,}  | total kept = {n_tot:,}")
    print(f"[split] Sample proportions: train={n_tr/n_tot:.1%}, val={n_va/n_tot:.1%}, test={n_te/n_tot:.1%}")

    # NOW create the ID based on successfully downloaded stocks (after error handling)
    # Include all parameters in ID generation
    # Serialize data source for ID generation
    data_source_str = _serialize_data_source(data_source)
    data_id = _get_data_id(
        successfully_downloaded_stocks, 
        args, 
        use_nlp=use_nlp, 
        nlp_method=nlp_method,
        prediction_type=prediction_type,
        period_type=period_type,
        seq_len=seq_len,
        data_source_str=data_source_str
    )
    
    # Save datasets separately
    def _to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    
    # Use simple filename format: {data_id}_{set_type}.npz
    train_path = os.path.join(DATA_DIR, f"{data_id}_train.npz")
    val_path = os.path.join(DATA_DIR, f"{data_id}_val.npz")
    test_path = os.path.join(DATA_DIR, f"{data_id}_test.npz")
    metrics_path = os.path.join(DATA_DIR, f"{data_id}_metrics.npz")
    
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
    # Include all parameters in mapping
    _save_id_mapping(
        data_id, 
        successfully_downloaded_stocks, 
        args, 
        use_nlp=use_nlp, 
        nlp_method=nlp_method,
        prediction_type=prediction_type,
        period_type=period_type,
        seq_len=seq_len,
        data_source_str=data_source_str
    )
    
    return (Xtr_f, Xva_f, Xte_f, Ytr_f, Yva_f, Yte_f, Dtrain_f, Dvalidation_f, Dtest_f, Rev_f, Returns_f, Sp500_f)

def load_data_from_cache(stocks, args, data_source: DataSource, prediction_type="classification", use_nlp=False, nlp_method="aggregated", period_type="LS", seq_len=240):
    """
    Load data from cache files if they exist.
    
    This function filters out problematic stocks before checking for cache,
    matching the behavior of get_data() which saves cache with filtered stocks.
    It also uses the mapping file to find cache by matching stocks and args.
    
    Args:
        stocks: List of stock tickers (may include problematic stocks - will be filtered)
        args: Date range arguments
        data_source: DataSource instance (required). Used for matching cache entries and fetching
                    S&P 500 data if needed for old cache formats.
        prediction_type: Type of prediction (default: "classification")
        use_nlp: Whether NLP features were used (default: False)
        nlp_method: NLP method used - "aggregated" or "individual" (default: "aggregated")
        period_type: Period type for feature extraction ("LS" or "full"). Default: "LS"
                    - "LS": Creates long periods (stepped) + short period (last 20% of seq_len)
                    - "full": Uses full sequence length window
        seq_len: Sequence length (lookback window size) for period window calculation.
                Default: 240 (for backward compatibility with existing caches)
    
    Returns:
        data tuple if cache exists, None otherwise.
    """
    # Step 1: Load problematic stocks for this time period (if they exist)
    problematic_stocks = _load_problematic_stocks(args)
    
    # Step 2: Filter out problematic stocks from input stocks (matching get_data behavior)
    filtered_stocks = [stock for stock in stocks if stock not in problematic_stocks]
    
    # Step 3: Try to find cache using the mapping file
    # We verify all 8 conditions explicitly:
    # 1. Cleaned stock list matches
    # 2. Time period matches
    # 3. use_nlp matches
    # 4. nlp_method matches
    # 5. prediction_type matches
    # 6. period_type matches
    # 7. seq_len matches
    # 8. data_source matches
    mapping = _load_id_mapping()
    found_cache = False
    data_id = None
    
    # Serialize data_source for comparison
    data_source_str = _serialize_data_source(data_source)
    
    for cached_id, cached_info in mapping.items():
        cached_stocks = set(cached_info.get('stocks', []))
        cached_args = cached_info.get('args', [])
        cached_use_nlp = cached_info.get('use_nlp', False)
        cached_nlp_method = cached_info.get('nlp_method', 'aggregated')
        cached_prediction_type = cached_info.get('prediction_type', 'classification')
        cached_period_type = cached_info.get('period_type', 'LS')
        cached_seq_len = cached_info.get('seq_len', 240)
        cached_data_source = cached_info.get('data_source')
        if cached_data_source is None:
            # Old cache format without data_source - skip this cache entry
            continue
        
        # Normalize args for comparison (convert to list if needed)
        cached_args_list = list(cached_args) if cached_args else []
        args_list = list(args) if args else []
        
        # Verify all 8 conditions:
        # 1. Cleaned stock list matches
        stocks_match = set(filtered_stocks) == cached_stocks
        # 2. Time period matches
        time_period_match = cached_args_list == args_list
        # 3. use_nlp matches
        use_nlp_match = cached_use_nlp == use_nlp
        # 4. nlp_method matches
        nlp_method_match = cached_nlp_method == nlp_method
        # 5. prediction_type matches
        prediction_type_match = cached_prediction_type == prediction_type
        # 6. period_type matches
        period_type_match = cached_period_type == period_type
        # 7. seq_len matches
        seq_len_match = cached_seq_len == seq_len
        # 8. data_source matches
        data_source_match = cached_data_source == data_source_str
        
        # All 8 conditions must be satisfied
        if (stocks_match and time_period_match and use_nlp_match and nlp_method_match and 
            prediction_type_match and period_type_match and seq_len_match and data_source_match):
            # Found matching cache - verify files exist using the unique ID
            data_id = cached_id
            
            # Construct file paths using the unique ID
            train_path = os.path.join(DATA_DIR, f"{data_id}_train.npz")
            val_path = os.path.join(DATA_DIR, f"{data_id}_val.npz")
            test_path = os.path.join(DATA_DIR, f"{data_id}_test.npz")
            metrics_path = os.path.join(DATA_DIR, f"{data_id}_metrics.npz")
            
            # Verify all required files exist
            if all(os.path.exists(p) for p in [train_path, val_path, test_path, metrics_path]):
                found_cache = True
                break
    
    if not found_cache:
        # No cache found that satisfies all 7 conditions
        return None
    
    # At this point, we have verified all 7 conditions and confirmed files exist
    # data_id is set to the matching unique ID from the mapping file
    # Load data using the unique ID
    print(f"[cache] Loading cached data using unique ID: {data_id}")
    
    # Construct file paths using the unique ID (paths already set in loop, but set again for clarity)
    train_path = os.path.join(DATA_DIR, f"{data_id}_train.npz")
    val_path = os.path.join(DATA_DIR, f"{data_id}_val.npz")
    test_path = os.path.join(DATA_DIR, f"{data_id}_test.npz")
    metrics_path = os.path.join(DATA_DIR, f"{data_id}_metrics.npz")
    
    # Load from cache using the unique ID
    train_data = _load_npz_progress(train_path, ["X", "Y", "D"], desc="Loading training dataset (.npz)")
    val_data = _load_npz_progress(val_path, ["X", "Y", "D"], desc="Loading validation dataset (.npz)")
    test_data = _load_npz_progress(test_path, ["X", "Y", "D"], desc="Loading test dataset (.npz)")
    
    # Validate that cached data has expected NLP features if NLP is requested
    if use_nlp:
        from nlp_features import get_nlp_feature_dim
        sample_X = train_data.get("X")
        if sample_X is not None:
            actual_features = sample_X.shape[2] if len(sample_X.shape) >= 3 else sample_X.shape[1]
            expected_nlp_dim = get_nlp_feature_dim(nlp_method)
            expected_features = 3 + expected_nlp_dim
            
            if actual_features != expected_features:
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
            Sp500 = _get_sp500_returns_for_dates(Dte_temp, None, None, data_source)
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
        Sp500 = _get_sp500_returns_for_dates(Dte_temp, None, None, data_source)
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

def save_data_locally(stocks, args, seq_len, data_source: DataSource, force=False, prediction_type="classification"):
    """
    Force a rebuild/save and return the data tuple.
    
    Args:
        stocks: List of stock tickers
        args: Date range arguments
        seq_len: Sequence length (lookback window size)
        data_source: DataSource instance (required)
        force: Force rebuild even if cache exists
        prediction_type: Type of prediction (default: "classification")
    
    Returns:
        Data tuple
    """
    return get_data(stocks, args, seq_len=seq_len, data_source=data_source, force=True, prediction_type=prediction_type)



########################################################
# get feature input
########################################################

def get_period(end_t, seq_len, mask_type="LS"):
    """
    Generate period indices for feature extraction based on sequence length.
    
    Args:
        end_t: End time index
        seq_len: Sequence length (total history window size) to consider
        mask_type: Period type ("LS" or "full")
                 - "LS": Long-Short sampling pattern:
                        1. Get the full seq_len window
                        2. Short term: int(seq_len/12) + 1 consecutive days at the end (stride 1)
                           Relative indices: [0, 1, 2, ..., int(seq_len/12)]
                        3. Long term: Remaining days sampled with stride int(seq_len/12),
                           starting from 2 * int(seq_len/12) (relative to window start)
                           Example for seq_len=240: stride=20, short term=[0-20], 
                           long term sampled at [40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
                           Gaps are filled to ensure exactly seq_len points total
                        * Returns exactly seq_len points total
                 - "full": Uses full sequence length window (all consecutive days)
    
    Returns:
        Array of period indices with exactly seq_len points
    """
    if mask_type == "LS":
        # LS mode algorithm:
        # 1. Get the full seq_len window
        # 2. Short term: int(seq_len/12) + 1 consecutive days at the end (stride 1)
        #    Relative indices: [0, 1, 2, ..., int(seq_len/12)]
        # 3. Long term: Remaining days sampled with stride int(seq_len/12),
        #    starting from 2 * int(seq_len/12) (relative to window start)
        
        # Calculate stride (step size)
        stride = int(seq_len / 12)
        
        # Window boundaries (relative to end_t)
        # Total window: [end_t - seq_len + 1, end_t]
        window_start = end_t - seq_len + 1
        
        # Step 3: Long term window - sampled with stride, starting from 2 * stride
        # Starting index relative to window_start: 2 * stride
        # Absolute starting index: window_start + 2 * stride
        # We need exactly 10 points: [40, 60, 80, 100, 120, 140, 160, 180, 200, 220] relative indices
        # Generate: long_start + i*stride for i in [0, 9] = 10 points
        long_start = window_start + 2 * stride
        num_long_points = 10  # For seq_len=240, we want 10 long-term points
        period_long = np.array([long_start + i * stride for i in range(num_long_points)], dtype=int)
        
        # Step 2: Short term window - last (stride + 1) consecutive days (stride 1)
        # Relative indices from end of window: [0, 1, 2, ..., stride]
        # For window [window_start, end_t], the last (stride+1) points are [end_t - stride, end_t]
        # This gives us relative indices [seq_len - stride - 1, seq_len - stride, ..., seq_len - 1]
        # For seq_len=240, stride=20: relative indices [219, 220, 221, ..., 239] = 21 points
        # Note: relative index 220 overlaps with the last long-term point, which is correct
        # The total will be 31 points: 10 long-term + 21 short-term, with 1 overlap = 30 unique + 1 duplicate = 31 total
        short_start = end_t - stride
        period_short = np.arange(short_start, end_t + 1, step=1, dtype=int)
        
        # Combine long and short periods
        # For LS mode, return only the sampled points (no gap filling)
        # Note: There may be an overlap between long-term and short-term (e.g., relative index 220 for seq_len=240)
        # This is intentional and results in exactly (num_long_points + num_short_points) total points
        # For seq_len=240: 10 long-term + 21 short-term = 31 points (with 1 overlap at relative index 220)
        period = np.concatenate([period_long, period_short])
        period = np.sort(period)  # Ensure sorted order
        
        # Return the sampled period as-is (no gap filling for LS mode)
        # The shape will be (num_samples, len(period), num_features)
        # For seq_len=240: shape will be (num_samples, 31, num_features)
        return period
    elif mask_type == "full":
        # Full window: use entire sequence length (all consecutive points)
        return np.arange(end_t - seq_len + 1, end_t + 1, step=1, dtype=int)
    else:
        validate_mask_type(mask_type)  # Will raise ValueError if invalid
        return None  # Should not reach here

# op[x] is the op vector for stock x
# op and cp has indices from time 0 to T_study-1
def get_feature_input_classification(op, cp, seq_len, study_period, num_stocks, date_index, nlp_features=None, use_nlp=False, nlp_method=None, successfully_downloaded_stocks=None, period_type="LS", normalize_nlp_separately=True, normalize_per_stock=False):
    """
    Get feature input for classification task.
    
    Args:
        op: Open prices DataFrame
        cp: Close prices DataFrame
        seq_len: Sequence length for period window calculation
        study_period: Total study period length
        num_stocks: Number of stocks
        date_index: Date index
        nlp_features: NLP features dictionary (optional)
        use_nlp: Whether to use NLP features
        nlp_method: NLP method ("aggregated" or "individual")
        successfully_downloaded_stocks: List of successfully downloaded stocks
        period_type: Period type ("LS" or "full")
                    - "LS": Creates long periods (stepped) + short period (last 20% of seq_len)
                    - "full": Uses full sequence length window
        normalize_nlp_separately: If True, normalize NLP features separately from price features
        normalize_per_stock: If True, normalize price features per stock (instead of globally)
    """
    T = study_period
    
    print(f"[features] Computing features for {num_stocks} stocks over {T} time periods...")
    print(f"[features] Using sequence length: {seq_len} for period window calculation")
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
    total_windows = (T - seq_len - 2) * num_stocks
    pbar = tqdm(total=total_windows, desc="Building feature windows", unit="window")
    
    for end_t in range(seq_len + 2, T):
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

            period = get_period(end_t, seq_len, mask_type=period_type)

            # Build 3-feature window; skip if any feature is invalid over the window
            window = np.empty((len(period), 3), dtype=float)
            valid = True
            for i in range(3):  # ir, cpr, opr (robust z-score)
                vec = f_t1[n, i, period]
                if np.isnan(vec).any():
                    dropped_feature_nan += 1
                    valid = False; break
                
                # Window-based robust z-score normalization
                # Note: normalize_per_stock option would require a different normalization approach
                # (e.g., normalizing all windows per stock before windowing). For now, we use
                # window-based normalization which normalizes each window independently.
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
                
                # Determine feature dimension and method based on feature structure
                # This must match what get_nlp_feature_vector will return
                if use_simple or is_individual_method:
                    # Individual method: 4 features
                    nlp_feature_dim = 4
                    actual_method_for_features = "individual"
                else:
                    # Aggregated method: 10 features
                    nlp_feature_dim = 10
                    actual_method_for_features = "aggregated"
                
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
                                nlp_vec = get_nlp_feature_vector(nlp_row, nlp_method=actual_method_for_features)
                                nlp_window[idx, :] = nlp_vec
                            else:
                                # No NLP data for this stock-date - use zeros
                                nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                        else:
                            # Aggregated method: date -> row
                            if date_at_t in nlp_features_clean:
                                nlp_row = nlp_features_clean[date_at_t]
                                nlp_vec = get_nlp_feature_vector(nlp_row, nlp_method=actual_method_for_features)
                                nlp_window[idx, :] = nlp_vec
                            else:
                                # No NLP data for this date - use zeros
                                nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                    else:
                        # Date index out of range - use zeros
                        nlp_window[idx, :] = np.zeros(nlp_feature_dim, dtype=float)
                
                # Normalize NLP features separately if requested
                if normalize_nlp_separately:
                    # Standardize NLP features: (x - mean) / std per feature
                    for feat_idx in range(nlp_feature_dim):
                        nlp_feat = nlp_window[:, feat_idx]
                        feat_mean = np.mean(nlp_feat)
                        feat_std = np.std(nlp_feat)
                        if feat_std > 1e-8:  # Avoid division by zero
                            nlp_window[:, feat_idx] = (nlp_feat - feat_mean) / feat_std
                        else:
                            # If std is too small, just center (set to zero)
                            nlp_window[:, feat_idx] = nlp_feat - feat_mean
                
                # Concatenate price features with NLP features
                window = np.concatenate([window, nlp_window], axis=1)  # (len(period), 3 + nlp_feature_dim)
                
                # Debug: Print feature dimensions on first window
                if len(X_list) == 0:
                    norm_info = []
                    if normalize_per_stock:
                        norm_info.append("per-stock price normalization")
                    else:
                        norm_info.append("global price normalization")
                    if normalize_nlp_separately:
                        norm_info.append("separate NLP normalization")
                    print(f"[features] NLP features added: {nlp_feature_dim} features ({actual_method_for_features} method)")
                    print(f"[features] Normalization: {', '.join(norm_info)}")
                    print(f"[features] Total features per timestep: {window.shape[1]} (3 base + {nlp_feature_dim} NLP)")
            else:
                # Debug: Print when NLP features are NOT added
                if len(X_list) == 0:
                    if use_nlp:
                        print(f"[features] ⚠️  Warning: use_nlp=True but nlp_features is None. Only base features will be used.")
                    print(f"[features] Total features per timestep: {window.shape[1]} (base features only)")

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

    X = np.array(X_list, dtype=float)   # (num_stocks, period_length, 3)
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
