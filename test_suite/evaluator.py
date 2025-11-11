"""
Comprehensive Model Evaluator for Stock Classification

This module provides a flexible evaluation framework that can test multiple metrics
on saved models and log results to TensorBoard for analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, log_loss, brier_score_loss
from typing import Dict, List, Tuple, Optional, Union
import os
import sys
import yfinance as yf
import hashlib
from zipfile import ZipFile, ZIP_STORED
from io import BytesIO
from scipy import stats as scipy_stats
from scipy.special import expit
from collections import defaultdict

# Add parent directory and logan-version to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logan_version_dir = os.path.join(parent_dir, "logan-version")
if logan_version_dir not in sys.path:
    sys.path.insert(0, logan_version_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import LSTMModel
import util


class ModelEvaluator:
    """
    A comprehensive evaluator for stock classification models.
    
    This class provides:
    - Paper-aligned evaluation (Fischer-Krauss/Ghosh methodology)
    - Multiple evaluation metrics (accuracy, trading performance, etc.)
    - TensorBoard logging of evaluation results
    - Visualization capabilities
    - Support for loading saved models
    - Time-series specific analysis
    
    Paper-Aligned Evaluation (Fischer-Krauss/Ghosh):
    - Portfolio construction: Daily top-k/flop-k ranking and equal-weighted long-short portfolios
    - Transaction costs: Fixed formula 4 * cost_per_side for equal-weight long-short with full rebalance
    - Accuracy: Computed on traded set only with DM/PT statistical tests (using paper labels)
    - Target labels: Binary classification based on outperforming cross-sectional median per day
    - Risk metrics: Sharpe, Sortino, VaR/CVaR (1% & 5%), max drawdown, skewness, kurtosis
    - Benchmarks: S&P 500 (aligned by date) and random k-long/k-short portfolios
    - Max drawdown: Positive convention on wealth curve (cumprod(1+r))
    - Random benchmark: Includes distribution stats (percentiles) and percentile rank
    
    Important Notes:
    - Returns vs Revenues: The util pipeline now persists both revenues (Close-Open dollars)
      and returns ((Close-Open)/Open). Paper-aligned evaluation uses the return series exclusively.
    
    Additional diagnostics (separated from paper metrics):
    - MSE/MAE, residual autocorrelation, slope accuracy (not in papers)
    """
    
    def __init__(self, 
                 model_path: str,
                 stocks: List[str] = ["MSFT", "AAPL"], 
                 time_args: List[str] = ["3y"],
                 data_dir: str = None,
                 log_dir: str = "runs/evaluation",
                 device: Optional[torch.device] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the saved model (.pth file)
            stocks: List of stock symbols to evaluate on
            time_args: Time arguments for data loading
            data_dir: Directory containing data files (defaults to logan-version/data)
            log_dir: Directory for TensorBoard logs
            device: Device to run evaluation on (auto-detect if None)
        """
        # Default to logan-version/data if not specified
        if data_dir is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logan_version_dir = os.path.join(parent_dir, "logan-version")
            data_dir = os.path.join(logan_version_dir, "data")
        
        # Resolve model path - check logan-version if not absolute and not found locally
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logan_version_dir = os.path.join(parent_dir, "logan-version")
            logan_model_path = os.path.join(logan_version_dir, model_path)
            if os.path.exists(logan_model_path):
                model_path = logan_model_path
        
        self.model_path = model_path
        self.stocks = stocks
        self.time_args = time_args
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        print(f"[Evaluator] Using data directory: {os.path.abspath(self.data_dir)}")
        print(f"[Evaluator] Using model path: {os.path.abspath(self.model_path)}")
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"[Evaluator] Using device: {self.device}")
        
        # Load data
        self._load_data()
        
        # Initialize model
        self._load_model()
        
        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Metrics storage
        self.results = {}
        
    def _load_data(self):
        """Load test data for evaluation."""
        print("[Evaluator] Loading test data...")
        
        # Load data using the same method as training
        # Try cache first, then download if needed
        input_data = util.load_data_from_cache(self.stocks, self.time_args, data_dir=self.data_dir, prediction_type="classification")
        if input_data is None:
            input_data = util.get_data(self.stocks, self.time_args, data_dir=self.data_dir, prediction_type="classification")
        if isinstance(input_data, int):
            raise RuntimeError("Error getting data from util.get_data()")
        
        if len(input_data) != 12:
            raise ValueError(
                "Expected 12-element dataset (with returns). Please regenerate data with the current util pipeline."
            )

        (
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
            D_train,
            D_val,
            D_test,
            Rev_test,
            Returns_test,
            Sp500_test,
        ) = input_data
        
        # Store test data
        self.X_test = X_test
        self.Y_test = Y_test
        self.test_dates = D_test
        
        # Ensure revenues are numpy arrays and aligned with test inputs
        self.test_revenues = np.array(Rev_test) if not isinstance(Rev_test, np.ndarray) else Rev_test
        
        # Returns (Close-Open)/Open from util metrics (mandatory)
        self.test_returns = np.array(Returns_test) if not isinstance(Returns_test, np.ndarray) else Returns_test
        
        # Store S&P 500 returns if available
        if Sp500_test is not None:
            self.test_sp500 = np.array(Sp500_test) if not isinstance(Sp500_test, np.ndarray) else Sp500_test
        else:
            self.test_sp500 = None
        
        # Verify alignment between test inputs and revenues
        if len(self.X_test) != len(self.test_revenues):
            raise ValueError(f"Mismatch: test inputs ({len(self.X_test)}) and revenues ({len(self.test_revenues)}) must have same length")
        if len(self.X_test) != len(self.test_returns):
            raise ValueError(f"Mismatch: test inputs ({len(self.X_test)}) and returns ({len(self.test_returns)}) must have same length")
        
        if self.test_sp500 is not None and len(self.X_test) != len(self.test_sp500):
            raise ValueError(f"Mismatch: test inputs ({len(self.X_test)}) and S&P 500 returns ({len(self.test_sp500)}) must have same length")
        
        print(f"[Evaluator] Loaded {len(self.X_test)} test samples")
        print(f"[Evaluator] Test revenues shape: {self.test_revenues.shape}, aligned with test inputs: {len(self.X_test) == len(self.test_revenues)}")
        if self.test_sp500 is not None:
            print(f"[Evaluator] Test S&P 500 returns shape: {self.test_sp500.shape}, aligned with test inputs: {len(self.X_test) == len(self.test_sp500)}")
        
    def _get_sp500_cache_id(self, min_date, max_date):
        """
        Generate a cache ID for S&P 500 data based on date range.
        
        Args:
            min_date: Minimum date
            max_date: Maximum date
            
        Returns:
            A short hash string for the date range
        """
        # Create a stable string representation
        date_str = f"{min_date}|{max_date}"
        
        # Generate a short hash
        hash_obj = hashlib.sha256(date_str.encode())
        return hash_obj.hexdigest()[:10]
    
    def _save_sp500_cache(self, sp500_returns: np.ndarray, min_date, max_date, data_dir: str):
        """
        Save S&P 500 returns to cache using the same scheme as util.py.
        
        Args:
            sp500_returns: S&P 500 returns array
            min_date: Minimum date
            max_date: Maximum date
            data_dir: Directory for data files
        """
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        cache_id = self._get_sp500_cache_id(min_date, max_date)
        cache_path = os.path.join(data_dir, f"sp500_{cache_id}.npz")
        
        # Use the same save function pattern as util.py
        def _save_npz_progress(path: str, arrays: dict, desc="Saving dataset"):
            with ZipFile(path, mode="w", compression=ZIP_STORED) as zf:
                bar = tqdm(total=len(arrays), desc=desc)
                for name, arr in arrays.items():
                    buf = BytesIO()
                    np.save(buf, np.asanyarray(arr), allow_pickle=False)
                    zf.writestr(f"{name}.npy", buf.getvalue())
                    bar.update(1)
                bar.close()
        
        _save_npz_progress(cache_path, {
            "returns": sp500_returns,
            "min_date": np.array([min_date], dtype="datetime64[ns]"),
            "max_date": np.array([max_date], dtype="datetime64[ns]")
        }, desc="Saving S&P 500 cache (.npz)")
        
        print(f"[Evaluator] S&P 500 cache saved: {os.path.abspath(cache_path)}")
    
    def _load_sp500_cache(self, min_date, max_date, data_dir: str) -> Optional[np.ndarray]:
        """
        Load S&P 500 returns from cache if available.
        
        Args:
            min_date: Minimum date
            max_date: Maximum date
            data_dir: Directory for data files
            
        Returns:
            S&P 500 returns array if cache exists, None otherwise
        """
        cache_id = self._get_sp500_cache_id(min_date, max_date)
        cache_path = os.path.join(data_dir, f"sp500_{cache_id}.npz")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Use the same load function pattern as util.py
            def _load_npz_progress(path: str, names: list, desc="Loading dataset"):
                with np.load(path, allow_pickle=False) as z:
                    out = {}
                    bar = tqdm(total=len(names), desc=desc)
                    for n in names:
                        out[n] = z[n]
                        bar.update(1)
                    bar.close()
                    return out
            
            cache_data = _load_npz_progress(cache_path, ["returns", "min_date", "max_date"], desc="Loading S&P 500 cache (.npz)")
            
            # Verify date range matches
            cached_min = pd.Timestamp(cache_data["min_date"][0]).to_pydatetime()
            cached_max = pd.Timestamp(cache_data["max_date"][0]).to_pydatetime()
            
            if cached_min == min_date and cached_max == max_date:
                print(f"[Evaluator] S&P 500 cache loaded from: {os.path.abspath(cache_path)}")
                return cache_data["returns"]
            else:
                print(f"[Evaluator] S&P 500 cache date mismatch, will refetch")
                return None
                
        except Exception as e:
            print(f"[Evaluator] Warning: Error loading S&P 500 cache: {e}")
            return None
    
    def _get_sp500_returns(self) -> Optional[np.ndarray]:
        """
        Fetch S&P 500 returns aligned with test dates.
        Priority order:
        1. From loaded metrics data (saved by util.py)
        2. From standalone cache file
        3. Fetch from yfinance and cache
        
        Returns:
            S&P 500 daily returns aligned with test_dates, or None if fetch fails
        """
        if not self.test_dates or len(self.test_dates) == 0:
            print("[Evaluator] No test dates available for S&P 500 comparison")
            return None
        
        # Priority 1: Check if S&P 500 data is available from loaded metrics data
        if hasattr(self, 'test_sp500') and self.test_sp500 is not None:
            try:
                sp500_from_metrics = np.asarray(self.test_sp500, dtype=float)
            except (TypeError, ValueError):
                sp500_from_metrics = None

            if sp500_from_metrics is not None and sp500_from_metrics.shape == (len(self.test_dates),):
                std_metrics = np.std(sp500_from_metrics)
                if std_metrics > 1e-12:
                    print(f"[Evaluator] Using S&P 500 returns from loaded metrics data: {len(sp500_from_metrics)} values, mean={np.mean(sp500_from_metrics):.6f}, std={std_metrics:.6f}")
                    return sp500_from_metrics
                else:
                    print("[Evaluator] Warning: Loaded S&P 500 metrics appear constant; refetching from source...")
            else:
                print("[Evaluator] Warning: Loaded S&P 500 metrics have unexpected shape; refetching from source...")
        
        # Get date range from test dates
        min_date = min(self.test_dates)
        max_date = max(self.test_dates)
        
        # Priority 2: Try to load from standalone cache file
        sp500_returns = self._load_sp500_cache(min_date, max_date, self.data_dir)
        if sp500_returns is not None and len(sp500_returns) == len(self.test_dates):
            print(f"[Evaluator] S&P 500 returns loaded from standalone cache: {len(sp500_returns)} values, mean={np.mean(sp500_returns):.6f}, std={np.std(sp500_returns):.6f}")
            return sp500_returns
        
        # Cache miss - fetch from yfinance
        try:
            print("[Evaluator] Fetching S&P 500 data for benchmark comparison...")
            
            # Fetch S&P 500 data (^GSPC is the ticker for S&P 500)
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(start=min_date, end=max_date, repair=True)
            
            if sp500_data is None or sp500_data.empty:
                print("[Evaluator] Warning: Could not fetch S&P 500 data")
                return None
            
            # Calculate returns from S&P 500
            # Note: If revenues are intraday (open→close), we should compute intraday returns
            # For now, we use close-to-close returns. In a full implementation, we'd compute
            # intraday returns: (Close - Open) / Open for each day
            if 'Open' in sp500_data.columns and 'Close' in sp500_data.columns:
                # Try intraday returns if available (for alignment with intraday revenues)
                sp500_open = sp500_data['Open']
                sp500_close = sp500_data['Close']
                # Intraday return: (Close - Open) / Open
                sp500_daily_returns = ((sp500_close - sp500_open) / sp500_open).dropna()
                if len(sp500_daily_returns) == 0:
                    # Fallback to close-to-close
                    sp500_daily_returns = sp500_close.pct_change().dropna()
            else:
                # Fallback: close-to-close returns
                sp500_close = sp500_data['Close']
                sp500_daily_returns = sp500_close.pct_change().dropna()

            if isinstance(sp500_daily_returns.index, pd.DatetimeIndex) and sp500_daily_returns.index.tz is not None:
                sp500_daily_returns.index = sp500_daily_returns.index.tz_localize(None)
            
            # Convert test dates to pandas Timestamp for alignment
            test_dates_pd = pd.to_datetime(self.test_dates)
            if isinstance(test_dates_pd, pd.DatetimeIndex) and test_dates_pd.tz is not None:
                test_dates_pd = test_dates_pd.tz_localize(None)

            # Align S&P 500 returns with test dates using forward-fill
            aligned = sp500_daily_returns.reindex(test_dates_pd, method="ffill").fillna(0.0)
            sp500_returns = aligned.to_numpy()
            
            # Save to cache
            self._save_sp500_cache(sp500_returns, min_date, max_date, self.data_dir)
            
            print(f"[Evaluator] S&P 500 returns fetched: {len(sp500_returns)} values, mean={np.mean(sp500_returns):.6f}, std={np.std(sp500_returns):.6f}")
            return sp500_returns
            
        except Exception as e:
            print(f"[Evaluator] Warning: Error fetching S&P 500 data: {e}")
            return None
        
    def _load_model(self):
        """Load the saved model."""
        print(f"[Evaluator] Loading model from {self.model_path}")
        
        # Initialize model
        self.model = LSTMModel()
        self.model = self.model.to(self.device)
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.model.eval()
        
        print("[Evaluator] Model loaded successfully")
        
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array in a device-agnostic way."""
        return tensor.detach().cpu().numpy()

    def _group_indices_by_date(self, dates: List) -> Tuple[Dict, List]:
        """Group indices by date and return sorted unique dates."""
        groups = defaultdict(list)
        for idx, date in enumerate(dates):
            groups[date].append(idx)
        return groups, sorted(groups.keys())

    def _map_first_occurrence_returns(self, dates: List, daily_returns_array: np.ndarray) -> Dict:
        """Map each date to the first corresponding return value."""
        mapped = {}
        for date, value in zip(pd.to_datetime(dates), daily_returns_array):
            date_key = date.date() if hasattr(date, 'date') else pd.Timestamp(date).date()
            if date_key not in mapped:
                mapped[date_key] = value
        return mapped

    def _sign_arrays(self, predictions: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return sign arrays for predictions and targets."""
        return np.sign(predictions), np.sign(targets)

    def _wealth_curve(self, returns: np.ndarray) -> np.ndarray:
        """Compute cumulative wealth curve from returns."""
        returns_array = np.asarray(returns, dtype=float)
        if returns_array.size == 0:
            return np.array([])
        return np.cumprod(1.0 + returns_array)

    def _flatten_metric_dict(self, data: Dict, prefix: str = "") -> Dict[str, float]:
        """Flatten nested metric dictionaries, keeping only scalar-like values."""
        flat: Dict[str, float] = {}

        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else str(key)

            if isinstance(value, dict):
                flat.update(self._flatten_metric_dict(value, full_key))
            elif value is None:
                continue
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                if value.size == 1:
                    flat[full_key] = float(value.item())
            elif isinstance(value, (np.integer, int, np.floating, float)):
                flat[full_key] = float(value)
            elif isinstance(value, (np.bool_, bool)):
                flat[full_key] = float(value)

        return flat
        
    def predict(self, batch_size: int = 32, return_raw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for the test set.
        
        Args:
            batch_size: Batch size for prediction
            return_raw: If True, return raw predictions (for ranking). If False, return thresholded predictions.
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        self.model.eval()
        predictions = []
        targets = []
        
        # Create data loader for batching
        dataset = torch.utils.data.TensorDataset(self.X_test, self.Y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print("[Evaluator] Generating predictions...")
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(dataloader, desc="Predicting"):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Get predictions
                Y_pred = self.model(X_batch)
                
                if return_raw:
                    # Return raw predictions for ranking (paper methodology)
                    predictions.extend(self._tensor_to_numpy(Y_pred.squeeze()))
                else:
                    # Legacy thresholding behavior
                    abs_pred = torch.abs(Y_pred)
                    Y_pred_thresholded = torch.where(
                        abs_pred >= 0.1,
                        torch.sign(Y_pred),
                        torch.zeros_like(Y_pred)
                    )
                    predictions.extend(self._tensor_to_numpy(Y_pred_thresholded.squeeze()))
                
                targets.extend(self._tensor_to_numpy(Y_batch.squeeze()))
        
        return np.array(predictions), np.array(targets)
    
    def construct_portfolio_returns(self, 
                                     raw_predictions: np.ndarray,
                                     returns: np.ndarray,
                                     dates: List,
                                     k: int = 10,
                                     cost_bps_per_side: float = 5.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Construct daily k-long/k-short portfolios following Fischer-Krauss/Ghosh methodology.
        
        For each day:
        1. Rank all stocks by predicted probability (raw_predictions)
        2. Select top-k (long) and flop-k (short)
        3. Equal-weight positions
        4. Apply transaction costs
        
        Args:
            raw_predictions: Raw model outputs (for ranking)
            returns: Actual returns (Close - Open) / Open for each stock
            dates: List of dates corresponding to each prediction
            k: Number of stocks in long/short legs
            cost_bps_per_side: Transaction cost in basis points per half-turn (default: 5 bps)
            
        Returns:
            Tuple of (portfolio_returns, traded_indices_mask, portfolio_info_dict)
        """
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "pred": np.asarray(raw_predictions),
            "ret": np.asarray(returns)
        })
        df["row_id"] = np.arange(len(df))

        cost_per_side = cost_bps_per_side / 10000.0  # Convert bps to decimal
        fixed_daily_cost = 4.0 * cost_per_side  # 4 half-turns for full rebalance

        portfolio_returns = []
        traded_mask = np.zeros(len(df), dtype=bool)
        daily_info = []
        unique_dates = []
        long_leg_returns = []
        short_leg_returns = []
        gross_returns = []
        net_returns = []
        daily_costs = []
        traded_days = []

        for dt, group in df.groupby("date", sort=True):
            unique_dates.append(dt)
            n_rows = len(group)

            if n_rows < 2 * k:
                portfolio_returns.append(0.0)
                long_leg_returns.append(0.0)
                short_leg_returns.append(0.0)
                gross_returns.append(0.0)
                net_returns.append(0.0)
                daily_costs.append(0.0)
                traded_days.append(False)
                daily_info.append({
                    "date": dt,
                    "n_stocks": n_rows,
                    "top_k": [],
                    "flop_k": [],
                    "long_return": 0.0,
                    "short_return": 0.0,
                    "portfolio_return": 0.0,
                    "portfolio_return_after_cost": 0.0,
                    "portfolio_return_before_cost": 0.0,
                    "cost": 0.0,
                    "traded": False
                })
                continue

            top = group.nlargest(k, "pred")
            flop = group.nsmallest(k, "pred")

            long_return = float(top["ret"].mean()) if not top.empty else 0.0
            short_return = float(-flop["ret"].mean()) if not flop.empty else 0.0
            gross_return = long_return + short_return
            net_return = gross_return - fixed_daily_cost

            portfolio_returns.append(net_return)
            long_leg_returns.append(long_return)
            short_leg_returns.append(short_return)
            gross_returns.append(gross_return)
            net_returns.append(net_return)
            daily_costs.append(fixed_daily_cost)
            traded_days.append(True)
            traded_mask[top["row_id"].to_numpy()] = True
            traded_mask[flop["row_id"].to_numpy()] = True

            daily_info.append({
                "date": dt,
                "n_stocks": n_rows,
                "top_k": top["row_id"].astype(int).tolist(),
                "flop_k": flop["row_id"].astype(int).tolist(),
                "long_return": long_return,
                "short_return": short_return,
                "portfolio_return": net_return,
                "portfolio_return_after_cost": net_return,
                "portfolio_return_before_cost": gross_return,
                "cost": fixed_daily_cost,
                "traded": True
            })

        portfolio_returns = np.asarray(portfolio_returns, dtype=float)
        long_leg_returns = np.asarray(long_leg_returns, dtype=float)
        short_leg_returns = np.asarray(short_leg_returns, dtype=float)
        gross_returns = np.asarray(gross_returns, dtype=float)
        net_returns = np.asarray(net_returns, dtype=float)
        daily_costs = np.asarray(daily_costs, dtype=float)
        traded_days = np.asarray(traded_days, dtype=bool)
        portfolio_info = {
            "daily_info": daily_info,
            "unique_dates": unique_dates,
            "traded_mask": traded_mask,
            "long_leg_returns": long_leg_returns,
            "short_leg_returns": short_leg_returns,
            "gross_returns": gross_returns,
            "net_returns": net_returns,
            "daily_costs": daily_costs,
            "traded_days_mask": traded_days
        }

        return portfolio_returns, traded_mask, portfolio_info
    
    def _build_cross_sectional_frame(self,
                                     raw_predictions: np.ndarray,
                                     paper_targets: np.ndarray,
                                     returns: np.ndarray,
                                     dates: List,
                                     additional_metadata: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """Build a tidy DataFrame containing per-sample cross-sectional data."""
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "raw_score": np.asarray(raw_predictions, dtype=float),
            "ret": np.asarray(returns, dtype=float),
        })
        targets_array = np.asarray(paper_targets, dtype=float)
        binary_labels = (targets_array > 0).astype(int)
        df["y_true"] = binary_labels
        df["y_score"] = expit(df["raw_score"])
        if additional_metadata:
            for key, value in additional_metadata.items():
                if value is not None and len(value) == len(df):
                    df[key] = value
        return df

    def _safe_auc(self, y_true: pd.Series, y_score: pd.Series) -> float:
        """Compute AUC but return NaN when both classes are not present."""
        if y_true.nunique() < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)

    def _safe_log_loss(self, y_true: pd.Series, y_score: pd.Series) -> float:
        """Compute log loss with clipping, return NaN if not defined."""
        if y_true.nunique() < 2:
            return np.nan
        clipped = np.clip(y_score, 1e-6, 1 - 1e-6)
        return log_loss(y_true, clipped)

    def _safe_brier(self, y_true: pd.Series, y_score: pd.Series) -> float:
        """Compute Brier score, return NaN if not defined."""
        if y_true.nunique() < 2:
            return np.nan
        return brier_score_loss(y_true, y_score)

    def _topk_hit_rate(self, scores: pd.Series, labels: pd.Series, k: int) -> float:
        """Fraction of positives in the top-k ranked by score."""
        if len(scores) == 0:
            return np.nan
        k = max(1, min(k, len(scores)))
        top_idx = np.argsort(-scores.to_numpy())[:k]
        return float(labels.iloc[top_idx].mean())

    def _topk_long_short(self, scores: pd.Series, returns: pd.Series, k: int) -> float:
        """Long top-k, short bottom-k spread."""
        if len(scores) == 0:
            return np.nan
        k = max(1, min(k, len(scores) // 2 if len(scores) >= 2 else 1))
        score_array = scores.to_numpy()
        ret_array = returns.to_numpy()
        top_idx = np.argsort(-score_array)[:k]
        bottom_idx = np.argsort(score_array)[:k]
        long_ret = ret_array[top_idx].mean() if len(top_idx) else np.nan
        short_ret = ret_array[bottom_idx].mean() if len(bottom_idx) else np.nan
        if np.isnan(long_ret) or np.isnan(short_ret):
            return np.nan
        return float(long_ret - short_ret)

    def _fp_fn_counts(self, scores: pd.Series, labels: pd.Series, threshold: float = 0.5) -> Dict[str, int]:
        """Return FP/FN counts given continuous scores."""
        preds = (scores >= threshold).astype(int)
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return {"fp": fp, "fn": fn}

    def _serialize_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """Convert dataframe to list of dicts (JSON friendly)."""
        if df is None:
            return []
        reset = df.reset_index()
        reset.columns = [str(c) for c in reset.columns]
        reset = reset.replace({np.nan: None})
        for col in reset.columns:
            if pd.api.types.is_datetime64_any_dtype(reset[col]):
                reset[col] = reset[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
            else:
                reset[col] = reset[col].apply(
                    lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime)) else x
                )
        return reset.to_dict(orient='records')

    def _compute_cross_sectional_diagnostics(self,
                                             raw_predictions: np.ndarray,
                                             paper_targets: np.ndarray,
                                             returns: np.ndarray,
                                             dates: List,
                                             k: int) -> Dict[str, any]:
        """Compute daily cross-sectional diagnostics and supporting artifacts."""
        cross_sectional_df = self._build_cross_sectional_frame(raw_predictions, paper_targets, returns, dates)

        grouped = cross_sectional_df.groupby("date", sort=True)

        def _spearman_ic(group: pd.DataFrame) -> float:
            if group["raw_score"].nunique() < 2 or group["ret"].nunique() < 2:
                return np.nan
            return group["raw_score"].corr(group["ret"], method="spearman")

        def _apply(grouped_obj, func):
            try:
                return grouped_obj.apply(func, include_groups=False)
            except TypeError:
                return grouped_obj.apply(func)

        daily_metrics = pd.DataFrame({
            "auc": _apply(grouped, lambda g: self._safe_auc(g["y_true"], g["y_score"])),
            "ic": _apply(grouped, _spearman_ic),
            "logloss": _apply(grouped, lambda g: self._safe_log_loss(g["y_true"], g["y_score"])),
            "brier": _apply(grouped, lambda g: self._safe_brier(g["y_true"], g["y_score"])),
            "topk_hit_rate": _apply(grouped, lambda g: self._topk_hit_rate(g["y_score"], g["y_true"], k)),
            "long_short_spread": _apply(grouped, lambda g: self._topk_long_short(g["y_score"], g["ret"], k)),
            "n": grouped.size()
        })

        daily_metrics["difficulty"] = -daily_metrics["ic"]
        daily_metrics["ic_rolling_mean"] = daily_metrics["ic"].rolling(window=10, min_periods=3).mean()
        daily_metrics["ic_rolling_std"] = daily_metrics["ic"].rolling(window=10, min_periods=3).std()

        hardest_days = daily_metrics.sort_values("difficulty", ascending=False).head(20)

        error_counts = _apply(grouped, lambda g: pd.Series(self._fp_fn_counts(g["y_score"], g["y_true"])))

        calendar_data = daily_metrics.copy()
        calendar_data["weekday"] = calendar_data.index.weekday
        calendar_data["iso_week"] = calendar_data.index.isocalendar().week.astype(int)
        calendar_pivot = calendar_data.pivot_table(index="iso_week", columns="weekday", values="ic", aggfunc="mean")

        week_hour_heatmap = None
        if "hour" in cross_sectional_df.columns:
            wh = cross_sectional_df.assign(weekday=cross_sectional_df["date"].dt.dayofweek)
            week_hour_heatmap = wh.groupby(["weekday", "hour"]).apply(
                lambda g: _spearman_ic(g)
            ).unstack("hour")

        diagnostics_serialized = {
            "daily_metrics": self._serialize_dataframe(daily_metrics),
            "hardest_days": self._serialize_dataframe(hardest_days),
            "error_counts": self._serialize_dataframe(error_counts),
            "calendar_heatmap": self._serialize_dataframe(calendar_pivot),
        }
        if week_hour_heatmap is not None:
            diagnostics_serialized["weekday_hour_ic"] = self._serialize_dataframe(week_hour_heatmap)

        return {
            "frame": cross_sectional_df,
            "daily_metrics": daily_metrics,
            "hardest_days": hardest_days,
            "error_counts": error_counts,
            "calendar_matrix": calendar_pivot,
            "weekday_hour": week_hour_heatmap,
            "summary": diagnostics_serialized
        }

    def _newey_west_variance(self, series: np.ndarray, max_lags: int = None) -> float:
        """
        Compute Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) variance.
        
        Args:
            series: Time series data
            max_lags: Maximum number of lags (default: floor(4*(T/100)^(2/9)))
            
        Returns:
            HAC variance estimate
        """
        T = len(series)
        if T < 2:
            return np.var(series) if T > 0 else 0.0
        
        # Default max_lags following Newey-West recommendation
        if max_lags is None:
            max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
        
        max_lags = min(max_lags, T - 1)
        
        # Sample variance
        sample_var = np.var(series, ddof=1)
        
        # Autocovariances
        autocovs = []
        for lag in range(1, max_lags + 1):
            if lag < T:
                autocov = np.mean((series[lag:] - np.mean(series)) * (series[:-lag] - np.mean(series)))
                # Bartlett kernel weights
                weight = 1 - (lag / (max_lags + 1))
                autocovs.append(2 * weight * autocov)
        
        hac_var = sample_var + sum(autocovs)
        return max(hac_var, 0.0)  # Ensure non-negative
    
    def diebold_mariano_test(self, 
                             predictions: np.ndarray,
                             targets: np.ndarray,
                             loss_type: str = 'classification') -> Dict[str, float]:
        """
        Diebold-Mariano test for predictive accuracy.
        
        Following Fischer-Krauss, uses classification error (0/1 loss) when loss_type='classification',
        or squared error when loss_type='mse'.
        
        Args:
            predictions: Model predictions (binary or continuous)
            targets: True targets
            loss_type: 'classification' for 0/1 loss, 'mse' for squared error
            
        Returns:
            Dictionary with DM statistic and p-value
        """
        if loss_type == 'classification':
            # Classification error: 1 if prediction != target sign, 0 otherwise
            pred_sign = np.sign(predictions)
            target_sign = np.sign(targets)
            losses = (pred_sign != target_sign).astype(float)
        else:
            # Squared error
            losses = (predictions - targets) ** 2
        
        # Compare against naive forecast (median/mode)
        if loss_type == 'classification':
            # Naive forecast: always predict the majority class
            majority_class = scipy_stats.mode(target_sign, keepdims=True)[0][0]
            naive_losses = (majority_class != target_sign).astype(float)
        else:
            # Naive forecast: mean target
            naive_loss = np.mean((targets - np.mean(targets)) ** 2)
            naive_losses = np.full_like(losses, naive_loss)
        
        # Loss differential
        d = losses - naive_losses
        d_bar = np.mean(d)
        
        if len(d) < 2:
            return {'dm_statistic': 0.0, 'p_value': 1.0, 'mean_loss_diff': d_bar}
        
        # HAC variance
        var_hac = self._newey_west_variance(d)
        
        if var_hac <= 0:
            return {'dm_statistic': 0.0, 'p_value': 1.0, 'mean_loss_diff': d_bar}
        
        # DM statistic
        n = len(d)
        dm_stat = d_bar / np.sqrt(var_hac / n)
        
        # Two-sided p-value
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
        
        return {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'mean_loss_diff': float(d_bar),
            'significant': p_value < 0.05
        }
    
    def pesaran_timmermann_test(self,
                                predictions: np.ndarray,
                                targets: np.ndarray) -> Dict[str, float]:
        """
        Pesaran-Timmermann test for directional accuracy.
        
        Tests whether predictions have predictive power for the direction of targets.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Dictionary with PT statistic and p-value
        """
        pred_sign = np.sign(predictions)
        target_sign = np.sign(targets)
        
        # Counts
        n = len(predictions)
        n_pred_up = np.sum(pred_sign > 0)
        n_pred_down = np.sum(pred_sign < 0)
        n_target_up = np.sum(target_sign > 0)
        n_target_down = np.sum(target_sign < 0)
        
        # Correct predictions
        n_correct = np.sum(pred_sign == target_sign)
        
        if n < 2 or n_pred_up == 0 or n_pred_down == 0 or n_target_up == 0 or n_target_down == 0:
            return {'pt_statistic': 0.0, 'p_value': 1.0, 'accuracy': n_correct / n if n > 0 else 0.0}
        
        # Expected correct predictions under independence
        p_pred_up = n_pred_up / n
        p_pred_down = n_pred_down / n
        p_target_up = n_target_up / n
        p_target_down = n_target_down / n
        
        p_correct = p_pred_up * p_target_up + p_pred_down * p_target_down
        n_correct_expected = n * p_correct
        
        # Variance
        var_n_correct = n * p_correct * (1 - p_correct)
        
        if var_n_correct <= 0:
            return {'pt_statistic': 0.0, 'p_value': 1.0, 'accuracy': n_correct / n}
        
        # PT statistic
        pt_stat = (n_correct - n_correct_expected) / np.sqrt(var_n_correct)
        
        # Two-sided p-value
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(pt_stat)))
        
        return {
            'pt_statistic': float(pt_stat),
            'p_value': float(p_value),
            'accuracy': float(n_correct / n),
            'significant': p_value < 0.05
        }
    
    def calculate_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Classification accuracy
        pred_sign, target_sign = self._sign_arrays(predictions, targets)
        correct = np.sum(pred_sign == target_sign)
        total = len(predictions)
        metrics['accuracy'] = (correct / total) * 100 if total > 0 else 0
        
        # MSE for continuous predictions
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        return metrics
        
    def calculate_directional_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics for classification.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of directional metrics
        """
        metrics = {}
        
        # Classification accuracy
        pred_sign, target_sign = self._sign_arrays(predictions, targets)
        correct = np.sum(pred_sign == target_sign)
        total_predictions = len(predictions)
        metrics['directional_accuracy'] = (correct / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Upward movement accuracy
        up_mask = target_sign > 0
        if np.sum(up_mask) > 0:
            up_accuracy = np.sum((pred_sign > 0) & up_mask) / np.sum(up_mask) * 100
            metrics['upward_accuracy'] = up_accuracy
        else:
            metrics['upward_accuracy'] = 0
        
        # Downward movement accuracy
        down_mask = target_sign < 0
        if np.sum(down_mask) > 0:
            down_accuracy = np.sum((pred_sign < 0) & down_mask) / np.sum(down_mask) * 100
            metrics['downward_accuracy'] = down_accuracy
        else:
            metrics['downward_accuracy'] = 0
        
        return metrics
        
    def calculate_portfolio_risk_metrics(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for portfolio returns (paper-aligned).
        
        Following Fischer-Krauss/Ghosh, includes:
        - Sharpe ratio
        - Sortino ratio (downside deviation)
        - VaR and CVaR (1% and 5%)
        - Maximum drawdown (positive convention on wealth curve)
        - Skewness and kurtosis
        
        Args:
            portfolio_returns: Daily portfolio returns (can include zeros)
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        returns = np.asarray(portfolio_returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        if returns.size == 0:
            return {
                'volatility_annualized': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'var_1pct': 0.0,
                'cvar_1pct': 0.0,
                'var_5pct': 0.0,
                'cvar_5pct': 0.0,
                'max_drawdown': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'mean_return': 0.0,
                'annualized_return': 0.0,
                'standard_deviation': 0.0,
                'share_positive': 0.0,
                'newey_west_std_error': 0.0,
                'newey_west_t_stat': 0.0,
                'standard_error': 0.0,
                't_stat': 0.0,
                'min_return': 0.0,
                'max_return': 0.0,
                'quantile_25': 0.0,
                'median': 0.0,
                'quantile_75': 0.0,
                'downside_deviation_annualized': 0.0,
                'downside_deviation_daily': 0.0
            }
        # Include all returns (papers don't filter zeros)
        
        # Mean and volatility
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if returns.size > 1 else 0.0
        
        metrics['mean_return'] = float(mean_return)
        metrics['annualized_return'] = float(mean_return * 252)
        metrics['volatility_annualized'] = float(std_return * np.sqrt(252)) if std_return > 0 else 0.0
        metrics['standard_deviation'] = float(std_return)
        metrics['share_positive'] = float(np.mean(returns > 0)) if returns.size > 0 else 0.0
        metrics['min_return'] = float(np.min(returns))
        metrics['max_return'] = float(np.max(returns))
        metrics['quantile_25'] = float(np.percentile(returns, 25))
        metrics['median'] = float(np.percentile(returns, 50))
        metrics['quantile_75'] = float(np.percentile(returns, 75))
        
        # Sharpe ratio (assuming zero risk-free rate)
        if std_return > 0:
            metrics['sharpe_ratio'] = float(mean_return / std_return * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if downside_returns.size > 1:
            downside_std = np.std(downside_returns, ddof=1)
            if downside_std > 0:
                metrics['sortino_ratio'] = float(mean_return / downside_std * np.sqrt(252))
                metrics['downside_deviation_daily'] = float(downside_std)
                metrics['downside_deviation_annualized'] = float(downside_std * np.sqrt(252))
            else:
                metrics['sortino_ratio'] = 0.0
                metrics['downside_deviation_daily'] = 0.0
                metrics['downside_deviation_annualized'] = 0.0
        else:
            metrics['sortino_ratio'] = 0.0
            metrics['downside_deviation_daily'] = 0.0
            metrics['downside_deviation_annualized'] = 0.0
        
        # VaR and CVaR (Value at Risk and Conditional VaR)
        if returns.size > 0:
            # VaR at 1%
            var_1pct = np.percentile(returns, 1.0)
            metrics['var_1pct'] = float(var_1pct)
            
            # CVaR at 1% (expected loss given loss exceeds VaR)
            cvar_1pct = np.mean(returns[returns <= var_1pct]) if np.any(returns <= var_1pct) else var_1pct
            metrics['cvar_1pct'] = float(cvar_1pct)
            
            # VaR at 5%
            var_5pct = np.percentile(returns, 5.0)
            metrics['var_5pct'] = float(var_5pct)
            
            # CVaR at 5%
            cvar_5pct = np.mean(returns[returns <= var_5pct]) if np.any(returns <= var_5pct) else var_5pct
            metrics['cvar_5pct'] = float(cvar_5pct)
        else:
            metrics['var_1pct'] = 0.0
            metrics['cvar_1pct'] = 0.0
            metrics['var_5pct'] = 0.0
            metrics['cvar_5pct'] = 0.0
        
        # Maximum drawdown (positive convention on wealth curve)
        # Convert returns to wealth: start with 1, multiply by (1 + return)
        wealth = self._wealth_curve(returns)
        if wealth.size > 0 and np.all(wealth > 0):
            # Running maximum (peak)
            peak = np.maximum.accumulate(wealth)
            # Drawdown: (peak - wealth) / peak (positive value)
            drawdown = (peak - wealth) / peak
            max_drawdown = np.max(drawdown)  # Maximum drawdown as positive value
            metrics['max_drawdown'] = float(max_drawdown * 100)  # Convert to percentage
        else:
            metrics['max_drawdown'] = 0.0
        
        # Skewness and kurtosis
        if returns.size > 2:
            metrics['skewness'] = float(scipy_stats.skew(returns))
            metrics['kurtosis'] = float(scipy_stats.kurtosis(returns))  # Excess kurtosis
        else:
            metrics['skewness'] = 0.0
            metrics['kurtosis'] = 0.0

        # Newey-West adjusted standard error and t-statistic of mean returns
        if returns.size > 1:
            centered_returns = returns - mean_return
            nw_variance = self._newey_west_variance(centered_returns)
            if nw_variance < 0:
                nw_variance = 0.0
            nw_std_error = np.sqrt(nw_variance / returns.size) if returns.size > 0 else 0.0
            metrics['newey_west_std_error'] = float(nw_std_error)
            metrics['newey_west_t_stat'] = float(mean_return / nw_std_error) if nw_std_error > 0 else 0.0
        else:
            metrics['newey_west_std_error'] = 0.0
            metrics['newey_west_t_stat'] = 0.0

        # Naive (iid) standard error and t-stat for reference
        if returns.size > 1 and std_return > 0:
            standard_error = std_return / np.sqrt(returns.size)
            metrics['standard_error'] = float(standard_error)
            metrics['t_stat'] = float(mean_return / standard_error) if standard_error > 0 else 0.0
        else:
            metrics['standard_error'] = 0.0
            metrics['t_stat'] = 0.0
        
        return metrics
        
    def calculate_time_series_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate time series specific metrics.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Dictionary of time series metrics
        """
        metrics = {}
        
        # Autocorrelation of residuals
        residuals = targets - predictions
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        metrics['residual_autocorrelation'] = autocorr
        
        # Directional consistency of consecutive changes
        if len(predictions) > 1 and len(targets) > 1:
            pred_diff = np.diff(predictions)
            target_diff = np.diff(targets)
            valid_mask = (target_diff != 0) | (pred_diff != 0)
            if np.any(valid_mask):
                trend_matches = np.sign(pred_diff[valid_mask]) == np.sign(target_diff[valid_mask])
                metrics['trend_direction_consistency_pct'] = float(np.mean(trend_matches) * 100)
            else:
                metrics['trend_direction_consistency_pct'] = 0.0
        else:
            metrics['trend_direction_consistency_pct'] = 0.0

        # Relative slope error (still useful but only when informative)
        if len(predictions) > 1 and len(targets) > 1:
            x = np.arange(len(predictions))
            pred_slope = np.polyfit(x, predictions, 1)[0]
            true_slope = np.polyfit(x, targets, 1)[0]
            if true_slope != 0:
                metrics['slope_error_pct'] = float(abs(pred_slope - true_slope) / abs(true_slope) * 100)
            else:
                metrics['slope_error_pct'] = None
        else:
            metrics['slope_error_pct'] = None
        
        return metrics
    
    def generate_random_portfolio_benchmark(self,
                                            returns: np.ndarray,
                                            dates: List,
                                            k: int = 10,
                                            cost_bps_per_side: float = 5.0,
                                            n_portfolios: int = 100000,
                                            seed: int = 42) -> Tuple[np.ndarray, Dict]:
        """
        Generate random k-long/k-short portfolio benchmark (Fischer-Krauss style).
        
        Creates n_portfolios random portfolios by randomly selecting k stocks for long
        and k stocks for short each day, then averages their returns and computes distribution stats.
        
        Args:
            returns: Actual returns (Close - Open) / Open for each stock
            dates: List of dates corresponding to each revenue
            k: Number of stocks in long/short legs
            cost_bps_per_side: Transaction cost in basis points per half-turn
            n_portfolios: Number of random portfolios to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (average_random_portfolio_returns, benchmark_info_dict with distribution stats)
        """
        np.random.seed(seed)
        
        # Group by date
        date_groups, unique_dates = self._group_indices_by_date(dates)
        cost_per_side = cost_bps_per_side / 10000.0
        fixed_daily_cost = 4.0 * cost_per_side

        all_portfolio_returns = []
        for date in unique_dates:
            idxs = date_groups[date]
            if len(idxs) < 2 * k:
                all_portfolio_returns.append(np.zeros(n_portfolios))
                continue

            r = returns[idxs]
            m = len(r)

            rand_order = np.argsort(np.random.rand(n_portfolios, m), axis=1)
            picks = rand_order[:, :2 * k]
            picked_returns = np.take(r, picks)

            long_mean = picked_returns[:, :k].mean(axis=1)
            short_mean = -picked_returns[:, k:].mean(axis=1)
            portfolio = long_mean + short_mean - fixed_daily_cost
            all_portfolio_returns.append(portfolio)

        if all_portfolio_returns:
            stacked = np.vstack(all_portfolio_returns)
            average_returns = stacked.mean(axis=1)
            percentile_5 = np.percentile(stacked, 5, axis=1)
            percentile_50 = np.percentile(stacked, 50, axis=1)
            percentile_95 = np.percentile(stacked, 95, axis=1)
        else:
            stacked = None
            average_returns = np.array([])
            percentile_5 = percentile_50 = percentile_95 = np.array([])

        benchmark_info = {
            'n_portfolios': n_portfolios,
            'unique_dates': unique_dates,
            'method': 'random_k_long_short',
            'percentile_5': percentile_5,
            'percentile_50': percentile_50,
            'percentile_95': percentile_95,
            'all_returns': stacked
        }

        return average_returns, benchmark_info

    def evaluate_paper_aligned_metrics(self,
                                       k: int = 10,
                                       cost_bps_per_side: float = 5.0,
                                       batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model using Fischer-Krauss/Ghosh methodology (paper-aligned).
        
        This method implements the exact evaluation protocol from the papers:
        1. Portfolio construction: top-k/flop-k ranking
        2. Transaction costs: 5 bps per half-turn (fixed formula: 4 * cost_per_side)
        3. Accuracy on traded set with DM/PT tests (using paper labels)
        4. Comprehensive risk metrics
        5. Random benchmark: random k-long/k-short portfolios
        
        Args:
            k: Number of stocks in long/short legs (default: 10)
            cost_bps_per_side: Transaction cost in basis points per side (default: 5.0)
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary of paper-aligned metrics
        """
        print("[Evaluator] Running paper-aligned evaluation (Fischer-Krauss/Ghosh methodology)...")
        
        # Get raw predictions (for ranking)
        raw_predictions, paper_targets = self.predict(batch_size=batch_size, return_raw=True)
        
        # Use pre-computed returns from util
        returns = self.test_returns
        
        # Construct portfolios
        print(f"[Evaluator] Constructing {k}-long/{k}-short portfolios...")
        portfolio_returns, traded_mask, portfolio_info = self.construct_portfolio_returns(
            raw_predictions, returns, self.test_dates,
            k=k, cost_bps_per_side=cost_bps_per_side
        )
        
        # Calculate accuracy on traded set only (using paper labels)
        traded_predictions = raw_predictions[traded_mask]
        traded_paper_targets = paper_targets[traded_mask]
        
        # Convert raw predictions to binary for accuracy calculation
        # Use sign of raw prediction
        pred_sign = np.sign(traded_predictions)
        target_sign = np.sign(traded_paper_targets)
        accuracy_traded = np.mean(pred_sign == target_sign) * 100 if len(traded_predictions) > 0 else 0.0
        
        # Statistical tests on traded set (using paper labels)
        print("[Evaluator] Running statistical tests...")
        dm_test = self.diebold_mariano_test(traded_predictions, traded_paper_targets, loss_type='classification')
        pt_test = self.pesaran_timmermann_test(traded_predictions, traded_paper_targets)
        
        # Portfolio risk metrics
        print("[Evaluator] Calculating portfolio risk metrics...")
        risk_metrics = self.calculate_portfolio_risk_metrics(portfolio_returns)
        gross_returns = portfolio_info.get('gross_returns')
        daily_costs = portfolio_info.get('daily_costs')
        if daily_costs is None:
            daily_costs = np.full_like(portfolio_returns, fill_value=(4.0 * cost_bps_per_side / 10000.0))
        if gross_returns is None:
            gross_returns = portfolio_returns + daily_costs
        traded_days_mask = portfolio_info.get('traded_days_mask')
        if traded_days_mask is None:
            traded_days_mask = np.ones_like(portfolio_returns, dtype=bool)

        gross_risk_metrics = self.calculate_portfolio_risk_metrics(gross_returns)
        long_leg_returns = portfolio_info.get('long_leg_returns')
        short_leg_returns = portfolio_info.get('short_leg_returns')
        if long_leg_returns is None:
            long_leg_returns = np.zeros_like(portfolio_returns)
        if short_leg_returns is None:
            short_leg_returns = np.zeros_like(portfolio_returns)

        traded_long_returns = long_leg_returns[traded_days_mask]
        traded_short_returns = short_leg_returns[traded_days_mask]
        traded_gross_returns = gross_returns[traded_days_mask]
        traded_costs = daily_costs[traded_days_mask]

        mean_long_return = float(np.mean(traded_long_returns)) if traded_long_returns.size > 0 else 0.0
        mean_short_return = float(np.mean(traded_short_returns)) if traded_short_returns.size > 0 else 0.0
        mean_gross_return = float(np.mean(traded_gross_returns)) if traded_gross_returns.size > 0 else float(np.mean(gross_returns)) if np.size(gross_returns) > 0 else 0.0
        annualized_gross_return = mean_gross_return * 252
        gross_share_positive = float(np.mean(traded_gross_returns > 0)) if traded_gross_returns.size > 0 else float(np.mean(gross_returns > 0)) if np.size(gross_returns) > 0 else 0.0

        mean_daily_cost = float(np.mean(traded_costs)) if traded_costs.size > 0 else float(np.mean(daily_costs)) if np.size(daily_costs) > 0 else 0.0
        total_cost = float(np.sum(traded_costs)) if traded_costs.size > 0 else float(np.sum(daily_costs)) if np.size(daily_costs) > 0 else 0.0
        annualized_cost = mean_daily_cost * 252

        leg_distribution = {
            'mean_long_return': mean_long_return,
            'mean_short_return': mean_short_return,
            'median_long_return': float(np.median(traded_long_returns)) if traded_long_returns.size > 0 else 0.0,
            'median_short_return': float(np.median(traded_short_returns)) if traded_short_returns.size > 0 else 0.0,
            'std_long_return': float(np.std(traded_long_returns, ddof=1)) if traded_long_returns.size > 1 else 0.0,
            'std_short_return': float(np.std(traded_short_returns, ddof=1)) if traded_short_returns.size > 1 else 0.0,
            'share_positive_long': float(np.mean(traded_long_returns > 0)) if traded_long_returns.size > 0 else 0.0,
            'share_positive_short': float(np.mean(traded_short_returns > 0)) if traded_short_returns.size > 0 else 0.0
        }

        net_distribution = {
            'mean_return_after_cost': risk_metrics['mean_return'],
            'annualized_return_after_cost': risk_metrics['annualized_return'],
            'std_return_after_cost': risk_metrics['standard_deviation'],
            'share_positive_after_cost': risk_metrics['share_positive'],
            'min_return_after_cost': risk_metrics['min_return'],
            'quantile_25_after_cost': risk_metrics['quantile_25'],
            'median_return_after_cost': risk_metrics['median'],
            'quantile_75_after_cost': risk_metrics['quantile_75'],
            'max_return_after_cost': risk_metrics['max_return'],
            'newey_west_std_error_after_cost': risk_metrics['newey_west_std_error'],
            'newey_west_t_stat_after_cost': risk_metrics['newey_west_t_stat'],
            'standard_error_after_cost': risk_metrics['standard_error'],
            't_stat_after_cost': risk_metrics['t_stat']
        }

        gross_distribution = {
            'mean_return_before_cost': mean_gross_return,
            'annualized_return_before_cost': annualized_gross_return,
            'std_return_before_cost': gross_risk_metrics['standard_deviation'],
            'share_positive_before_cost': gross_share_positive,
            'min_return_before_cost': gross_risk_metrics['min_return'],
            'quantile_25_before_cost': gross_risk_metrics['quantile_25'],
            'median_return_before_cost': gross_risk_metrics['median'],
            'quantile_75_before_cost': gross_risk_metrics['quantile_75'],
            'max_return_before_cost': gross_risk_metrics['max_return'],
            'newey_west_std_error_before_cost': gross_risk_metrics['newey_west_std_error'],
            'newey_west_t_stat_before_cost': gross_risk_metrics['newey_west_t_stat'],
            'standard_error_before_cost': gross_risk_metrics['standard_error'],
            't_stat_before_cost': gross_risk_metrics['t_stat']
        }
        
        # S&P 500 benchmark comparison (aligned by date)
        sp500_returns = self._get_sp500_returns()
        sp500_metrics = {}
        if sp500_returns is not None and len(portfolio_returns) > 0:
            unique_dates = portfolio_info['unique_dates']

            # Align S&P 500 returns by date (not just length)
            date_to_sp500 = {}
            if len(sp500_returns) == len(self.test_dates):
                date_to_sp500 = self._map_first_occurrence_returns(self.test_dates, sp500_returns)

            sp500_aligned = []
            missing_dates = []
            for date in unique_dates:
                date_key = date.date() if hasattr(date, 'date') else pd.Timestamp(date).date()
                value = date_to_sp500.get(date_key)
                if value is None:
                    missing_dates.append(date_key)
                    value = 0.0
                sp500_aligned.append(value)
            
            if missing_dates:
                print(f"[Evaluator] Warning: {len(missing_dates)} portfolio dates not found in S&P 500 data (using 0.0)")
            
            sp500_aligned = np.array(sp500_aligned)
            
            if len(sp500_aligned) == len(portfolio_returns):
                sp500_risk = self.calculate_portfolio_risk_metrics(sp500_aligned)
                sp500_metrics = {
                    'sp500_annualized_return': sp500_risk['annualized_return'],
                    'sp500_volatility': sp500_risk['volatility_annualized'],
                    'sp500_sharpe': sp500_risk['sharpe_ratio'],
                    'sp500_max_drawdown': sp500_risk['max_drawdown'],
                    'sp500_distribution': {
                        'std_return': sp500_risk['standard_deviation'],
                        'share_positive': sp500_risk['share_positive'],
                        'min_return': sp500_risk['min_return'],
                        'quantile_25': sp500_risk['quantile_25'],
                        'median': sp500_risk['median'],
                        'quantile_75': sp500_risk['quantile_75'],
                        'max_return': sp500_risk['max_return'],
                        'newey_west_std_error': sp500_risk['newey_west_std_error'],
                        'newey_west_t_stat': sp500_risk['newey_west_t_stat']
                    },
                    'excess_return_vs_sp500': risk_metrics['annualized_return'] - sp500_risk['annualized_return'],
                    'excess_sharpe_vs_sp500': risk_metrics['sharpe_ratio'] - sp500_risk['sharpe_ratio']
                }
                self.sp500_returns = sp500_aligned
            else:
                print(f"[Evaluator] Warning: S&P 500 alignment failed (expected {len(portfolio_returns)}, got {len(sp500_aligned)})")
        
        # Random benchmark
        # Note: Using fewer portfolios for computational efficiency
        # Papers use 100,000 but average over all days, which is very slow
        # For practical purposes, we use a smaller number (can be increased for accuracy)
        print("[Evaluator] Generating random portfolio benchmark...")
        n_random_portfolios = 1000  # Reduced for efficiency (papers use 100,000)
        random_portfolio_returns, random_info = self.generate_random_portfolio_benchmark(
            self.test_returns, self.test_dates, k=k, cost_bps_per_side=cost_bps_per_side,
            n_portfolios=n_random_portfolios, seed=42
        )
        
        random_metrics = {}
        if len(random_portfolio_returns) > 0 and len(random_portfolio_returns) == len(portfolio_returns):
            random_risk = self.calculate_portfolio_risk_metrics(random_portfolio_returns)
            
            # Compute percentile rank of strategy's mean return in random distribution
            percentile_rank = None
            if random_info.get('all_returns') is not None:
                # Compute mean return across all days for strategy
                strategy_mean_return = np.mean(portfolio_returns)
                # Compute mean return for each random portfolio (across all days)
                random_means = np.mean(random_info['all_returns'], axis=0)  # Shape: (n_portfolios,)
                # Percentile rank: what % of random portfolios have mean <= strategy mean
                percentile_rank = (random_means <= strategy_mean_return).mean() * 100
            
            random_metrics = {
                'random_annualized_return': random_risk['annualized_return'],
                'random_sharpe': random_risk['sharpe_ratio'],
                'random_max_drawdown': random_risk['max_drawdown'],
                'random_distribution': {
                    'std_return': random_risk['standard_deviation'],
                    'share_positive': random_risk['share_positive'],
                    'min_return': random_risk['min_return'],
                    'quantile_25': random_risk['quantile_25'],
                    'median': random_risk['median'],
                    'quantile_75': random_risk['quantile_75'],
                    'max_return': random_risk['max_return'],
                    'newey_west_std_error': random_risk['newey_west_std_error'],
                    'newey_west_t_stat': random_risk['newey_west_t_stat']
                },
                'outperformance_vs_random': risk_metrics['annualized_return'] - random_risk['annualized_return'],
                'percentile_rank_in_random_distribution': percentile_rank,
                'random_percentile_5_annualized': np.mean(random_info.get('percentile_5', [])) * 252 if len(random_info.get('percentile_5', [])) > 0 else None,
                'random_percentile_50_annualized': np.mean(random_info.get('percentile_50', [])) * 252 if len(random_info.get('percentile_50', [])) > 0 else None,
                'random_percentile_95_annualized': np.mean(random_info.get('percentile_95', [])) * 252 if len(random_info.get('percentile_95', [])) > 0 else None
            }
        
        # Compile results
        paper_metrics = {
            'portfolio_performance': {
                'mean_daily_return': risk_metrics['mean_return'],
                'annualized_return': risk_metrics['annualized_return'],
                'volatility_annualized': risk_metrics['volatility_annualized'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'sortino_ratio': risk_metrics['sortino_ratio'],
                'max_drawdown_pct': risk_metrics['max_drawdown'],
                'var_1pct': risk_metrics['var_1pct'],
                'cvar_1pct': risk_metrics['cvar_1pct'],
                'var_5pct': risk_metrics['var_5pct'],
                'cvar_5pct': risk_metrics['cvar_5pct'],
                'skewness': risk_metrics['skewness'],
                'kurtosis': risk_metrics['kurtosis'],
                'standard_deviation': risk_metrics['standard_deviation'],
                'share_positive': risk_metrics['share_positive'],
                'min_return': risk_metrics['min_return'],
                'quantile_25': risk_metrics['quantile_25'],
                'median_return': risk_metrics['median'],
                'quantile_75': risk_metrics['quantile_75'],
                'max_return': risk_metrics['max_return'],
                'downside_deviation_daily': risk_metrics['downside_deviation_daily'],
                'downside_deviation_annualized': risk_metrics['downside_deviation_annualized'],
                'newey_west_std_error': risk_metrics['newey_west_std_error'],
                'newey_west_t_stat': risk_metrics['newey_west_t_stat'],
                'standard_error': risk_metrics['standard_error'],
                't_stat': risk_metrics['t_stat'],
                'mean_daily_return_before_cost': gross_distribution['mean_return_before_cost'],
                'annualized_return_before_cost': gross_distribution['annualized_return_before_cost'],
                'std_return_before_cost': gross_distribution['std_return_before_cost'],
                'share_positive_before_cost': gross_distribution['share_positive_before_cost'],
                'min_return_before_cost': gross_distribution['min_return_before_cost'],
                'quantile_25_before_cost': gross_distribution['quantile_25_before_cost'],
                'median_return_before_cost': gross_distribution['median_return_before_cost'],
                'quantile_75_before_cost': gross_distribution['quantile_75_before_cost'],
                'max_return_before_cost': gross_distribution['max_return_before_cost'],
                'newey_west_std_error_before_cost': gross_distribution['newey_west_std_error_before_cost'],
                'newey_west_t_stat_before_cost': gross_distribution['newey_west_t_stat_before_cost'],
                'standard_error_before_cost': gross_distribution['standard_error_before_cost'],
                't_stat_before_cost': gross_distribution['t_stat_before_cost'],
                'mean_long_leg_return': leg_distribution['mean_long_return'],
                'mean_short_leg_return': leg_distribution['mean_short_return'],
                'median_long_leg_return': leg_distribution['median_long_return'],
                'median_short_leg_return': leg_distribution['median_short_return'],
                'std_long_leg_return': leg_distribution['std_long_return'],
                'std_short_leg_return': leg_distribution['std_short_return'],
                'share_positive_long_leg': leg_distribution['share_positive_long'],
                'share_positive_short_leg': leg_distribution['share_positive_short'],
                'mean_daily_transaction_cost': mean_daily_cost,
                'annualized_transaction_cost': annualized_cost,
                'total_transaction_cost': total_cost,
                'traded_days': int(traded_days_mask.sum()),
                'non_traded_days': int(traded_days_mask.size - traded_days_mask.sum()),
                'n_trading_days': len(portfolio_returns),
                'k': k,
                'cost_bps_per_side': cost_bps_per_side
            },
            'portfolio_distribution': {
                **gross_distribution,
                **net_distribution,
                **leg_distribution,
                'costs': {
                    'mean_daily_cost': mean_daily_cost,
                    'annualized_cost': annualized_cost,
                    'total_cost': total_cost
                }
            },
            'before_cost_risk_metrics': gross_risk_metrics,
            'accuracy': {
                'accuracy_traded_set_pct': accuracy_traded,
                'n_traded_samples': int(np.sum(traded_mask)),
                'n_total_samples': len(raw_predictions)
            },
            'statistical_tests': {
                'dm_statistic': dm_test['dm_statistic'],
                'dm_p_value': dm_test['p_value'],
                'dm_significant': dm_test['significant'],
                'pt_statistic': pt_test['pt_statistic'],
                'pt_p_value': pt_test['p_value'],
                'pt_significant': pt_test['significant'],
                'pt_accuracy': pt_test['accuracy'] * 100
            },
            'benchmarks': {
                **sp500_metrics,
                **random_metrics
            }
        }
        
        diagnostics_payload = self._compute_cross_sectional_diagnostics(
            raw_predictions, paper_targets, returns, self.test_dates, k
        )
        paper_metrics['cross_sectional_diagnostics'] = diagnostics_payload['summary']

        self.cross_sectional_frame = diagnostics_payload['frame']
        self.cross_sectional_daily_metrics = diagnostics_payload['daily_metrics']
        self.cross_sectional_hardest_days = diagnostics_payload['hardest_days']
        self.cross_sectional_error_counts = diagnostics_payload['error_counts']
        self.cross_sectional_calendar_matrix = diagnostics_payload['calendar_matrix']
        self.cross_sectional_weekday_hour = diagnostics_payload['weekday_hour']
        self.cross_sectional_k = k

        # Store for visualization
        self.portfolio_returns = portfolio_returns
        self.portfolio_returns_before_cost = gross_returns
        self.daily_transaction_costs = daily_costs
        self.portfolio_info = portfolio_info
        
        return paper_metrics

    def calculate_real_world_metrics(self, predictions: np.ndarray, targets: np.ndarray, revenues: np.ndarray) -> Dict[str, float]:
        """
        Calculate real-world metrics for financial predictions.
        
        Args:
            predictions: Model predictions
            targets: True values
            revenues: Actual revenues (close - open) aligned with predictions
        """
        metrics = {}

        # Calculate returns based on predictions and revenues
        # Predictions and targets are already classified (thresholded to -1, 0, or +1)
        # So we can directly multiply: prediction * revenue gives the return
        pred_returns = predictions * revenues
        target_returns = targets * revenues

        # Mean return
        metrics['mean_prediction_return'] = np.mean(pred_returns) if len(pred_returns) > 0 else 0.0
        metrics['mean_target_return'] = np.mean(target_returns) if len(target_returns) > 0 else 0.0

        # Annualized return (assuming daily data, 252 trading days per year)
        metrics['annualized_prediction_return'] = metrics['mean_prediction_return'] * 252
        metrics['annualized_target_return'] = metrics['mean_target_return'] * 252

        # Excess return (prediction return - target return)
        excess_returns = pred_returns - target_returns
        metrics['mean_excess_return'] = np.mean(excess_returns) if len(excess_returns) > 0 else 0.0
        metrics['annualized_excess_return'] = metrics['mean_excess_return'] * 252

        # Share of positive returns
        if len(pred_returns) > 0:
            metrics['share_positive_prediction_returns'] = np.sum(pred_returns > 0) / len(pred_returns) * 100
        else:
            metrics['share_positive_prediction_returns'] = 0.0
            
        if len(target_returns) > 0:
            metrics['share_positive_target_returns'] = np.sum(target_returns > 0) / len(target_returns) * 100
        else:
            metrics['share_positive_target_returns'] = 0.0

        # Cumulative money growth (assuming starting with $0, cumulative sum of returns)
        if len(pred_returns) > 0:
            pred_cumulative_growth = np.cumsum(pred_returns)  # Cumulative sum
            metrics['final_prediction_growth'] = pred_cumulative_growth[-1]  # Final value
            metrics['total_prediction_return'] = pred_cumulative_growth[-1]  # Total return (absolute)
        else:
            metrics['final_prediction_growth'] = 0.0
            metrics['total_prediction_return'] = 0.0
            
        if len(target_returns) > 0:
            target_cumulative_growth = np.cumsum(target_returns)
            metrics['final_target_growth'] = target_cumulative_growth[-1]  # Final value
            metrics['total_target_return'] = target_cumulative_growth[-1]  # Total return (absolute)
        else:
            metrics['final_target_growth'] = 0.0
            metrics['total_target_return'] = 0.0

        # Random "monkey" benchmark (random walk with same volatility as target)
        if len(target_returns) > 0 and np.std(target_returns) > 0:
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.normal(np.mean(target_returns), np.std(target_returns), len(target_returns))
            random_cumulative_growth = np.cumsum(random_returns)
            
            metrics['random_benchmark_final_growth'] = random_cumulative_growth[-1]
            metrics['random_benchmark_total_return'] = random_cumulative_growth[-1]
            metrics['random_benchmark_annualized_return'] = np.mean(random_returns) * 252
            
            # Performance vs random benchmark
            metrics['outperformance_vs_random'] = metrics['final_prediction_growth'] - metrics['random_benchmark_final_growth']
        else:
            metrics['random_benchmark_final_growth'] = 0.0
            metrics['random_benchmark_total_return'] = 0.0
            metrics['random_benchmark_annualized_return'] = 0.0
            metrics['outperformance_vs_random'] = 0.0
        
        # S&P 500 benchmark comparison
        # _get_sp500_returns() handles priority: metrics data -> standalone cache -> fetch
        sp500_returns = self._get_sp500_returns()
        
        if sp500_returns is not None and len(sp500_returns) == len(pred_returns):
            # Calculate S&P 500 metrics
            metrics['sp500_mean_return'] = np.mean(sp500_returns)
            metrics['sp500_annualized_return'] = metrics['sp500_mean_return'] * 252
            metrics['sp500_volatility'] = np.std(sp500_returns) * np.sqrt(252) if len(sp500_returns) > 1 else 0.0
            metrics['sp500_sharpe'] = (metrics['sp500_mean_return'] / np.std(sp500_returns) * np.sqrt(252)) if np.std(sp500_returns) > 0 else 0.0
            
            # Cumulative S&P 500 growth
            sp500_cumulative = np.cumsum(sp500_returns)
            metrics['sp500_final_growth'] = sp500_cumulative[-1]
            metrics['sp500_total_return'] = sp500_cumulative[-1]
            
            # Performance vs S&P 500
            metrics['outperformance_vs_sp500'] = metrics['final_prediction_growth'] - metrics['sp500_final_growth']
            metrics['excess_return_vs_sp500'] = metrics['mean_prediction_return'] - metrics['sp500_mean_return']
            metrics['annualized_excess_return_vs_sp500'] = metrics['excess_return_vs_sp500'] * 252
            
            # Store for visualization
            self.sp500_returns = sp500_returns
            self.sp500_cumulative = sp500_cumulative
        else:
            metrics['sp500_mean_return'] = None
            metrics['sp500_annualized_return'] = None
            metrics['sp500_volatility'] = None
            metrics['sp500_sharpe'] = None
            metrics['sp500_final_growth'] = None
            metrics['sp500_total_return'] = None
            metrics['outperformance_vs_sp500'] = None
            metrics['excess_return_vs_sp500'] = None
            metrics['annualized_excess_return_vs_sp500'] = None
            self.sp500_returns = None
            self.sp500_cumulative = None
        
        return metrics
        
    def log_metrics_to_tensorboard(self, metrics: Dict[str, Dict[str, float]]):
        """
        Log all metrics to TensorBoard.
        
        Args:
            metrics: Nested dictionary of metric categories and values
        """
        print("[Evaluator] Logging metrics to TensorBoard...")
        
        flat_metrics = self._flatten_metric_dict(metrics)

        for metric_name, value in flat_metrics.items():
            self.writer.add_scalar(f'Evaluation/{metric_name}', value)

        sanitized_metrics = {metric_name.replace('/', '_'): value for metric_name, value in flat_metrics.items()}

        if sanitized_metrics:
            self.writer.add_hparams(
                hparam_dict={},
                metric_dict=sanitized_metrics
            )
        
        self.writer.flush()
        print("[Evaluator] Metrics logged to TensorBoard")
        
    def create_visualizations(self, predictions: np.ndarray, targets: np.ndarray, revenues: Optional[np.ndarray] = None):
        """
        Create and log visualizations to TensorBoard.
        
        Args:
            predictions: Model predictions
            targets: True values
            revenues: Actual revenues (close - open) aligned with predictions
        """
        print("[Evaluator] Creating visualizations...")
        
        # 1. Time series plot
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.test_dates, targets, label='Target', alpha=0.7, linewidth=1)
        ax.plot(self.test_dates, predictions, label='Predicted', alpha=0.7, linewidth=1)
        ax.set_title('Stock Classification Predictions vs Target Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Classification Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        self._format_date_axis(plt, self.test_dates)
        plt.tight_layout()
        
        self.writer.add_figure('Evaluation/Time_Series_Prediction', fig)
        plt.close()
        
        # 2. Scatter plot: Predictions vs Target
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('Target Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predictions vs Target Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.writer.add_figure('Evaluation/Scatter_Predictions_vs_Target', fig)
        plt.close()
        
        # 3. Residuals plot
        residuals = targets - predictions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals over time
        ax1.plot(self.test_dates, residuals, alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residuals (Target - Predicted)')
        ax1.grid(True, alpha=0.3)
        self._format_date_axis(plt, self.test_dates)
        
        # Residuals histogram
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure('Evaluation/Residuals_Analysis', fig)
        plt.close()
        
        # 4. Error distribution by time periods
        if len(self.test_dates) > 100:  # Only if we have enough data
            # Split into quarters and analyze error
            dates_array = np.array(self.test_dates)
            errors = np.abs(residuals)
            
            # Create quarterly bins
            quarters = []
            quarter_errors = []
            
            for i in range(0, len(dates_array), len(dates_array)//4):
                end_idx = min(i + len(dates_array)//4, len(dates_array))
                quarter_errors.append(np.mean(errors[i:end_idx]))
                quarters.append(f"Q{len(quarters)+1}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(quarters, quarter_errors, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Mean Absolute Error by Time Period')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Mean Absolute Error')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, error in zip(bars, quarter_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{error:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            self.writer.add_figure('Evaluation/Error_by_Time_Period', fig)
            plt.close()
        
        # 5. Model vs S&P 500 cumulative returns comparison
        if hasattr(self, 'sp500_cumulative') and self.sp500_cumulative is not None and self.test_revenues is not None:
            # Calculate model cumulative returns
            # Predictions are already classified (thresholded to -1, 0, or +1)
            # So we can directly multiply: prediction * revenue gives the return
            model_returns = predictions * self.test_revenues
            model_cumulative = np.cumsum(model_returns)
            
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(self.test_dates, model_cumulative, label='Model Strategy', alpha=0.8, linewidth=2)
            ax.plot(self.test_dates, self.sp500_cumulative, label='S&P 500', alpha=0.8, linewidth=2, linestyle='--')
            ax.set_title('Model Performance vs S&P 500 Benchmark')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._format_date_axis(plt, self.test_dates)
            plt.tight_layout()
            
            self.writer.add_figure('Evaluation/Model_vs_SP500', fig)
            plt.close()
        
        print("[Evaluator] Visualizations created and logged")
        
    def _format_date_axis(self, plt, datetime_dates):
        """Format x-axis dates with adaptive resolution based on data span."""
        if not datetime_dates:
            return
            
        # Calculate the span of dates
        min_date = min(datetime_dates)
        max_date = max(datetime_dates)
        date_span = (max_date - min_date).days
        num_points = len(datetime_dates)
        
        # Determine appropriate tick resolution based on data span
        if date_span <= 30:  # Less than 1 month
            if num_points <= 15:
                locator = mdates.DayLocator(interval=1)
                formatter = mdates.DateFormatter('%Y-%m-%d')
            else:
                locator = mdates.DayLocator(interval=2)
                formatter = mdates.DateFormatter('%m-%d')
                
        elif date_span <= 90:  # 1-3 months
            locator = mdates.WeekdayLocator(interval=1)
            formatter = mdates.DateFormatter('%m-%d')
            
        elif date_span <= 365:  # 3-12 months
            locator = mdates.MonthLocator(interval=1)
            formatter = mdates.DateFormatter('%Y-%m')
            
        elif date_span <= 730:  # 1-2 years
            locator = mdates.MonthLocator(interval=2)
            formatter = mdates.DateFormatter('%Y-%m')
            
        else:  # More than 2 years
            if date_span <= 1095:  # 3 years
                locator = mdates.MonthLocator(interval=6)
                formatter = mdates.DateFormatter('%Y-%m')
            else:
                locator = mdates.YearLocator()
                formatter = mdates.DateFormatter('%Y')
        
        # Apply the formatting
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        
        # For small datasets, show all points as ticks
        if num_points <= 20 and date_span <= 90:
            plt.xticks(datetime_dates, [d.strftime('%m-%d') for d in datetime_dates], rotation=45)
            
    def evaluate_all_metrics(self, 
                             batch_size: int = 32, 
                             create_plots: bool = True,
                             k: int = 10,
                             cost_bps_per_side: float = 5.0,
                             use_paper_aligned: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            batch_size: Batch size for prediction
            create_plots: Whether to create and log visualizations
            k: Number of stocks in long/short legs (for paper-aligned evaluation)
            cost_bps_per_side: Transaction cost in basis points per side (for paper-aligned evaluation)
            use_paper_aligned: If True, use paper-aligned evaluation (Fischer-Krauss/Ghosh methodology)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("[Evaluator] Starting comprehensive evaluation...")
        
        metrics = {}
        
        if use_paper_aligned:
            # Paper-aligned evaluation (primary)
            print("\n" + "="*60)
            print("PAPER-ALIGNED EVALUATION (Fischer-Krauss/Ghosh)")
            print("="*60)
            paper_metrics = self.evaluate_paper_aligned_metrics(
                k=k, cost_bps_per_side=cost_bps_per_side, batch_size=batch_size
            )
            metrics['paper_aligned'] = paper_metrics
            
            # Get predictions for additional diagnostics
            raw_predictions, targets = self.predict(batch_size=batch_size, return_raw=True)
            
            # Additional diagnostics (separated from paper metrics)
            print("\n" + "="*60)
            print("ADDITIONAL DIAGNOSTICS (Not in papers)")
            print("="*60)
            metrics['additional_diagnostics'] = {}
            
            # Basic metrics on all predictions
            predictions_binary = np.sign(raw_predictions)
            metrics['additional_diagnostics']['basic'] = self.calculate_basic_metrics(predictions_binary, targets)
            metrics['additional_diagnostics']['directional'] = self.calculate_directional_metrics(predictions_binary, targets)
            metrics['additional_diagnostics']['time_series'] = self.calculate_time_series_metrics(raw_predictions, targets)
            
            # Store for visualization
            self.predictions = raw_predictions
            self.targets = targets
            
        else:
            # Legacy evaluation (thresholded predictions)
            predictions, targets = self.predict(batch_size, return_raw=False)
            
            # Ensure revenues are aligned with predictions (same length)
            if len(predictions) != len(self.test_revenues):
                raise ValueError(f"Predictions ({len(predictions)}) and revenues ({len(self.test_revenues)}) must have same length")
            
            revenues = self.test_revenues[:len(predictions)]
            
            # Calculate all metrics
            metrics['basic'] = self.calculate_basic_metrics(predictions, targets)
            metrics['directional'] = self.calculate_directional_metrics(predictions, targets)
            metrics['time_series'] = self.calculate_time_series_metrics(predictions, targets)
            metrics['real_world'] = self.calculate_real_world_metrics(predictions, targets, revenues)
            
            # Store results
            self.predictions = predictions
            self.targets = targets
        
        # Store results
        self.results = metrics
        
        # Log to TensorBoard
        self.log_metrics_to_tensorboard(metrics)
        
        # Create visualizations
        if create_plots:
            if use_paper_aligned and hasattr(self, 'portfolio_returns'):
                # Visualize portfolio performance
                self._create_portfolio_visualizations()
            else:
                revenues = self.test_revenues[:len(self.predictions)] if hasattr(self, 'predictions') else None
                self.create_visualizations(self.predictions, self.targets, revenues)
        
        # Print summary
        self._print_evaluation_summary(metrics)
        
        return metrics
    
    def _create_portfolio_visualizations(self):
        """Create visualizations for portfolio-based evaluation."""
        if not hasattr(self, 'portfolio_returns') or len(self.portfolio_returns) == 0:
            return
        
        print("[Evaluator] Creating portfolio visualizations...")
        unique_dates = self.portfolio_info['unique_dates']
        
        # Portfolio cumulative returns
        portfolio_wealth = self._wealth_curve(self.portfolio_returns)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(unique_dates, portfolio_wealth, label='Portfolio Strategy', alpha=0.8, linewidth=2)
        
        # Add S&P 500 if available
        if hasattr(self, 'sp500_returns') and self.sp500_returns is not None and len(self.sp500_returns) == len(portfolio_wealth):
            sp500_wealth = self._wealth_curve(self.sp500_returns)
            ax.plot(unique_dates, sp500_wealth, label='S&P 500', alpha=0.8, linewidth=2, linestyle='--')
        
        ax.set_title('Portfolio Performance (Paper-Aligned)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Wealth (Starting at 1.0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._format_date_axis(plt, unique_dates)
        plt.tight_layout()
        
        self.writer.add_figure('Evaluation/Portfolio_Performance', fig)
        plt.close()
        
        if hasattr(self, 'cross_sectional_daily_metrics'):
            self._create_cross_sectional_visualizations()
        
        print("[Evaluator] Portfolio visualizations created")

    def _create_cross_sectional_visualizations(self):
        """Generate plots for cross-sectional diagnostics."""
        daily_metrics = getattr(self, 'cross_sectional_daily_metrics', None)
        frame = getattr(self, 'cross_sectional_frame', None)
        hardest = getattr(self, 'cross_sectional_hardest_days', None)
        error_counts = getattr(self, 'cross_sectional_error_counts', None)
        calendar_matrix = getattr(self, 'cross_sectional_calendar_matrix', None)
        weekday_hour = getattr(self, 'cross_sectional_weekday_hour', None)

        if daily_metrics is None or frame is None:
            return

        dates = daily_metrics.index

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(dates, daily_metrics["ic"], label='IC', alpha=0.7)
        if daily_metrics["ic_rolling_mean"].notna().any():
            ax.plot(dates, daily_metrics["ic_rolling_mean"], label='IC Rolling Mean (10d)', linewidth=2)
            rolling_std = daily_metrics["ic_rolling_std"].fillna(0.0)
            ax.fill_between(dates,
                            daily_metrics["ic_rolling_mean"] - 2 * rolling_std,
                            daily_metrics["ic_rolling_mean"] + 2 * rolling_std,
                            color='gray', alpha=0.2, label='±2σ')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title('Daily Information Coefficient (IC)')
        ax.set_xlabel('Date')
        ax.set_ylabel('IC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._format_date_axis(plt, list(dates))
        plt.tight_layout()
        self.writer.add_figure('Evaluation/CrossSectional_IC', fig)
        plt.close()

        if {'topk_hit_rate', 'long_short_spread'}.issubset(daily_metrics.columns):
            fig, ax1 = plt.subplots(figsize=(15, 6))
            ax1.plot(dates, daily_metrics['topk_hit_rate'], color='tab:blue', label='Top-k Hit Rate')
            ax1.set_ylabel('Hit Rate', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(dates, daily_metrics['long_short_spread'], color='tab:orange', label='Long-Short Spread')
            ax2.set_ylabel('Long-Short Spread', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax1.set_title('Top-k Diagnostics')
            ax1.set_xlabel('Date')
            self._format_date_axis(plt, list(dates))
            fig.tight_layout()
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            self.writer.add_figure('Evaluation/CrossSectional_TopK', fig)
            plt.close()

        if hardest is not None and error_counts is not None and not hardest.empty:
            hardest_idx = hardest.index
            err_subset = error_counts.loc[hardest_idx].fillna(0)
            fig, ax = plt.subplots(figsize=(15, 6))
            bottom = np.zeros(len(err_subset))
            for col, color in zip(['fp', 'fn'], ['tab:red', 'tab:green']):
                values = err_subset[col].to_numpy()
                ax.bar(range(len(err_subset)), values, bottom=bottom, label=col.upper(), color=color)
                bottom += values
            ax.set_xticks(range(len(err_subset)))
            ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in hardest_idx], rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('FP/FN Composition on Hardest Days (lowest IC)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            self.writer.add_figure('Evaluation/CrossSectional_ErrorComposition', fig)
            plt.close()

        if calendar_matrix is not None and not calendar_matrix.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(calendar_matrix.values, aspect='auto', cmap='coolwarm', interpolation='nearest')
            ax.set_title('Calendar Heatmap (Weekly x Weekday) - IC')
            ax.set_xlabel('Weekday (Mon=0)')
            ax.set_ylabel('ISO Week')
            ax.set_xticks(range(calendar_matrix.shape[1]))
            ax.set_xticklabels(calendar_matrix.columns)
            ax.set_yticks(range(calendar_matrix.shape[0]))
            ax.set_yticklabels(calendar_matrix.index)
            plt.colorbar(im, ax=ax, label='IC')
            plt.tight_layout()
            self.writer.add_figure('Evaluation/CrossSectional_Calendar', fig)
            plt.close()

        if weekday_hour is not None and not weekday_hour.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(weekday_hour.values, aspect='auto', cmap='coolwarm', interpolation='nearest')
            ax.set_title('Average IC by Weekday × Hour')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Weekday (Mon=0)')
            ax.set_xticks(range(len(weekday_hour.columns)))
            ax.set_xticklabels(weekday_hour.columns)
            ax.set_yticks(range(len(weekday_hour.index)))
            ax.set_yticklabels(weekday_hour.index)
            plt.colorbar(im, ax=ax, label='IC')
            plt.tight_layout()
            self.writer.add_figure('Evaluation/CrossSectional_WeekdayHour', fig)
            plt.close()

        if "sector" in frame.columns:
            hardest_dates = hardest.index if hardest is not None else []
            if len(hardest_dates) > 0:
                sector_df = frame[frame["date"].isin(hardest_dates) & frame["sector"].notna()]
                if not sector_df.empty:
                    pivot = sector_df.pivot_table(index=sector_df["date"].dt.strftime('%Y-%m-%d'),
                                                  columns='sector',
                                                  values='ret',
                                                  aggfunc='mean')
                    fig, ax = plt.subplots(figsize=(15, 6))
                    im = ax.imshow(pivot.values, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    ax.set_title('Sector Return Heatmap on Hardest Days')
                    ax.set_xlabel('Sector')
                    ax.set_ylabel('Date')
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_yticklabels(pivot.index)
                    plt.colorbar(im, ax=ax, label='Return')
                    plt.tight_layout()
                    self.writer.add_figure('Evaluation/CrossSectional_SectorHeatmap', fig)
                    plt.close()
        else:
            print("[Evaluator] Sector data unavailable; skipping sector heatmap.")

        
    def _print_evaluation_summary(self, metrics: Dict[str, Dict[str, float]]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for category, metric_dict in metrics.items():
            print(f"\n{category.upper()} METRICS:")
            print("-" * 30)
            for metric_name, value in metric_dict.items():
                if metric_name == 'cross_sectional_diagnostics':
                    print(f"  {metric_name:25}: summarized ({', '.join(value.keys())})")
                    continue
                if isinstance(value, float):
                    if 'accuracy' in metric_name or 'mape' in metric_name or 'smape' in metric_name:
                        print(f"  {metric_name:25}: {value:8.2f}%")
                    else:
                        print(f"  {metric_name:25}: {value:8.6f}")
                elif isinstance(value, dict):
                    print(f"  {metric_name:25}: dict[{len(value)}]")
                elif isinstance(value, list):
                    print(f"  {metric_name:25}: list[{len(value)}]")
                else:
                    print(f"  {metric_name:25}: {value}")
        
        print("\n" + "="*60)
        
    def save_results(self, filepath: str):
        """
        Save evaluation results to a file.
        
        Args:
            filepath: Path to save results (JSON format)
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                 return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create serializable results
        def convert_nested(data):
            if isinstance(data, dict):
                return {key: convert_nested(value) for key, value in data.items() if value is not None}
            return convert_numpy(data)

        serializable_results = convert_nested(self.results)
        
        # Add metadata
        output = {
            'model_path': self.model_path,
            'stocks': self.stocks,
            'time_args': self.time_args,
            'evaluation_date': datetime.now().isoformat(),
            'num_test_samples': len(self.X_test),
            'metrics': serializable_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[Evaluator] Results saved to {filepath}")
        
    def close(self):
        """Close TensorBoard writer and clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        print("[Evaluator] Evaluation completed and resources cleaned up")


class ModelComparator:
    """
    Compare multiple models using their Trainer instances.
    
    This class allows you to compare multiple models side-by-side using all
    the comprehensive metrics from the Trainer class including:
    - Trading performance metrics
    - Statistical tests
    - Time series diagnostics
    """
    
    def __init__(self, 
                 trainers: Union[Dict[str, any], List[any], set],
                 log_dir: str = "runs/comparison",
                 device: Optional[torch.device] = None):
        """
        Initialize the ModelComparator.
        
        Args:
            trainers: Dictionary mapping model names to Trainer instances,
                     List of Trainer instances, or set of Trainer instances.
                     For lists/sets, model names will be auto-generated as "model_0", "model_1", etc.
                     Example: {"model1": trainer1, "model2": trainer2}
                     or [trainer1, trainer2] or {trainer1, trainer2}
            log_dir: Directory for TensorBoard logs
            device: Device to run evaluation on (auto-detect if None)
        """
        # Convert list/set to dict if needed
        if isinstance(trainers, (list, set)):
            self.trainers = {f"model_{i}": t for i, t in enumerate(trainers)}
        else:
            self.trainers = trainers
        self.model_names = list(self.trainers.keys())
        self.log_dir = log_dir
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Results storage
        self.results = {}
        
        print(f"[Comparator] Initialized with {len(self.trainers)} models: {', '.join(self.model_names)}")
        print(f"[Comparator] Using device: {self.device}")
    
    def compare_all_metrics(self, run_evaluation: bool = True) -> Dict[str, Dict]:
        """
        Compare all metrics across all models.
        
        Args:
            run_evaluation: If True, run evaluate() on each trainer first
        
        Returns:
            Dictionary of results for each model
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON - ALL METRICS")
        print("="*80)
        
        for model_name, trainer in self.trainers.items():
            print(f"\n{'='*80}")
            print(f"Evaluating Model: {model_name}")
            print(f"{'='*80}")
            
            # Run evaluation if requested
            if run_evaluation:
                trainer.evaluate()
            
            # Collect all metrics
            model_results = {
                'trading_performance': trainer.get_trading_performance_metrics(),
                'statistical_tests': trainer.get_statistical_tests(),
                'time_series_diagnostics': trainer.get_time_series_diagnostics()
            }
            
            self.results[model_name] = model_results
        
        return self.results
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary of all models."""
        if not self.results:
            print("[Comparator] No results to compare. Run compare_all_metrics() first.")
            return
        
        print("\n" + "="*100)
        print("MODEL COMPARISON SUMMARY")
        print("="*100)
        
        # Trading Performance Metrics
        print("\n1. TRADING PERFORMANCE METRICS")
        print("-" * 100)
        trading_metrics_to_compare = [
            'accuracy', 'sharpe_ratio', 'sortino_ratio', 'annualized_return',
            'value_at_risk_1pct', 'maximum_drawdown', 'excess_return_annualized'
        ]
        
        for metric in trading_metrics_to_compare:
            print(f"\n  {metric.replace('_', ' ').title()}:")
            values = {}
            for model_name in self.model_names:
                if model_name in self.results:
                    val = self.results[model_name]['trading_performance'].get(metric)
                    if val is not None:
                        values[model_name] = val
            
            if values:
                # Find best and worst
                if 'ratio' in metric.lower() or 'accuracy' in metric.lower():
                    best_model = max(values, key=values.get)
                    worst_model = min(values, key=values.get)
                else:
                    best_model = max(values, key=values.get)
                    worst_model = min(values, key=values.get)
                
                for model_name in self.model_names:
                    if model_name in values:
                        is_best = "🏆 BEST" if model_name == best_model else ""
                        is_worst = "⚠️ WORST" if model_name == worst_model and best_model != worst_model else ""
                        marker = is_best or is_worst
                        print(f"    {model_name:20s}: {values[model_name]:12.6f} {marker}")
        
        # Statistical Tests
        print("\n2. STATISTICAL TESTS")
        print("-" * 100)
        
        for test_name in ['diebold_mariano', 'pesaran_timmermann', 'newey_west_mean_return']:
            print(f"\n  {test_name.replace('_', ' ').title()}:")
            for model_name in self.model_names:
                if model_name in self.results:
                    test_result = self.results[model_name]['statistical_tests'].get(test_name, {})
                    stat = test_result.get('statistic')
                    p_val = test_result.get('p_value')
                    if stat is not None and p_val is not None:
                        significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
                        print(f"    {model_name:20s}: stat={stat:8.4f}, p={p_val:8.4f} {significance}")
        
        # Time Series Diagnostics
        print("\n3. TIME SERIES DIAGNOSTICS")
        print("-" * 100)
        diagnostics_to_compare = ['aic', 'bic', 'ljung_box', 'adf_residuals']
        
        for diag in diagnostics_to_compare:
            print(f"\n  {diag.replace('_', ' ').title()}:")
            for model_name in self.model_names:
                if model_name in self.results:
                    diag_result = self.results[model_name]['time_series_diagnostics'].get(diag, {})
                    if isinstance(diag_result, dict):
                        p_val = diag_result.get('p_value')
                        if p_val is not None:
                            significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
                            print(f"    {model_name:20s}: p={p_val:8.4f} {significance}")
                    else:
                        val = diag_result
                        if val is not None:
                            print(f"    {model_name:20s}: {val:12.4f}")
        
        print("\n" + "="*100)
        print("Comparison complete!")
        print("="*100)
    
    def get_best_model(self, metric: str) -> Tuple[str, float]:
        """
        Get the model that performs best on a specific metric.
        
        Args:
            metric: Name of the metric (e.g., 'sharpe_ratio', 'accuracy')
        
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            return None, None
        
        values = {}
        for model_name in self.model_names:
            if model_name in self.results:
                # Try trading performance first
                val = self.results[model_name]['trading_performance'].get(metric)
                if val is not None:
                    values[model_name] = val
        
        if not values:
            return None, None
        
        best_model = max(values, key=values.get)
        return best_model, values[best_model]
    
    def compare_diebold_mariano(self, model1_name: str, model2_name: str) -> Dict:
        """
        Compare two models using the Diebold-Mariano test.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
        
        Returns:
            Dictionary with comparison results
        """
        if model1_name not in self.results or model2_name not in self.results:
            print(f"Error: One or both models not found in results")
            return None
        
        # Get predictions and targets for both models
        trainer1 = self.trainers[model1_name]
        trainer2 = self.trainers[model2_name]
        
        pred1 = np.array(trainer1.predicted_values)
        target1 = np.array(trainer1.actual_values)
        pred2 = np.array(trainer2.predicted_values)
        target2 = np.array(trainer2.actual_values)
        
        if len(pred1) != len(pred2) or len(target1) != len(target2):
            print("Error: Models have different prediction lengths")
            return None
        
        # Calculate losses
        loss1 = (pred1 - target1) ** 2
        loss2 = (pred2 - target2) ** 2
        
        # Difference in losses
        d = loss1 - loss2
        d_bar = np.mean(d)
        
        # DM test from trainer1 (it should have the _newey_west_variance method)
        var_hac = trainer1._newey_west_variance(d)
        n = len(d)
        
        if var_hac > 0:
            dm_stat = d_bar / np.sqrt(var_hac / n)
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        else:
            dm_stat = 0
            p_value = 1.0
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'mean_loss_diff': d_bar,
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'interpretation': 'Model 1 better' if d_bar < 0 and p_value < 0.05 else (
                'Model 2 better' if d_bar > 0 and p_value < 0.05 else 'Equal performance'
            )
        }
    
    def save_comparison(self, filepath: str = "model_comparison.json"):
        """
        Save comparison results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        import json
        
        if not self.results:
            print("[Comparator] No results to save. Run compare_all_metrics() first.")
            return
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        output = {
            'comparison_date': datetime.now().isoformat(),
            'models': self.model_names,
            'results': convert_numpy(self.results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[Comparator] Comparison results saved to {filepath}")
    
    def close(self):
        """Close TensorBoard writer and clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        print("[Comparator] Comparison completed and resources cleaned up")
    
    def generate_csv_panels(self, 
                           output_dir: str = "comparison_csvs",
                           run_evaluation: bool = True,
                           horizon: str = "1d",
                           k: Optional[int] = None,
                           costs_bps_per_side: Optional[float] = None,
                           test_period: Optional[str] = None,
                           baseline_model: Optional[str] = None) -> Dict[str, str]:
        """
        Generate CSV files for all 6 panels (A-F) as defined in the evaluation structure.
        
        Args:
            output_dir: Directory to save CSV files
            run_evaluation: If True, run evaluate() on each trainer first
            horizon: Prediction horizon (e.g., "1d", "5min") for Panel A
            k: Number of long/short positions for Panel B (if None, will try to infer)
            costs_bps_per_side: Trading costs in basis points per side for Panel B
            test_period: Test period identifier for metadata (if None, auto-generated)
            baseline_model: Model name to use as baseline for DM tests (if None, uses first model)
        
        Returns:
            Dictionary mapping panel names to CSV file paths
        """
        print("\n" + "="*80)
        print("GENERATING CSV PANELS (A-F)")
        print("="*80)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation if needed
        if run_evaluation or not self.results:
            self.compare_all_metrics(run_evaluation=run_evaluation)
        
        # Auto-generate test_period if not provided
        if test_period is None:
            test_period = datetime.now().strftime("%Y-%m")
        
        # Set baseline model
        if baseline_model is None and len(self.model_names) > 0:
            baseline_model = self.model_names[0]
        
        csv_paths = {}
        
        # Panel A: Forecasting metrics
        csv_paths['A'] = self._generate_panel_a(output_dir, horizon, test_period, baseline_model)
        
        # Panel B: Trading performance
        csv_paths['B'] = self._generate_panel_b(output_dir, k, costs_bps_per_side, test_period)
        
        # Panel C: Distribution diagnostics
        csv_paths['C'] = self._generate_panel_c(output_dir, k, test_period)
        
        # Panel D: Significance tests
        csv_paths['D'] = self._generate_panel_d(output_dir, horizon)
        
        # Panel E: Setup metadata
        csv_paths['E'] = self._generate_panel_e(output_dir, k, costs_bps_per_side, test_period)
        
        # Panel F: Efficiency
        csv_paths['F'] = self._generate_panel_f(output_dir)
        
        print("\n" + "="*80)
        print(f"All CSV panels generated in: {output_dir}")
        print("="*80)
        for panel, path in csv_paths.items():
            print(f"  Panel {panel}: {path}")
        
        return csv_paths
    
    def _generate_panel_a(self, output_dir: str, horizon: str, test_period: str, baseline_model: str) -> str:
        """Generate Panel A: Forecasting metrics"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            stats = result.get('statistical_tests', {})
            trainer = self.trainers[model_name]
            
            # Get number of test observations
            n = len(trainer.predicted_values) if hasattr(trainer, 'predicted_values') and len(trainer.predicted_values) > 0 else 0
            
            # Directional accuracy
            acc_dir_pct = trading.get('accuracy', 0)
            
            # PT p-value
            pt_test = stats.get('pesaran_timmermann', {})
            pt_p = pt_test.get('p_value', None) if pt_test else None
            
            # RMSE, MAE, NMSE
            if hasattr(trainer, 'residuals') and len(trainer.residuals) > 0:
                residuals = np.array(trainer.residuals)
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                
                # NMSE: Normalized MSE (MSE / variance of targets)
                if hasattr(trainer, 'actual_values') and len(trainer.actual_values) > 0:
                    targets = np.array(trainer.actual_values)
                    var_targets = np.var(targets)
                    nmse = (rmse**2) / var_targets if var_targets > 0 else 0
                else:
                    nmse = 0
            else:
                rmse = mae = nmse = None
            
            # MI (Mutual Information) - simplified approximation using correlation
            mi_bits = None
            if hasattr(trainer, 'predicted_values') and hasattr(trainer, 'actual_values'):
                preds = np.array(trainer.predicted_values)
                targets = np.array(trainer.actual_values)
                if len(preds) > 0 and len(targets) > 0:
                    # Use correlation as proxy for MI (0 to ~0.3 bits for strong correlation)
                    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
                    mi_bits = abs(corr) * 0.3 if not np.isnan(corr) else 0
            
            # DM p-val vs baseline
            dm_p = None
            if baseline_model and baseline_model != model_name and baseline_model in self.results:
                dm_result = self.compare_diebold_mariano(model_name, baseline_model)
                if dm_result:
                    dm_p = dm_result.get('p_value')
            elif model_name == baseline_model:
                # For baseline model, compare to naive forecast (from its own stats)
                dm_test = stats.get('diebold_mariano', {})
                dm_p = dm_test.get('p_value', None) if dm_test else None
            
            row = {
                'model_id': model_name,
                'horizon': horizon,
                'test_period': test_period,
                'n': n,
                'acc_dir_pct': acc_dir_pct if acc_dir_pct else None,
                'pt_p': pt_p if pt_p is not None else None,
                'rmse': rmse if rmse is not None else None,
                'mae': mae if mae is not None else None,
                'nmse': nmse if nmse is not None else None,
                'mi_bits': mi_bits if mi_bits is not None else None,
                'dm_p': dm_p if dm_p is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "A_forecast.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel A: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_b(self, output_dir: str, k: Optional[int], costs_bps_per_side: Optional[float], test_period: str) -> str:
        """Generate Panel B: Trading performance"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            trainer = self.trainers[model_name]
            
            # If k not provided, try to infer or use default
            k_val = k if k is not None else 10  # Default to 10
            
            # If costs not provided, use default or try to infer
            costs = costs_bps_per_side if costs_bps_per_side is not None else 5.0  # Default 5 bps
            
            # Mean daily return (%)
            mean_r_daily = trading.get('mean_daily_return', 0) * 100
            
            # Annualized return (%)
            ann_return = trading.get('annualized_return', 0) * 100
            
            # Annualized volatility (%)
            ann_vol = trading.get('standard_deviation_annualized', 0) * 100
            
            # Sharpe ratio
            sharpe = trading.get('sharpe_ratio', 0)
            
            # Sortino ratio
            sortino = trading.get('sortino_ratio', 0)
            
            # VaR 1% (%/day)
            var_1pct = trading.get('value_at_risk_1pct', 0) * 100
            
            # Maximum drawdown (%)
            maxdd = trading.get('maximum_drawdown', 0) * 100
            
            # % Positive days
            share_pos = trading.get('fraction_positive_returns', 0)
            
            # SE(μ_d) - Standard error of mean daily return
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                returns = np.array(trainer.real_world_returns)
            elif hasattr(trainer, 'actual_returns') and len(trainer.actual_returns) > 0:
                returns = np.array(trainer.actual_returns)
            else:
                returns = np.array([])
            
            if len(returns) > 0:
                se_mean_daily = np.std(returns, ddof=1) / np.sqrt(len(returns)) * 100
            else:
                se_mean_daily = None
            
            row = {
                'model_id': model_name,
                'k': k_val,
                'costs_bps_per_side': costs,
                'test_period': test_period,
                'mean_r_daily_pct': mean_r_daily if mean_r_daily else None,
                'ann_return_pct': ann_return if ann_return else None,
                'ann_vol_pct': ann_vol if ann_vol else None,
                'sharpe': sharpe if sharpe else None,
                'sortino': sortino if sortino else None,
                'var_1pct_daily_pct': var_1pct if var_1pct else None,
                'maxdd_pct': maxdd if maxdd else None,
                'share_pos_pct': share_pos if share_pos else None,
                'se_mean_daily_pct': se_mean_daily if se_mean_daily is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "B_trading.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel B: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_c(self, output_dir: str, k: Optional[int], test_period: str) -> str:
        """Generate Panel C: Distribution diagnostics"""
        rows = []
        
        for model_name in self.model_names:
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            trading = result.get('trading_performance', {})
            trainer = self.trainers[model_name]
            
            k_val = k if k is not None else 10
            
            # Get returns
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                returns = np.array(trainer.real_world_returns)
            elif hasattr(trainer, 'actual_returns') and len(trainer.actual_returns) > 0:
                returns = np.array(trainer.actual_returns)
            else:
                returns = np.array([])
            
            if len(returns) == 0:
                continue
            
            n_days = len(returns)
            
            # Convert to percentage
            returns_pct = returns * 100
            
            # Percentiles
            min_pct = np.min(returns_pct)
            q1_pct = np.percentile(returns_pct, 25)
            median_pct = np.median(returns_pct)
            q3_pct = np.percentile(returns_pct, 75)
            max_pct = np.max(returns_pct)
            
            # Skewness and kurtosis
            skew = trading.get('skewness', None)
            kurt = trading.get('kurtosis', None)
            
            row = {
                'model_id': model_name,
                'k': k_val,
                'test_period': test_period,
                'n_days': n_days,
                'min_pct': min_pct,
                'q1_pct': q1_pct,
                'median_pct': median_pct,
                'q3_pct': q3_pct,
                'max_pct': max_pct,
                'skew': skew if skew is not None else None,
                'kurt': kurt if kurt is not None else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "C_distribution.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel C: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_d(self, output_dir: str, horizon: str) -> str:
        """Generate Panel D: Significance tests"""
        rows = []
        
        # Generate comparisons between all pairs of models
        for i, model1_name in enumerate(self.model_names):
            for j, model2_name in enumerate(self.model_names[i+1:], start=i+1):
                # Get DM test result
                dm_result = self.compare_diebold_mariano(model1_name, model2_name)
                if not dm_result:
                    continue
                
                # Get PT stats from both models
                model1_result = self.results.get(model1_name, {})
                model2_result = self.results.get(model2_name, {})
                
                model1_stats = model1_result.get('statistical_tests', {})
                model2_stats = model2_result.get('statistical_tests', {})
                
                # Use average PT stat if both available, otherwise use one
                pt_stat = None
                pt_p = None
                
                pt1 = model1_stats.get('pesaran_timmermann', {})
                pt2 = model2_stats.get('pesaran_timmermann', {})
                
                if pt1.get('statistic') is not None and pt2.get('statistic') is not None:
                    pt_stat = (pt1['statistic'] + pt2['statistic']) / 2
                    pt_p = min(pt1.get('p_value', 1.0), pt2.get('p_value', 1.0))
                elif pt1.get('statistic') is not None:
                    pt_stat = pt1['statistic']
                    pt_p = pt1.get('p_value')
                elif pt2.get('statistic') is not None:
                    pt_stat = pt2['statistic']
                    pt_p = pt2.get('p_value')
                
                row = {
                    'comparison': f"{model1_name} vs {model2_name}",
                    'horizon': horizon,
                    'dm_stat': dm_result.get('dm_statistic'),
                    'dm_p': dm_result.get('p_value'),
                    'pt_stat': pt_stat,
                    'pt_p': pt_p
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "D_significance.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel D: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_e(self, output_dir: str, k: Optional[int], costs_bps_per_side: Optional[float], test_period: str) -> str:
        """Generate Panel E: Setup metadata"""
        rows = []
        
        for model_name in self.model_names:
            trainer = self.trainers[model_name]
            
            # Extract setup information from trainer
            universe = ", ".join(trainer.stocks) if hasattr(trainer, 'stocks') else "Unknown"
            
            # Train/test years from time_args
            train_test_str = "/".join(trainer.time_args) if hasattr(trainer, 'time_args') else "Unknown"
            
            # Rebalance frequency
            rebalance = "Daily"  # Default assumption
            
            # Rule
            k_val = k if k is not None else 10
            rule = f"Top{k_val}/Bottom{k_val}, equal-weight"
            
            # Costs
            costs = costs_bps_per_side if costs_bps_per_side is not None else 5.0
            
            # Neutrality (market-neutral)
            neutral = "Yes"
            
            # Exclusions
            exclusions = "Zero-volume"  # Default assumption
            
            # Avg names/day - try to infer from data
            avg_names_per_day = len(trainer.stocks) if hasattr(trainer, 'stocks') else None
            
            # Turnover - approximate from returns if available
            turnover_pct = None
            if hasattr(trainer, 'real_world_returns') and len(trainer.real_world_returns) > 0:
                # Rough approximation: assume daily rebalancing with k positions
                # This is a simplified estimate
                turnover_pct = (k_val * 2 / avg_names_per_day * 100) if avg_names_per_day and avg_names_per_day > 0 else None
            
            row = {
                'model_id': model_name,
                'universe': universe,
                'train_years': train_test_str.split('/')[0] if '/' in train_test_str else train_test_str,
                'test_years': train_test_str.split('/')[1] if '/' in train_test_str and len(train_test_str.split('/')) > 1 else test_period,
                'rebalance': rebalance,
                'rule': rule,
                'costs_bps_per_side': costs,
                'neutral': neutral,
                'exclusions': exclusions,
                'avg_names_per_day': avg_names_per_day if avg_names_per_day else None,
                'turnover_pct': turnover_pct if turnover_pct else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "E_setup.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel E: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _generate_panel_f(self, output_dir: str) -> str:
        """Generate Panel F: Efficiency"""
        rows = []
        
        for model_name in self.model_names:
            trainer = self.trainers[model_name]
            
            # Model parameters
            if hasattr(trainer, 'Model'):
                model = trainer.Model.module if hasattr(trainer.Model, 'module') else trainer.Model
                params_m = sum(p.numel() for p in model.parameters()) / 1e6
            else:
                params_m = None
            
            # Hardware
            if hasattr(trainer, 'device'):
                device_type = str(trainer.device)
                if 'cuda' in device_type:
                    if torch.cuda.is_available():
                        hardware = torch.cuda.get_device_name(0)
                    else:
                        hardware = "CUDA (unknown)"
                elif 'mps' in device_type:
                    hardware = "Apple Silicon"
                else:
                    hardware = "CPU"
            else:
                hardware = "Unknown"
            
            # Training time - try to get from trainer attributes
            train_hours = None
            if hasattr(trainer, 'training_time'):
                train_hours = trainer.training_time / 3600  # Convert seconds to hours
            
            # Epochs
            epochs = None
            if hasattr(trainer, 'num_epochs'):
                epochs = trainer.num_epochs
            elif hasattr(trainer, 'stopper') and hasattr(trainer.stopper, 'counter'):
                # Try to infer from early stopping
                epochs = trainer.stopper.counter if hasattr(trainer.stopper, 'min_validation_loss') else None
            
            # Decision latency and inference time - these would need to be measured
            # For now, provide placeholder values
            decision_latency_ms = None  # Would need actual measurement
            infer_seconds_per_day = None  # Would need actual measurement
            
            row = {
                'model_id': model_name,
                'params_m': params_m if params_m else None,
                'hardware': hardware,
                'train_hours': train_hours if train_hours else None,
                'epochs': epochs if epochs else None,
                'decision_latency_ms': decision_latency_ms,
                'infer_seconds_per_day': infer_seconds_per_day
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(output_dir, "F_efficiency.csv")
        df.to_csv(filepath, index=False)
        print(f"  Generated Panel F: {filepath} ({len(rows)} rows)")
        return filepath


def generate_csv_panels_from_trainers(trainers: Union[Dict[str, any], List[any], set],
                                     output_dir: str = "comparison_csvs",
                                     run_evaluation: bool = True,
                                     horizon: str = "1d",
                                     k: Optional[int] = None,
                                     costs_bps_per_side: Optional[float] = None,
                                     test_period: Optional[str] = None,
                                     baseline_model: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to generate CSV panels from a list, set, or dict of Trainer instances.
    
    This is a standalone function that creates a ModelComparator internally and generates
    all 6 CSV panels (A-F) as defined in the evaluation structure.
    
    Args:
        trainers: Dictionary mapping model names to Trainer instances,
                 List of Trainer instances, or set of Trainer instances.
                 For lists/sets, model names will be auto-generated as "model_0", "model_1", etc.
        output_dir: Directory to save CSV files
        run_evaluation: If True, run evaluate() on each trainer first
        horizon: Prediction horizon (e.g., "1d", "5min") for Panel A
        k: Number of long/short positions for Panel B (if None, defaults to 10)
        costs_bps_per_side: Trading costs in basis points per side for Panel B (default: 5.0)
        test_period: Test period identifier for metadata (if None, auto-generated)
        baseline_model: Model name to use as baseline for DM tests (if None, uses first model)
    
    Returns:
        Dictionary mapping panel names to CSV file paths
    
    Example:
        >>> from evaluator import generate_csv_panels_from_trainers
        >>> from trainer import Trainer
        >>> 
        >>> # Create trainers
        >>> trainer1 = Trainer(stocks=["AAPL", "MSFT"], time_args=["3y"])
        >>> trainer2 = Trainer(stocks=["AAPL", "MSFT"], time_args=["3y"])
        >>> 
        >>> # Generate CSVs from list
        >>> csv_paths = generate_csv_panels_from_trainers(
        ...     [trainer1, trainer2],
        ...     output_dir="results_csvs",
        ...     k=10,
        ...     costs_bps_per_side=5.0
        ... )
        >>> 
        >>> # Or from dict with custom names
        >>> csv_paths = generate_csv_panels_from_trainers(
        ...     {"LSTM": trainer1, "RF": trainer2},
        ...     output_dir="results_csvs"
        ... )
    """
    comparator = ModelComparator(trainers)
    return comparator.generate_csv_panels(
        output_dir=output_dir,
        run_evaluation=run_evaluation,
        horizon=horizon,
        k=k,
        costs_bps_per_side=costs_bps_per_side,
        test_period=test_period,
        baseline_model=baseline_model
    )


def main_example():
    """
    Example usage of the ModelEvaluator.
    """
    # Example: Evaluate a saved model
    model_path = "savedmodel.pth"  # Update with your model path
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train a model first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        stocks=["AAPL", "MSFT"],
        time_args=["3y"],
        log_dir="runs/evaluation_example"
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_all_metrics()
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    
    # Clean up
    evaluator.close()
    
    print("\nEvaluation completed! Check TensorBoard at 'runs/evaluation_example' for visualizations.")


if __name__ == "__main__":
    main_example()
