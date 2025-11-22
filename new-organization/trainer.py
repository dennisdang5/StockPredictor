import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
try:
    from nlp_features import get_nlp_feature_dim
except ImportError:
    # Fallback function if nlp_features module is not available
    def get_nlp_feature_dim(nlp_method: str) -> int:
        """Return NLP feature dimension based on method."""
        if nlp_method == "aggregated":
            return 10
        elif nlp_method == "individual":
            return 10
        else:
            return 0
# Import model registry system
from models import create_model, ModelRegistry, get_available_models
from models.configs import (
    BaseModelConfig,
    LSTMConfig, CNNLSTMConfig, AELSTMConfig, CNNAELSTMConfig, TimesNetConfig
)
import util
from data_sources import YFinanceDataSource, DataSource
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from tqdm import tqdm
# Removed helpers_workers import - using simple defaults for workers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

# Import metrics computation helpers from evaluator
try:
    from evaluation.evaluator import ModelEvaluator
    _EVALUATOR_AVAILABLE = True
except ImportError:
    _EVALUATOR_AVAILABLE = False
    print("[WARNING] evaluation.evaluator not available, metrics computation will be limited")


def _compute_basic_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute basic classification metrics (reusing evaluator logic).
    
    Args:
        predictions: Model predictions (numpy array)
        targets: True values (numpy array)
        
    Returns:
        Dictionary with accuracy, MSE, RMSE, MAE
    """
    metrics = {}
    
    # Classification accuracy (sign-based)
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
    correct = np.sum(pred_sign == target_sign)
    total = len(predictions)
    metrics['accuracy'] = (correct / total) * 100 if total > 0 else 0
    
    # MSE for continuous predictions
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets, predictions)
    
    return metrics


def _compute_directional_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute directional accuracy metrics (reusing evaluator logic).
    
    Args:
        predictions: Model predictions (numpy array)
        targets: True values (numpy array)
        
    Returns:
        Dictionary with directional accuracy, upward/downward accuracy
    """
    metrics = {}
    
    # Classification accuracy
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
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

class TrainerConfig:
    """
    Configuration class for Trainer.
    Contains all training parameters and model-specific configurations.
    
    Users must create both TrainerConfig and model config objects (e.g., LSTMConfig, TimesNetConfig).
    
    Example usage:
        from models.configs import LSTMConfig, TimesNetConfig
        
        # Example 1: LSTM model
        # Create model-specific config
        lstm_config = LSTMConfig(parameters={
            'input_shape': (31, 13),  # Will be overridden by TrainerConfig based on data
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        })
        
        # Create trainer config
        from data_sources import YFinanceDataSource
        trainer_config = TrainerConfig(
            stocks=["AAPL", "MSFT"],
            time_args=["1990-01-01", "2015-12-31"],
            batch_size=32,
            num_epochs=1000,
            model_type="LSTM",
            model_config=lstm_config,
            period_type="LS",
            lookback=240,  # Days of historical data used for feature extraction
            data_source=YFinanceDataSource(),  # Required: specify data source
            use_nlp=True,
            nlp_method="aggregated"
        )
        
        # Use with Trainer
        trainer = Trainer(config=trainer_config)
        
        # Example 2: TimesNet model
        timesnet_config = TimesNetConfig(parameters={
            'input_shape': (31, 13),  # Will be overridden by TrainerConfig based on data
            'task_name': 'classification',
            'enc_in': 13,
            'num_class': 2,
            'd_model': 256,
            'e_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'd'
        })
        
        trainer_config = TrainerConfig(
            stocks=["AAPL", "MSFT"],
            time_args=["1990-01-01", "2015-12-31"],
            batch_size=32,
            num_epochs=1000,
            model_type="TimesNet",
            model_config=timesnet_config,
            period_type="LS",
            lookback=240,  # Days of historical data used for feature extraction
            use_nlp=True,
            nlp_method="aggregated"
        )
        
        trainer = Trainer(config=trainer_config)
    """
    def __init__(
        self,
        stocks=["MSFT", "AAPL"],
        time_args=["3y"],
        batch_size=8,
        num_epochs=10000,
        saved_model=None,
        prediction_type="classification",
        k=10,
        cost_bps_per_side=5.0,
        save_every_epochs=25,
        model_type="LSTM",
        model_config=None,
        early_stop_patience=7,
        early_stop_min_delta=0.001,
            period_type="LS",
        lookback=None,
        data_source: DataSource = None,
        **model_args
    ):
        """
        Initialize TrainerConfig.
        
        Args:
            stocks: List of stock tickers
            time_args: Time range arguments (e.g., ["3y"] or ["1990-01-01", "2015-12-31"])
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            saved_model: Path to saved model to load (optional)
            prediction_type: Type of prediction task ("classification" or "regression")
            k: Number of top/bottom positions for long-short portfolio
            cost_bps_per_side: Transaction costs per side in basis points
            save_every_epochs: Save model every N epochs (0 to disable periodic saves)
            model_type: Type of model ("LSTM", "CNNLSTM", "AELSTM", "CNNAELSTM", "TimesNet", etc.)
            model_config: Model-specific configuration object. Must be an instance of the appropriate
                         config class (e.g., LSTMConfig for "LSTM", TimesNetConfig for "TimesNet").
                         Users must create this config object themselves - see class docstring for examples.
                         The config type will be validated to match the model_type.
            early_stop_patience: Number of epochs to wait before early stopping (default: 7)
            early_stop_min_delta: Minimum change in validation loss to qualify as improvement (default: 0.001)
            period_type: Period type for feature extraction ("LS" or "full"). Default: "LS"
                    - "LS": Samples timesteps from lookback window (e.g., 31 timesteps from 240-day lookback)
                    - "full": Uses all days from lookback window (e.g., 240 timesteps from 240-day lookback)
            lookback: Days of historical data used for feature extraction (default: None).
                    This is the window size used to extract features. For example:
                    - lookback=240 means use 240 days of historical data
                    - With period_type="LS": samples ~31 timesteps from this 240-day window
                    - With period_type="full": uses all 240 timesteps from this window
                    If None, defaults to 240.
                    Note: The actual seq_len (model input timesteps) is determined by period_type
                    and data, not by lookback directly.
            data_source: DataSource instance to use for fetching stock data (required)
            **model_args: Additional model arguments (e.g., use_nlp, nlp_method, kernel_size)
        """
        if data_source is None:
            raise ValueError("data_source is required for TrainerConfig. Please specify a DataSource instance (e.g., YFinanceDataSource()).")
        
        self.stocks = stocks
        self.time_args = time_args
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.saved_model = saved_model
        self.prediction_type = prediction_type
        self.k = k
        self.cost_bps_per_side = cost_bps_per_side
        self.save_every_epochs = save_every_epochs
        self.model_type = model_type
        self.model_config = model_config  # Model-specific config (e.g., TimesNet config)
        self.model_args = model_args  # Additional model arguments
        
        # Early stopping parameters
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        
        # Period type for feature extraction
        self.period_type = period_type
        
        # Lookback: Days of historical data used for feature extraction
        # If not provided, defaults to 240
        self.lookback = lookback if lookback is not None else 240
        
        # Data source: Required parameter
        self.data_source = data_source
        
        # Extract common model args for convenience
        self.use_nlp = model_args.get("use_nlp", True)  # Default to True
        self.nlp_method = model_args.get("nlp_method", "aggregated")  # Default to aggregated
    
    def to_dict(self):
        """Convert config to dictionary for easy inspection."""
        return {
            'stocks': self.stocks,
            'time_args': self.time_args,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'saved_model': self.saved_model,
            'prediction_type': self.prediction_type,
            'k': self.k,
            'cost_bps_per_side': self.cost_bps_per_side,
            'save_every_epochs': self.save_every_epochs,
            'model_type': self.model_type,
            'model_config': self.model_config,
            'model_args': self.model_args,
            'early_stop_patience': self.early_stop_patience,
            'early_stop_min_delta': self.early_stop_min_delta,
            'period_type': self.period_type,
            'lookback': self.lookback,
            'data_source': type(self.data_source).__name__ if self.data_source else None,
        }
    
    def __repr__(self):
        """String representation of config."""
        return f"TrainerConfig(model_type={self.model_type}, batch_size={self.batch_size}, num_epochs={self.num_epochs})"

########################################################
# validation helper functions
########################################################

def validate_trainer_config(config):
    """
    Validate that config is an instance of TrainerConfig.
    
    Args:
        config: Config object to validate
    
    Returns:
        bool: True if valid
    
    Raises:
        TypeError: If config is not an instance of TrainerConfig
    """
    if not isinstance(config, TrainerConfig):
        raise TypeError(
            f"config must be an instance of TrainerConfig, got {type(config).__name__}. "
            f"Please create a TrainerConfig object and pass it to Trainer."
        )
    return True

def validate_model_config(model_config, model_type, expected_config_class):
    """
    Validate that model_config matches the expected model type.
    
    Args:
        model_config: Model configuration object
        model_type: Expected model type string
        expected_config_class: Expected config class
    
    Returns:
        bool: True if valid
    
    Raises:
        ValueError: If model_config is missing or doesn't match model_type
    """
    if expected_config_class is None:
        from models import get_available_models
        available_models = get_available_models()
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {', '.join(available_models) if available_models else 'None registered'}"
        )
    
    if model_config is None:
        raise ValueError(
            f"model_config is required for model type '{model_type}'. "
            f"Please create a {expected_config_class.__name__} instance and pass it to TrainerConfig."
        )
    
    if not isinstance(model_config, expected_config_class):
        raise ValueError(
            f"Config type mismatch: model_type='{model_type}' requires {expected_config_class.__name__}, "
            f"but got {type(model_config).__name__}. "
            f"Please create a {expected_config_class.__name__} instance and pass it to TrainerConfig."
        )
    
    return True

def setup_dist():
    if not torch.cuda.is_available() or os.getenv("RANK") is None:
        return None, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
    
    # CRITICAL: Set device BEFORE initializing process group to avoid device mapping issues
    # This prevents NCCL from guessing which GPU to use, which can cause hangs
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Now initialize process group with explicit device_id to ensure correct GPU mapping
    dist.init_process_group(
        backend="nccl", 
        timeout=dt.timedelta(minutes=30),
        device_id=torch.cuda.current_device()  # Explicitly specify the device
    )
    
    return local_rank, torch.device(f"cuda:{local_rank}"), True

class IndexedDataset(data.Dataset):
    """
    Custom dataset that tracks indices to maintain correspondence with dates
    even when data is shuffled.
    """
    def __init__(self, X, Y, dates):
        self.X = X
        self.Y = Y
        self.dates = dates
        assert len(X) == len(Y) == len(dates), "All inputs must have same length"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], idx  # Return data + index

# ---------------------------
# Device selection helper
# ---------------------------
def pick_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon (Metal)
        return torch.device("mps")
    else:
        return torch.device("cpu")

class EarlyStopper():
    def __init__(self, patience=10, min_delta=0, is_dist=False, rank=0, save_path="savedmodel.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.is_dist = is_dist
        self.rank = rank
        self.is_main = (rank == 0)
        self.save_path = save_path

    def early_stop(self, validation_loss, model):
        # In distributed mode, we need to synchronize early stopping across all ranks
        if self.is_dist:
            # Create tensors for synchronization on the model's device
            device = next(model.parameters()).device
            should_stop_tensor = torch.tensor(0, dtype=torch.int, device=device)
            
            # Check if this rank should stop
            # All ranks must evaluate the same condition to maintain synchronization
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
                if self.is_main:
                    torch.save((model.module if hasattr(model, "module") else model).state_dict(), self.save_path)
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    should_stop_tensor = torch.tensor(1, dtype=torch.int, device=device)
            
            # Synchronize validation loss across all ranks first to ensure consistent state
            # This ensures all ranks see the same min_validation_loss
            val_loss_tensor = torch.tensor(validation_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.MIN)
            
            # If any rank has a lower validation loss, update all ranks
            if val_loss_tensor.item() < self.min_validation_loss:
                self.min_validation_loss = val_loss_tensor.item()
                self.counter = 0
            
            # Broadcast the early stopping decision from all ranks
            # Use MAX to ensure if any rank wants to stop, all ranks stop
            try:
                dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                # If any rank wants to stop, all ranks should stop
                return should_stop_tensor.item() == 1
            except Exception as e:
                if self.is_main:
                    print(f"[ERROR] Failed to synchronize early stopping decision: {e}")
                # On error, don't stop to avoid hanging
                return False
        else:
            # Non-distributed mode - original logic
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
                torch.save(model.state_dict(), self.save_path)
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        

class Trainer():
    def __init__(self, config):
        """
        Initialize Trainer.
        
        Users must create a TrainerConfig object (with a model config object inside it)
        and pass it to the Trainer.
        
        Example workflow:
            1. Create model config:
               from models.configs import LSTMConfig
               model_config = LSTMConfig(parameters={'hidden_size': 64, ...})
            
            2. Create trainer config:
               trainer_config = TrainerConfig(
                   stocks=["AAPL", "MSFT"],
                   model_type="LSTM",
                   model_config=model_config,
                   ...
               )
            
            3. Create trainer:
               trainer = Trainer(config=trainer_config)
        
        Args:
            config: TrainerConfig instance containing all training parameters.
                    The TrainerConfig must contain a model_config that matches the model_type.
            
        Raises:
            TypeError: If config is not an instance of TrainerConfig.
            ValueError: If model_config is missing or doesn't match the model_type.
        """
        validate_trainer_config(config)
        
        self.config = config
        
        # Extract parameters from config
        self.stocks = config.stocks
        self.time_args = config.time_args
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.saved_model = config.saved_model
        self.prediction_type = config.prediction_type
        self.k = config.k
        self.cost_bps_per_side = config.cost_bps_per_side
        self.save_every_epochs = config.save_every_epochs
        self.model_type = config.model_type
        self.model_config = config.model_config
        self.model_args = config.model_args.copy() if config.model_args else {}
        self.use_nlp = config.use_nlp
        self.nlp_method = config.nlp_method
        
        # Setup distributed training
        self.local_rank, device, self.is_dist = setup_dist()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.is_main = (self.rank == 0)

        # Only rank 0 creates TensorBoard writer
        self.writer = SummaryWriter() if self.is_main else None

        # Use distributed device if available, otherwise pick best device
        if self.is_dist:
            self.device = device
        else:
            self.device = pick_device()
        
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        

        # Set simple defaults for DataLoader workers
        # Use 0 workers on macOS/MPS to avoid multiprocessing import issues
        if self.device.type == "mps":
            self.num_workers = 0
            self.persistent_workers = False
        else:
            self.num_workers = 1
            self.persistent_workers = True  # Use persistent workers when num_workers > 0
        self.pin_memory = True if self.device.type == "cuda" else False  # Pin memory for CUDA
        
        # Create loader_args dict for DataLoader
        self.loader_args = {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }

        # Only rank 0 prints info
        if self.is_main:
            print(f"[distributed] is_dist={self.is_dist}, rank={self.rank}/{self.world_size}")
            print(f"[device] using {self.device}")
            if self.device.type == "cuda":
                device_idx = self.local_rank if self.is_dist else 0
                print(f"[cuda] {torch.cuda.get_device_name(device_idx)} (count={torch.cuda.device_count()})")
            elif self.device.type == "mps":
                print("[mps] Apple Metal Performance Shaders backend")
            print(f"[dataloader] num_workers={self.num_workers}, persistent_workers={self.persistent_workers}, pin_memory={self.pin_memory}")

        # Extract lookback for data loading: days of historical data used for feature extraction
        lookback = self.config.lookback  # Days of historical data window

        # Data loading: use util.get_data which handles cache checking, problematic stock filtering,
        # downloading, and processing. In distributed mode, only rank 0 downloads/processes.
        # Use data source from config
        data_source = self.config.data_source
        
        if self.is_dist:
            # Distributed mode: rank 0 downloads/processes, other ranks load from cache
            if self.is_main:
                # Rank 0: get_data handles everything (cache check, filtering, download, processing)
                input_data = util.get_data(
                    self.stocks, self.time_args, 
                    seq_len=lookback,  # util.get_data uses seq_len parameter name for backward compatibility
                    data_source=data_source,
                    prediction_type=self.prediction_type, 
                    use_nlp=self.use_nlp, 
                    nlp_method=self.nlp_method, 
                    period_type=self.config.period_type
                )
            
            # Synchronize - ensure rank 0 finishes downloading/processing before others proceed
            dist.barrier()
            
            # Other ranks: load from cache (load_data_from_cache already filters problematic stocks)
            if not self.is_main:
                input_data = util.load_data_from_cache(
                    self.stocks, self.time_args, 
                    prediction_type=self.prediction_type, 
                    use_nlp=self.use_nlp, 
                    nlp_method=self.nlp_method,
                    period_type=self.config.period_type,
                    seq_len=lookback,  # util.get_data uses seq_len parameter name for backward compatibility
                    data_source=data_source
                )
                if input_data is None:
                    raise RuntimeError(f"Cache files not found after rank 0 processing. Expected cache should exist.")
            
            # Synchronize again - ensure all ranks have loaded data
            dist.barrier()
        else:
            # Non-distributed: get_data handles everything (cache check, filtering, download, processing)
            input_data = util.get_data(
                self.stocks, self.time_args, 
                seq_len=lookback,  # util.get_data uses seq_len parameter name for backward compatibility
                data_source=data_source,
                prediction_type=self.prediction_type, 
                use_nlp=self.use_nlp, 
                nlp_method=self.nlp_method, 
                period_type=self.config.period_type
            )
        
        # Unpack data tuple (12 elements: X_train, X_val, X_test, Y_train, Y_val, Y_test, 
        # D_train, D_val, D_test, Rev_test, Returns_test, Sp500_test)
        X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test, Rev_test, Returns_test, Sp500_test = input_data
        
        # Determine input_shape from actual data dimensions
        if self.is_main:
            print(f"[data] Data loaded successfully")
            seq_len = X_train.shape[1] if hasattr(X_train, 'shape') and len(X_train) > 0 else 0
            print(f"  X_train shape: {X_train.shape if hasattr(X_train, 'shape') else f'list with {len(X_train)} samples'}")
            print(f"  X_val shape: {X_val.shape if hasattr(X_val, 'shape') else f'list with {len(X_val)} samples'}")
            print(f"  X_test shape: {X_test.shape if hasattr(X_test, 'shape') else f'list with {len(X_test)} samples'}")
            if self.config.period_type == "full":
                print(f"  Lookback: {lookback} days, Sequence length (seq_len): {seq_len} timesteps")
                print(f"  Period type: full (all consecutive days from lookback window)")
            else:
                print(f"  Lookback: {lookback} days, Sequence length (seq_len): {seq_len} timesteps (sampled)")
                print(f"  Period type: LS (sampled timesteps from lookback window)")
        
        # Store dates for later reference (even when data is shuffled)
        self.train_dates = D_train
        self.val_dates = D_val
        self.test_dates = D_test
        self.test_revenues = Rev_test
        self.test_returns = Returns_test  # Store Returns_test if available
        self.train_ds = IndexedDataset(X_train, Y_train, D_train)
        val_ds   = IndexedDataset(X_val,   Y_val,   D_val)
        test_ds  = IndexedDataset(X_test,  Y_test,  D_test)

        self.train_sampler = DistributedSampler(self.train_ds, shuffle=True) if self.is_dist else None
        self.val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if self.is_dist else None
        self.test_sampler   = DistributedSampler(test_ds,   shuffle=False, drop_last=False) if self.is_dist else None


        # =========================
        # BUILD DATALOADERS (use heuristics)
        # =========================
        self.trainLoader = data.DataLoader(
            self.train_ds, shuffle=False, sampler=self.train_sampler,  batch_size=self.batch_size, **self.loader_args
        )
        self.validationLoader = data.DataLoader(
            val_ds,   shuffle=False, sampler=self.val_sampler, batch_size=self.batch_size, **self.loader_args
        )
        self.testLoader = data.DataLoader(
            test_ds,  shuffle=False, sampler=self.test_sampler, batch_size=self.batch_size, **self.loader_args
        )
        
        # Validate dataloader lengths are equal across ranks in distributed mode
        if self.is_dist:
            train_len = len(self.trainLoader)
            val_len = len(self.validationLoader)
            test_len = len(self.testLoader)
            
            # Gather lengths from all ranks
            train_len_tensor = torch.tensor(train_len, device=self.device)
            val_len_tensor = torch.tensor(val_len, device=self.device)
            test_len_tensor = torch.tensor(test_len, device=self.device)
            
            train_lens = [torch.zeros_like(train_len_tensor) for _ in range(self.world_size)]
            val_lens = [torch.zeros_like(val_len_tensor) for _ in range(self.world_size)]
            test_lens = [torch.zeros_like(test_len_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(train_lens, train_len_tensor)
            dist.all_gather(val_lens, val_len_tensor)
            dist.all_gather(test_lens, test_len_tensor)
            
            # Check if all ranks have the same length
            train_uniform = all(t.item() == train_len for t in train_lens)
            val_uniform = all(v.item() == val_len for v in val_lens)
            test_uniform = all(t.item() == test_len for t in test_lens)
            
            if self.is_main and not (train_uniform and val_uniform and test_uniform):
                print(f"[WARNING] Dataloader lengths differ across ranks!")
                print(f"  Train: {[t.item() for t in train_lens]}")
                print(f"  Val: {[v.item() for v in val_lens]}")
                print(f"  Test: {[t.item() for t in test_lens]}")
                print(f"  This may cause DDP synchronization issues.")
            
            # Synchronize before continuing
            dist.barrier()

        # =========================
        # MODEL
        # =========================
        # Determine input_shape from actual data dimensions
        try:
            if len(X_train) > 0:
                actual_features = X_train.shape[2]
                seq_len = X_train.shape[1]  # Actual model input timesteps
                input_shape = (seq_len, actual_features)
                
                if self.is_main:
                    print(f"[config] Determined input_shape from data: {input_shape} (seq_len={seq_len}, features={actual_features})")
            else:
                # Fallback if no data available (shouldn't happen, but handle gracefully)
                # Estimate seq_len based on period_type
                if self.config.period_type == "full":
                    estimated_seq_len = lookback
                else:
                    # For LS, estimate ~31 timesteps from 240-day lookback
                    estimated_seq_len = int(lookback / 12) + 1 + 10  # Approximate LS sampling
                
                estimated_features = 3 + (get_nlp_feature_dim(self.nlp_method) if self.use_nlp and self.nlp_method else 0)
                input_shape = (estimated_seq_len, estimated_features)
                if self.is_main:
                    print(f"[config] No data available, using estimated input_shape: {input_shape} (lookback={lookback}, estimated_seq_len={estimated_seq_len})")
        except Exception as e:
            print(f"[ERROR] Failed to determine input_shape: {e}")
            return None
        
        # Move model to device first
        # Create appropriate ModelConfig class instance based on model_type
        final_model_config = self._create_model_config(input_shape)
        
        # Initialize model using registry system
        try:
            self.Model = create_model(self.model_type, final_model_config).to(self.device)
            
            # Handle TimesNet-specific encoder freezing if requested
            if self.model_type.upper() == "TIMESNET":
                freeze_encoder = getattr(final_model_config, 'freeze_encoder', False)
                if freeze_encoder:
                    # Freeze encoder components: enc_embedding, model (TimesBlock layers), and layer_norm
                    # Keep projection (classifier head) trainable
                    model_to_freeze = self.Model.module if hasattr(self.Model, 'module') else self.Model
                    if hasattr(model_to_freeze, 'timesnet'):
                        # If using adapter, access the underlying TimesNet model
                        timesnet_model = model_to_freeze.timesnet
                    else:
                        timesnet_model = model_to_freeze
                    
                    if hasattr(timesnet_model, 'enc_embedding'):
                        for param in timesnet_model.enc_embedding.parameters():
                            param.requires_grad = False
                    if hasattr(timesnet_model, 'model'):
                        for param in timesnet_model.model.parameters():
                            param.requires_grad = False
                    if hasattr(timesnet_model, 'layer_norm'):
                        for param in timesnet_model.layer_norm.parameters():
                            param.requires_grad = False
                    
                    # Ensure projection remains trainable
                    if hasattr(timesnet_model, 'projection'):
                        for param in timesnet_model.projection.parameters():
                            param.requires_grad = True
                
                if self.is_main:
                    print(f"[TimesNet] Model initialized with config:")
                    if hasattr(final_model_config, 'lookback'):
                        print(f"  lookback: {final_model_config.lookback} days")
                    if hasattr(final_model_config, 'seq_len'):
                        print(f"  seq_len: {final_model_config.seq_len} timesteps")
                    if hasattr(final_model_config, 'enc_in'):
                        print(f"  enc_in: {final_model_config.enc_in}")
                    if hasattr(final_model_config, 'num_class'):
                        print(f"  num_class: {final_model_config.num_class}")
                    if hasattr(final_model_config, 'd_model'):
                        print(f"  d_model: {final_model_config.d_model}")
                    print(f"  freeze_encoder: {freeze_encoder}")
        except ValueError as e:
            # Provide helpful error message with available models
            available_models = get_available_models()
            if available_models:
                available_str = ", ".join(available_models)
                raise ValueError(
                    f"Failed to create model '{self.model_type}': {e}\n"
                    f"Available models: {available_str}"
                )
            else:
                raise ValueError(
                    f"Failed to create model '{self.model_type}': {e}\n"
                    f"No models are currently registered. Make sure model modules are imported."
                )
        
        # Wrap with DDP if in distributed mode
        if self.is_dist:
            self.Model = DDP(self.Model, device_ids=[self.local_rank])
            if self.is_main:
                print(f"[DDP] using {self.world_size} processes across {dist.get_world_size()} GPUs")
        # For single-GPU or single-node multi-GPU without torchrun, use DataParallel
        elif self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.Model = nn.DataParallel(self.Model)
            if self.is_main:
                print(f"[DataParallel] using {torch.cuda.device_count()} GPUs")

        # Count and print parameters after wrapping (for consistency)
        if self.is_main:
            # Get unwrapped model for accurate parameter counting
            model_to_count = self.Model.module if hasattr(self.Model, 'module') else self.Model
            
            # Count total parameters
            total_params = sum(param.numel() for param in model_to_count.parameters())
            print(f"{total_params:,} total parameters")
            
            # If TimesNet with frozen encoder, show detailed breakdown
            if self.model_type.upper() == "TIMESNET":
                freeze_encoder = getattr(final_model_config, 'freeze_encoder', False)
                if freeze_encoder:
                    # Access the underlying TimesNet model
                    if hasattr(model_to_count, 'timesnet'):
                        timesnet_model = model_to_count.timesnet
                    else:
                        timesnet_model = model_to_count
                    
                    # Count frozen and trainable parameters
                    frozen_params = sum(p.numel() for p in model_to_count.parameters() if not p.requires_grad)
                    trainable_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
                    print(f"[TimesNet] Encoder frozen: {frozen_params:,} parameters frozen, {trainable_params:,} parameters trainable")
        
        # Determine the save path for the model using unique ID system
        # First check if a model exists with matching config
        existing_model_path = None
        
        if self.config.saved_model is not None:
            # User explicitly provided a path - use it (legacy behavior)
            self.save_path = self.config.saved_model
            if os.path.exists(self.config.saved_model):
                existing_model_path = self.config.saved_model
        else:
            # Use unique ID system: check if model exists with matching config
            model_id, existing_model_path = util.find_model_by_config(self.config.model_config)
            
            if existing_model_path is not None:
                # Found matching model - use its ID-based path
                self.save_path = existing_model_path
                if self.is_main:
                    print(f"[model] Found existing model with matching config (ID: {model_id})")
                    print(f"[model] Using model path: {self.save_path}")
            else:
                # No matching model found - generate new ID and create new path
                model_id = util._get_model_id(self.config.model_config)
                # Ensure models directory exists
                os.makedirs(util.MODELS_DIR, exist_ok=True)
                self.save_path = os.path.join(util.MODELS_DIR, f"{model_id}.pth")
                if self.is_main:
                    print(f"[model] No matching model found, creating new model (ID: {model_id})")
                    print(f"[model] Model will be saved to: {self.save_path}")
                # Save mapping for new model
                util._save_model_mapping(model_id, self.config.model_config)
        
        # Load model weights if file exists
        if existing_model_path is not None and os.path.exists(existing_model_path):
            state_dict = torch.load(existing_model_path, map_location="cpu")
            # Load into underlying module if DataParallel/DDP is active
            target_module = self.Model.module if hasattr(self.Model, "module") else self.Model
            target_module.load_state_dict(state_dict)
            if self.is_main:
                print(f"[load] Restored weights from {existing_model_path}")
        else:
            if self.is_main:
                if self.config.saved_model is not None:
                    print(f"[load] No saved model found at {self.config.saved_model}, starting with random weights")
                else:
                    print(f"[load] Starting training with random weights (new model)")

        # Use a lower learning rate to prevent instability
        # Even lower learning rate for more stability
        self.optimizer = optim.Adam(self.Model.parameters(), lr=5e-5, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        # More aggressive gradient clipping to prevent exploding gradients
        self.max_grad_norm = 0.5
        # Use early stopping parameters from config
        early_stop_patience = self.config.early_stop_patience
        early_stop_min_delta = self.config.early_stop_min_delta
        self.stopper = EarlyStopper(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            is_dist=self.is_dist,
            rank=self.rank,
            save_path=self.save_path
        )
        
        # Storage for evaluation metrics
        self.predicted_returns = []
        self.actual_returns = []
        self.predicted_values = []
        self.actual_values = []
        self.evaluation_dates = []
        self.equity_curve = []
        self.residuals = []
        self.real_world_returns = []
        self.daily_returns = []  # Daily portfolio returns (one per trading day)
        
        # Store model parameter count for AIC/BIC
        self.num_model_parameters = sum(p.numel() for p in self.Model.parameters())
        
        # Track recovery attempts to prevent infinite loops
        self.nan_recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.last_good_checkpoint = None  # Track last known good checkpoint path
    
    def _create_model_config(self, input_shape):
        """
        Validate that the provided config object matches the model type.
        
        The user must create the config object themselves and provide it to TrainerConfig.
        This method validates that the config type matches the model type and updates
        input_shape, seq_len, and lookback from TrainerConfig/data.
        
        Args:
            input_shape (tuple): Input shape determined from data (seq_len, num_features)
            
        Returns:
            BaseModelConfig instance (the provided config with updated attributes):
            - input_shape: (seq_len, num_features) - actual model input dimensions
            - seq_len: number of timesteps fed to model (from data)
            - lookback: days of historical data used for feature extraction (from TrainerConfig)
            
        Raises:
            ValueError: If config is not provided or doesn't match the model type
        """
        model_type_upper = self.model_type.upper()
        
        # Get expected config class from registry
        expected_config_class = ModelRegistry.get_config_class(model_type_upper)
        
        # Validate model config using helper function
        validate_model_config(self.model_config, self.model_type, expected_config_class)
        
        # Config is valid - use it and update attributes from TrainerConfig/data
        final_config = self.model_config
        
        # Extract values
        seq_len = input_shape[0]  # Actual model input timesteps (from data)
        lookback = self.config.lookback  # Historical data window (from TrainerConfig)
        
        # Update input_shape and attributes (these come from TrainerConfig/data, not user config)
        if hasattr(final_config, 'parameters'):
            final_config.parameters['input_shape'] = input_shape
            final_config.parameters['seq_len'] = seq_len
            final_config.parameters['lookback'] = lookback
        
        # Always set as attributes for easy access
        final_config.input_shape = input_shape  # (seq_len, num_features)
        final_config.seq_len = seq_len  # Actual model input timesteps
        final_config.lookback = lookback  # Historical data window
        
        return final_config
    
    def _has_nan_weights(self):
        """Check if any model parameters contain NaN or Inf values."""
        model = self.Model.module if hasattr(self.Model, "module") else self.Model
        for param in model.parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                return True
        return False
    
    def _recover_from_nan(self):
        """
        Attempt to recover from NaN by reloading from checkpoint and reducing learning rate.
        Returns True if recovery was successful, False otherwise.
        """
        self.nan_recovery_attempts += 1
        
        if self.nan_recovery_attempts > self.max_recovery_attempts:
            if self.is_main:
                print(f"[ERROR] Maximum recovery attempts ({self.max_recovery_attempts}) reached. Stopping training.")
            return False
        
        # Try to reload from checkpoint
        checkpoint_path = self.save_path
        if not os.path.exists(checkpoint_path):
            # Try to find the last periodic save
            if self.is_main:
                print(f"[WARNING] No checkpoint found at {checkpoint_path}. Cannot recover from NaN.")
            return False
        
        try:
            if self.is_main:
                print(f"[RECOVERY] Attempt {self.nan_recovery_attempts}/{self.max_recovery_attempts}: "
                      f"Reloading from checkpoint {checkpoint_path}")
            
            # Load checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            target_module = self.Model.module if hasattr(self.Model, "module") else self.Model
            target_module.load_state_dict(state_dict)
            
            # Reduce learning rate by factor of 2
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Reset optimizer state to avoid carrying over corrupted momentum
            self.optimizer = optim.Adam(self.Model.parameters(), lr=new_lr, weight_decay=1e-5)
            
            if self.is_main:
                print(f"[RECOVERY] Successfully reloaded checkpoint. Reduced learning rate: {current_lr:.2e} -> {new_lr:.2e}")
            
            # Verify weights are good now
            if self._has_nan_weights():
                if self.is_main:
                    print(f"[ERROR] Checkpoint also contains NaN weights! Cannot recover.")
                return False
            
            return True
            
        except Exception as e:
            if self.is_main:
                print(f"[ERROR] Failed to recover from NaN: {e}")
            return False

    def train_one_epoch(self, epoch):

        # Ensure DDP shuffles differently each epoch for all samplers
        if self.is_dist:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(epoch)
            # Note: test_sampler doesn't need set_epoch as it's only used during evaluation

        if self.is_main:
            print("Epoch: {}".format(epoch+1))
            print("--------------------------------------------")
        
        self.Model.train()
        train_loss = 0
        val_loss = 0
        stop_condition=None
        train_batches_processed = 0  # Track actual batches processed (excluding skipped NaN batches)
        
        # Collect predictions and targets for metrics computation
        train_predictions = []
        train_targets = []
        
        # Track gradient norms for diagnostics (only on main rank)
        grad_norms = [] if self.is_main else None
        
        # Count NaN/Inf occurrences in training
        train_nan_x_batch = 0
        train_nan_y_batch = 0
        train_nan_model_output = 0
        train_nan_loss = 0

        start_time = time.perf_counter()

        # Use stored trainLoader, only show progress bar on rank 0
        pbar = tqdm(self.trainLoader, total=len(self.trainLoader), 
                   desc=f"train {epoch+1}/{self.num_epochs}", disable=not self.is_main)
        for X_batch, Y_batch, indices in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            # move data to device
            X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
            Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
            
            # Check for NaN/Inf in input data
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                train_nan_x_batch += 1
                if self.is_main:
                    print(f"[WARNING] NaN/Inf found in X_batch at batch {pbar.n}")
                continue
            if torch.isnan(Y_batch).any() or torch.isinf(Y_batch).any():
                train_nan_y_batch += 1
                if self.is_main:
                    print(f"[WARNING] NaN/Inf found in Y_batch at batch {pbar.n}")
                continue
            
            # Get dates for this batch (for debugging/analysis)
            batch_dates = [self.train_dates[idx] for idx in indices.tolist()]

            # forward + loss (+AMP on CUDA)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    Y_pred = self.Model(X_batch)
                    # Check for NaN/Inf in model output
                    if torch.isnan(Y_pred).any() or torch.isinf(Y_pred).any():
                        train_nan_model_output += 1
                        # Check if weights are corrupted
                        if self._has_nan_weights():
                            if self.is_main:
                                print(f"[ERROR] NaN in model output and weights detected at batch {pbar.n}!")
                            # Attempt recovery
                            if self._recover_from_nan():
                                if self.is_main:
                                    print(f"[RECOVERY] Continuing training after recovery...")
                                self.optimizer.zero_grad(set_to_none=True)
                                continue
                            else:
                                if self.is_main:
                                    print(f"[ERROR] Cannot recover from NaN. Stopping training.")
                                avg_train = float('nan')
                                avg_val = float('nan')
                                break
                        else:
                            # Output is NaN but weights are OK - might be a numerical issue with this specific batch
                            if self.is_main:
                                print(f"[WARNING] NaN/Inf in model output at batch {pbar.n} (weights OK, skipping batch)")
                            continue
                    loss = self.loss_fn(Y_pred, Y_batch)
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    train_nan_loss += 1
                    if self.is_main:
                        print(f"[WARNING] NaN/Inf loss at batch {pbar.n}, skipping")
                    continue
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                # Compute gradient norm before clipping (for diagnostics)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.Model.parameters(), self.max_grad_norm)
                if self.is_main:
                    grad_norms.append(grad_norm.item())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Check for NaN in model weights after optimizer step
                if self._has_nan_weights():
                    train_nan_model_output += 1
                    if self.is_main:
                        print(f"[ERROR] NaN detected in model weights after batch {pbar.n}!")
                    # Attempt recovery
                    if self._recover_from_nan():
                        if self.is_main:
                            print(f"[RECOVERY] Continuing training after recovery...")
                        # Skip this batch and continue
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        # Recovery failed, stop training
                        if self.is_main:
                            print(f"[ERROR] Cannot recover from NaN. Stopping training.")
                        # Mark epoch as failed
                        avg_train = float('nan')
                        avg_val = float('nan')
                        break
            else:
                Y_pred = self.Model(X_batch)
                # Check for NaN/Inf in model output
                if torch.isnan(Y_pred).any() or torch.isinf(Y_pred).any():
                    train_nan_model_output += 1
                    # Check if weights are corrupted
                    if self._has_nan_weights():
                        if self.is_main:
                            print(f"[ERROR] NaN in model output and weights detected at batch {pbar.n}!")
                        # Attempt recovery
                        if self._recover_from_nan():
                            if self.is_main:
                                print(f"[RECOVERY] Continuing training after recovery...")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        else:
                            if self.is_main:
                                print(f"[ERROR] Cannot recover from NaN. Stopping training.")
                            avg_train = float('nan')
                            avg_val = float('nan')
                            break
                    else:
                        # Output is NaN but weights are OK - might be a numerical issue with this specific batch
                        if self.is_main:
                            print(f"[WARNING] NaN/Inf in model output at batch {pbar.n} (weights OK, skipping batch)")
                        continue
                loss = self.loss_fn(Y_pred, Y_batch)
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    train_nan_loss += 1
                    if self.is_main:
                        print(f"[WARNING] NaN/Inf loss at batch {pbar.n}, skipping")
                    continue
                loss.backward()
                # Gradient clipping
                # Compute gradient norm before clipping (for diagnostics)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.Model.parameters(), self.max_grad_norm)
                if self.is_main:
                    grad_norms.append(grad_norm.item())
                self.optimizer.step()
                
                # Check for NaN in model weights after optimizer step
                if self._has_nan_weights():
                    train_nan_model_output += 1
                    if self.is_main:
                        print(f"[ERROR] NaN detected in model weights after batch {pbar.n}!")
                    # Attempt recovery
                    if self._recover_from_nan():
                        if self.is_main:
                            print(f"[RECOVERY] Continuing training after recovery...")
                        # Skip this batch and continue
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        # Recovery failed, stop training
                        if self.is_main:
                            print(f"[ERROR] Cannot recover from NaN. Stopping training.")
                        # Mark epoch as failed
                        avg_train = float('nan')
                        avg_val = float('nan')
                        break


            train_loss += loss.item()
            train_batches_processed += 1
            
            # Collect predictions and targets for metrics (only on main rank to avoid duplication)
            if self.is_main:
                # Detach and convert to numpy for metrics computation
                train_predictions.append(Y_pred.detach().cpu().numpy().flatten())
                train_targets.append(Y_batch.detach().cpu().numpy().flatten())
            
            if self.is_main:
                pbar.set_postfix(avg_train=train_loss/(train_batches_processed or 1))
        
        # Aggregate training loss and NaN counts across all ranks
        if self.is_dist:
            try:
                train_loss_tensor = torch.tensor(train_loss, device=self.device)
                train_batches_tensor = torch.tensor(train_batches_processed, device=self.device, dtype=torch.int)
                train_nan_x_tensor = torch.tensor(train_nan_x_batch, device=self.device, dtype=torch.int)
                train_nan_y_tensor = torch.tensor(train_nan_y_batch, device=self.device, dtype=torch.int)
                train_nan_output_tensor = torch.tensor(train_nan_model_output, device=self.device, dtype=torch.int)
                train_nan_loss_tensor = torch.tensor(train_nan_loss, device=self.device, dtype=torch.int)
                
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_batches_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_nan_x_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_nan_y_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_nan_output_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_nan_loss_tensor, op=dist.ReduceOp.SUM)
                
                train_loss = train_loss_tensor.item()
                train_batches_processed = train_batches_tensor.item()
                train_nan_x_batch = train_nan_x_tensor.item()
                train_nan_y_batch = train_nan_y_tensor.item()
                train_nan_model_output = train_nan_output_tensor.item()
                train_nan_loss = train_nan_loss_tensor.item()
            except Exception as e:
                if self.is_main:
                    print(f"[ERROR] Failed to aggregate training loss: {e}")
                # Fallback: use local values only
                pass
        
        if train_batches_processed == 0:
            if self.is_main:
                print("[ERROR] No valid batches processed in training! All batches contained NaN/Inf.")
            avg_train = float('nan')
        else:
            avg_train = train_loss / train_batches_processed

        self.Model.eval()
        val_batches_processed = 0  # Track actual batches processed (excluding skipped NaN batches)
        
        # Collect predictions and targets for metrics computation
        val_predictions = []
        val_targets = []
        
        # Count NaN/Inf occurrences in validation
        val_nan_x_batch = 0
        val_nan_y_batch = 0
        val_nan_model_output = 0
        val_nan_loss = 0
        
        with torch.no_grad():
            vbar = tqdm(self.validationLoader, total=len(self.validationLoader), 
                       desc=f"validation", disable=not self.is_main)
            for X_batch, Y_batch, indices in vbar:
                X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
                Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
                
                # Check for NaN/Inf in validation data
                if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                    val_nan_x_batch += 1
                    continue
                if torch.isnan(Y_batch).any() or torch.isinf(Y_batch).any():
                    val_nan_y_batch += 1
                    continue
                
                # Get dates for this batch (for debugging/analysis)
                batch_dates = [self.val_dates[idx] for idx in indices.tolist()]
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        Y_pred = self.Model(X_batch)
                        if torch.isnan(Y_pred).any() or torch.isinf(Y_pred).any():
                            val_nan_model_output += 1
                            continue
                        loss = self.loss_fn(Y_pred, Y_batch)
                else:
                    Y_pred = self.Model(X_batch)
                    if torch.isnan(Y_pred).any() or torch.isinf(Y_pred).any():
                        val_nan_model_output += 1
                        continue
                    loss = self.loss_fn(Y_pred, Y_batch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    val_nan_loss += 1
                    continue
                val_loss += loss.item()
                val_batches_processed += 1
                
                # Collect predictions and targets for metrics (only on main rank)
                if self.is_main:
                    # Detach and convert to numpy for metrics computation
                    val_predictions.append(Y_pred.detach().cpu().numpy().flatten())
                    val_targets.append(Y_batch.detach().cpu().numpy().flatten())
                
                if self.is_main:
                    vbar.set_postfix(avg_val=val_loss/(val_batches_processed or 1))
            
            # Aggregate validation loss and NaN counts across all ranks
            if self.is_dist:
                try:
                    val_loss_tensor = torch.tensor(val_loss, device=self.device)
                    val_batches_tensor = torch.tensor(val_batches_processed, device=self.device, dtype=torch.int)
                    val_nan_x_tensor = torch.tensor(val_nan_x_batch, device=self.device, dtype=torch.int)
                    val_nan_y_tensor = torch.tensor(val_nan_y_batch, device=self.device, dtype=torch.int)
                    val_nan_output_tensor = torch.tensor(val_nan_model_output, device=self.device, dtype=torch.int)
                    val_nan_loss_tensor = torch.tensor(val_nan_loss, device=self.device, dtype=torch.int)
                    
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_nan_x_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_nan_y_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_nan_output_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_nan_loss_tensor, op=dist.ReduceOp.SUM)
                    
                    val_loss = val_loss_tensor.item()
                    val_batches_processed = val_batches_tensor.item()
                    val_nan_x_batch = val_nan_x_tensor.item()
                    val_nan_y_batch = val_nan_y_tensor.item()
                    val_nan_model_output = val_nan_output_tensor.item()
                    val_nan_loss = val_nan_loss_tensor.item()
                except Exception as e:
                    if self.is_main:
                        print(f"[ERROR] Failed to aggregate validation loss: {e}")
                    # Fallback: use local values only
                    pass
            
            if val_batches_processed == 0:
                if self.is_main:
                    print("[ERROR] No valid batches processed in validation! All batches contained NaN/Inf.")
                avg_val = float('nan')
            else:
                avg_val = val_loss / val_batches_processed
            
            # Synchronize before early stopping check
            if self.is_dist:
                try:
                    dist.barrier()
                except Exception as e:
                    if self.is_main:
                        print(f"[ERROR] Barrier failed before early stopping check: {e}")
            
            stop_condition = self.stopper.early_stop(avg_val, self.Model)   
            
            # If early stopping is triggered, synchronize all ranks before exiting
            if stop_condition and self.is_dist:
                try:
                    dist.barrier()  # Ensure all ranks are synchronized before any rank exits
                except Exception as e:
                    if self.is_main:
                        print(f"[ERROR] Barrier failed after early stopping: {e}")

        end_time = time.perf_counter()
        
        # Compute metrics if we have predictions and targets
        train_metrics = {}
        val_metrics = {}
        if self.is_main and len(train_predictions) > 0 and len(train_targets) > 0:
            try:
                train_pred_array = np.concatenate(train_predictions)
                train_target_array = np.concatenate(train_targets)
                train_metrics = _compute_basic_metrics(train_pred_array, train_target_array)
                train_dir_metrics = _compute_directional_metrics(train_pred_array, train_target_array)
                train_metrics.update(train_dir_metrics)
            except Exception as e:
                if self.is_main:
                    print(f"[WARNING] Failed to compute training metrics: {e}")
        
        if self.is_main and len(val_predictions) > 0 and len(val_targets) > 0:
            try:
                val_pred_array = np.concatenate(val_predictions)
                val_target_array = np.concatenate(val_targets)
                val_metrics = _compute_basic_metrics(val_pred_array, val_target_array)
                val_dir_metrics = _compute_directional_metrics(val_pred_array, val_target_array)
                val_metrics.update(val_dir_metrics)
            except Exception as e:
                if self.is_main:
                    print(f"[WARNING] Failed to compute validation metrics: {e}")
        
        # Only rank 0 prints and logs
        if self.is_main:
            print("Training Loss: {}".format(avg_train))
            print("Validation Loss: {}".format(avg_val))
            if train_metrics:
                print("Training Metrics - Accuracy: {:.2f}%, Directional Accuracy: {:.2f}%".format(
                    train_metrics.get('accuracy', 0), train_metrics.get('directional_accuracy', 0)))
            if val_metrics:
                print("Validation Metrics - Accuracy: {:.2f}%, Directional Accuracy: {:.2f}%".format(
                    val_metrics.get('accuracy', 0), val_metrics.get('directional_accuracy', 0)))
            print("Training Time: {:.6f}s".format(end_time-start_time))
            
            # Print NaN/Inf statistics
            total_train_batches = len(self.trainLoader) * (self.world_size if self.is_dist else 1)
            total_val_batches = len(self.validationLoader) * (self.world_size if self.is_dist else 1)
            
            print("\n[NaN/Inf Statistics]")
            print(f"Training - Batches skipped due to NaN/Inf:")
            print(f"  X_batch: {train_nan_x_batch}/{total_train_batches} ({100*train_nan_x_batch/max(total_train_batches,1):.2f}%)")
            print(f"  Y_batch: {train_nan_y_batch}/{total_train_batches} ({100*train_nan_y_batch/max(total_train_batches,1):.2f}%)")
            print(f"  Model output: {train_nan_model_output}/{total_train_batches} ({100*train_nan_model_output/max(total_train_batches,1):.2f}%)")
            print(f"  Loss: {train_nan_loss}/{total_train_batches} ({100*train_nan_loss/max(total_train_batches,1):.2f}%)")
            print(f"Validation - Batches skipped due to NaN/Inf:")
            print(f"  X_batch: {val_nan_x_batch}/{total_val_batches} ({100*val_nan_x_batch/max(total_val_batches,1):.2f}%)")
            print(f"  Y_batch: {val_nan_y_batch}/{total_val_batches} ({100*val_nan_y_batch/max(total_val_batches,1):.2f}%)")
            print(f"  Model output: {val_nan_model_output}/{total_val_batches} ({100*val_nan_model_output/max(total_val_batches,1):.2f}%)")
            print(f"  Loss: {val_nan_loss}/{total_val_batches} ({100*val_nan_loss/max(total_val_batches,1):.2f}%)")
            print(f"Valid batches processed - Training: {train_batches_processed}, Validation: {val_batches_processed}")
            
            if stop_condition:
                print("Early stop at epoch: {}".format(epoch))
            print("--------------------------------------------")
            
            # Only log to TensorBoard if writer exists
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', avg_train, epoch+1)
                self.writer.add_scalars('Loss/trainVSvalidation', {"Training":(avg_train), "Validation":(avg_val)}, epoch+1)
                self.writer.add_scalar('Train Time', (end_time-start_time), epoch+1)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', current_lr, epoch+1)
                
                # Log metrics
                if train_metrics:
                    self.writer.add_scalar('Metrics/Train/Accuracy', train_metrics.get('accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Train/DirectionalAccuracy', train_metrics.get('directional_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Train/UpwardAccuracy', train_metrics.get('upward_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Train/DownwardAccuracy', train_metrics.get('downward_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Train/RMSE', train_metrics.get('rmse', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Train/MAE', train_metrics.get('mae', 0), epoch+1)
                
                if val_metrics:
                    self.writer.add_scalar('Metrics/Val/Accuracy', val_metrics.get('accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Val/DirectionalAccuracy', val_metrics.get('directional_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Val/UpwardAccuracy', val_metrics.get('upward_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Val/DownwardAccuracy', val_metrics.get('downward_accuracy', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Val/RMSE', val_metrics.get('rmse', 0), epoch+1)
                    self.writer.add_scalar('Metrics/Val/MAE', val_metrics.get('mae', 0), epoch+1)
                
                # Log batch statistics
                self.writer.add_scalar('Batches/Train/Processed', train_batches_processed, epoch+1)
                self.writer.add_scalar('Batches/Val/Processed', val_batches_processed, epoch+1)
                
                # Log gradient norm statistics
                if grad_norms and len(grad_norms) > 0:
                    self.writer.add_scalar('Gradients/Norm/Mean', np.mean(grad_norms), epoch+1)
                    self.writer.add_scalar('Gradients/Norm/Max', np.max(grad_norms), epoch+1)
                    self.writer.add_scalar('Gradients/Norm/Min', np.min(grad_norms), epoch+1)
                
                self.writer.flush()
        
        # Periodic save regardless of validation loss improvement
        # Check on all ranks, but only save on main rank after synchronization
        should_save_periodic = (self.save_every_epochs > 0 and 
                               ((epoch == 0) or ((epoch + 1) % self.save_every_epochs == 0)))
        
        if should_save_periodic:
            # Synchronize all ranks before saving (in distributed mode)
            if self.is_dist:
                try:
                    dist.barrier()
                except Exception as e:
                    if self.is_main:
                        print(f"[ERROR] Barrier failed before periodic save: {e}")
            
            # Only main rank saves the model
            if self.is_main:
                self._save_model_periodic(epoch + 1)
        
        return stop_condition
    
    def _save_model_periodic(self, epoch):
        """
        Save the model periodically regardless of validation loss improvement.
        This ensures we have checkpoints even if the model doesn't improve.
        """
        if not self.is_main:
            return  # Only main rank saves
        
        # Check for NaN weights before saving
        if self._has_nan_weights():
            print(f"[WARNING] Skipping periodic save at epoch {epoch} - model contains NaN weights!")
            return
        
        try:
            # Get the underlying model (unwrap DDP/DataParallel if needed)
            model_to_save = self.Model.module if hasattr(self.Model, "module") else self.Model
            torch.save(model_to_save.state_dict(), "checkpoint_" + self.save_path )
            print(f"[periodic save] Model saved at epoch {epoch} to checkpoint_{self.save_path}")
            # Reset recovery counter on successful save (model is stable)
            self.nan_recovery_attempts = 0
            self.last_good_checkpoint = "checkpoint_" + self.save_path
        except Exception as e:
            print(f"[ERROR] Failed to save model periodically at epoch {epoch}: {e}")
    
    def get_summary(self):
        summary(self.Model, (240,3), self.batch_size, device=self.device.type)
    
    def get_batch_dates(self, indices, dataset_type='train'):
        """
        Get dates for a batch of indices.
        
        Args:
            indices: List or tensor of indices
            dataset_type: 'train', 'val', or 'test'
        
        Returns:
            List of dates corresponding to the indices
        """
        if dataset_type == 'train':
            dates = self.train_dates
        elif dataset_type == 'val':
            dates = self.val_dates
        elif dataset_type == 'test':
            dates = self.test_dates
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")
        
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        return [dates[idx] for idx in indices]
    
    def _tensor_to_numpy(self, tensor):
        """
        Convert a tensor to numpy array in a device-agnostic way.
        
        Args:
            tensor: PyTorch tensor on any device (CPU, CUDA, MPS)
            
        Returns:
            numpy array
        """
        # Detach from computation graph and move to CPU before converting to numpy
        return tensor.detach().cpu().numpy()
    
    def _format_date_axis(self, plt, datetime_dates):
        """
        Format x-axis dates with adaptive resolution based on data span.
        
        Args:
            plt: matplotlib pyplot object
            datetime_dates: List of datetime objects
        """
        if not datetime_dates:
            return
            
        # Calculate the span of dates
        min_date = min(datetime_dates)
        max_date = max(datetime_dates)
        date_span = (max_date - min_date).days
        num_points = len(datetime_dates)
        
        # Determine appropriate tick resolution based on data span
        if date_span <= 30:  # Less than 1 month
            # Show every few days or every day if few points
            if num_points <= 15:
                locator = mdates.DayLocator(interval=1)
                formatter = mdates.DateFormatter('%Y-%m-%d')
            else:
                locator = mdates.DayLocator(interval=2)
                formatter = mdates.DateFormatter('%m-%d')
                
        elif date_span <= 90:  # 1-3 months
            # Show every week
            locator = mdates.WeekdayLocator(interval=1)
            formatter = mdates.DateFormatter('%m-%d')
            
        elif date_span <= 365:  # 3-12 months
            # Show every month
            locator = mdates.MonthLocator(interval=1)
            formatter = mdates.DateFormatter('%Y-%m')
            
        elif date_span <= 730:  # 1-2 years
            # Show every 2-3 months
            locator = mdates.MonthLocator(interval=2)
            formatter = mdates.DateFormatter('%Y-%m')
            
        else:  # More than 2 years
            # Show every 6 months or year
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

    def _log_loss_vs_date_to_tensorboard(self, losses, dates):
        """
        Log individual test losses against their corresponding dates to TensorBoard as a figure.
        
        Args:
            losses: List of individual loss values
            dates: List of corresponding dates

        """
        try:
            # Convert dates to datetime objects for proper plotting
            if hasattr(dates[0], 'to_pydatetime'):
                # pandas DatetimeIndex or Timestamp
                datetime_dates = [date.to_pydatetime() for date in dates]
            else:
                # Already datetime or string
                import pandas as pd
                datetime_dates = [pd.Timestamp(date).to_pydatetime() for date in dates]
            
            # Create matplotlib figure
            plt.figure(figsize=(12, 6))
            plt.plot(datetime_dates, losses, 'b-', alpha=0.7, linewidth=1, marker='o', markersize=3)
            plt.scatter(datetime_dates, losses, c='red', alpha=0.6, s=20)
            
            # Customize the plot
            plt.title(f'Individual Test Loss vs Date')
            plt.xlabel('Date')
            plt.ylabel('Individual Test Loss (MSE)')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates with adaptive resolution
            self._format_date_axis(plt, datetime_dates)
            plt.tight_layout()
            
            # Log the figure to TensorBoard
            self.writer.add_figure('Loss/Test_Individual_vs_Date', plt.gcf())
            
            # Close the figure to free memory
            plt.close()
            
            print(f"Logged {len(losses)} individual test losses vs dates as figure to TensorBoard")
            
        except Exception as e:
            print(f"Warning: Could not log loss vs date figure to TensorBoard: {e}")

            


    def get_trading_performance_metrics(self):
        """
        Calculate comprehensive trading performance metrics.
        
        Returns:
            dict: Dictionary containing all trading performance metrics
        """
        # Use daily portfolio returns if available, otherwise fall back to old data
        if len(self.daily_returns) > 0:
            returns = np.array(self.daily_returns)
        elif len(self.real_world_returns) > 0:
            returns = np.array(self.real_world_returns)
        elif len(self.actual_returns) > 0:
            returns = np.array(self.actual_returns)
        else:
            returns = np.array([])
        
        if len(returns) == 0:
            return {}
        
        returns = returns.flatten()
        n = len(returns)
        
        # Accuracy (for classification: correct sign predictions)
        if len(self.predicted_values) > 0 and len(self.actual_values) > 0:
            pred_vals = np.array(self.predicted_values).flatten()
            actual_vals = np.array(self.actual_values).flatten()
            # For classification: check if signs match
            accuracy = np.mean(np.sign(pred_vals) == np.sign(actual_vals)) * 100
        else:
            accuracy = 0
        
        # Mean daily return
        mu_daily = np.mean(returns)
        
        # Annualized return (assuming 252 trading days)
        mu_ann = mu_daily * 252
        
        # Total return
        total_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
        
        # Excess returns: Compare our returns vs baseline (actual returns or S&P 500)
        # Use actual returns as baseline if available, otherwise use simple mean
        baseline_returns = None
        if len(self.actual_returns) > 0:
            baseline_returns = np.array(self.actual_returns)
        
        if baseline_returns is not None and len(baseline_returns) > 0:
            # Calculate excess returns vs baseline
            if len(returns) == len(baseline_returns):
                excess_returns = returns - baseline_returns
                excess_daily = np.mean(excess_returns)
            elif len(returns) == len(baseline_returns) + 1:
                # Returns are one shorter due to diff
                excess_returns = returns[:-1] - baseline_returns
                excess_daily = np.mean(excess_returns)
            else:
                # Mismatched lengths, fall back to simple comparison
                baseline_daily = np.mean(baseline_returns)
                excess_daily = mu_daily - baseline_daily
        else:
            # No baseline available, set excess returns to 0
            excess_daily = 0.0
        
        excess_ann = excess_daily * 252
        
        # Standard deviation (annualized)
        sigma_daily = np.std(returns, ddof=1)
        sigma_ann = sigma_daily * np.sqrt(252)
        
        # Risk-free rate (for Sharpe and Sortino ratios)
        rf_ann = 0.0
        rf_daily = rf_ann / 252
        
        # Sharpe ratio
        sharpe = (mu_ann - rf_ann) / sigma_ann if sigma_ann > 0 else 0
        
        # Downside deviation (annualized) - MAR = 0 (minimum acceptable return)
        MAR = 0
        downside_returns = np.minimum(0, returns - MAR)
        downside_var = np.mean(downside_returns ** 2)
        downside_ann = np.sqrt(252 * downside_var)
        
        # Sortino ratio
        sortino = (mu_ann - rf_ann) / downside_ann if downside_ann > 0 else 0
        
        # Value at Risk (VaR) at 1% (alpha = 0.01)
        alpha = 0.01
        var_alpha = -np.percentile(returns, alpha * 100)
        
        # Maximum drawdown
        if len(self.equity_curve) > 0:
            equity = np.array(self.equity_curve)
        else:
            # Build equity curve from returns
            equity = np.cumprod(1 + returns)
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        mdd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Fraction of days with positive returns
        share_pos = np.mean(returns > 0) * 100
        
        # Distribution summaries
        rmin = np.min(returns)
        q1 = np.percentile(returns, 25)
        med = np.median(returns)
        q3 = np.percentile(returns, 75)
        rmax = np.max(returns)
        
        # Skewness
        if sigma_daily > 0:
            skew = np.mean(((returns - mu_daily) / sigma_daily) ** 3)
        else:
            skew = 0
        
        # Kurtosis
        if sigma_daily > 0:
            kurt = np.mean(((returns - mu_daily) / sigma_daily) ** 4)
        else:
            kurt = 0
        
        # Outperformance vs random (compare to zero-mean random walk)
        random_benchmark_return = 0  # Random walk has zero expected return
        outperformance = mu_ann - random_benchmark_return
        
        metrics = {
            'accuracy': accuracy,
            'mean_daily_return': mu_daily,
            'annualized_return': mu_ann,
            'total_return': total_return,
            'excess_return_daily': excess_daily,
            'excess_return_annualized': excess_ann,
            'standard_deviation_daily': sigma_daily,
            'standard_deviation_annualized': sigma_ann,
            'sharpe_ratio': sharpe,
            'downside_deviation_annualized': downside_ann,
            'sortino_ratio': sortino,
            'value_at_risk_1pct': var_alpha,
            'maximum_drawdown': mdd,
            'fraction_positive_returns': share_pos,
            'outperformance_vs_random': outperformance,
            'quantiles': {'q1': q1, 'median': med, 'q3': q3},
            'skewness': skew,
            'kurtosis': kurt,
            'min_return': rmin,
            'max_return': rmax
        }
        
        return metrics

    def _newey_west_variance(self, data, max_lag=None):
        """
        Compute Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) variance.
        
        Args:
            data: 1D numpy array
            max_lag: Maximum lag for autocorrelation (default: floor(4*(n/100)^(2/9)))
        
        Returns:
            float: Newey-West variance estimate
        """
        data = np.array(data).flatten()
        n = len(data)
        
        if n < 2:
            return np.var(data, ddof=1)
        
        # Default lag selection rule
        if max_lag is None:
            max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
        
        max_lag = min(max_lag, n - 1)
        
        # Mean-centered data
        data_centered = data - np.mean(data)
        
        # Base variance (sample variance)
        var_base = np.mean(data_centered ** 2)
        
        # Add autocorrelation adjustments
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
            autocov = np.mean(data_centered[:-lag] * data_centered[lag:])
            var_base += 2 * weight * autocov
        
        return var_base
    
    def _pesaran_timmermann_stat(self, pred_sign, actual_sign):
        """
        Compute Pesaran-Timmermann test statistic for sign predictions.
        
        Args:
            pred_sign: Array of predicted signs (+1 or -1)
            actual_sign: Array of actual signs (+1 or -1)
        
        Returns:
            float: PT test statistic
        """
        pred_sign = np.array(pred_sign).flatten()
        actual_sign = np.array(actual_sign).flatten()
        
        n = len(pred_sign)
        if n == 0:
            return 0
        
        # Count successes (correct predictions)
        n11 = np.sum((pred_sign > 0) & (actual_sign > 0))  # Pred +, Actual +
        n12 = np.sum((pred_sign > 0) & (actual_sign <= 0))  # Pred +, Actual -
        n21 = np.sum((pred_sign <= 0) & (actual_sign > 0))  # Pred -, Actual +
        n22 = np.sum((pred_sign <= 0) & (actual_sign <= 0))  # Pred -, Actual -
        
        # Proportions
        p1 = (n11 + n12) / n  # Proportion of positive predictions
        p2 = (n11 + n21) / n  # Proportion of positive actuals
        p = (n11 + n22) / n   # Proportion of correct predictions
        
        # Expected proportion under independence
        p_star = p1 * p2 + (1 - p1) * (1 - p2)
        
        # Variance under independence
        var_p = p_star * (1 - p_star) / n
        
        if var_p <= 0:
            return 0
        
        # PT statistic
        pt_stat = (p - p_star) / np.sqrt(var_p)
        
        return pt_stat

    def get_statistical_tests(self):
        """
        Calculate statistical tests for model evaluation.
        
        Returns:
            dict: Dictionary containing test results
        """
        tests = {}
        
        # Diebold-Mariano test
        # Compare loss from model predictions vs baseline (e.g., naive forecast or actual mean)
        if len(self.predicted_values) > 0 and len(self.actual_values) > 0:
            pred_vals = np.array(self.predicted_values).flatten()
            actual_vals = np.array(self.actual_values).flatten()
            
            # Loss from model predictions
            loss1 = (pred_vals - actual_vals) ** 2
            
            # Baseline loss (e.g., naive forecast: use previous value or mean)
            if len(actual_vals) > 1:
                # Naive forecast: use previous actual value
                baseline_pred = np.concatenate([[actual_vals[0]], actual_vals[:-1]])
                loss2 = (baseline_pred - actual_vals) ** 2
            else:
                # Single value: compare to mean
                baseline_pred = np.full_like(actual_vals, np.mean(actual_vals))
                loss2 = (baseline_pred - actual_vals) ** 2
            
            # Difference in losses
            d = loss1 - loss2
            d_bar = np.mean(d)
            
            # Newey-West variance
            var_hac = self._newey_west_variance(d)
            n = len(d)
            
            # DM statistic
            if var_hac > 0:
                dm_stat = d_bar / np.sqrt(var_hac / n)
                # Two-sided p-value
                p_value_dm = 2 * (1 - norm.cdf(abs(dm_stat)))
            else:
                dm_stat = 0
                p_value_dm = 1.0
            
            tests['diebold_mariano'] = {
                'statistic': dm_stat,
                'p_value': p_value_dm,
                'mean_loss_diff': d_bar,
                'interpretation': 'H0: Models have equal predictive ability. Reject if p < 0.05'
            }
        else:
            tests['diebold_mariano'] = {'statistic': None, 'p_value': None}
        
        # Pesaran-Timmermann test
        if len(self.predicted_values) > 0 and len(self.actual_values) > 0:
            pred_vals = np.array(self.predicted_values).flatten()
            actual_vals = np.array(self.actual_values).flatten()
            
            pred_sign = np.sign(pred_vals)
            actual_sign = np.sign(actual_vals)
            
            pt_stat = self._pesaran_timmermann_stat(pred_sign, actual_sign)
            
            # Two-sided p-value
            p_value_pt = 2 * (1 - norm.cdf(abs(pt_stat)))
            
            tests['pesaran_timmermann'] = {
                'statistic': pt_stat,
                'p_value': p_value_pt,
                'interpretation': 'H0: No predictive power for sign. Reject if p < 0.05'
            }
        else:
            tests['pesaran_timmermann'] = {'statistic': None, 'p_value': None}
        
        # Newey-West test for mean return
        if len(self.real_world_returns) > 0:
            returns = np.array(self.real_world_returns).flatten()
        elif len(self.actual_returns) > 0:
            returns = np.array(self.actual_returns).flatten()
        else:
            returns = np.array([])
        
        if len(returns) > 0:
            mean_return = np.mean(returns)
            n = len(returns)
            
            # Newey-West variance
            var_nw = self._newey_west_variance(returns)
            
            # t-statistic
            if var_nw > 0:
                se_nw = np.sqrt(var_nw / n)
                t_stat = mean_return / se_nw
                
                # p-value (two-sided t-test)
                p_value_nw = 2 * (1 - norm.cdf(abs(t_stat)))
            else:
                t_stat = 0
                p_value_nw = 1.0
                se_nw = 0
            
            tests['newey_west_mean_return'] = {
                'statistic': t_stat,
                'p_value': p_value_nw,
                'mean_return': mean_return,
                'standard_error': se_nw,
                'interpretation': 'H0: Mean return is zero. Reject if p < 0.05'
            }
        else:
            tests['newey_west_mean_return'] = {'statistic': None, 'p_value': None}
        
        return tests

    def get_time_series_diagnostics(self):
        """
        Calculate time series diagnostics for residuals and returns.
        
        Returns:
            dict: Dictionary containing time series diagnostic metrics
        """
        diagnostics = {}
        
        # Get residuals (prediction errors)
        if len(self.residuals) > 0:
            residuals = np.array(self.residuals).flatten()
        elif len(self.predicted_values) > 0 and len(self.actual_values) > 0:
            pred_vals = np.array(self.predicted_values).flatten()
            actual_vals = np.array(self.actual_values).flatten()
            residuals = actual_vals - pred_vals
        else:
            residuals = np.array([])
        
        # Get returns for stationarity tests
        if len(self.real_world_returns) > 0:
            returns = np.array(self.real_world_returns).flatten()
        elif len(self.actual_returns) > 0:
            returns = np.array(self.actual_returns).flatten()
        else:
            returns = np.array([])
        
        # AIC and BIC (for residuals assuming normal distribution)
        if len(residuals) > 0:
            n = len(residuals)
            
            # Estimate log-likelihood assuming normal distribution
            sigma_sq = np.var(residuals, ddof=1)
            if sigma_sq > 0:
                loglik = -0.5 * n * (np.log(2 * np.pi * sigma_sq) + 1)
            else:
                loglik = 0
            
            # Number of parameters - use stored model parameter count
            k = self.num_model_parameters if hasattr(self, 'num_model_parameters') else 100
            
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
            
            diagnostics['aic'] = aic
            diagnostics['bic'] = bic
            diagnostics['log_likelihood'] = loglik
            diagnostics['num_parameters'] = k  # Should be updated with actual parameter count
        else:
            diagnostics['aic'] = None
            diagnostics['bic'] = None
            diagnostics['log_likelihood'] = None
        
        # Ljung-Box test for autocorrelation in residuals
        if len(residuals) > 0:
            try:
                # Use max lag based on data size (common: min(10, n/5))
                max_lag = min(10, max(1, len(residuals) // 5))
                
                # Use statsmodels acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
                
                # Extract Q statistic and p-value for the maximum lag
                q_stat = lb_result['lb_stat'].iloc[-1]
                p_value_lb = lb_result['lb_pvalue'].iloc[-1]
                
                diagnostics['ljung_box'] = {
                    'q_statistic': q_stat,
                    'p_value': p_value_lb,
                    'max_lag': max_lag,
                    'interpretation': 'H0: No autocorrelation. Reject if p < 0.05'
                }
            except Exception as e:
                diagnostics['ljung_box'] = {
                    'error': str(e),
                    'interpretation': 'Could not compute Ljung-Box test'
                }
        else:
            diagnostics['ljung_box'] = {'q_statistic': None, 'p_value': None}
        
        # ADF (Augmented Dickey-Fuller) stationarity test
        if len(residuals) > 0:
            try:
                adf_result = adfuller(residuals, autolag='AIC')
                adf_stat = adf_result[0]
                p_value_adf = adf_result[1]
                adf_critical = adf_result[4]
                
                diagnostics['adf_residuals'] = {
                    'statistic': adf_stat,
                    'p_value': p_value_adf,
                    'critical_values': adf_critical,
                    'interpretation': 'H0: Series is non-stationary. Reject if p < 0.05'
                }
            except Exception as e:
                diagnostics['adf_residuals'] = {
                    'error': str(e),
                    'interpretation': 'Could not compute ADF test for residuals'
                }
        else:
            diagnostics['adf_residuals'] = {'statistic': None, 'p_value': None}
        
        # ADF test for returns
        if len(returns) > 0:
            try:
                adf_result_returns = adfuller(returns, autolag='AIC')
                adf_stat_returns = adf_result_returns[0]
                p_value_adf_returns = adf_result_returns[1]
                adf_critical_returns = adf_result_returns[4]
                
                diagnostics['adf_returns'] = {
                    'statistic': adf_stat_returns,
                    'p_value': p_value_adf_returns,
                    'critical_values': adf_critical_returns,
                    'interpretation': 'H0: Series is non-stationary. Reject if p < 0.05'
                }
            except Exception as e:
                diagnostics['adf_returns'] = {
                    'error': str(e),
                    'interpretation': 'Could not compute ADF test for returns'
                }
        else:
            diagnostics['adf_returns'] = {'statistic': None, 'p_value': None}
        
        # ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)
        if len(residuals) > 0:
            try:
                max_lag_acf = min(40, max(1, len(residuals) // 4))
                
                # ACF
                acf_vals = acf(residuals, nlags=max_lag_acf, fft=True)
                
                # PACF
                pacf_vals = pacf(residuals, nlags=max_lag_acf)
                
                diagnostics['acf'] = {
                    'values': acf_vals.tolist(),
                    'max_lag': max_lag_acf,
                    'first_lag': acf_vals[1] if len(acf_vals) > 1 else 0
                }
                
                diagnostics['pacf'] = {
                    'values': pacf_vals.tolist(),
                    'max_lag': max_lag_acf,
                    'first_lag': pacf_vals[1] if len(pacf_vals) > 1 else 0
                }
            except Exception as e:
                diagnostics['acf'] = {'error': str(e)}
                diagnostics['pacf'] = {'error': str(e)}
        else:
            diagnostics['acf'] = {'values': None}
            diagnostics['pacf'] = {'values': None}
        
        # ACF/PACF for returns
        if len(returns) > 0:
            try:
                max_lag_acf = min(40, max(1, len(returns) // 4))
                
                # ACF
                acf_returns = acf(returns, nlags=max_lag_acf, fft=True)
                
                # PACF
                pacf_returns = pacf(returns, nlags=max_lag_acf)
                
                diagnostics['acf_returns'] = {
                    'values': acf_returns.tolist(),
                    'max_lag': max_lag_acf,
                    'first_lag': acf_returns[1] if len(acf_returns) > 1 else 0
                }
                
                diagnostics['pacf_returns'] = {
                    'values': pacf_returns.tolist(),
                    'max_lag': max_lag_acf,
                    'first_lag': pacf_returns[1] if len(pacf_returns) > 1 else 0
                }
            except Exception as e:
                diagnostics['acf_returns'] = {'error': str(e)}
                diagnostics['pacf_returns'] = {'error': str(e)}
        else:
            diagnostics['acf_returns'] = {'values': None}
            diagnostics['pacf_returns'] = {'values': None}
        
        return diagnostics

    def calculate_real_world_metrics(self, rev_pred, actual_rev=None, indices=None):
        """
        Calculate real-world trading metrics based on revenue predictions.
        
        Args:
            rev_pred: Predicted revenue values (tensor or array)
            actual_rev: Actual revenue values (optional, uses test_revenues if None)
            indices: Batch indices to map to test data
        """
        # Convert to numpy
        rev_pred = self._tensor_to_numpy(rev_pred)
        if actual_rev is None:
            # Use stored test revenues if available
            if indices is not None and hasattr(self, 'test_revenues') and self.test_revenues is not None:
                # Convert test_revenues to numpy if it's a tensor
                if isinstance(self.test_revenues, torch.Tensor):
                    test_revenues_np = self._tensor_to_numpy(self.test_revenues)
                else:
                    test_revenues_np = np.array(self.test_revenues)
                
                if hasattr(indices, 'tolist'):
                    indices_list = indices.tolist()
                else:
                    indices_list = indices
                actual_rev = np.array([test_revenues_np[idx] if idx < len(test_revenues_np) else 0 
                                     for idx in indices_list])
            else:
                # If no actual revenue available, skip
                return
        
        rev_pred = np.array(rev_pred).flatten()
        actual_rev = np.array(actual_rev).flatten()
        
        if len(rev_pred) == 0 or len(actual_rev) == 0:
            return
        
        # Get top k and bottom k stocks by revenue prediction
        # For now, use k = min(10, len(stocks)//2) or a reasonable fraction
        k = max(1, min(10, len(rev_pred) // 10))
        
        # Get top k indices (highest predicted revenue)
        top_k_indices = np.argsort(rev_pred)[-k:]
        # Get bottom k indices (lowest predicted revenue)
        bottom_k_indices = np.argsort(rev_pred)[:k]
        
        # Calculate real revenue for top k and bottom k
        # For classification: prediction is sign (-1 or 1), actual_rev is actual revenue
        # Real return = sign(prediction) * actual_rev
        top_k_real_rev = rev_pred[top_k_indices] * actual_rev[top_k_indices]
        bottom_k_real_rev = rev_pred[bottom_k_indices] * actual_rev[bottom_k_indices]
        
        # Combined real returns (long top k, short bottom k)
        real_returns = np.concatenate([top_k_real_rev, -bottom_k_real_rev])  # Short bottom k
        
        # Keep track of real-world returns
        self.real_world_returns.extend(real_returns.tolist())
        
        # Calculate percentage of accurate predictions (real revenue > 0)
        if len(real_returns) > 0:
            accuracy = np.mean(real_returns > 0) * 100
        else:
            accuracy = 0
        
        # Update equity curve
        if len(self.equity_curve) == 0:
            self.equity_curve = [1.0]
        else:
            last_equity = self.equity_curve[-1]
            # Update equity with average return for this batch
            avg_return = np.mean(real_returns) if len(real_returns) > 0 else 0
            self.equity_curve.append(last_equity * (1 + avg_return))






    def evaluate(self):
        self.Model.eval()
        test_loss = 0
        individual_losses = []
        individual_dates = []
        
        # Clear previous evaluation data
        self.predicted_values = []
        self.actual_values = []
        self.predicted_returns = []
        self.actual_returns = []
        self.residuals = []
        self.evaluation_dates = []
        self.real_world_returns = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Collect data by date for cross-sectional ranking
        by_date = {}  # date -> list of (score, realized_return)
        
        with torch.no_grad():
            tbar = tqdm(self.testLoader, total=len(self.testLoader), 
                       desc=f"test", disable=not self.is_main)
            for X_batch, Y_batch, indices in tbar:
                X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
                Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
                
                # Get dates for this batch
                batch_dates = [self.test_dates[idx] for idx in indices.tolist()]
                self.evaluation_dates.extend(batch_dates)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        Y_pred = self.Model(X_batch)
                else:
                    Y_pred = self.Model(X_batch)
                
                # Store predictions and actual values
                Y_pred_np = self._tensor_to_numpy(Y_pred)
                Y_batch_np = self._tensor_to_numpy(Y_batch)
                
                self.predicted_values.extend(Y_pred_np.flatten().tolist())
                self.actual_values.extend(Y_batch_np.flatten().tolist())
                
                # Store residuals
                residuals_np = Y_batch_np.flatten() - Y_pred_np.flatten()
                self.residuals.extend(residuals_np.tolist())
                
                # Collect (date, score, realized_return) for cross-sectional ranking
                if hasattr(self, 'test_revenues') and self.test_revenues is not None:
                    # Convert test_revenues to numpy if it's a tensor
                    if isinstance(self.test_revenues, torch.Tensor):
                        test_revenues_np = self._tensor_to_numpy(self.test_revenues)
                    else:
                        test_revenues_np = np.array(self.test_revenues)
                    
                    # Get actual revenues for this batch
                    indices_list = indices.tolist()
                    scores = Y_pred_np.flatten()
                    realized_rets = np.array([test_revenues_np[idx] if idx < len(test_revenues_np) else 0 
                                            for idx in indices_list])
                    
                    # Collect by date
                    for d, score, ret in zip(batch_dates, scores, realized_rets):
                        by_date.setdefault(d, []).append((score, ret))
                
                # Calculate test loss
                loss = self.loss_fn(Y_pred, Y_batch)
                test_loss += loss.item()
                
                if self.is_main:
                    tbar.set_postfix(avg_loss=test_loss/(tbar.n or 1))
                
                # Calculate individual losses for each sample in the batch
                # Y_pred and Y_batch have shape [batch_size, 1], so we squeeze to [batch_size]
                individual_loss = ((Y_pred - Y_batch) ** 2).squeeze()
                # Convert to numpy in a device-agnostic way
                individual_losses.extend(self._tensor_to_numpy(individual_loss))
                individual_dates.extend(batch_dates)
        
        # Aggregate test loss across all ranks
        if self.is_dist:
            test_loss_tensor = torch.tensor(test_loss, device=self.device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            test_loss = test_loss_tensor.item()
            test_samples = len(self.testLoader) * self.world_size
        else:
            test_samples = len(self.testLoader)
        
        avg_test_loss = test_loss / test_samples
        
        # Build daily portfolio returns from cross-sectional ranking
        if len(by_date) > 0 and self.is_main:
            for d in sorted(by_date.keys()):
                pairs = by_date[d]
                # Rank by score (ascending order)
                pairs.sort(key=lambda x: x[0])
                # Get top-k and bottom-k returns (but adjust k if not enough data)
                # Need at least 2 stocks (one to long, one to short)
                if len(pairs) >= 2:
                    # Use k or adjust down if necessary
                    actual_k = min(self.k, len(pairs) // 2)
                    top_k_returns = [r for (_, r) in pairs[-actual_k:]]
                    bottom_k_returns = [r for (_, r) in pairs[:actual_k]]
                    
                    # Equal-weight long-short portfolio return
                    r_d = np.mean(top_k_returns) - np.mean(bottom_k_returns)
                    
                    # Subtract transaction costs
                    if self.cost_bps_per_side > 0:
                        r_d -= 2 * self.cost_bps_per_side / 10000.0
                    
                    self.daily_returns.append(r_d)
        
        # Build equity curve from daily returns
        if len(self.daily_returns) > 0:
            self.equity_curve = np.cumprod(1 + np.array(self.daily_returns)).tolist()
        
        # Only rank 0 prints and logs
        if self.is_main:
            print("Test Loss: {}".format(avg_test_loss))
            # Note: Individual test losses are logged vs timestamps below
            
            # Log individual losses against dates to TensorBoard
            if individual_losses and individual_dates and self.writer is not None:
                self._log_loss_vs_date_to_tensorboard(individual_losses, individual_dates)
            
            # Calculate and print all metrics
            print("\n" + "="*60)
            print("TRADING PERFORMANCE METRICS")
            print("="*60)
            trading_metrics = self.get_trading_performance_metrics()
            for key, value in trading_metrics.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
            
            print("\n" + "="*60)
            print("STATISTICAL TESTS")
            print("="*60)
            statistical_tests = self.get_statistical_tests()
            for test_name, test_results in statistical_tests.items():
                print(f"{test_name}:")
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {test_results}")
            
            print("\n" + "="*60)
            print("TIME SERIES DIAGNOSTICS")
            print("="*60)
            diagnostics = self.get_time_series_diagnostics()
            for key, value in diagnostics.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        if sub_key != 'values' or len(str(sub_value)) < 100:  # Don't print full ACF/PACF arrays
                            print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
            
            if self.writer is not None:
                self.writer.flush()
        
        return avg_test_loss

    def load_separate_datasets(self, stocks, time_args):
        """
        Load train, validation, and test datasets from separate .npz files.
        This method can be used to explicitly load separate datasets.
        """
        # Use load_data_from_cache which handles separate .npz files
        # Use default seq_len=240 (lookback window) for backward compatibility
        input_data = util.load_data_from_cache(stocks, time_args, prediction_type=self.prediction_type, use_nlp=self.use_nlp, nlp_method=self.nlp_method, seq_len=self.config.lookback, data_source=self.config.data_source)
        if input_data is None:
            raise RuntimeError(f"Could not load separate datasets from cache in {util.DATA_DIR}. Data may need to be downloaded first.")
        return input_data

    def stop(self):
        if self.writer is not None:
            self.writer.close()
        
        # Clean up distributed process group
        if self.is_dist:
            dist.destroy_process_group()

