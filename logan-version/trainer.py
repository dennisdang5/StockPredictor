import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
from model import LSTMModel, CNNLSTMModel
import util
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from tqdm import tqdm
from helpers_workers import dataloader_kwargs, autotune_num_workers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self, stocks=["MSFT", "AAPL"], time_args=["3y"], batch_size=8, num_epochs=10000, saved_model=None, prediction_type="classification", k=10, cost_bps_per_side=5.0, save_every_epochs=25):
        # Setup distributed training
        self.local_rank, device, self.is_dist = setup_dist()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.is_main = (self.rank == 0)

        self.stocks = stocks
        self.time_args = time_args
        self.batch_size = batch_size
        self.prediction_type = prediction_type
        self.k = k  # Number of top/bottom positions for long-short portfolio
        self.cost_bps_per_side = cost_bps_per_side  # Transaction costs per side in basis points
        # Only rank 0 creates TensorBoard writer
        self.writer = SummaryWriter() if self.is_main else None

        # Use distributed device if available, otherwise pick best device
        if self.is_dist:
            self.device = device
        else:
            self.device = pick_device()
        
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        

        self.num_workers = 0
        # =========================
        # RECOMMENDED WORKERS HERE
        # =========================
        # Choose an I/O profile for your pipeline (edit as needed)
        io_profile = "medium"   # {"light","medium","heavy"}

        # Get heuristic kwargs for DataLoader based on device + SLURM CPU budget
        loader_args = dataloader_kwargs(self.device, io=io_profile)

        # Save to instance attributes for consistency + logging
        #self.num_workers        = loader_args["num_workers"]
        self.num_workers        = 1
        self.persistent_workers = loader_args["persistent_workers"]
        self.pin_memory         = loader_args["pin_memory"]

        # Update loader_args with the overridden num_workers value
        loader_args["num_workers"] = self.num_workers
        loader_args["persistent_workers"] = bool(self.num_workers)

        # DataLoader doesn't accept None for prefetch_factor, so strip it if absent
        if loader_args.get("prefetch_factor", None) is None:
            loader_args.pop("prefetch_factor", None)
        self.loader_args = loader_args

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

        # data - distributed loading: only rank 0 downloads/processes, all ranks load from cache
        # Optimized flow:
        # Step 1: Check cache first with full stock list (if exact match exists, use it - saves all time!)
        # Step 2: Load saved problematic stocks for this time period (if exists)
        # Step 3: Remove problematic stocks from input set
        # Step 4: Check cache with filtered stocks
        # Step 5: If cache doesn't exist, download data (which will save problematic stocks)
        # Step 6: Process and save data
        if self.is_dist:
            # Only rank 0 processes data
            if self.is_main:
                # Step 1: Check cache first with full stock list (optimization: if exact match exists, use it)
                input_data = util.load_data_from_cache(stocks, time_args, data_dir="data", prediction_type=self.prediction_type)
                
                if input_data is None:
                    # Step 2: Load saved problematic stocks for this time period
                    problematic_stocks_saved = util._load_problematic_stocks(time_args, data_dir="data")
                    
                    # Step 3: Remove problematic stocks from input set
                    if problematic_stocks_saved:
                        valid_stocks = [stock for stock in stocks if stock not in problematic_stocks_saved]
                        print(f"[data] Loaded {len(problematic_stocks_saved)} previously identified problematic stocks for this time period")
                        print(f"[data] Filtered input: {len(stocks)} -> {len(valid_stocks)} stocks")
                    else:
                        valid_stocks = stocks
                    
                    if len(valid_stocks) == 0:
                        raise RuntimeError("No valid stocks found after filtering problematic stocks.")
                    
                    # Step 4: Check cache with filtered stocks
                    input_data = util.load_data_from_cache(valid_stocks, time_args, data_dir="data", prediction_type=self.prediction_type)
                    
                    if input_data is None:
                        # Step 5: Download data (will save problematic stocks for future runs)
                        print("[data] Cache not found, downloading data (rank 0 only)...")
                        open_close, failed_stocks_dict = util.handle_yfinance_errors(valid_stocks, time_args, max_retries=1)
                        
                        if open_close is None:
                            raise RuntimeError("ERROR: Failed to download any stock data. Cannot proceed.")
                        
                        # Calculate problematic stocks from this download (new problematic stocks found in valid_stocks)
                        new_problematic = [stock for stock in valid_stocks if stock not in open_close["Open"].columns]
                        # Combine with previously known problematic stocks
                        all_problematic = list(problematic_stocks_saved) + new_problematic
                        
                        # Step 6: Process and save data (get_data will save problematic stocks)
                        print("[data] Processing downloaded data and saving to cache...")
                        input_data = util.get_data(valid_stocks, time_args, data_dir="data", prediction_type=self.prediction_type, open_close_data=open_close, problematic_stocks=all_problematic if all_problematic else None)
                        if isinstance(input_data, int):
                            raise RuntimeError("Error getting data from util.get_data()")
                    else:
                        print("[data] Found cache for filtered stocks (rank 0 only)")
                else:
                    print("[data] Loaded from cache (rank 0 only)")
            
            # Synchronize - ensure rank 0 finishes downloading/processing before others proceed
            dist.barrier()
            
            # Now all ranks load from cache (they need to filter the same way to get the same cache key)
            if not self.is_main:
                # Step 2: Load saved problematic stocks (same as rank 0)
                problematic_stocks_saved = util._load_problematic_stocks(time_args, data_dir="data")
                
                # Step 3: Remove problematic stocks from input set (same as rank 0)
                if problematic_stocks_saved:
                    valid_stocks = [stock for stock in stocks if stock not in problematic_stocks_saved]
                else:
                    valid_stocks = stocks
                
                if len(valid_stocks) == 0:
                    raise RuntimeError("No valid stocks found after filtering problematic stocks.")
                
                # Step 4: Load from cache with filtered stocks
                input_data = util.load_data_from_cache(valid_stocks, time_args, data_dir="data", prediction_type=self.prediction_type)
                if input_data is None:
                    raise RuntimeError(f"Cache files not found after rank 0 processing. Expected cache should exist.")
            
            # Synchronize again - ensure all ranks have loaded data
            dist.barrier()
        else:
            # Non-distributed: optimized flow
            # Step 1: Check cache first with full stock list
            input_data = util.load_data_from_cache(stocks, time_args, data_dir="data", prediction_type=self.prediction_type)
            
            if input_data is None:
                # Step 2: Load saved problematic stocks for this time period
                problematic_stocks_saved = util._load_problematic_stocks(time_args, data_dir="data")
                
                # Step 3: Remove problematic stocks from input set
                if problematic_stocks_saved:
                    valid_stocks = [stock for stock in stocks if stock not in problematic_stocks_saved]
                    print(f"[data] Loaded {len(problematic_stocks_saved)} previously identified problematic stocks for this time period")
                    print(f"[data] Filtered input: {len(stocks)} -> {len(valid_stocks)} stocks")
                else:
                    valid_stocks = stocks
                
                if len(valid_stocks) == 0:
                    raise RuntimeError("No valid stocks found after filtering problematic stocks.")
                
                # Step 4: Check cache with filtered stocks
                input_data = util.load_data_from_cache(valid_stocks, time_args, data_dir="data", prediction_type=self.prediction_type)
                
                if input_data is None:
                    # Step 5: Download data (will save problematic stocks for future runs)
                    print("[data] Cache not found, downloading data...")
                    open_close, failed_stocks_dict = util.handle_yfinance_errors(valid_stocks, time_args, max_retries=1)
                    
                    if open_close is None:
                        raise RuntimeError("ERROR: Failed to download any stock data. Cannot proceed.")
                    
                    # Calculate problematic stocks from this download (new problematic stocks found in valid_stocks)
                    new_problematic = [stock for stock in valid_stocks if stock not in open_close["Open"].columns]
                    # Combine with previously known problematic stocks
                    all_problematic = list(problematic_stocks_saved) + new_problematic if problematic_stocks_saved else new_problematic
                    
                    # Step 6: Process and save data (get_data will save problematic stocks)
                    print("[data] Processing downloaded data and saving to cache...")
                    input_data = util.get_data(valid_stocks, time_args, data_dir="data", prediction_type=self.prediction_type, open_close_data=open_close, problematic_stocks=all_problematic if all_problematic else None)
                    if isinstance(input_data, int):
                        raise RuntimeError("Error getting data from util.get_data()")
                else:
                    print("[data] Found cache for filtered stocks")
            else:
                print("[data] Loaded from cache")
        
        # Handle multiple data formats:
        # - 10 elements: old format (no S&P 500, no Returns)
        # - 11 elements: format with S&P 500 but no Returns
        # - 12 elements: new format with S&P 500 and Returns
        if len(input_data) == 10:
            X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test, Rev_test = input_data
            Returns_test = None  # Old format doesn't have Returns
            Sp500_test = None  # Old format doesn't have S&P 500
        elif len(input_data) == 11:
            X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test, Rev_test, Sp500_test = input_data
            Returns_test = None  # Format doesn't have Returns
        elif len(input_data) == 12:
            X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test, Rev_test, Returns_test, Sp500_test = input_data
        else:
            raise ValueError(f"Unexpected number of elements in input_data: {len(input_data)}. Expected 10, 11, or 12 elements.")
        
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

        # (Optional) quick autotune once; uncomment if you want to probe
        #tuned = autotune_num_workers(self.train_ds, self.batch_size, self.device, candidates=(0,1,2,4,8))
        #print(f"[dataloader] autotuned num_workers={tuned}")
        #self.num_workers = tuned
        #self.persistent_workers = bool(tuned)
        ## Update loader_args with tuned values, handling prefetch_factor correctly
        #self.loader_args.update(num_workers=tuned, persistent_workers=bool(tuned))
        #if tuned == 0:
        #    self.loader_args.pop("prefetch_factor", None)
        #else:
        #    self.loader_args["prefetch_factor"] = 2

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
        self.Model = CNNLSTMModel()
        
        # Move model to device first
        self.Model = self.Model.to(self.device)
        
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

        if self.is_main:
            print("{} total parameters".format(sum(param.numel() for param in self.Model.parameters())))

        # Determine the save path for the model
        # If saved_model is provided, use it for both loading and saving
        # Otherwise, default to "savedmodel.pth"
        if saved_model is not None:
            self.save_path = saved_model
        else:
            self.save_path = "savedmodel.pth"
        
        if saved_model is not None:
            if os.path.exists(saved_model):
                state_dict = torch.load(saved_model, map_location="cpu")
                # Load into underlying module if DataParallel/DDP is active
                target_module = self.Model.module if hasattr(self.Model, "module") else self.Model
                target_module.load_state_dict(state_dict)
                if self.is_main:
                    print(f"[load] restored weights from {saved_model}")
            else:
                if self.is_main:
                    print(f"[load] No saved model found at {saved_model}, starting with random weights")

        # Use a lower learning rate to prevent instability
        # Even lower learning rate for more stability
        self.optimizer = optim.Adam(self.Model.parameters(), lr=5e-5, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        # More aggressive gradient clipping to prevent exploding gradients
        self.max_grad_norm = 0.5
        self.stopper = EarlyStopper(patience=50, min_delta=0.001, is_dist=self.is_dist, rank=self.rank, save_path=self.save_path)
        self.num_epochs = num_epochs
        self.save_every_epochs = save_every_epochs
        
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
                torch.nn.utils.clip_grad_norm_(self.Model.parameters(), self.max_grad_norm)
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
                torch.nn.utils.clip_grad_norm_(self.Model.parameters(), self.max_grad_norm)
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
        
        # Only rank 0 prints and logs
        if self.is_main:
            print("Training Loss: {}".format(avg_train))
            print("Validation Loss: {}".format(avg_val))
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
            torch.save(model_to_save.state_dict(), self.save_path)
            print(f"[periodic save] Model saved at epoch {epoch} to {self.save_path}")
            # Reset recovery counter on successful save (model is stable)
            self.nan_recovery_attempts = 0
            self.last_good_checkpoint = self.save_path
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

    def load_separate_datasets(self, stocks, time_args, data_dir="data"):
        """
        Load train, validation, and test datasets from separate .npz files.
        This method can be used to explicitly load separate datasets.
        """
        # Use load_data_from_cache which handles separate .npz files
        input_data = util.load_data_from_cache(stocks, time_args, data_dir=data_dir, prediction_type=self.prediction_type)
        if input_data is None:
            raise RuntimeError(f"Could not load separate datasets from cache in {data_dir}. Data may need to be downloaded first.")
        return input_data

    def stop(self):
        if self.writer is not None:
            self.writer.close()
        
        # Clean up distributed process group
        if self.is_dist:
            dist.destroy_process_group()

