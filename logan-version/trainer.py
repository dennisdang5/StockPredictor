import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
from model import LSTMModelPricePredict
import util
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
from helpers_workers import dataloader_kwargs, autotune_num_workers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

def setup_dist():
    if not torch.cuda.is_available() or os.getenv("RANK") is None:
        return None, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
    dist.init_process_group("nccl")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
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
    def __init__(self, patience=10, min_delta=0, is_dist=False, rank=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.is_dist = is_dist
        self.rank = rank
        self.is_main = (rank == 0)

    def early_stop(self, validation_loss, model):
        # In distributed mode, we need to synchronize early stopping across all ranks
        if self.is_dist:
            # Create tensors for synchronization
            device = next(model.parameters()).device
            should_stop_tensor = torch.tensor(0, dtype=torch.int, device=device)
            
            # Check if this rank should stop
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
                if self.is_main:
                    torch.save((model.module if hasattr(model, "module") else model).state_dict(), "savedmodel.pth")
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    should_stop_tensor = torch.tensor(1, dtype=torch.int, device=device)
            
            # Broadcast the early stopping decision from rank 0 to all ranks
            # First, gather all ranks' decisions
            dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
            
            # If any rank wants to stop, all ranks should stop
            return should_stop_tensor.item() == 1
        else:
            # Non-distributed mode - original logic
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
                torch.save(model.state_dict(), "savedmodel.pth")
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        

class Trainer():
    def __init__(self, stocks=["MSFT", "AAPL"], time_args=["3y"], batch_size=8, num_epochs=10000, saved_model=None, prediction_type="classification"):
        # Setup distributed training
        self.local_rank, device, self.is_dist = setup_dist()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.is_main = (self.rank == 0)

        self.stocks = stocks
        self.time_args = time_args
        self.batch_size = batch_size
        self.prediction_type = prediction_type
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

        # data - now using local data loading (includes dates)
        # This will automatically try separate format first, then fall back to combined format
        input_data = util.get_data(stocks, time_args, self.prediction_type)
        if isinstance(input_data, int):
            raise RuntimeError("Error getting data from util.get_data()")
        X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test = input_data
        # Store dates for later reference (even when data is shuffled)
        self.train_dates = D_train
        self.val_dates = D_val
        self.test_dates = D_test
        
        self.train_ds = IndexedDataset(X_train, Y_train, D_train)
        val_ds   = IndexedDataset(X_val,   Y_val,   D_val)
        test_ds  = IndexedDataset(X_test,  Y_test,  D_test)

        self.train_sampler = DistributedSampler(self.train_ds, shuffle=True) if self.is_dist else None
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if self.is_dist else None
        test_sampler   = DistributedSampler(test_ds,   shuffle=False, drop_last=False) if self.is_dist else None

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
            val_ds,   shuffle=False, sampler=val_sampler, batch_size=self.batch_size, **self.loader_args
        )
        self.testLoader = data.DataLoader(
            test_ds,  shuffle=False, sampler=test_sampler, batch_size=self.batch_size, **self.loader_args
        )

        # =========================
        # MODEL
        # =========================
        self.Model = LSTMModelPricePredict()
        
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

        self.optimizer = optim.Adam(self.Model.parameters())
        self.loss_fn = nn.MSELoss()
        self.stopper = EarlyStopper(patience=50, min_delta=0.001, is_dist=self.is_dist, rank=self.rank)
        self.num_epochs = num_epochs

    def train_one_epoch(self, epoch):

        # Ensure DDP shuffles differently each epoch
        if self.is_dist and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        if self.is_main:
            print("Epoch: {}".format(epoch+1))
            print("--------------------------------------------")
        
        self.Model.train()
        train_loss = 0
        val_loss = 0
        stop_condition=None

        start_time = time.perf_counter()

        # Use stored trainLoader, only show progress bar on rank 0
        pbar = tqdm(self.trainLoader, total=len(self.trainLoader), 
                   desc=f"train {epoch+1}/{self.num_epochs}", disable=not self.is_main)
        for X_batch, Y_batch, indices in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            # move data to device
            X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
            Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
            
            # Get dates for this batch (for debugging/analysis)
            batch_dates = [self.train_dates[idx] for idx in indices.tolist()]

            # forward + loss (+AMP on CUDA)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    Y_pred = self.Model(X_batch)
                    loss = self.loss_fn(Y_pred, Y_batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                Y_pred = self.Model(X_batch)
                loss = self.loss_fn(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()


            train_loss += loss.item()
            
            if self.is_main:
                pbar.set_postfix(avg_train=train_loss/(pbar.n or 1))
        
        # Aggregate training loss across all ranks
        if self.is_dist:
            train_loss_tensor = torch.tensor(train_loss, device=self.device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item()
            train_samples = len(self.trainLoader) * self.world_size
        else:
            train_samples = len(self.trainLoader)
        
        avg_train = train_loss / train_samples

        self.Model.eval()
        
        with torch.no_grad():
            vbar = tqdm(self.validationLoader, total=len(self.validationLoader), 
                       desc=f"validation", disable=not self.is_main)
            for X_batch, Y_batch, indices in vbar:
                X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
                Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
                
                # Get dates for this batch (for debugging/analysis)
                batch_dates = [self.val_dates[idx] for idx in indices.tolist()]
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        Y_pred = self.Model(X_batch)
                        loss = self.loss_fn(Y_pred, Y_batch)
                else:
                    Y_pred = self.Model(X_batch)
                    loss = self.loss_fn(Y_pred, Y_batch)
                val_loss += loss.item()
                if self.is_main:
                    vbar.set_postfix(avg_val=val_loss/(vbar.n or 1))
            
            # Aggregate validation loss across all ranks
            if self.is_dist:
                val_loss_tensor = torch.tensor(val_loss, device=self.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item()
                val_samples = len(self.validationLoader) * self.world_size
            else:
                val_samples = len(self.validationLoader)
            
            avg_val = val_loss / val_samples
            
            # Synchronize before early stopping check
            if self.is_dist:
                dist.barrier()
            
            stop_condition = self.stopper.early_stop(avg_val, self.Model)   
            
            # If early stopping is triggered, synchronize all ranks before exiting
            if stop_condition and self.is_dist:
                dist.barrier()  # Ensure all ranks are synchronized before any rank exits

        end_time = time.perf_counter()
        
        # Only rank 0 prints and logs
        if self.is_main:
            print("Training Loss: {}".format(avg_train))
            print("Validation Loss: {}".format(avg_val))
            print("Training Time: {:.6f}s".format(end_time-start_time))
            if stop_condition:
                print("Early stop at epoch: {}".format(epoch))
            print("--------------------------------------------")
            
            # Only log to TensorBoard if writer exists
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', avg_train, epoch+1)
                self.writer.add_scalars('Loss/trainVSvalidation', {"Training":(avg_train), "Validation":(avg_val)}, epoch+1)
                self.writer.add_scalar('Train Time', (end_time-start_time), epoch+1)
                self.writer.flush()

        return stop_condition
    
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
                # pandas DatetimeIndex
                datetime_dates = [date.to_pydatetime() for date in dates]
            elif hasattr(dates[0], 'timestamp'):
                # pandas Timestamp
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

        

    def evaluate(self):
        self.Model.eval()
        test_loss = 0
        individual_losses = []
        individual_dates = []
        
        with torch.no_grad():
            tbar = tqdm(self.testLoader, total=len(self.testLoader), 
                       desc=f"test", disable=not self.is_main)
            for X_batch, Y_batch, indices in tbar:
                X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
                Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
                
                # Get dates for this batch
                batch_dates = [self.test_dates[idx] for idx in indices.tolist()]
                
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        Y_pred = self.Model(X_batch)
                        loss = self.loss_fn(Y_pred, Y_batch)
                else:
                    Y_pred = self.Model(X_batch)
                    loss = self.loss_fn(Y_pred, Y_batch)
                
                test_loss += loss.item()
                if self.is_main:
                    tbar.set_postfix(avg_test=test_loss/(tbar.n or 1))
                
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
        
        # Only rank 0 prints and logs
        if self.is_main:
            print("Test Loss: {}".format(avg_test_loss))
            # Note: Individual test losses are logged vs timestamps below
            
            # Log individual losses against dates to TensorBoard
            if individual_losses and individual_dates and self.writer is not None:
                self._log_loss_vs_date_to_tensorboard(individual_losses, individual_dates)
            
            if self.writer is not None:
                self.writer.flush()
        
        return avg_test_loss

    def load_separate_datasets(self, stocks, time_args, data_dir="data"):
        """
        Load train, validation, and test datasets from separate .npz files.
        This method can be used to explicitly load separate datasets.
        """
        input_data = util.load_separate_datasets(stocks, time_args, data_dir)
        if isinstance(input_data, int):
            raise RuntimeError("Error loading separate datasets from util.load_separate_datasets()")
        return input_data

    def stop(self):
        if self.writer is not None:
            self.writer.close()
        
        # Clean up distributed process group
        if self.is_dist:
            dist.destroy_process_group()
