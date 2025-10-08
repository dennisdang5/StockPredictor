import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
import model
import util
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
from helpers_workers import dataloader_kwargs, autotune_num_workers

class IndexedDataset(data.Dataset):
    """
    Custom dataset that tracks indices to maintain correspondence with dates
    even when data is shuffled.
    """
    def __init__(self, X, Y, dates):
        self.X = X
        self.Y = Y
        self.dates = dates
        print(len(X), len(Y), len(dates))
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
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), 'savedmodel.pth')

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        

class Trainer():
    def __init__(self, stocks=["MSFT", "AAPL"], time_args=["3y"], batch_size=8, num_epochs=10000, saved_model=None):
        self.stocks = stocks
        self.time_args = time_args
        self.batch_size= batch_size
        self.writer = SummaryWriter()

        self.device = pick_device()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type,enabled=self.use_amp)
        

        self.num_workers = 0# =========================
        # RECOMMENDED WORKERS HERE
        # =========================
        # Choose an I/O profile for your pipeline (edit as needed)
        io_profile = "medium"   # {"light","medium","heavy"}

        # Get heuristic kwargs for DataLoader based on device + SLURM CPU budget
        loader_args = dataloader_kwargs(self.device, io=io_profile)

        # Save to instance attributes for consistency + logging
        self.num_workers        = loader_args["num_workers"]
        self.persistent_workers = loader_args["persistent_workers"]
        self.pin_memory         = loader_args["pin_memory"]

        # DataLoader doesn't accept None for prefetch_factor, so strip it if absent
        if loader_args.get("prefetch_factor", None) is None:
            loader_args.pop("prefetch_factor", None)
        self.loader_args = loader_args

        print(f"[device] using {self.device}")
        if self.device.type == "cuda":
            print(f"[cuda] {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
        elif self.device.type == "mps":
            print("[mps] Apple Metal Performance Shaders backend")
        print(f"[dataloader] num_workers={self.num_workers}, persistent_workers={self.persistent_workers}, pin_memory={self.pin_memory}")

        # data - now using local data loading (includes dates)
        input_data = util.get_data(stocks, time_args)
        if isinstance(input_data, int):
            print("Error getting data")
            return 1
        X_train, X_val, X_test, Y_train, Y_val, Y_test, D_train, D_val, D_test = input_data
        # Store dates for later reference (even when data is shuffled)
        self.train_dates = D_train
        self.val_dates = D_val
        self.test_dates = D_test
        
        train_ds = IndexedDataset(X_train, Y_train, D_train)
        val_ds   = IndexedDataset(X_val,   Y_val,   D_val)
        test_ds  = IndexedDataset(X_test,  Y_test,  D_test)

        # (Optional) quick autotune once; uncomment if you want to probe
        tuned = autotune_num_workers(train_ds, self.batch_size, self.device, candidates=(0,1,2,4,8))
        print(f"[dataloader] autotuned num_workers={tuned}")
        self.num_workers = tuned
        self.persistent_workers = bool(tuned)
        # Update loader_args with tuned values, handling prefetch_factor correctly
        self.loader_args.update(num_workers=tuned, persistent_workers=bool(tuned))
        if tuned == 0:
            self.loader_args.pop("prefetch_factor", None)
        else:
            self.loader_args["prefetch_factor"] = 2

        # =========================
        # BUILD DATALOADERS (use heuristics)
        # =========================
        self.trainLoader = data.DataLoader(
            train_ds, shuffle=True,  batch_size=self.batch_size, **self.loader_args
        )
        self.validationLoader = data.DataLoader(
            val_ds,   shuffle=False, batch_size=self.batch_size, **self.loader_args
        )
        self.testLoader = data.DataLoader(
            test_ds,  shuffle=False, batch_size=self.batch_size, **self.loader_args
        )

        # =========================
        # MODEL
        # =========================
        self.Model = model.LSTMModelPricePredict()

        # Multi-GPU (CUDA) support via DataParallel; safe no-op on single GPU/CPU/MPS
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.Model = nn.DataParallel(self.Model)
            print(f"[dataparallel] using {torch.cuda.device_count()} GPUs")
        # Move model to the chosen device
        self.Model = self.Model.to(self.device)

        print("{} total parameters".format(sum(param.numel() for param in self.Model.parameters())))

        if saved_model is not None:
            state_dict = torch.load(saved_model, map_location="cpu")
            # Load into underlying module if DataParallel is active
            target_module = self.Model.module if hasattr(self.Model, "module") else self.Model
            target_module.load_state_dict(state_dict)
            print(f"[load] restored weights from {saved_model}")

        self.optimizer = optim.Adam(self.Model.parameters())
        self.loss_fn = nn.MSELoss()
        self.stopper = EarlyStopper()
        self.num_epochs = num_epochs

    def train_one_epoch(self, epoch):

        print("Epoch: {}".format(epoch+1))
        print("--------------------------------------------")
        self.Model.train()
        train_loss = 0
        val_loss = 0
        stop_condition=None

        start_time = time.perf_counter()
        
        pbar = tqdm(self.trainLoader, total=len(self.trainLoader), desc=f"train {epoch+1}/{self.num_epochs}")
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

            """            
            # keep loss for later
            print("X_batch: ", X_batch)
            print("Y_batch: ", Y_batch)
            print("Y_pred: ", Y_pred)
            print("loss: ", loss)
            """

            train_loss += loss.item()
            

            pbar.set_postfix(avg_train=train_loss/(pbar.n or 1))
        avg_train = train_loss / len(self.trainLoader)

        self.Model.eval()
        
        with torch.no_grad():
            vbar = tqdm(self.validationLoader, total=len(self.validationLoader), desc=f"validation")
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
                vbar.set_postfix(avg_val=val_loss/(vbar.n or 1))
            avg_val = val_loss / len(self.validationLoader)
            stop_condition = self.stopper.early_stop(avg_val, self.Model)   
            

        end_time = time.perf_counter()
        
        print("Training Loss: {}".format(avg_train))
        print("Validation Loss: {}".format(avg_val))
        print("Training Time: {:.6f}s".format(end_time-start_time))
        print("--------------------------------------------")
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
            tbar = tqdm(self.testLoader, total=len(self.testLoader), desc=f"test")
            for X_batch, Y_batch, indices in tbar:
                self.optimizer.zero_grad(set_to_none=True)
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
                tbar.set_postfix(avg_test=test_loss/(tbar.n or 1))
                # Calculate individual losses for each sample in the batch
                # Y_pred and Y_batch have shape [batch_size, 1], so we squeeze to [batch_size]
                individual_loss = ((Y_pred - Y_batch) ** 2).squeeze()
                # Convert to numpy in a device-agnostic way
                individual_losses.extend(self._tensor_to_numpy(individual_loss))
                individual_dates.extend(batch_dates)
        
        avg_test_loss = test_loss / len(self.testLoader)
        print("Test Loss: {}".format(avg_test_loss))
        # Note: Individual test losses are logged vs timestamps below
        
        # Log individual losses against dates to TensorBoard
        if individual_losses and individual_dates:
            self._log_loss_vs_date_to_tensorboard(individual_losses, individual_dates)
        
        self.writer.flush()
        return avg_test_loss

    def stop(self):
        self.writer.close()
