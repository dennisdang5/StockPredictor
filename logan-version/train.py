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
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), 'savedmodel.pth')
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        

class Trainer():
    def __init__(self, stocks=["MSFT", "AAPL"], time_args=["3y"], batch_size=8, num_epochs=10000, saved_model=None):
        self.stocks = stocks
        self.time_args = time_args
        self.batch_size=batch_size
        self.writer = SummaryWriter()
        self.device = pick_device()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type,enabled=self.use_amp)
        self.pin_memory = self.device.type == "cuda"

        # -------- NEW: set workers = number of CUDA GPUs detected --------
        if self.device.type == "cuda":
            self.num_workers = torch.cuda.device_count()  # 0,1,2,...
        else:
            self.num_workers = 0
        self.persistent_workers = self.num_workers > 0
        # ------------------------------------------------------------------

        print(f"[device] using {self.device}")
        if self.device.type == "cuda":
            print(f"[cuda] {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
        elif self.device.type == "mps":
            print("[mps] Apple Metal Performance Shaders backend")

        print(f"[dataloader] num_workers={self.num_workers}, persistent_workers={self.persistent_workers}, pin_memory={self.pin_memory}")

        # data - now using local data loading
        input_data = util.save_data_locally(stocks, time_args)
        if (input_data == 1):
            return 1
        else:
            X_train, X_val, X_test, Y_train, Y_val, Y_test = input_data
        

        self.trainLoader = data.DataLoader(
            data.TensorDataset(X_train, Y_train),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
        self.validationLoader = data.DataLoader(
            data.TensorDataset(X_val, Y_val),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
        self.testLoader = data.DataLoader(
            data.TensorDataset(X_test, Y_test),
            shuffle=False,
            batch_size=batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        self.lstmModel = model.LSTMModelPricePredict()

        # Multi-GPU (CUDA) support via DataParallel; safe no-op on single GPU/CPU/MPS
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.lstmModel = nn.DataParallel(self.lstmModel)
            print(f"[dataparallel] using {torch.cuda.device_count()} GPUs")
        # Move model to the chosen device
        self.lstmModel = self.lstmModel.to(self.device)

        print("{} total parameters".format(sum(param.numel() for param in self.lstmModel.parameters())))

        if saved_model is not None:
            state_dict = torch.load(saved_model, map_location="cpu")
            # Load into underlying module if DataParallel is active
            target_module = self.lstmModel.module if hasattr(self.lstmModel, "module") else self.lstmModel
            target_module.load_state_dict(state_dict)
            print(f"[load] restored weights from {saved_model}")

        self.optimizer = optim.Adam(self.lstmModel.parameters())
        self.loss_fn = nn.MSELoss()

        self.stopper = EarlyStopper()
        self.num_epochs = num_epochs

    def train_one_epoch(self, epoch):

        print("Epoch: {}".format(epoch+1))
        print("--------------------------------------------")
        self.lstmModel.train()
        train_loss = 0
        val_loss = 0
        stop_condition=None

        start_time = time.perf_counter()
        
        for X_batch, Y_batch in self.trainLoader:
            # move data to device
            X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
            Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)

            # forward + loss (+AMP on CUDA)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    Y_pred = self.lstmModel(X_batch)
                    loss = self.loss_fn(Y_pred, Y_batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                Y_pred = self.lstmModel(X_batch)
                loss = self.loss_fn(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()

            # keep loss for later
            train_loss += loss.item()

        self.lstmModel.eval()
        
        with torch.no_grad():
            for X_batch, Y_batch in self.validationLoader:
                X_batch = X_batch.to(self.device, non_blocking=self.pin_memory)
                Y_batch = Y_batch.to(self.device, non_blocking=self.pin_memory)
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        Y_pred = self.lstmModel(X_batch)
                        loss = self.loss_fn(Y_pred, Y_batch)
                else:
                    Y_pred = self.lstmModel(X_batch)
                    loss = self.loss_fn(Y_pred, Y_batch)
                val_loss += loss.item()

            stop_condition = self.stopper.early_stop(val_loss, self.lstmModel)    

        end_time = time.perf_counter()
        
        print("Training Loss: {}".format(train_loss/len(self.trainLoader)))
        print("Validation Loss: {}".format(val_loss/len(self.validationLoader)))
        print("Training Time: {:.6f}s".format(end_time-start_time))
        print("--------------------------------------------")
        self.writer.add_scalar('Loss/train', train_loss/len(self.trainLoader), epoch+1)
        self.writer.add_scalars('Loss/trainVSvalidation', {"Training":(train_loss/len(self.trainLoader)), "Validation":(val_loss/len(self.validationLoader))}, epoch+1)
        self.writer.add_scalar('Train Time', (end_time-start_time), epoch+1)
        self.writer.flush()

        return stop_condition
    
    def get_summary(self):
        summary(self.lstmModel, (240,3), self.batch_size, device=self.device.type)

    def stop(self):
        self.writer.close()