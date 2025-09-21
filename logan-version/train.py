import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
import model
import util
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time

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
            torch.save(model.state_dict(), 'savedmodel.pth')

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            torch.save(model.state_dict(), 'savedmodel.pth')
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

        #data
        input_data = util.get_data(stocks, time_args)
        if (input_data == 1):
            return 1
        else:
            X_train, X_val, X_test, Y_train, Y_val, Y_test = input_data
        

        self.trainLoader = data.DataLoader(data.TensorDataset(X_train, Y_train), shuffle=True, batch_size=batch_size)
        self.validationLoader = data.DataLoader(data.TensorDataset(X_val, Y_val), shuffle=True, batch_size=batch_size)
        self.testLoader = data.DataLoader(data.TensorDataset(X_test, Y_test), shuffle=True, batch_size=batch_size)

        self.lstmModel = model.LSTMModelPricePredict()

        print("{} total parameters".format(sum(param.numel() for param in self.lstmModel.parameters())))

        if saved_model != None:
            state_dict = torch.load(saved_model)
            self.lstmModel.load_state_dict(state_dict)

        self.optimizer = optim.Adam(self.lstmModel.parameters())
        self.loss_fn = nn.MSELoss()

        self.stopper = EarlyStopper()
        self.num_epochs = num_epochs

        print(torch.accelerator.is_available())
        print(torch.accelerator.current_accelerator())

        if torch.accelerator.is_available():
            accelerator_type = torch.accelerator.current_accelerator()
            self.accelerator = torch.device(accelerator_type)
            self.lstmModel = torch.nn.DataParallel(self.lstmModel,device_ids=[i for i in range(torch.accelerator.device_count())])
            print("{} device found".format(accelerator_type))
        else:
            print ("MPS device not found.")
            self.accelerator = None

    def train_one_epoch(self, epoch):

        print("Epoch: {}".format(epoch+1))
        print("--------------------------------------------")
        train_loss = 0
        val_loss = 0
        stop_condition=None

        start_time = time.perf_counter()
        
        for X_batch, Y_batch in self.trainLoader:
            
            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            Y_pred = self.lstmModel(X_batch)

            # get and add loss for batches
            loss = self.loss_fn(Y_batch, Y_pred)
            
            # backprop from loss
            loss.backward()

            # update weights
            self.optimizer.step()

            # keep loss for later
            train_loss += loss.item()

        self.lstmModel.eval()
        
        with torch.no_grad():
            for X_batch, Y_batch in self.validationLoader:
                # forward pass
                Y_pred = self.lstmModel(X_batch)

                # get loss
                loss = self.loss_fn(Y_batch, Y_pred)

                # add to total validation loss
                val_loss += loss.item()

            stop_condition = self.stopper.early_stop(val_loss,self.lstmModel)    

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
        summary(self.lstmModel, (240,3), self.batch_size)

    def stop(self):
        self.writer.close()
        