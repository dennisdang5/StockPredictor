import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.utils.data as data
import model
import util
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

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
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        

class Trainer():
    def __init__(self, stocks=["MSFT", "AAPL"], sstudy_period="1y", batch_size=8, num_epochs=10000):
        self.stocks = stocks
        self.sstudy_period = sstudy_period
        self.batch_size=batch_size
        self.writer = SummaryWriter()

        #data
        X_train, X_val, X_test, Y_train, Y_val, Y_test = util.get_data(stocks, sstudy_period)

        self.trainLoader = data.DataLoader(data.TensorDataset(X_train, Y_train), shuffle=True, batch_size=batch_size)
        self.validationLoader = data.DataLoader(data.TensorDataset(X_val, Y_val), shuffle=True, batch_size=batch_size)
        self.testLoader = data.DataLoader(data.TensorDataset(X_test, Y_test), shuffle=True, batch_size=batch_size)
        
        self.lstmModel = model.LSTMModelPricePredict()
        print("{} total parameters".format(sum(param.numel() for param in self.lstmModel.parameters())))

        self.optimizer = optim.Adam(self.lstmModel.parameters())
        self.loss_fn = nn.MSELoss()

        self.stopper = EarlyStopper()
        self.num_epochs = num_epochs

        if torch.backends.mps.is_available():
            self.mps_device = torch.device("mps")
            print("MPS device found")
        else:
            print ("MPS device not found.")

    def train_one_epoch(self, epoch):
        print("Epoch: {}".format(epoch+1))
        print("--------------------------------------------")
        train_loss = 0
        val_loss = 0
        stop_condition=None
        
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
        
        print("Training Loss: {}".format(train_loss/len(self.trainLoader)))
        print("Validation Loss: {}".format(val_loss/len(self.validationLoader)))
        print("--------------------------------------------")
        self.writer.add_scalar('Loss/train', train_loss/len(self.trainLoader), epoch+1)
        self.writer.add_scalars('Loss/trainVSvalidation', {"Training":(train_loss/len(self.trainLoader)), "Validation":(val_loss/len(self.validationLoader))}, epoch+1)
        self.writer.flush()

        return stop_condition
    
    def get_summary(self):
        summary(self.lstmModel, (240,3), self.batch_size)
