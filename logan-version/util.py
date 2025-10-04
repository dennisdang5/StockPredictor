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
import pickle


def get_data(stocks, args):
    
    dat = yf.Tickers(stocks)

    try:

        if len(args) == 1:
            # only take open and close times
            open_close = (dat.history(period=args[0]))[['Open','Close']]
        elif len(args) == 2:
            open_close = (dat.history(period=None, start=args[0], end=args[1], interval="1d"))[['Open','Close']]
        else:
            print("Invalid Data Input Arguments")
            return 1
    except:
        print("Data Error")
        return 1
    

    lookback = 240

    op = open_close["Open"].T
    cp = open_close["Close"].T

    print(op.shape)

    if lookback >= (op.shape)[1]:
        print("study period too short")

    # get 

    xdata, ydata = get_feature_input(op,cp, lookback, op.shape[1], len(stocks))
    xdata = torch.Tensor.to(torch.from_numpy(xdata),dtype=torch.float32)
    ydata = torch.Tensor.to(torch.from_numpy(ydata),dtype=torch.float32)

    total_points = xdata.shape[1]
    total_points_buffer = total_points - 2*lookback

    print("Total Points No Overlap: {}".format(total_points_buffer))

    train_val_size = int(total_points_buffer * (2/3))
    train_size = int(train_val_size * 0.8)

    Xtrain, Xvalidation, Xtest = xdata[:,:train_size], xdata[:,train_size+lookback:train_val_size+lookback], xdata[:,train_val_size+2*lookback:]
    Ytrain, Yvalidation, Ytest = ydata[:,:train_size], ydata[:,train_size+lookback:train_val_size+lookback], ydata[:,train_val_size+2*lookback:]

    print("Training set size: {}".format(Xtrain.shape[1]))
    print("Validation set size: {}".format(Xvalidation.shape[1]))
    print("Test set size: {}".format(Xtest.shape[1]))

    batch_flatten = lambda Xs: [torch.flatten(x,start_dim=0,end_dim=1) for x in Xs]

    return tuple(batch_flatten([Xtrain, Xvalidation, Xtest, Ytrain, Yvalidation, Ytest]))

# op[x] is the op vector for stock x
# op and cp has indices from time 0 to T_study-1
def get_feature_input(op, cp, lookback, study_period, num_stocks):

    T_study = study_period
    lookback = 240

    # one day/two day back calculate ir, cpr, opr
    f_t1 = np.empty((num_stocks, 4, T_study))
    for n in range(num_stocks):
        for t in range(2,T_study):
            f_t1[n][0][t] = cp.iloc[n,t-1]/op.iloc[n,t-1] - 1   #ir
            f_t1[n][1][t] = cp.iloc[n,t-1]/cp.iloc[n,t-2] - 1   #cpr
            f_t1[n][2][t] = op.iloc[n,t]/op.iloc[n,t-1] - 1     #opr
            f_t1[n][3][t] = op.iloc[n,t]                        #op
  

    # get all end times last end t is T_study-1 (can use end_t values as index)
    end_t = list(range(lookback + 2,T_study))

    # 4 features ir, cpr, opr, op
    rss = np.empty((num_stocks, len(end_t),4,lookback))
    target = np.empty((num_stocks,len(end_t),1))


    # loop over all end times to generate all stacks

    #over all stocks
    for n in range(num_stocks):
        #over all end times
        for k in range(len(end_t)):

            target[n][k][0] = cp.iloc[n,end_t[k]]

            period = [l for l in range(end_t[k]-lookback+1,end_t[k]+1)]
            #print(len(period)==lookback)

            # calculate features
            q = np.array([statistics.quantiles(f_t1[n][i][period]) for i in range(3)])
            for j in range(len(period)):
                for i in range(3):

                    rss[n][k][i][j] = (f_t1[n][i][period[j]] - q[i][1])/(q[i][2]-q[i][0])
                #opening price
                rss[n][k][3][j] = f_t1[n][i][period[j]]

            #print(op.iloc[n,end_t[k]]==op.iloc[n,period[-1]])
                

    rss = np.transpose(rss,(0,1,3,2))
    return rss, target
    """
    trying to get prediction at time t
    looking back at previous 241 days to predict the 242nd day

    start with opening prices (op) and closing prices(cp)
    prices for both ordered from 0-t

    intraday returns (ir)= cp/op -1
    returns wrt to last cp = 

    """

def save_data_locally(stocks, args, data_dir="data"):
    """
    Import data using yfinance and save it locally for future use.
    Returns the processed data and saves it to pickle files.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filename based on stocks and time args
    stocks_str = "_".join(stocks)
    time_str = "_".join(args)
    filename = f"{data_dir}/{stocks_str}_{time_str}.pkl"
    
    # Check if data already exists
    if os.path.exists(filename):
        print(f"Data file {filename} already exists. Loading from local file...")
        return load_data_from_local(filename)
    
    print(f"Downloading data for {stocks} with args {args}...")
    
    # Get data using existing get_data function
    data = get_data(stocks, args)
    
    if data == 1:
        print("Error downloading data")
        return 1
    
    # Save data to local file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to {filename}")
    return data

def load_data_from_local(filename):
    """
    Load data from a local pickle file.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return 1
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return 1

def get_model_summary(model):
    torchsummary.summary(model,(240,3),batch_size=8)