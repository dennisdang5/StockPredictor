# https://doi.org/10.1016/j.frl.2021.102280
# "Forecasting directional movements of stock prices for intraday trading using LSTM and random forests" Ghosh, Neufeld, Sahoo 2022


import pandas as pd
import torch
import yfinance as yf
import numpy as np
import statistics
import torchsummary


def get_data(stocks, sstudy_period):
    
    dat = yf.Tickers(stocks)

    open_close = (dat.history(period=sstudy_period))[['Open','Close']]

    lookback = 240

    op = open_close["Open"].T
    cp = open_close["Close"].T

    if lookback >= (op.shape)[1]:
        print("study period too short")

    xdata, ydata = get_feature_input(op,cp, lookback, op.shape[1], len(stocks))
    xdata = torch.Tensor.to(torch.from_numpy(xdata),dtype=torch.float32)
    ydata = torch.Tensor.to(torch.from_numpy(ydata),dtype=torch.float32)

    print(xdata.dtype)
    print(ydata.dtype)

    train_val_size = int(xdata.shape[1] * (2/3))
    Xtrain_val, Xtest = xdata[:,:train_val_size], xdata[:,train_val_size:]
    Ytrain_val, Ytest = ydata[:,:train_val_size], ydata[:,train_val_size:]

    train_size = int(Xtrain_val.shape[1] * 0.8)
    Xtrain, Xvalidation = Xtrain_val[:,:train_size], Xtrain_val[:,train_size:]
    Ytrain, Yvalidation = Ytrain_val[:,:train_size], Ytrain_val[:,train_size:]

    batch_flatten = lambda Xs: [torch.flatten(x,start_dim=0,end_dim=1) for x in Xs]

    return tuple(batch_flatten([Xtrain, Xvalidation, Xtest, Ytrain, Yvalidation, Ytest]))

# op[x] is the op vector for stock x
# op and cp has values from time 0 to T_study
def get_feature_input(op, cp, lookback, study_period, num_stocks):

    T_study = study_period
    lookback = 240

    # one day back calculate ir, cpr, opr
    f_t1 = np.empty((num_stocks, 3, T_study))
    for n in range(num_stocks):
        for t in range(2,T_study):
            f_t1[n][0][t] = cp.iloc[n,t-1]/op.iloc[n,t-1] - 1   #ir
            f_t1[n][1][t] = cp.iloc[n,t-1]/cp.iloc[n,t-2] - 1   #cpr
            f_t1[n][2][t] = op.iloc[n,t]/op.iloc[n,t-1] - 1     #opr

    # get all end times
    end_t = list(range(lookback + 2,study_period + 1))

    rss = np.empty((num_stocks, len(end_t),3,lookback))
    target = np.empty((num_stocks,len(end_t),1))
    print(len(end_t))

    # loop over all end times to generate all stacks
    for n in range(num_stocks):
        for k in range(len(end_t)):
            target[n][k][0] = cp.iloc[n,end_t[k]-1]
            period = [l for l in range(end_t[k]-lookback,end_t[k])]
            for i in range(3):
                q1,q2,q3 = statistics.quantiles(f_t1[n][i][period])
                for j in range(len(period) - 1):
                    rss[n][k][i][j] = (f_t1[n][i][period[j]] - q2)/(q3-q1)
                

    rss = np.transpose(rss,(0,1,3,2))
    print(rss.shape)
    return rss, target
    """
    trying to get prediction at time t
    looking back at previous 241 days to predict the 242nd day

    start with opening prices (op) and closing prices(cp)
    prices for both ordered from 0-t

    intraday returns (ir)= cp/op -1
    returns wrt to last cp = 

    """

def get_model_summary(model):
    torchsummary.summary(model,(240,3),batch_size=8)