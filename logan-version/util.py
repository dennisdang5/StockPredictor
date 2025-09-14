# https://doi.org/10.1016/j.frl.2021.102280
# "Forecasting directional movements of stock prices for intraday trading using LSTM and random forests" Ghosh, Neufeld, Sahoo 2022


import pandas as pd
import torch
import yfinance as yf
import numpy as np
import statistics


def get_data(stocks, sstudy_period):
    
    dat = yf.Tickers(stocks)

    open_close = (dat.history(period=sstudy_period))[['Open','Close']]

    lookback = 240

    op = open_close["Open"].T
    cp = open_close["Close"].T

    if lookback >= (op.shape)[1]:
        print("study period too short")

    data = get_feature_input(op,cp, lookback, op.shape[1], len(stocks))
    
    return data

   


# op[x] is the op vector for stock x
# op and cp has values from time 0 to T_study
def get_feature_input(op, cp, lookback, study_period, num_stocks):

    T_study = study_period
    lookback = 240

    # one day back calculate ir, cpr, opr
    f_t1 = np.empty((num_stocks, 3, T_study))
    for n in range(num_stocks):
        for t in range(2,T_study):
            f_t1[n][0][t] = cp.iloc[n,t-1]/op.iloc[n,t-1] - 1
            f_t1[n][1][t] = cp.iloc[n,t-1]/cp.iloc[n,t-2] - 1
            f_t1[n][2][t] = op.iloc[n,t]/op.iloc[n,t-1] - 1

    # get all end times
    end_t = list(range(lookback + 2,study_period + 1))

    rss = np.empty((num_stocks, len(end_t),3,lookback))

    # loop over all end times to generate all stacks
    for n in range(num_stocks):
        for k in range(len(end_t)):
            period = [l for l in range(end_t[k]-lookback,end_t[k])]
            for i in range(3):
                q1,q2,q3 = statistics.quantiles(f_t1[n][i][period])
                for j in range(len(period)):
                    rss[n][k][i][j] = (f_t1[n][i][period[j]] - q2)/(q3-q1)

    rss = np.transpose(rss,(0,1,3,2))
    print(rss.shape)
    return(rss)
    """
    trying to get prediction at time t
    looking back at previous 241 days to predict the 242nd day

    start with opening prices (op) and closing prices(cp)
    prices for both ordered from 0-t

    intraday returns (ir)= cp/op -1
    returns wrt to last cp = 

    """


    