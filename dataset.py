# plt.show()import numpy as np
import pandas as pd
import yfinance as yf
import ta as ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
import tensorflow as tf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import numpy as np
# from tqdm._tqdm_notebook import tqdm_notebook
from parameter import *
# TIME_STEPS = 50
# NUM_TIME_SERIES = 502
# NUM_FEATURES = 5

def dataset1():
    """<h1><b>Sample Data set from Apple</h1></b>

    Traders use them to study the short-term price movement since they do not prove very useful for long-term investors. They are employed primarily to predict future price levels.
    """

    data = yf.download("BAC", start="2017-01-01", end="2019-01-01")
    data2 = yf.download("SPY", start="2017-01-01", end="2019-01-01")
    pd.DataFrame(data['Close']).plot()
    plt.grid(True)
    plt.title('BAC Close')
    plt.show()

    """The technical indicators used are Average True Range
    (ATR), Bollinger Bands (BB), Force Index (FI), Rate of Change (ROC), Williams percentage
    Range (WR), Moving Average Convergence Divergence (MACD), and 5 days moving average
    (MA5)
    """

    # Clean nan values
    df = ta.utils.dropna(data)

    # ROC
    k=ta.momentum.roc(close=df['Close'],n=5,fillna=True)
    k
    k.plot()
    plt.grid(True)
    plt.title('Rate of Change')
    plt.show()

    # Williams percentage Range (WR)
    WR = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"],fillna=True)
    WR._wr
    WR._wr.plot()
    plt.grid(True)
    plt.title('Williams percentage Range')
    plt.show()

    # Moving Average Convergence Divergence
    # subtracting the 26-period EMA from the 12-period EMA.
    macd = ta.trend.MACD(close=df['Close'], n_slow= 26, n_fast=12, n_sign=9, fillna=True)
    plt.plot(macd._macd, label='MACD')
    plt.plot(macd._macd_signal, label='Signal Line (red)', color='red')
    plt.plot(macd._macd_diff, label='Difference (yellow)', color='yellow')
    plt.legend(loc=2,prop={'size':11})
    plt.grid(True)
    plt.title('Moving Averages Convergence Divergence')
    plt.show()
    # a bullish crossover happens when the MACD crosses above the signal line
    # a bearish crossover happens when the MACD crosses below the signal line.

    # MA
    ma = ta.volatility.bollinger_mavg(close=df['Close'], n=5, fillna=True)
    ma100 = ta.volatility.bollinger_mavg(close=df['Close'], n=100, fillna=True)
    plt.plot(df['Close'], lw=1, label='Closing Prices')
    ma.plot(label='5-day MA (red)', color='red')
    ma100.plot(label='100-day MA (black)', color='black')
    plt.legend(loc=2, prop={'size': 11})
    plt.grid(True)
    plt.title('Moving Averages')
    plt.show()

    """<b><h1>Bollinger Bands</h1></b>"""

    low = ta.volatility.bollinger_hband(close=df["Close"],
                                        n=5,
                                        ndev=2,
                                        fillna=True)
    # pd.DataFrame(low, columns=["hband"])

    high = ta.volatility.bollinger_lband(close=df["Close"],
                                         n=5,
                                         ndev=2,
                                         fillna=True)
    pd.concat([low, df["Close"], high], axis=1).plot(grid=True)
    plt.title('Bollinger Bands Plot')
    plt.grid(True)
    plt.show()

    """<b><h1>ATR</h1></b>"""

    atr = ta.volatility.AverageTrueRange(high=df["High"],
                                         low=df["Low"],
                                         close=df["Close"],
                                         fillna=True)
    atr.average_true_range().plot()
    plt.title('ATR')
    plt.grid(True)
    plt.show()

    """<b><h1>Force Index</h1></b>

    The force index takes into account the direction of the stock price, the extent of the stock price movement, and the volume. Using these three elements it forms an oscillator that measures the buying and the selling pressure.

    Each of these three factors plays an important role in the determination of the force index. For example, a big advance in prices, which is given by the extent of the price movement, shows a strong buying pressure. A big decline in heavy volume indicates a strong selling pressure.
    """

    force_index = ta.volume.ForceIndexIndicator(close=df["Close"],
                                                volume=df["Volume"],
                                                fillna=True,
                                                n=13).force_index()
    force_index.plot()
    plt.title('Force Index')
    plt.grid(True)
    plt.show()

    # LSTM
    aggregate_data = pd.concat([data["Close"],
                                data["High"],
                                data["Low"],
                                data["Volume"],
                                data["Open"],
                                data2["Close"],
                                force_index,
                                atr.average_true_range(),
                                k,
                                WR._wr,
                                ma,
                                low,
                                high,
                                macd._macd],
                               axis=1)
    # aggregate_data.columns = ["Stock_Close","Stock_High", "Stock_Low", "Stock_Volume", "SPY_Open","SPY", "FI","ATR","ROC","WR","MA","BB_Low","BB_High","MACD"]
    ip = aggregate_data.to_numpy()
    inputs = ip.reshape(NUM_TIME_SERIES, NUM_FEATURES)  ## number of time series x num of features

    normalize_inputs, _, __, = normalize_data(inputs)
    return normalize_inputs


def normalize_data(inputs):
    minimum, maximum = inputs.min(axis=0), inputs.max(axis=0)
    print('train data maximum minimum: ', inputs[0:400,0].min(axis=0), inputs[0:400,0].max(axis=0))
    normalize_inputs = np.divide(inputs - minimum, maximum - minimum)
    print('test data maximum minimum: ', inputs[:,0].min(axis=0), inputs[:,0].max(axis=0))
    # print(normalize_inputs)
    return normalize_inputs, minimum, maximum

def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0",dim_0)
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
#         if i < 10:
#           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o",x.shape,y.shape)
    return x, y


if __name__ == '__main__':
    inputs=dataset1()
    # print(inputs.shape)
    # print(inputs)
    # x, y = build_timeseries(inputs, 0)
    # print(x.shape, y.shape)
