# Stock-Price-Prediction
Predicting stock price using historical data of a company, using Neural networks (LSTM).
unfinished 
## Table of content
* [Why are we interested in stock price prediction?](#abstract)
* [Introduction](#overview)
* [The data](#thedata)
    * [Technical indicators](#technicalind)
* [The result](#theresult)
* [What is next?](#whatisnext)
* [Disclaimer](#disclaimer)

# 1. Why are we interested in stock price prediction? <a class="anchor" id="abstract"></a>
Stock return forecasting is one of the core issues in financial research. It is closely related to many important financial issues, such as portfolio management, capital cost and market efficiency.The financial market tremendously impacts our daily lives in many perspectives. Our group wants to forecast the stock through the sequential data. People invest in exchange-traded funds against the inflation rate. Netflix produces TV series to reveal Wall Street’s life. Time series forecasting is one of the most challenging missions by deep learning. In this research, we aim to find an appropriate model for stock price prediction along with a profit-maximizing trading strategy. Long short term memory is the main technique used on the targets of stock price of two corporations: The Procter & Gamble Company and Bank of America. As comparison, some data de-noising is finished by one-dimension residual convolutional networks before passing into the LSTM as input features. Final results show that CNN successfully tackles random noise problems and uncertain information in time series but a single LSTM expresses a better performance.

Honestly speaking, we would like to apply netural networks on stock data to make some prediction on stock price. Try to find some chances to earn some money in the market. This is our inital thoughts. 
# 2. Introduction <a class="anchor" id="overview"></a>
Many parametric approaches are developed but fail to produce precise results. Instead, long short term memory in deep learning allows nonlinear characters and leads to a higher predictive accuracy. Some researchers also utilize the convolutional neural networks to solve the problem of noise in the waveform data. Our group aims to compare the prediction of future prices of The Procter & Gamble Company (PG) and Bank of America (BAC). The work is done by grid search on different parameters for LSTM only or combined CNN & LSTM model

## 2.1. Idea of financial Market: 

"Buy low, sell high" is the oldest and the most famous criterion in the market. However, the real world is more complicated. In order to mimic transactions as close as the reality, four trading strategies are developed. In other words, the ultimate objective of this experiment is to maximize profits based on the outputs and strategies.

# 3. Dataset <a class="anchor" id="thedata"></a>

Our group aims to compare the prediction of future prices of The Procter & Gamble Company (PG) and Bank of America (BAC). The work is done by grid search on different parameters for LSTM only or combined CNN & LSTM model. By understanding the behaviors of the stock, investors may improve their investment decisions.

```python
import numpy as np
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

data = yf.download("PG", start="2017-01-01", end="2019-01-01")
data2 = yf.download("SPY", start="2017-01-01", end="2019-01-01")
pd.DataFrame(data)
```

Our data is the historical daily stock price of PG and BAC from 01/01/2017 to 01/01/2019, downloaded from Yahoo Finance. The datasets include a daily open price, the daily highest price, the daily lowest price, a close price, an adjusted close price and the volume.

<center><img src='pics/sampledata.PNG' width=500></img></center>

_figure 1: sample data_

<center><img src='pics/pgclose.png' width=500></img></center>

_figure 2: close price for PG_

<center><img src='pics/bacclose.png' width=500></img></center>

_figure 3: close price for BAC_

## 3.1. Technical indicators <a class="anchor" id="technicalind"></a>
5 days and 100 days moving averages are applied to smooth temporary and random price fluctuations over time. A buy signal happens when the short-duration MA crosses above the long-duration MA. In professional terms, this is called a "golden cross." On the contrary, the trend of price drops and generates a sell signal when two lines cross the other way. This is known as a "dead cross. " Some other technical indicators tracked are Average True Range, Bollinger Bands, Rate of Change, Force Index, Williams percentage Range and Moving Average Convergence Divergence. The third dataset is the S&P 500 in Figure 5, as a benchmark to represent the overall economy.



















