# Stock-Price-Prediction
Predicting stock price using historical data of a company, using Neural networks (LSTM).
unfinished 
## Table of content
* [Why are we interested in stock price prediction?](#abstract)
* [Introduction](#overview)
* [The data](#thedata)
    * [Correlated assets](#corrassets)
    * [Technical indicators](#technicalind)
    * [Fundamental analysis](#fundamental)
        - [Bidirectional Embedding Representations from Transformers - BERT](#bidirnlp)
    * [Fourier transforms for trend analysis](#fouriertransform)
    * [ARIMA as a feature](#arimafeature)
    * [Statistical checks](#statchecks)
        - [Heteroskedasticity, multicollinearity, serial correlation](#hetemultiser)
    * [Feature Engineering](#featureeng)
        * [Feature importance with XGBoost](#xgboost)
    * [Extracting high-level features with Stacked Autoencoders](#stacked_ae)
        * [Activation function - GELU (Gaussian Error)](#gelu)
        * [Eigen portfolio with PCA](#pca)
    * [Deep Unsupervised Learning for anomaly detection in derivatives pricing](#dulfaddp)
* [Generative Adversarial Network - GAN](#qgan)
    * [Why GAN for stock market prediction?](#whygan)
    * [Metropolis-Hastings GAN and Wasserstein GAN](#mhganwgan)
    * [The Generator - One layer RNN](#thegenerator)
        - [LSTM or GRU](#lstmorgru)
        - [The LSTM architecture](#lstmarchitecture)
        - [Learning rate scheduler](#lrscheduler)
        - [How to prevent overfitting and the bias-variance trade-off](#preventoverfitting)
        - [Custom weights initializers and custom loss metric](#customfns)
    * [The Discriminator - 1D CNN](#thediscriminator)
        - [Why CNN as a discriminator?](#why_cnn_architecture)
        - [The CNN architecture](#the_cnn_architecture)
    * [Hyperparameters](#hyperparams)
* [Hyperparameters optimization](#hyperparams_optim)
    * [Reinforcement learning for hyperparameters optimization](#reinforcementlearning)
        - [Theory](#reinforcementlearning_theory)
            - [Rainbow](#rl_rainbow)
            - [PPO](#rl_ppo)
        - [Further work on Reinforcement learning](#reinforcementlearning_further)
    * [Bayesian optimization](#bayesian_opt)
        - [Gaussian process](#gaussprocess)
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
