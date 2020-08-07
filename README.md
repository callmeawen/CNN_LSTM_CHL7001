# Stock-Price-Prediction

If you would like to try our code, just download all and run the main file.

``` python
python main.py
```

There are two things we can adjut for, one is time steps(in parameter called TIMESTEP, need to be even number and greater than 2) and the other one is change model since we have 3 different models. 

``` python
    # comment here to change models
    # model = create_cnn_model()
    # model = create_lstm_model()
    model = create_cnn_lstm_model()
```
From line 28-31 at main file. 

It's already aggregated enough for now, but we can still imporve these using arguements or parse. Due to the time constrat, we will do this next time. 

**We are happy if you can make money using our model, but we are not resposible for any loss if you decide to use the model**

## Table of content
* [Why are we interested in stock price prediction?](#abstract)
* [Introduction](#overview)
* [The data](#thedata)
    * [Technical indicators](#technicalind)
* [Model Explanation](#Model)
    * [Model Explanation LSTM continued](#Model.c)
    * [1D-CNN](#cnn)
    * [Test trading](#trading)
* [Results](#Result)
* [Conclusion](#Conclusion)
* [References](#references)

# 1. Why are we interested in stock price prediction? <a class="anchor" id="abstract"></a>

Stock return prediction is one of the most attractive issues in money related exploration. It is firmly identified with numerous significant budgetary issues, for example, portfolio the executives, capital expense and market efficiency.The monetary market colossally impacts our every day lives in numerous points of view. Our gathering needs to conjecture the stock through the consecutive information. Individuals put resources into trade exchanged assets against the swelling rate. Netflix produces TV arrangement to uncover Wall Street's life. Time arrangement guaging is one of the most testing missions by profound learning. In this exploration, we expect to locate a proper model for stock value forecast alongside a benefit expanding exchanging technique. Long momentary memory is the principle strategy utilized on the objectives of stock cost of two companies: The Procter and Gamble Company and Bank of America. As examination, a few information de-noising is done by one-dimentional residual convolutional network before going into the LSTM as input features

Honestly speaking, we would like to apply netural networks on stock data to make some prediction on stock price. Try to find some chances to earn some money in the market. This is our inital thoughts. 

# 2. Introduction <a class="anchor" id="overview"></a>

Numerous parametric methodologies are grown however neglect to create exact outcomes. Rather, long momentary memory in profound learning permits nonlinear characters and prompts a higher prescient precision. A few analysts likewise use the convolutional neural systems to tackle the issue of commotion in the waveform information. Our gathering intends to look at the expectation of future costs of The Procter and Gamble Company (PG) and Bank of America (BAC). The work is finished by matrix look on changed boundaries for LSTM just, CNN just or consolidated CNN and LSTM model.

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
5 days and 100 days moving averages are applied to smooth temporary and random price fluctuations over time. A buy signal happens when the short-duration MA crosses above the long-duration MA. In professional terms, this is called a "golden cross." On the contrary, the trend of price drops and generates a sell signal when two lines cross the other way. This is known as a "dead cross. " Some other technical indicators tracked are Average True Range, Bollinger Bands, Rate of Change, Force Index, Williams percentage Range and Moving Average Convergence Divergence. The third dataset is the S&P 500, as a benchmark to represent the overall economy.

A lot of investors follow technical indicators. We included the most popular indicators as independent features.

**NOTE:** We are not showing Technical Indicator Graphs here. Since They will be shown in our report. 

# 4. Model Explanation <a class="anchor" id="Model"></a>

## 4.1 Model Explanation LSTM <a class="anchor" id="Model"></a>

Long short term memory is similar to Recurrent Neural Network (RNN) in deep learning. It captures dynamic nonlinear characters and transfers previous relevant things to the present.

<center><img src='pics/lstmdiagram.png' width=500></img></center>

_figure 4:  architecture of LSTM_


## 4.2 Model Explanation LSTM continued  <a class="anchor" id="Model.c"></a>

As the new information flows through different gates (the input gate zi , the forget gate zf and the output gate zo) in memory blocks, it is read, forgotten and stored. Then, the cell state and the hidden state are updated and transferred to the next cell. For instance, previous cell state ct-1 is used to store the information kept from the last step: an increasing trend of the stock price in the past. Previous hidden state ht-1 is used to receive outputs from last cells: the closing price of the stock yesterday. Next, they are combined with the current input state at xt, which can fresh information: an unexpected major personnel change today. Finally, an accurate output is received.


## 4.3 1D-CNN  <a class="anchor" id="cnn"></a>

CNN is famous for diagram recognization. But Can be also used in dimentional reduction or feature aggregation. So we are going to pass features into sparse autoencoders with a convolution neural network through a 1 dimension convolution layer and global max-pooling layer before LSTM, we can reduce overfitting and improve forecasting performance if we can.

CNN or Convolutional Neural Network is also a big innovation in machine learning, with most applications in image recognition, image classification and natural language processing. However, with recent breakthroughs in data science, some studies show a better performance of convolutional neural networks in stock prices modeling compared with RNN. The advantage is especially reflected in “automatically and adaptively learning in spatial hierarchies of features through a backpropagation algorithm” [Rikiya et al., 2018]. Specifically, if features are not informative enough, they may hinder the extraction. A total number of 5 layers in CNN are constructed.(More details in the report.)

After transformations through each layer, the number of parameters that must be learned are decreased. Therefore, we can reduce the risk of overfitting and improve forecasting performance. 

## 4.4 CNN + LSTM  <a class="anchor" id="cnn"></a>

Traditional methods such as weighted moving average are largely introduced to smooth and de-noise datasets. The one-dimensional convolution is defined in Eq. (2)(detail in the report), where f is the input vector with length n and g is the kernel with length m. According to formula, the convolution operation can be viewed as a smoothing operator if the parameters are all positive, and hence, we propose to employ a CNN as a deeper input gate before LSTM to learn smoothing parameters from the inputs. Note that previous work already shows that by reducing the dimension, an “important” input gate before the LSTM will benefit the modeling of temporal structures of LSTM [Graves et al., 2013; Wu et al., 2018].

## 4.5 Test trading  <a class="anchor" id="trading"></a>

Four trading algorithms are developed with an initial capital of $1000 in the research in comparison of the rate of returns by different models and parameters. 

<center><img src='pics/strategy.png' width=500></img></center>

# 5. Results <a class="anchor" id="Result"></a>

We used permutation method before, and with permutation every steps and every movement can be perfectly predicted. But the method could be somehow not correct. Therefore we get rid of permutation. 

## 5.1 Test data against Predicted values in PG and BAC


<center><img src='pics/5.1.1.png' width=500></img></center>
<center><img src='pics/5.1.2.png' width=500></img></center>

_We only updated 2 pics here with timestep = 4 for both stock_

As appeared in Figure 7-12, We tuned the time steps of 4, 14 and 24 days to check whether the memory time would influence the expectations on three strategies. The dark line shows our test information, and the prescient bends for each market list are spoken to by strong lines in various hues. As per these plots, we see that the bend of the CNN model is a lot nearer to our genuine qualities than that of the other two methodologies. For BAC, the bend of LSTM sometimes gets a long way from the test esteems during the trial, while CNN+LSTM plays out the most noticeably awful by and large. With respect to PG, the perceptions are backwards. Regarding value pattern expectation, the exhibition of every one of the three models is commonly acceptable. Then again, there is no conspicuous contrast in yields while modifying memory time. Review that in the joined model, we utilize a CNN before going into LSTM. We can presume that CNN effectively brings the advantages of overfitting decrease and information de-noising. Despite the fact that, its gauge despite everything turned more terrible at long last.

## 5.2 Loss function comparison


<center><img src='pics/5.2.1.png' width=500 diplay:inline></img></center>
<center><img src='pics/5.2.2.png' width=500 diplay:inline></img></center>

<center><img src='pics/5.2.2.3.png' width=300></img></center>

Our motivation is to utilize CNN to cause the components in stock to can be coordinated and produce the impact of measurement decrease, in order to accomplish the impact of denoise. From table 1, No issue which timestep is utilized, the MSE of CNN with LSTM model is higher than CNN and LSTM. From the picture of misfortune work, we can likewise find that the LSTM model combines quicker than CNN and LSTM and MSE is moderately littler.

The assessment measurements utilized for correlation are MSE and misfortune to decrease likely predisposition in the investigation. All things considered, the least MSE was caught in CNN model, and the second best one happened on the LSTM model for the file BAC. CNN+LSTM has the biggest MSE around 0.0125, while its exhibition positions the second for the list PG. Sometimes, a closer prescient bend doesn't rise to a higher expectation precision. However the precision and the connection are emphatically related.

## 5.3 Test the earnings in using different models and strategies

<center><img src='pics/5.3.1.png' width=700></img></center>

Table 2 represents the pace of return utilizing 4 demonstrated techniques above. The majority of the paces of return are sure around 11% utilizing all in/full scale methodology in PG stock, yet this exchanging system is hazardous on the grounds that individuals are probably going to lose most of standards in beginning phases if the model forecast isn't sufficiently exact. Unexpectedly, by taking a gander at BAC pace of profits, most systems will in general have a deficiency. The second and the third calculation are both preservationist, yet the day by day venture technique can test our model expectation on patterns as it exceptionally relies upon value developments every day. Concerning the last exchanging calculation without utilizing any models, it can create benefits if and just if the general pattern is expanding. Generally speaking, from the model explicit pace of profits, CNN and LSTM will in general have better execution in exchanging with more income and less money related misfortune.

# 6. Conclusion and Discussion <a class="anchor" id="Conclusion"></a>

In synopsis, we address the usage and the examination of CNN and LSTM to monetary time arrangement expectation. As talked about over, the exchanging framework dependent on the forecast of a solitary CNN outflanks with a generally higher total returns contrasted with LSTM and CNN+LSTM. One reason that effects of CNN as a more profound information entryway are not clear is an absence of highlights and clamors in this trial. In addition, our investigation length is just multi year every day close cost, in this way, one of the further upgrades is the expansion of study length and profundity (i.e week by week, hourly exchanging). Because of the computational restriction, a set number of model boundaries is prepared. In this manner, future investigation will likewise present more arbitrary clamors and boundaries esteems. Simultaneously, we can consider the impacts of successive exchange costs, construct more expert exchanging calculations with earlier information to make beneficial portfolios, at that point step up certain API calls to make genuine records to perform day by day exchanging the genuine market.


# 7. References  <a class="anchor" id="references"></a>

Ganegedara, T. (2020, January 1st). Stock Market Predictions with LSTM in Python. Retrieved from DataCamp: https://www.datacamp.com/community/tutorials/lstm-python-stock-market

Geron, A. (2017). Hands-On Machine Learning with Scikit-Learn & TensorFlow.

Guanting Chen, Y. C. (n.d.). Application of Deep Learning to Algorithmic Trading.

Jialin Liu, F. C.-C.-M. (2019). Stock Prices Prediction using Deep Learning Models.

MITCHELL, C. (2020). How to Use a Moving Average to Buy Stocks. Investopedia.

PRANJAL SRIVASTAVA. (2017). Essentials of Deep Learning : Introduction to Long Short Term Memory. Analytics Vidhya.

Ugur Gudelek, A. B. (2017). A deep learning based stock trading model with 2-D CNN trend detection.

Wei Bao, J. Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory. National Library of Medicine. Retrieved from National Library of Medicine.

