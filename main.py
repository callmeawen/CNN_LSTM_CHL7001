from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

import os
from parameter import *
import pickle

from model import *
from keras import backend as K
import time
from dataset import *

from matplotlib import pyplot as plt

def train(x_train, y_train, x_val, y_val):

    time.tzset()
    print("Building model...")

    # comment here to change models
    # model = create_cnn_model()
    # model = create_lstm_model()
    model = create_cnn_lstm_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.00001)

    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,
                          "best_model.h5"), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

    history_callback = model.fit(x_train, y_train, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(x_val, y_val), callbacks=[es, mcp, csv_logger])
    loss_history = history_callback.history['loss']

    return model, loss_history

def predict(x_test, y_test=None, model=None):
    if model is None:
        model = load_model(os.path.join(OUTPUT_PATH, 'best_model.h5'))

    y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
    y_pred = y_pred.flatten()

    if y_test is not None:
        print(y_pred.shape, y_test.shape)
        error = mean_squared_error(y_test, y_pred)
        print("Error is", error)

    return y_pred

if __name__ == '__main__':
    inputs = dataset1()
    x_train, y_train = build_timeseries(inputs, 0)

    # x_test, y_test = x_train[400:480,:], y_train[400:480]
    x_test, y_test = x_train[-50:,:], y_train[-50:]
    x_train, y_train = x_train[0:400,:], y_train[0:400]
    # randomize the indice of data
    randIdx = np.random.permutation(range(x_train.shape[0]))
    print(randIdx)
    x_train, y_train = x_train[randIdx,:], y_train[randIdx]
    print(x_train.shape, y_train.shape)

    ## split the datasets !! define based on batch size
    x_train, y_train, x_val, y_val= x_train[0:360,:], y_train[0:360], x_train[360:400,:], y_train[360:400]

    ## model training
    model, loss_history = train(x_train, y_train, x_val, y_val)

    y_pred = predict(x_test, y_test, model)


    invert = y_pred * (32.84000015258789 - 22.049999237060547) + 22.049999237060547
    ytest_invert = y_test * (32.84000015258789 - 22.049999237060547) + 22.049999237060547

    ## graph
    plt.close()
    ## timesteps
    T = np.arange(y_test.shape[0])
    #print(T)
    #print(y_test)
    plt.plot(T, ytest_invert, lw=1, label='test')
    plt.plot(T, invert, lw=1, label='pred', color='red')
    plt.title('CNN Model test vs pred')
    plt.legend(loc=2, prop={'size': 11})
    plt.grid(True)
    plt.show()

## Method 3 for stock pg
money = 1000
unit = 1000/ytest_invert[0]

profit3 = (unit * ytest_invert[49] - money)/1000


# method 2 for stock PG (weight)
money = 1000
unit = 0
for i in range(48):
    rate_of_change = (invert[i + 1] - invert[i])/invert[i]
    w = rate_of_change
    if w >= 0:
        buy_price = money * w
        buy_unit = buy_price/ ytest_invert[i]
        money = money - buy_price
        unit = unit + buy_unit
    else:
        sell_unit = unit * w * -1
        unit = unit - sell_unit
        sell_price = sell_unit * ytest_invert[i]
        money = money + sell_price

profit2 = (money + unit * ytest_invert[49] - 1000)/1000

# method 1 for stock PG (all in/all out) earn more but loss more
exchange_rate = ytest_invert
cur_money = 1000
cur_bitcoin = 0

for day in range(48):
    if invert[day + 1] >= invert[day]:
        if cur_money > 0:
            cur_bitcoin = cur_money / exchange_rate[day]
            cur_money = 0
    else:
        if cur_bitcoin > 0:
            cur_money = cur_bitcoin * exchange_rate[day]
            cur_bitcoin = 0
if cur_money != 0:
    print(cur_money)
else:
    cur_money = cur_bitcoin * exchange_rate[49]
    print(cur_money)
# profit is 454.2939678208654
profit1 = (cur_money-1000)/1000


# method 4
r = 0
for i in range(48):
    change = (invert[i+1] - invert[i])/abs(invert[i+1] - invert[i])
    # rate for i+1
    r = r + change * (ytest_invert[i+1] - ytest_invert[i])/ytest_invert[i]


print(profit1,profit2,profit3, r)

loss_CL_24 = np.array(loss_history)
invert_CL_24 = invert

plt.plot(loss_CNN_4, lw=1,label='CNN', color='green')
plt.plot(loss_LSTM_4, lw=1, label='LSTM', color='red')
plt.plot(loss_CL_4, lw=1, label='CNN & LSTM', color='blue')
plt.title('loss plot: time steps =4')
plt.legend(loc=1, prop={'size': 11})
plt.grid(True)
plt.show()
####

plt.plot(ytest_invert, lw=1, label='test', color='black')
plt.plot(invert_CNN_4, lw=1, label='CNN', color='green')
plt.plot(invert_LSTM_4, lw=1, label='LSTM', color='red')
plt.plot(invert_CL_4, lw=1, label='CNN & LSTM', color='blue')
plt.title('TEST VS PRED: time steps =4')
plt.legend(loc=3, prop={'size': 8})
plt.grid(True)
plt.show()


plt.plot(loss_CNN_14, lw=1,label='CNN', color='green')
plt.plot(loss_LSTM_14, lw=1, label='LSTM', color='red')
plt.plot(loss_CL_14, lw=1, label='CNN & LSTM', color='blue')
plt.title('loss plot: time steps =14')
plt.legend(loc=1, prop={'size': 11})
plt.grid(True)
plt.show()
####

plt.plot(ytest_invert, lw=1, label='test', color='black')
plt.plot(invert_CNN_14, lw=1, label='CNN', color='green')
plt.plot(invert_LSTM_14, lw=1, label='LSTM', color='red')
plt.plot(invert_CL_14, lw=1, label='CNN & LSTM', color='blue')
plt.title('TEST VS PRED: time steps =14')
plt.legend(loc=3, prop={'size': 8})
plt.grid(True)
plt.show()

plt.plot(loss_CNN_24, lw=1,label='CNN', color='green')
plt.plot(loss_LSTM_24, lw=1, label='LSTM', color='red')
plt.plot(loss_CL_24, lw=1, label='CNN & LSTM', color='blue')
plt.title('loss plot: time steps =24')
plt.legend(loc=1, prop={'size': 11})
plt.grid(True)
plt.show()
####

plt.plot(ytest_invert, lw=1, label='test', color='black')
plt.plot(invert_CNN_24, lw=1, label='CNN', color='green')
plt.plot(invert_LSTM_24, lw=1, label='LSTM', color='red')
plt.plot(invert_CL_24, lw=1, label='CNN & LSTM', color='blue')
plt.title('TEST VS PRED: time steps =24')
plt.legend(loc=3, prop={'size': 8})
plt.grid(True)
plt.show()
