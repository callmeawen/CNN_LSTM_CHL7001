
import keras
from keras.models import Sequential, load_model, Input
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

from keras.layers import TimeDistributed, Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import LSTM


def create_lstm_model():
    inputs = Input(batch_input_shape=(BATCH_SIZE, TIME_STEPS, NUM_FEATURES))
    hidden1 = LSTM(40, dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform')(inputs)
    drop_hidden1 = Dropout(0.4)(hidden1)
    hidden2 = LSTM(20, dropout=0.0)(drop_hidden1)
    drop_hidden2 = Dropout(0.4)(hidden2)
    dense1 = Dense(20,activation='relu')(drop_hidden2)
    outputs = Dense(1, activation='sigmoid')(dense1)
    lstm_model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.RMSprop(lr=params["lr"])
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model

def create_cnn_lstm_model():
    inputs = Input(batch_input_shape=(BATCH_SIZE, TIME_STEPS, NUM_FEATURES))
    # (batch_size, timesteps, data_dim)
    filtered_inputs1 = Conv1D(filters = FILTER_SIZE, kernel_size = KERNEL_SIZE, activation='relu')(inputs)
    filtered_inputs2 = MaxPooling1D(pool_size = 2)(filtered_inputs1)
    # flatten_inputs = Flatten()(filtered_inputs2)
    # print(flatten_inputs)
    # exit()
    hidden1 = LSTM(40, dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform')(filtered_inputs2)
    drop_hidden1 = Dropout(0.4)(hidden1)
    hidden2 = LSTM(20, dropout=0.0)(drop_hidden1)
    drop_hidden2 = Dropout(0.4)(hidden2)
    dense1 = Dense(20,activation='relu')(drop_hidden2)
    outputs = Dense(1, activation='sigmoid')(dense1)
    cnn_lstm_model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.RMSprop(lr=params["lr"])
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return cnn_lstm_model

if __name__ == '__main__':
    from keras import backend as K
    import time
    time.tzset()
    from dataset import *
    inputs = dataset1()
    x_train, y_train = build_timeseries(inputs, 0)
    x_train, y_train, x_val, y_val = x_train[0:400,:], y_train[0:400], x_train[400:440,:], y_train[400:440]

    print("Building model...")
    # print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
    model = create_cnn_lstm_model()
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)

    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,
                          "best_model.h5"), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)
    # print(y_val)
    # print(model.predict(x_train, batch_size=BATCH_SIZE))
    history = model.fit(x_train, y_train, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(x_val, y_val), callbacks=[es, mcp, csv_logger])
    #
    # print("saving model...")
    # # pickle.dump(model, open("lstm_model", "wb"))
