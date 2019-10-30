import pandas as pd 
import tensorflow as tf 
from tqdm import trange

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import numpy as np

import preprocessing
import networks
import plotting

# df = pd.read_csv('/data/daily_csv.csv', parse_dates=['date']).dropna()
df = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').loc['Australia'].dropna()
# print(df)
dataset = df.price.values
dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
look_back = 100
X_train, Y_train = preprocessing.create_dataset(train, look_back)
X_test, Y_test = preprocessing.create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = networks.build_model(X_train.shape, neurons=128, layers=6, dropout_rate=0.1)
history = model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_test, Y_test), 
                    verbose=1, shuffle=False)

test_predict = model.predict(X_test)
num_predictions = 1


model.reset_states()
for i in trange(num_predictions, desc='Predicting'):
    series = preprocessing.create_next_seq(test_predict, look_back=look_back)
    prediction = model.predict(series)
    # assert False
    test_predict = np.append(test_predict, np.array([[prediction[-1][0]]]), axis=0)
    # test_predict = prediction

# Undo normalization
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

plotting.plot_dataset(Y_test, test_predict)