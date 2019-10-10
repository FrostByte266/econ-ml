import pandas as pd 
import tensorflow as tf 
from tqdm import trange

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import numpy as np

import preprocessing
import networks
import plotting

# df = pd.read_csv('/data/corn2013-2017.txt', parse_dates=['date'])
df = pd.read_csv('/data/daily_csv.csv', parse_dates=['date'])
df.dropna(inplace=True)
dataset = df.price.values
dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
look_back = 30
X_train, Y_train = preprocessing.create_dataset(train, look_back)
X_test, Y_test = preprocessing.create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = networks.build_model(X_train.shape)
history = model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_test, Y_test), 
                    verbose=1, shuffle=False)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
for i in trange(100, desc="Predicting"):
    example = preprocessing.create_next_seq(test_predict, look_back=look_back)
    example = np.reshape(example, (example.shape[0], 1, example.shape[1]))
    pred = model.predict(example)
    test_predict = np.append(test_predict, [pred[-1]], axis=0)

# Undo normalization
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

plotting.plot_train_errors(history)

plotting.plot_dataset(Y_test, test_predict)
# plotting.plot_dataset(Y_train, train_predict)