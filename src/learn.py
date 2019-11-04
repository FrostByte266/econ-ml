import pandas as pd 
import tensorflow as tf 
from tqdm import trange
import PySimpleGUI as sg 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import numpy as np

import preprocessing
import networks
import plotting

countries = [
    'Australia',
    'Brazil',
    'Canada',
    'China',
    'Denmark',
    'Euro',
    'Hong Kong',
    'India',
    'Japan',
    'Malaysia',
    'Mexico',
    'New Zealand',
    'Norway',
    'Singapore',
    'South Africa',
    'South Korea',
    'Sweden',
    'Switzerland',
    'Taiwan',
    'Thailand',
    'Venezuela'
]
layout = [
    [sg.Text('Select the dataset to train on'), sg.InputCombo(countries, key='select')],
    [sg.Ok()]
]
window = sg.Window('Select Datasource', layout=layout)
event, values = window.Read()

selection = values['select']
window.Close()

df = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').loc[selection].dropna()
dataset = df.price.values
dates = df.date.values

dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
look_back = 2
X_train, Y_train = preprocessing.create_dataset(train, look_back)
X_test, Y_test = preprocessing.create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
train_shape = X_train.shape

X_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))

X_train = X_train.shuffle(10000).batch(64, drop_remainder=True)


model = networks.build_model(train_shape, neurons=64, layers=6, dropout_rate=0.0, train=True, batch_size=64)

model.fit(X_train, epochs=20, verbose=1, shuffle=False, callbacks=[networks.ResetModelCallback()])

pred_model = networks.build_model(train_shape, neurons=64, layers=6, dropout_rate=0.0)
pred_model.set_weights(model.get_weights())

test_predict = pred_model.predict(X_test)
# num_predictions = 50
# for i in trange(num_predictions, desc='Predicting'):
#     # series = np.expand_dims(np.reshape(test_predict[-look_back:], (1, look_back)), axis=0)
#     series = preprocessing.create_next_seq(test_predict, look_back=look_back)
#     # assert False
#     series = tf.data.Dataset.from_tensor_slices(series).batch(1)
#     prediction = pred_model.predict(series)
#     test_predict = np.append(test_predict, np.array([[prediction[-1][0]]]), axis=0)

# Undo normalization
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

plotting.plot_dataset(Y_test, test_predict, country=selection)