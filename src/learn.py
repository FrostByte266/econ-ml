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
df = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').loc['Canada'].dropna()
# print(df)
dataset = df.price.values
dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
look_back = 1
X_train, Y_train = preprocessing.create_dataset(train, look_back)
X_test, Y_test = preprocessing.create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = networks.build_model(X_train.shape, neurons=64, layers=6, dropout_rate=0.0)
history = model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test), 
                    verbose=1, shuffle=False)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
num_predictions = 100
# for i in trange(num_predictions, desc="Predicting"):
#     example = preprocessing.create_next_seq(test_predict, look_back=look_back)
#     example = np.reshape(example, (example.shape[0], 1, example.shape[1]))
#     pred = model.predict(example)
#     test_predict = np.append(test_predict, [pred[-1]], axis=0)

pred_input = preprocessing.create_next_seq(test_predict, look_back=1)
model.reset_states()
for i in trange(num_predictions, desc='Predicting'):
    prediction = model.predict(pred_input)
    prediction = tf.squeeze(prediction, 0)
    pred_input = tf.expand_dims([prediction], 0)
    test_predict = np.append(test_predict, [prediction.numpy()], axis=0)

# Undo normalization
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

predictions = test_predict[-num_predictions:]
for i, item in enumerate(predictions):
    cur = item[0]
    previous = predictions[i-1][0] if i > 0 else test_predict[-(num_predictions+1):][0][0]
    delta_percent = ((cur - previous) / previous) * 100
    print(f'Current predicted price 1 week from previous point: {cur:.2f}, Delta: {delta_percent:.2f}%')

plotting.plot_train_errors(history)

plotting.plot_dataset(Y_test, test_predict)
plotting.plot_dataset(Y_train, train_predict)