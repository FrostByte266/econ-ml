import tensorflow as tf 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import max_norm

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler 

import matplotlib.pyplot as plt
from tqdm import trange

import networks
import preprocessing
from plotting import plot_dataset, plot_train_errors

# Load dataset to CSV
full_dataset = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').loc['Canada'].dropna()
dataset = full_dataset.price.values

# Scale data to range of 0-1
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create X and Y sets
train, test = preprocessing.split_dataset(dataset, ratio=0.8)
    
seq_len = 50
X_train, y_train = preprocessing.create_dataset(train, seq_len)
X_test, y_test = preprocessing.create_dataset(test, seq_len)

X_train = X_train.reshape((-1, seq_len, 1))
X_test = X_test.reshape((-1, seq_len, 1))

# Make training model
neurons = 8
batch_sz = 128

training_model = Sequential()
training_model.add(LSTM(neurons, input_shape=(seq_len, 1), return_sequences=True))
training_model.add(LSTM(neurons, return_sequences=True))
training_model.add(Dense(1, activation='linear'))
training_model.compile(loss='mean_squared_error', optimizer='adam')
history = training_model.fit(X_train, y_train, epochs=20, batch_size=batch_sz, verbose=True, validation_data=(X_test, y_test))

# plot_train_errors(history)

# Make prediction model
prediction_model = Sequential()
prediction_model.add(LSTM(neurons, stateful=True, batch_input_shape=(1, seq_len, 1), return_sequences=True))
prediction_model.add(LSTM(neurons, stateful=True, return_sequences=True))
prediction_model.add(Dense(1, activation='linear'))

# Transfer trained weights and reset internal states
prediction_model.set_weights(training_model.get_weights())
prediction_model.reset_states()

# Make the predictions
predictions = prediction_model.predict(X_test)

future_step = np.expand_dims(predictions[-1], axis=0)
future_steps = np.array(future_step)

for i in trange(10, desc='Making future predictions'):
    future_step = prediction_model.predict(future_step)
    future_steps = np.append(future_steps, future_step, axis=0)


# Undo scaling
X_test = scaler.inverse_transform(X_test.squeeze(axis=2))
preds = scaler.inverse_transform(np.append(predictions, future_steps, axis=0).squeeze(axis=2))

# Plot results
plt.plot(np.array([i[-1] for i in X_test]), label='Real')
plt.plot(np.array([i[-1] for i in preds]), label='Predicted')
plt.legend()
plt.show()