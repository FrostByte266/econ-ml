import tensorflow as tf 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler 

import matplotlib.pyplot as plt
from tqdm import trange

import networks
import preprocessing
from plotting import plot_dataset

# Load dataset to CSV
full_dataset = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').loc['Canada'].dropna()
dataset = full_dataset.price.values

# Scale data to range of 0-1
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create X and Y sets
dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
seq_len = 50
X_train, y_train = preprocessing.create_dataset(train, seq_len)
X_test, y_test = preprocessing.create_dataset(test, seq_len)

X_test_orig = X_test.copy()

# assert False

X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
X_test = X_test.reshape((X_test.shape[0], seq_len, 1))

# X_train = X_train.reshape((X_train.shape[0], 1, seq_len))
# X_test = X_test.reshape((X_test.shape[0], 1, seq_len))


# Make training model
neurons = 64
batch_sz = 64

training_model = Sequential()
training_model.add(GRU(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
training_model.add(Dense(1))
training_model.compile(loss='mean_squared_error', optimizer='adam')
training_model.fit(X_train, y_train, epochs=20, batch_size=batch_sz, verbose=True, validation_data=(X_test, y_test))

# Make prediction model
prediction_model = Sequential()
prediction_model.add(GRU(neurons, stateful=True, batch_input_shape=(1, seq_len, 1), return_sequences=True))
prediction_model.add(Dense(1))

# Transfer trained weights and reset internal states
prediction_model.set_weights(training_model.get_weights())
prediction_model.reset_states()

# Make the predictions
predictions = prediction_model.predict(X_test)

futureElement = predictions[-1]

futureElements = []
futureElements.append(futureElement)

for i in trange(100, desc='Making future predictions'):
    futureElement = prediction_model.predict(np.expand_dims(futureElement, axis=0))
    futureElements.append(futureElement.squeeze(axis=0))
    futureElement = futureElements[-1]

# Undo scaling
test = scaler.inverse_transform(test)
preds = scaler.inverse_transform(np.append(predictions, futureElements, axis=0).squeeze(axis=2))

# Plot results
plt.plot(test, label='Real')
plt.plot(np.array([i[-1] for i in preds]), label='Predicted')
plt.legend()
plt.show()