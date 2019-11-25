from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tqdm import trange

import preprocessing

# Load dataset and get all possible countries
full_dataset = pd.read_csv('/data/exchange.csv',
                           parse_dates=['date'],
                           index_col='country').dropna()

all_countries = list(full_dataset.index.unique())

# Create a parser for handling command arguments
parser = ArgumentParser()
parser.add_argument('--country', '-c',
                    type=str, nargs='+',
                    default=['Canada'],
                    required=False)

parser.add_argument('--seq-len', '-l',
                    type=int,
                    default=50,
                    required=False)

parser.add_argument('--train-epochs', '-e',
                    type=int,
                    default=20,
                    required=False)

parser.add_argument('--hidden-neurons', '-n',
                    type=int,
                    default=8,
                    required=False)

parser.add_argument('--batch-size', '-b',
                    type=int,
                    default=128,
                    required=False)

parser.add_argument('--future-predictions', '-p',
                    type=int,
                    default=10,
                    required=False)

parsed = parser.parse_args()

# Narrow down dataset to only selected country
country = ' '.join(parsed.country)
assert country in all_countries, f'\'{country}\' is not a valid choice'

dataset = full_dataset.loc[country].price.values

# Scale data to range of 0-1
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create X and Y sets
train, test = preprocessing.split_dataset(dataset, ratio=0.8)

seq_len = parsed.seq_len
X_train, y_train = preprocessing.create_dataset(train, seq_len)
X_test, y_test = preprocessing.create_dataset(test, seq_len)

X_train = X_train.reshape((-1, seq_len, 1))
X_test = X_test.reshape((-1, seq_len, 1))

# Make training model
neurons = parsed.hidden_neurons
batch_sz = parsed.batch_size

training_model = Sequential()
training_model.add(LSTM(neurons,
                   input_shape=(seq_len, 1),
                   return_sequences=True))

training_model.add(LSTM(neurons, return_sequences=True))
training_model.add(Dense(1, activation='linear'))
training_model.compile(loss='mean_squared_error', optimizer='adam')
training_model.fit(X_train, y_train,
                   epochs=parsed.train_epochs,
                   batch_size=batch_sz,
                   verbose=True,
                   validation_data=(X_test, y_test))

# Make prediction model
prediction_model = Sequential()
prediction_model.add(LSTM(neurons, stateful=True,
                          batch_input_shape=(1, seq_len, 1),
                          return_sequences=True))
prediction_model.add(LSTM(neurons, stateful=True, return_sequences=True))
prediction_model.add(Dense(1, activation='linear'))

# Transfer trained weights and reset internal states
prediction_model.set_weights(training_model.get_weights())
prediction_model.reset_states()

# Make the predictions
predictions = prediction_model.predict(X_test)

future_step = np.expand_dims(predictions[-1], axis=0)
future_steps = np.array(future_step)

num_preds = parsed.future_predictions
for i in trange(num_preds, desc='Making future predictions'):
    future_step = prediction_model.predict(future_step)
    future_steps = np.append(future_steps, future_step, axis=0)


# Undo scaling
X_test = scaler.inverse_transform(X_test.squeeze(axis=2))
preds = scaler.inverse_transform(np.append(predictions,
                                           future_steps,
                                           axis=0).squeeze(axis=2))

# Plot results
plt.title(f'US\u2192{country} exchange rates with ' +
          f'predictions for {num_preds} days')
plt.plot(np.array([i[-1] for i in X_test]), label='Real')
plt.plot(np.array([i[-1] for i in preds]), label='Predicted')
plt.legend()
plt.show()
