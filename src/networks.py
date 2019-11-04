import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def build_model(input_shape, dropout_rate=0.2, neurons=256, layers=3, train=False, batch_size=128):
    model = Sequential()
    batch_size = batch_size if train else 1
    model.add(Bidirectional(LSTM(neurons, batch_input_shape=(batch_size ,input_shape[1], input_shape[2]), return_sequences=True, recurrent_dropout=0, unroll=False, use_bias=True, stateful=True), batch_input_shape=(batch_size ,input_shape[1], input_shape[2])))
    model.add(Dropout(dropout_rate))

    for _ in range(layers-2):
        model.add(Bidirectional(LSTM(neurons, return_sequences=True, recurrent_dropout=0, unroll=False, use_bias=True, stateful=True)))
        model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(neurons, recurrent_dropout=0, unroll=False, use_bias=True, stateful=True)))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

class ResetModelCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()