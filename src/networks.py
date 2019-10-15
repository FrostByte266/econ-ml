from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def build_model(input_shape, dropout_rate=0.2, neurons=256, layers=3, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    for _ in range(layers-2):
        model.add(LSTM(neurons, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    model.add(LSTM(neurons))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model