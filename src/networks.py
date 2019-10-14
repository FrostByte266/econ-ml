from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def build_model(input_shape, dropout_rate=0.2):
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model