import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
    dataset_range = range(len(dataset)-look_back-1)
    X = [dataset[i:(i+look_back), 0] for i in dataset_range]
    Y = [dataset[i + look_back, 0] for i in dataset_range]
    return np.array(X), np.array(Y)

def create_next_seq(dataset, look_back=1):
    X = [dataset[i:(i+look_back)] for i in range(len(dataset)-look_back-1)]
    return np.array(X)

def split_dataset(dataset, ratio=0.8):
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test

def normalize_dataframe(df):
    dataset = np.reshape(df, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler
