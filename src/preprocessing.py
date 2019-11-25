import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
    dataset_range = range(len(dataset)-look_back)
    X = [dataset[i:(i+look_back), 0] for i in dataset_range]
    Y = [dataset[i+1:i+(look_back)+1, 0] for i in dataset_range]
    return np.array(X), np.array(Y)

def split_dataset(dataset, ratio=0.8):
    train_size = int(len(dataset) * ratio)
    train, test = dataset[:train_size], dataset[train_size:]
    return train, test