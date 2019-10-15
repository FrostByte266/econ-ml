import pandas as pd 
import tensorflow as tf 
from tqdm import trange

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping
from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt

import numpy as np

import preprocessing
import networks
import plotting

# df = pd.read_csv('/data/corn2013-2017.txt', parse_dates=['date']).dropna()
# df = pd.read_csv('/data/daily_csv.csv', parse_dates=['date']).dropna()
df = pd.read_csv('/data/exchange.csv', parse_dates=['date'], index_col='country').dropna().loc['Japan']
print(df)
dataset = df.price.values
dataset, scaler = preprocessing.normalize_dataframe(dataset)
train, test = preprocessing.split_dataset(dataset, ratio=0.6)
    
look_back = 100
X_train, Y_train = preprocessing.create_dataset(train, look_back)
X_test, Y_test = preprocessing.create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# print(type(X_train))
# print(type(X_test))
# print(type(Y_train))
# print(type(Y_train))

# assert False


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512, 1024]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.3, 0.4]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64]))

METRIC = 'loss'

with tf.summary.create_file_writer('/logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_BATCH_SIZE],
        metrics=[hp.Metric(METRIC, display_name='Loss')]
    )

def train_test_model(hparams, xtr, xte, ytr, yte, plot_title='Training Results', save_location=None):
    model = networks.build_model(xtr.shape, neurons=hparams[HP_NUM_UNITS], dropout_rate=hparams[HP_DROPOUT], layers=3)
    history = model.fit(xtr.copy(), ytr.copy(), epochs=10, batch_size=hparams[HP_BATCH_SIZE],
                        verbose=1, shuffle=False, callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])
    loss  = model.evaluate(xte, yte)

    # train_predict = model.predict(X_train)
    test_predict = model.predict(xte)
    # num_predictions = 5
    # for i in trange(num_predictions, desc="Predicting"):
    #     example = preprocessing.create_next_seq(test_predict, look_back=look_back)
    #     example = np.reshape(example, (example.shape[0], 1, example.shape[1]))
    #     pred = model.predict(example)
    #     test_predict = np.append(test_predict, [pred[-1]], axis=0)

    test_predict = scaler.inverse_transform(test_predict)
    yte = scaler.inverse_transform([yte])

    # predictions = test_predict[-num_predictions:]
    # for i, item in enumerate(predictions):
    #     cur = item[0]
    #     previous = predictions[i-1][0] if i > 0 else test_predict[-(num_predictions+1):][0][0]
    #     delta_percent = ((cur - previous) / previous) * 100
    #     print(f'Current predicted price 1 week from previous point: {cur:.2f}, Delta: {delta_percent:.2f}%')


    plotting.plot_dataset(yte, test_predict, plot_title=plot_title, save_location=save_location)
    return loss



def run(run_dir, hparams, xtr, xte, ytr, yte, plot_title='Training Results', save_location=None):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        loss = train_test_model(hparams, xtr, xte, ytr, yte, plot_title=plot_title, save_location=save_location)
        tf.summary.scalar(METRIC, loss, step=1)


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in HP_DROPOUT.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for batch_size in HP_BATCH_SIZE.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                    HP_BATCH_SIZE: batch_size
                }
                run_name = f"run-{session_num}"
                print(f'--- Starting trial: {run_name}')
                title = {h.name: hparams[h] for h in hparams}
                print(title)
                run(f'/logs/hparam_tuning/{run_name}', hparams, X_train, X_test, Y_train, Y_test, plot_title=title, save_location=f'/data/{run_name}')
                session_num += 1

# Undo normalization


# plotting.plot_train_errors(history)
# plotting.plot_dataset(Y_train, train_predict)