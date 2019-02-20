#!/opt/anaconda3/bin/python3

import pandas as pd
import numpy as np
import itertools
import sys

from datetime import datetime, timedelta, date
from collections import defaultdict

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=1))


def mse(predictions, targets):
    return((predictions - targets) ** 2).mean(axis=1)


def mae(predictions, targets):
    return (abs(predictions - targets)).mean(axis=1)


def noise_repeat(X, noise=0.2, repeat=5):
    
    X_real = np.repeat(X, repeat, axis=0)
    noise_array = noise * np.random.normal(loc=0.0, scale=1.0, size=[X.shape[i] * repeat if i == 0 else X.shape[i] if i == 1 else 1 for i in range(len(X.shape))])
    X_noise = np.clip(X_real + noise_array, 0, 1)
    
    return X, X_real, X_noise


def autoencoder_fit(X_predict, X_real, X_noise, model, verbose=0):
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    earlystopper = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_noise, 
                    X_real,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=verbose,
                    callbacks=[earlystopper])
    
    return model.predict(X_predict)


def noise_autoencoder(dim):
    
    X_input = Input(shape=(dim,))
    encoded = Dense(128, activation='relu')(X_input)
    encoded = Dense(64, activation='linear', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(X_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder


if __name__ == "__main__":
    
    df = pd.read_csv('../features/clear_data/rm_features_for_anomaly.csv', ';', infer_datetime_format=True, parse_dates=['time'])

    X = df.set_index(['time', 'host', 'consumer_group'])
    X_predict, X_real, X_noise = noise_repeat(X.values)
    noise_model = noise_autoencoder(X_predict.shape[1])
    autoencoder_predict = autoencoder_fit(X_predict, X_real, X_noise, noise_model, verbose=0)
    autoencoder_error = pd.concat([X.reset_index(), pd.DataFrame(mse(autoencoder_predict, X.values), columns=['error'])], axis=1, sort=False)
   
    autoencoder_error.to_csv('model_result/autoencoder_err.csv', ';', index=False)

