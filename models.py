import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, train):
        self.model = ARIMA(train, order=self.order)
        self.model_fit = self.model.fit()

    def forecast(self, steps):
        return self.model_fit.forecast(steps=steps)

class LSTMModel:
    def __init__(self, lstm_units, learning_rate=0.001):
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.LSTM(self.lstm_units, return_sequences=True, input_shape=(60, 1)))
        model.add(layers.LSTM(self.lstm_units, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, batch_size=1, epochs=1)

    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def fit(self, X_train, y_train):
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.reshape(X_test.shape[0], -1))

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor()

    def fit(self, X_train, y_train):
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.reshape(X_test.shape[0], -1))
