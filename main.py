import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import DataPreprocessor
from models import ARIMAModel, LSTMModel, RandomForestModel, XGBoostModel
from visualization import Visualizer
from sklearn.preprocessing import MinMaxScaler

def main():
    api_key = 'WRPV5DD35V0YGBUA'
    preprocessor = DataPreprocessor(api_key)
    preprocessor.fetch_data()

    data_close = preprocessor.get_data()
    train_size = int(len(data_close) * 0.8)
    train, test = data_close[0:train_size], data_close[train_size:]

    # Fit ARIMA model
    arima_model = ARIMAModel()
    arima_model.fit(train)
    forecast_arima = arima_model.forecast(steps=len(test))
    rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))
    print(f'ARIMA RMSE: {rmse_arima}')

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_close_scaled = scaler.fit_transform(data_close.values.reshape(-1, 1))
    train_size = int(len(data_close_scaled) * 0.8)
    train_scaled, test_scaled = data_close_scaled[0:train_size], data_close_scaled[train_size:]

    # Create dataset for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_scaled, time_step)
    X_test, y_test = create_dataset(test_scaled, time_step)

    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Fit LSTM model
    lstm_model = LSTMModel(lstm_units=50)
    lstm_model.fit(X_train, y_train)
    predictions_lstm = lstm_model.predict(X_test)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)

    rmse_lstm = np.sqrt(mean_squared_error(y_test, predictions_lstm))
    print(f'LSTM RMSE: {rmse_lstm}')

    # Fit Random Forest model
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    predictions_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
    print(f'Random Forest RMSE: {rmse_rf}')

    # Fit XGBoost model
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train)
    predictions_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
    print(f'XGBoost RMSE: {rmse_xgb}')

    # Visualization
    Visualizer.plot_data_and_predictions(data_close, predictions_lstm, 'LSTM', train_size + time_step)
    Visualizer.plot_data_and_predictions(data_close, predictions_rf, 'Random Forest', train_size + time_step)
    Visualizer.plot_data_and_predictions(data_close, predictions_xgb, 'XGBoost', train_size + time_step)
    Visualizer.plot_data_and_predictions(data_close, forecast_arima, 'ARIMA', train_size)

if __name__ == "__main__":
    main()
