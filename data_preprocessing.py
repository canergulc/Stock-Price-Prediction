import pandas as pd
from alpha_vantage.timeseries import TimeSeries

class DataPreprocessor:
    def __init__(self, api_key, symbol='AAPL'):
        self.api_key = api_key
        self.symbol = symbol
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.data_close = None

    def fetch_data(self):
        # Fetch daily stock data
        data, _ = self.ts.get_daily(symbol=self.symbol, outputsize='full')
        data.index = pd.to_datetime(data.index)
        self.data_close = data['4. close'].sort_index().asfreq('B').fillna(method='ffill')

    def get_moving_average(self, window=30):
        # Calculate moving average
        return self.data_close.rolling(window=window).mean()

    def get_data(self):
        return self.data_close
