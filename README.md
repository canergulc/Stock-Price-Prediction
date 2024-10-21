# Stock-Price-Prediction

This project aims to predict stock prices using different machine learning models, including ARIMA, LSTM, Random Forest, and XGBoost. The predictions are made using historical stock price data obtained from the Alpha Vantage API. The project is organized in a modular way, promoting clean code practices for better readability and maintainability.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Visualization](#visualization)
- [License](#license)

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - statsmodels
  - scikit-learn
  - xgboost
  - optuna
  - tensorflow
  - alpha_vantage

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
2. Create a virtual environment (optional but recommended):
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required libraries:
  pip install -r requirements.txt

## Usage

Obtain an API key from Alpha Vantage.
Update the API key in main.py:
  api_key = 'YOUR_API_KEY'
Run the main script:
  python main.py
The script will fetch historical stock data, train different models, and visualize the predictions.

## Models

The following models are implemented in this project:

ARIMA: A time series forecasting method that uses the relationship between an observation and a number of lagged observations.
LSTM: A type of recurrent neural network (RNN) that is well-suited for sequence prediction problems.
Random Forest: An ensemble learning method that constructs multiple decision trees at training time and outputs the mean prediction of the individual trees.
XGBoost: An optimized gradient boosting library designed to be highly efficient, flexible, and portable.

## Visualization

The results of the predictions from all models are visualized using matplotlib. The graphs display the actual stock prices alongside the predicted prices from each model for comparison.
