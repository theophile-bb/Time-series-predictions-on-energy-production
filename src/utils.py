
# =========================
# Standard library
# =========================
import json
from datetime import datetime, timedelta

# =========================
# Third-party libraries
# =========================
import requests
import numpy as np
import pandas as pd

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Statistics / Time Series
# =========================
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# =========================
# Machine Learning
# =========================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline

# =========================
# Prophet
# =========================
from prophet import Prophet

# =========================
# Deep Learning (LSTM)
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# AutoML / Chronos
# =========================
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


#--------------------- processing ------------------------

def load_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data

def create_df(data):
  all_data = []

  for region, region_data in data.items():
      for entry in region_data["data"]:
          entry["region"] = region
          all_data.append(entry)

  df = pd.DataFrame(all_data)
  columns = ['region'] + [col for col in df.columns if col != 'region']
  df = df[columns]
  return df

def get_countries(df, countries = ["United States"]):
  dfCountry = df[df['region'].isin(countries)]
  dfCountry = dfCountry.set_index('year')
  return dfCountry

def get_columns(df, col = ['region', 'year']):
  dfProduction = df[col]
  dfProduction = dfProduction.fillna(method='bfill')
  return dfProduction

def split_data(df, split = 0.8):
  train_size = int(len(df) * split)
  train, test = df[:train_size], df[train_size:]
  return train, test

def format_data_plot(data):

    if isinstance(data, pd.DataFrame):
        data_plot = data.reset_index()
        data_plot.columns = ['time', 'value']

    elif isinstance(data, pd.Series):
        data_plot = data.reset_index()
        data_plot.columns = ['time', 'value']

    elif isinstance(data, np.ndarray):
        data_plot = pd.DataFrame({
            'time': np.arange(len(data)),
            'value': data
        })

    else:
        raise TypeError("Unsupported data type")

    return data_plot


def format_forecast_plot(data, original_df, horizon=5):

    last_time = original_df.index[-1]

    if isinstance(original_df.index, pd.DatetimeIndex):
        freq = original_df.index.freq or pd.infer_freq(original_df.index)
        if freq is None:
            raise ValueError("Cannot infer datetime frequency from original_df index")

        future_time = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq
        )
    else:
        future_time = np.arange(last_time + 1, last_time + horizon + 1)

    if isinstance(data, pd.DataFrame):
        values = data.iloc[:, 0].values

    elif isinstance(data, pd.Series):
        values = data.values

    elif isinstance(data, np.ndarray):
        values = data

    else:
        raise TypeError("Unsupported data type")

    values = values[:horizon]

    data_plot = pd.DataFrame({
        'time': future_time,
        'forecast': values
    })
    return data_plot

def scale_data(df, col):
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(df[[col]].values)
  return scaled_data, scaler

#--------------------- ARIMA ------------------------

def stationarity(col):
  d = 0
  diff_col = col.dropna()
  res = adfuller(diff_col)

  while res[1] > 0.05:
    d += 1
    diff_col = diff_col.diff().dropna()
    res = adfuller(diff_col)
    print("\nTest Statistic:", res[0])
    print("P-Value:", res[1])
    if res[1] > 0.05:
      print("Non-Stationary")
    else:
      print("Stationary")
  return diff_col, d

def train_ARIMA(train, col, p,d,q):
  model = ARIMA(train[col], order=(p,d,q))
  model_fit = model.fit()
  return model_fit

def predict_ARIMA(model, test):
  forecast = model.forecast(steps=len(test))
  test['forecast'] = list(forecast)
  return test

#--------------------- SARIMAX ------------------------

def initialize_endog_exog(df,endogCol, exogCol):
  endog = df[endogCol]
  exog = df[exogCol]

  endog = endog.interpolate(method='linear', limit_direction='both')
  exog = exog.interpolate(method='linear', limit_direction='both')
  return endog, exog

def train_SARIMAX(endog, exog):

  auto_model = pm.auto_arima(
      endog,                    # Time series data
      start_p=0,               # Minimum p
      start_q=0,               # Minimum q
      max_p=10,                 # Maximum p
      max_q=10,                 # Maximum q
      d= None,                  # Let the model determine differencing (d)
      seasonal=False,          # Set to True if you want seasonal ARIMA
      trace=True,              # Show the selection process
      stepwise=True,           # Use stepwise search to reduce computation
      information_criterion='aic'  # Optimize based on AIC (can be 'bic' or 'hqic')
  )

  best_p, best_d, best_q = auto_model.order
  print(f"Optimal p, d, q: ({best_p}, {best_d}, {best_q})")

  model = SARIMAX(endog, exog=exog, order=(best_p, best_d, best_q))
  model_fit = model.fit()
  return model_fit


def predict_SARIMAX(model_fit, exog,  future_steps = 5):
  forecast = model_fit.forecast(steps=len(exog), exog=exog)
  return forecast

def predict_exog(exog, future_steps=5):
  future_years = np.arange(exog.index[-1] + 1, exog.index[-1] + future_steps + 1)
  prediction_years = np.concatenate([exog.index, future_years])
  prediction_years = prediction_years.reshape(-1, 1)

  exog_pred = pd.DataFrame(columns=exog.columns, index=future_years)

  for col in exog.columns:
    X = exog.index.values.reshape(-1, 1)
    y = exog[col].values

    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)

    y_pred = relu(poly_model.predict(prediction_years))

    exog_pred[col] = y_pred[-future_steps:].tolist()
  return exog_pred


#--------------------- Prophet ------------------------

def format_Prophet(df, col):
  data = pd.DataFrame({'ds': df.index,'y': df[col]})
  data = data.reset_index()

  data['ds'] = pd.to_datetime(data['ds'].astype(str) + '-01-01')

  data = data[['ds', 'y']]
  return data

def fit_Prophet(data):
  model = Prophet(
      weekly_seasonality=False,
      yearly_seasonality=False,
      daily_seasonality=False
      )
  model.fit(data)
  return model


def predict_Prophet(model, period):
  future = model.make_future_dataframe(periods=period, freq='YE')

  forecast = model.predict(future)
  return forecast

def format_Prophet_plot(original_df, predictions, period):
  last_original_year = original_df.index[-1]
  forecast_years = [i for i in range(last_original_year + 1, last_original_year + period + 1)]

  forecast_values = predictions['yhat'].iloc[-period:]

  dfForecast_result = format_forecast_plot(forecast_values, original_df, len(forecast_values))

  return dfForecast_result
#--------------------- LSTM ------------------------

def create_sequences(data, seq_length, horizon = 0):
    sequences, targets = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        sequence = data[i:i + seq_length, 0]
        target = data[i + seq_length:i + seq_length + horizon, 0]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def format_X_Y(train_data, test_data, seq_length, horizon):
  X_train, y_train = create_sequences(train_data, seq_length, horizon)
  X_test, y_test = create_sequences(test_data, seq_length, horizon)

  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
  return X_train, X_test, y_train, y_test


def fit_LSTM(train, test, seq_len, horizon):

  X_train, X_test, y_train, y_test = format_X_Y(train, test, seq_len, horizon)

  model = Sequential([
      LSTM(50, activation='relu', input_shape=(seq_len, 1)),
      Dense(horizon)
  ])

  model.compile(optimizer='adam', loss='mse')

  model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
  return model

def flatten_prediction(scaled_data, seq_len):
  input_sequence = scaled_data[-seq_len:, 0] # Shape (seq_length,)
  input_sequence = input_sequence.reshape(1, seq_len, 1)

  future_prediction = model.predict(input_sequence)[0] # Shape (horizon,)

  future_prediction_unscaled = scaler.inverse_transform(future_prediction.reshape(-1, 1))
  return future_prediction_unscaled.flatten()


#--------------------- Chronos ------------------------

def format_Chronos(df, col):
  data = df[[col]].copy()
  data['item_id'] = 'United States'
  years = pd.to_numeric(data.index)
  timestamps = pd.date_range(
        start=f"{years.min()}-01-01",
        periods=len(years),
        freq="YS"  # Year Start frequency (required!)
    )
  data["timestamp"] = timestamps

  #The model only accepts the time as 'timestamp' and values as 'target'
  data = data.rename(columns={col: 'target'})
  data = data.set_index(['item_id', 'timestamp'])
  ts = TimeSeriesDataFrame(data)
  return ts

def fit_Chronos(data, period=10):
    predictor = TimeSeriesPredictor(
        prediction_length=period,
        eval_metric="MASE"
    )

    predictor.fit(
        data,
        presets="high_quality",
        hyperparameters={
            "Chronos": {
                "model_path": "autogluon/chronos-bolt-small"
            }
        }
    )
    return predictor


def predict_Chronos(model, data):
  predictions = model.predict(data)
  return predictions


#--------------------- Graph plotting ------------------------

def plot_acf(diff_col):
  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(211)
  fig = sm.graphics.tsa.plot_acf(diff_col.iloc[13:],lags=40,ax=ax1)
  return None

def plot_pacf(diff_col):
  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(211)
  fig = sm.graphics.tsa.plot_pacf(diff_col.iloc[13:],lags=40,ax=ax1)
  return None

def plot_graph(df,title, cols=['coal_production', 'oil_production', 'gas_production']):
  plt.figure(figsize=(15, 7))
  for col in cols:
    if col in df.columns:
      sns.lineplot(data=df, x='year', y=col, marker='o', label=col.capitalize())

  plt.title(title, fontsize=16)
  plt.xlabel('Years', fontsize=12)
  plt.ylabel('Productions', fontsize=12)
  plt.xticks(rotation=90, ha='right')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.show()

def plot_forecasts(data_plot, forecast_plot, col):
    plt.figure(figsize=(15, 7))

    sns.lineplot(
        data=data_plot,
        x='time',
        y='value',
        marker='o',
        label=col.capitalize()
    )

    sns.lineplot(
        data=forecast_plot,
        x='time',
        y='forecast',
        marker='o',
        label=f'{col.capitalize()} Forecast'
    )

    plt.legend()
    plt.title('ARIMAX Model Forecast')
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Production', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    return None
#--------------------- other ------------------------

def relu(x):

  return np.maximum(0, x)




