import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from prophet import Prophet
from tensorflow.keras.models import Sequential
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)
    return data

def create_df(data: Dict[str, Any]) -> pd.DataFrame:
    all_data: List[Dict[str, Any]] = []
    for region, region_data in data.items():
        for entry in region_data["data"]:
            entry["region"] = region
            all_data.append(entry)

    df = pd.DataFrame(all_data)
    columns = ['region'] + [col for col in df.columns if col != 'region']
    df = df[columns]
    return df

def get_countries(df: pd.DataFrame, countries: List[str] = ["United States"]) -> pd.DataFrame:
    df_country = df[df['region'].isin(countries)]
    df_country = df_country.set_index('year')
    return df_country

def get_columns(df: pd.DataFrame, col: List[str] = ['region', 'year']) -> pd.DataFrame:
    df_production = df[col]
    df_production = df_production.bfill()
    return df_production

def split_data(df: pd.DataFrame, split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * split)
    train, test = df[:train_size], df[train_size:]
    return train, test


def format_data_plot(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    if isinstance(data, (pd.DataFrame, pd.Series)):
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

def format_forecast_plot(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], 
    original_df: pd.DataFrame, 
    horizon: int = 5
) -> pd.DataFrame:
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
    return pd.DataFrame({'time': future_time, 'forecast': values})


def scale_data(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[col]].values)
    return scaled_data, scaler

def stationarity(col: pd.Series) -> Tuple[pd.Series, int]:
    d = 0
    diff_col = col.dropna()
    res = adfuller(diff_col)

    while res[1] > 0.05:
        d += 1
        diff_col = diff_col.diff().dropna()
        res = adfuller(diff_col)
        print(f"\nTest Statistic: {res[0]}")
        print(f"P-Value: {res[1]}")
        print("Non-Stationary" if res[1] > 0.05 else "Stationary")
    return diff_col, d

def train_ARIMA(train: pd.DataFrame, col: str, p: int, d: int, q: int) -> ARIMAResults:
    model = ARIMA(train[col], order=(p, d, q))
    return model.fit()

def predict_ARIMA(model: ARIMAResults, test: pd.DataFrame) -> pd.DataFrame:
    forecast = model.forecast(steps=len(test))
    test['forecast'] = list(forecast)
    return test

def initialize_endog_exog(df: pd.DataFrame, endog_col: str, exog_col: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    endog = df[endog_col].interpolate(method='linear', limit_direction='both')
    exog = df[exog_col].interpolate(method='linear', limit_direction='both')
    return endog, exog

def train_SARIMAX(endog: pd.Series, exog: pd.DataFrame) -> SARIMAXResults:
    auto_model = pm.auto_arima(
        endog,
        start_p=0, start_q=0,
        max_p=10, max_q=10,
        seasonal=False,
        trace=True,
        stepwise=True,
        information_criterion='aic'
    )
    best_p, best_d, best_q = auto_model.order
    print(f"Optimal p, d, q: ({best_p}, {best_d}, {best_q})")
    model = SARIMAX(endog, exog=exog, order=(best_p, best_d, best_q))
    return model.fit()

def predict_SARIMAX(model_fit: SARIMAXResults, exog: pd.DataFrame, future_steps: int = 5) -> pd.Series:
    return model_fit.forecast(steps=len(exog), exog=exog)

def predict_exog(exog: pd.DataFrame, future_steps: int = 5) -> pd.DataFrame:
    future_years = np.arange(exog.index[-1] + 1, exog.index[-1] + future_steps + 1)
    prediction_years = np.concatenate([exog.index, future_years]).reshape(-1, 1)

    exog_pred = pd.DataFrame(columns=exog.columns, index=future_years)
    for col in exog.columns:
        X = exog.index.values.reshape(-1, 1)
        y = exog[col].values
        poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_model.fit(X, y)
        y_pred = relu(poly_model.predict(prediction_years))
        exog_pred[col] = y_pred[-future_steps:].tolist()
    return exog_pred


def format_Prophet(df: pd.DataFrame, col: str) -> pd.DataFrame:
    data = pd.DataFrame({'ds': df.index, 'y': df[col]}).reset_index()
    data['ds'] = pd.to_datetime(data['ds'].astype(str) + '-01-01')
    return data[['ds', 'y']]

def fit_Prophet(data: pd.DataFrame) -> Prophet:
    model = Prophet(weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    return model

def predict_Prophet(model: Prophet, period: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=period, freq='YE')
    return model.predict(future)

def format_Prophet_plot(original_df: pd.DataFrame, predictions: pd.DataFrame, period: int) -> pd.DataFrame:
    forecast_values = predictions['yhat'].iloc[-period:]
    return format_forecast_plot(forecast_values, original_df, len(forecast_values))


def create_sequences(data: np.ndarray, seq_length: int, horizon: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    sequences, targets = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        sequences.append(data[i:i + seq_length, 0])
        targets.append(data[i + seq_length:i + seq_length + horizon, 0])
    return np.array(sequences), np.array(targets)

  
def format_X_Y(train_data: np.ndarray, test_data: np.ndarray, seq_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = create_sequences(train_data, seq_length, horizon)
    X_test, y_test = create_sequences(test_data, seq_length, horizon)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test, y_train, y_test

  
def fit_LSTM(train: np.ndarray, test: np.ndarray, seq_len: int, horizon: int) -> Sequential:
    X_train, X_test, y_train, y_test = format_X_Y(train, test, seq_len, horizon)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len, 1)),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    return model

def flatten_prediction(scaled_data: np.ndarray, seq_len: int, model: Sequential, scaler: MinMaxScaler) -> np.ndarray:
    input_sequence = scaled_data[-seq_len:, 0].reshape(1, seq_len, 1)
    future_prediction = model.predict(input_sequence)[0]
    future_prediction_unscaled = scaler.inverse_transform(future_prediction.reshape(-1, 1))
    return future_prediction_unscaled.flatten()


def format_Chronos(df: pd.DataFrame, col: str) -> TimeSeriesDataFrame:
    data = df[[col]].copy()
    data['item_id'] = 'United States'
    years = pd.to_numeric(data.index)
    timestamps = pd.date_range(start=f"{years.min()}-01-01", periods=len(years), freq="YS")
    data["timestamp"] = timestamps
    data = data.rename(columns={col: 'target'}).set_index(['item_id', 'timestamp'])
    return TimeSeriesDataFrame(data)

def fit_Chronos(data: TimeSeriesDataFrame, period: int = 10) -> TimeSeriesPredictor:
    predictor = TimeSeriesPredictor(prediction_length=period, eval_metric="MASE")
    predictor.fit(
        data,
        presets="high_quality",
        hyperparameters={"Chronos": {"model_path": "autogluon/chronos-bolt-small"}}
    )
    return predictor

def predict_Chronos(model: TimeSeriesPredictor, data: TimeSeriesDataFrame) -> pd.DataFrame:
    return model.predict(data)


def plot_acf(diff_col: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    sm.graphics.tsa.plot_acf(diff_col.iloc[13:], lags=40, ax=ax)
    return fig

def plot_pacf(diff_col: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    sm.graphics.tsa.plot_pacf(diff_col.iloc[13:], lags=40, ax=ax)
    return fig

def plot_graph(df: pd.DataFrame, title: str, cols: List[str] = ['coal_production', 'oil_production', 'gas_production']) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 7))
    for col in cols:
        if col in df.columns:
            sns.lineplot(data=df, x='year', y=col, marker='o', label=col.capitalize(), ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Productions', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def plot_forecasts(data_plot: pd.DataFrame, forecast_plot: pd.DataFrame, col: str, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(data=data_plot, x='time', y='value', marker='o', label=col.capitalize(), ax=ax)
    sns.lineplot(data=forecast_plot, x='time', y='forecast', marker='o', label=f'{col.capitalize()} Forecast', ax=ax)
    ax.set_title(f'{model_name} Model Forecast')
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Production', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig



def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def save_figs(figs: List[plt.Figure], folder: str = "plots") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, fig in enumerate(figs, 1):
        fig.savefig(f"{folder}/plot_{i}_{timestamp}.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved {len(figs)} figures to {folder}/")

