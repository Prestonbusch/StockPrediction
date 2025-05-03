import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="S&P 500 Stock Price Prediction - End of 2025",
    page_icon="📈",
    layout="wide"
)

st.title("S&P 500 Stock Price Prediction")
st.markdown("Predict stock prices for the end of 2025 using historical data from January 2020 to December 2024.")

def load_data(file_path='Data/qgzpz8q9nyapvqp1_csv.zip'):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_name = zip_ref.namelist()[0]
            zip_ref.extract(file_name, path='temp_data')
            data = pd.read_csv(f'temp_data/{file_name}')
            os.remove(f'temp_data/{file_name}')
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def enforce_numeric_columns(ticker_data, returns_col, price_col, volume_col):
    if returns_col is not None:
        ticker_data[returns_col] = pd.to_numeric(ticker_data[returns_col], errors='coerce')
    if price_col is not None:
        ticker_data[price_col] = pd.to_numeric(ticker_data[price_col], errors='coerce')
    if volume_col is not None:
        ticker_data[volume_col] = pd.to_numeric(ticker_data[volume_col], errors='coerce')
    return ticker_data

def engineer_features(df, date_col, price_col, returns_col=None, volume_col=None):
    df = df.copy()
    df = df.sort_values(date_col)
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['MonthsSinceStart'] = (df[date_col].dt.year - df[date_col].dt.year.min()) * 12 + df[date_col].dt.month - df[date_col].dt.month.min() + 1

    for window in [3, 6, 12]:
        df[f'MA_{window}'] = df[price_col].rolling(window=window).mean()
        df[f'Volatility_{window}'] = df[price_col].rolling(window=window).std()

    if returns_col is not None:
        df[returns_col] = pd.to_numeric(df[returns_col], errors='coerce')
        for window in [3, 6, 12]:
            df[f'Cum_Return_{window}'] = (1 + df[returns_col]).rolling(window=window).apply(lambda x: np.prod(x) - 1)
            df[f'Avg_Return_{window}'] = df[returns_col].rolling(window=window).mean()

    if volume_col is not None:
        for window in [3, 6, 12]:
            df[f'Volume_MA_{window}'] = df[volume_col].rolling(window=window).mean()
        df['Volume_Ratio'] = df[volume_col] / df[volume_col].rolling(window=3).mean()

    for window in [3, 6, 12]:
        ma_col = f'MA_{window}'
        if ma_col in df.columns:
            df[f'Momentum_{window}'] = df[price_col] / df[ma_col] - 1

    for lag in range(1, 13):
        df[f'Price_Lag_{lag}'] = df[price_col].shift(lag)
        df[f'Price_Diff_{lag}'] = df[price_col] - df[f'Price_Lag_{lag}']
        df[f'Price_Ratio_{lag}'] = df[price_col] / df[f'Price_Lag_{lag}']

    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df = df.dropna()
    return df

def prepare_training_data(df, date_col, price_col):
    df['Target_Next_Year'] = df[price_col].shift(-12)
    exclude_cols = [date_col, price_col, 'Target_Next_Year', 'Month-Year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df_train = df.dropna(subset=['Target_Next_Year']).copy()
    X = df_train[feature_cols]
    y = df_train['Target_Next_Year']
    return X, y, feature_cols

def train_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    rf_metrics = {'MAE': mean_absolute_error(y_val, rf_pred), 'RMSE': np.sqrt(mean_squared_error(y_val, rf_pred)), 'R2': r2_score(y_val, rf_pred)}
    xgb_metrics = {'MAE': mean_absolute_error(y_val, xgb_pred), 'RMSE': np.sqrt(mean_squared_error(y_val, xgb_pred)), 'R2': r2_score(y_val, xgb_pred)}
    return rf, xgb_model, rf_metrics, xgb_metrics

def main():
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    data = load_data()
    if data.empty:
        return

    ticker_col = next((col for col in data.columns if col.lower() == 'ticker'), None)
    date_col = next((col for col in data.columns if col.lower() == 'date'), None)
    price_col = next((col for col in data.columns if col.lower() in ['prc', 'price']), None)
    returns_col = next((col for col in data.columns if col.lower() == 'ret'), None)
    volume_col = next((col for col in data.columns if col.lower() == 'vol'), None)

    if not all([ticker_col, date_col, price_col]):
        st.error("Missing necessary columns.")
        return

    st.sidebar.header("Configuration")
    available_tickers = data[ticker_col].unique().tolist()
    selected_ticker = st.sidebar.selectbox("Select Stock", available_tickers)
    model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Ensemble (RF + XGBoost)"])
    rf_weight = st.sidebar.slider("Random Forest Weight", 0.0, 1.0, 0.5, step=0.1) if model_type == "Ensemble (RF + XGBoost)" else 0.5

    ticker_data = data[data[ticker_col] == selected_ticker].copy()
    ticker_data[date_col] = pd.to_datetime(ticker_data[date_col], errors='coerce')
    ticker_data = ticker_data.dropna(subset=[date_col])
    ticker_data['Month-Year'] = ticker_data[date_col].dt.strftime('%b %Y')
    ticker_data = enforce_numeric_columns(ticker_data, returns_col, price_col, volume_col)

    st.subheader(f"{selected_ticker} Historical Price Data")
    fig = px.line(ticker_data, x=date_col, y=price_col, title=f"{selected_ticker} Stock Price History")
    st.plotly_chart(fig, use_container_width=True)

    featured_df = engineer_features(ticker_data, date_col, price_col, returns_col, volume_col)
    X, y, features = prepare_training_data(featured_df, date_col, price_col)

    if len(X) < 6:
        st.warning("Insufficient data for model training.")
        return
        st.write("Number of rows after feature engineering:", len(featured_df))

    rf_model, xgb_model, rf_metrics, xgb_metrics = train_models(X, y)

    last_price = ticker_data[price_col].iloc[-1]
    last_row = featured_df[features].iloc[-1:]
    rf_pred = rf_model.predict(last_row)[0]
    xgb_pred = xgb_model.predict(last_row)[0]
    ensemble_pred = rf_pred * rf_weight + xgb_pred * (1 - rf_weight)

    st.subheader("Price Predictions for End of 2025")
    st.metric("Last Known Price", f"${last_price:.2f}")
    if model_type == "Random Forest":
        st.metric("Random Forest Prediction", f"${rf_pred:.2f}")
    elif model_type == "XGBoost":
        st.metric("XGBoost Prediction", f"${xgb_pred:.2f}")
    else:
        st.metric("Ensemble Prediction", f"${ensemble_pred:.2f}")

    st.subheader("Model Performance")
    st.write("Random Forest:", rf_metrics)
    st.write("XGBoost:", xgb_metrics)

if __name__ == '__main__':
    main()
