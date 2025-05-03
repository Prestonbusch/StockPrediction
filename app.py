
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="S&P 500 Stock Price Prediction - End of 2025",
    page_icon="📈",
    layout="wide"
)

# Load data
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

# Engineer features
def engineer_features(df, date_col, price_col, returns_col=None, volume_col=None):
    df = df.copy()
    df = df.sort_values(date_col)
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['MonthsSinceStart'] = (df[date_col].dt.year - df[date_col].dt.year.min()) * 12 + df[date_col].dt.month - df[date_col].dt.month.min() + 1

    for window in [2, 3]:
        df[f'MA_{window}'] = df[price_col].rolling(window=window).mean()
        df[f'Volatility_{window}'] = df[price_col].rolling(window=window).std()

    if returns_col is not None:
        df[returns_col] = pd.to_numeric(df[returns_col], errors='coerce')
        for window in [2, 3]:
            df[f'Cum_Return_{window}'] = (1 + df[returns_col]).rolling(window=window).apply(lambda x: np.prod(x) - 1)
            df[f'Avg_Return_{window}'] = df[returns_col].rolling(window=window).mean()

    if volume_col is not None:
        df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
        for window in [2, 3]:
            df[f'Volume_MA_{window}'] = df[volume_col].rolling(window=window).mean()
        df['Volume_Ratio'] = df[volume_col] / df[volume_col].rolling(window=2).mean()

    for window in [2, 3]:
        df[f'Momentum_{window}'] = df[price_col] / df[f'MA_{window}'] - 1

    for lag in range(1, 4):
        df[f'Price_Lag_{lag}'] = df[price_col].shift(lag)
        df[f'Price_Diff_{lag}'] = df[price_col] - df[f'Price_Lag_{lag}']
        df[f'Price_Ratio_{lag}'] = df[price_col] / df[f'Price_Lag_{lag}']

    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df = df.dropna()
    return df

# Prepare training data
def prepare_training_data(df, date_col, price_col):
    df['Target_Next_Year'] = df[price_col].shift(-12)
    feature_cols = [col for col in df.columns if col not in [date_col, price_col, 'Target_Next_Year']]
    df_train = df.dropna(subset=['Target_Next_Year']).copy()
    X = df_train[feature_cols]
    y = df_train['Target_Next_Year']
    return X, y, feature_cols

# Train models
def train_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    rf_metrics = {'MAE': mean_absolute_error(y_val, rf_pred), 'RMSE': np.sqrt(mean_squared_error(y_val, rf_pred)), 'R2': r2_score(y_val, rf_pred)}
    xgb_metrics = {'MAE': mean_absolute_error(y_val, xgb_pred), 'RMSE': np.sqrt(mean_squared_error(y_val, xgb_pred)), 'R2': r2_score(y_val, xgb_pred)}
    return rf_model, xgb_model, rf_metrics, xgb_metrics

# Main Streamlit app logic
def main():
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    data = load_data()
    if data.empty:
        return
    ticker_col = next((c for c in data.columns if c.lower() == 'ticker'), None)
    date_col = next((c for c in data.columns if c.lower() == 'date'), None)
    price_col = next((c for c in data.columns if c.lower() in ['prc', 'price']), None)
    returns_col = next((c for c in data.columns if c.lower() == 'ret'), None)
    volume_col = next((c for c in data.columns if c.lower() == 'vol'), None)

    if None in [ticker_col, date_col, price_col]:
        st.error("Missing essential columns")
        return

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])

    for col in [price_col, returns_col, volume_col]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    tickers = data[ticker_col].unique().tolist()
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
    filtered = data[data[ticker_col] == selected_ticker].copy()
    st.write(f"Raw rows: {len(filtered)}")
    st.write(filtered[[date_col, price_col, returns_col, volume_col]].tail())

    st.subheader("Raw Price Chart")
    fig = px.line(filtered, x=date_col, y=price_col, title=f"{selected_ticker} Price History")
    st.plotly_chart(fig, use_container_width=True)

    df_feat = engineer_features(filtered, date_col, price_col, returns_col, volume_col)
    st.write("Rows after feature engineering:", len(df_feat))
    if len(df_feat) < 12:
        st.warning("Insufficient data for training.")
        return

    X, y, feat_cols = prepare_training_data(df_feat, date_col, price_col)
    rf, xgb_m, rf_mets, xgb_mets = train_models(X, y)
    st.subheader("Model Metrics")
    st.write("Random Forest:", rf_mets)
    st.write("XGBoost:", xgb_mets)

    latest_feat = df_feat[feat_cols].iloc[-1:].copy()
    rf_pred = rf.predict(latest_feat)[0]
    xgb_pred = xgb_m.predict(latest_feat)[0]
    ensemble = (rf_pred + xgb_pred) / 2

    st.subheader("Predictions for Dec 2025")
    st.metric("Random Forest", f"${rf_pred:.2f}")
    st.metric("XGBoost", f"${xgb_pred:.2f}")
    st.metric("Ensemble Avg", f"${ensemble:.2f}")

if __name__ == '__main__':
    main()
