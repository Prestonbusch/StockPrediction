import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="1-Year Stock Price Forecast", page_icon="📈", layout="wide")
st.title("📈 1-Year Stock Price Forecast Using Random Forest")

@st.cache_data
def fetch_data(ticker, start="2018-01-01", end="2024-12-31"):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("Downloaded data is empty")
        data.reset_index(inplace=True)
        if 'Close' not in data.columns:
            raise ValueError("Column 'Close' is missing from downloaded data")
        data = data.dropna(subset=['Close'])
        return data
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

def engineer_features(data):
    try:
        if 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError("Missing required columns in data.")
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year
        data['Price'] = data['Close']
        for lag in range(1, 13):
            data[f'Lag_{lag}'] = data['Price'].shift(lag)
        data['Target'] = data['Price'].shift(-12)
        required_cols = [f'Lag_{lag}' for lag in range(1, 13)] + ['Target']
        data = data.dropna(subset=required_cols)
        return data
    except Exception as e:
        st.error(f"Feature engineering error: {e}")
        return pd.DataFrame()

def train_model(data):
    try:
        features = [col for col in data.columns if 'Lag' in col]
        X = data[features]
        y = data['Target']
        if X.empty or y.empty:
            raise ValueError("Training data is empty.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_tree_preds = np.stack([est.predict(X_test) for est in model.estimators_])
        pred_std = all_tree_preds.std(axis=0)
        last_input = X.iloc[[-1]]
        prediction = model.predict(last_input)[0]
        return model, features, data.iloc[[-1]], prediction, y_test.iloc[-1], r2_score(y_test, preds), pred_std[-1]
    except Exception as e:
        st.error(f"Model training error: {e}")
        return None, [], None, None, None, None, None

ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
if ticker:
    df = fetch_data(ticker)
    if df.empty:
        st.error("Failed to load data or ticker not valid.")
    else:
        st.write(f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}")
        st.subheader("Raw Price Chart")
        if 'Close' in df.columns:
            st.line_chart(df.set_index("Date")["Close"])
        else:
            st.warning("Column 'Close' not found in data.")

        df_feat = engineer_features(df)
        st.write("Rows after feature engineering:", len(df_feat))

        if df_feat.empty or len(df_feat) < 50:
            st.warning("Not enough data after feature engineering.")
        else:
            model, feat_cols, last_row, prediction, actual, r2, conf_std = train_model(df_feat)
            if prediction is not None:
                st.subheader("🔮 Forecast for 1 Year Ahead")
                st.metric("Predicted Price (in 12 months)", f"${prediction:.2f}", f"± ${conf_std:.2f}")
                st.metric("Last Actual Price Used", f"${last_row['Price'].values[0]:.2f}")
                st.metric("R² Score", f"{r2:.4f}")

                st.subheader("📊 Feature Importance")
                importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
                fig = px.bar(importances, orientation='h', labels={'value': 'Importance', 'index': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Model could not be trained properly.")
