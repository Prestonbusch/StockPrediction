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

def fetch_data(ticker, start="2018-01-01", end="2024-12-31"):
    data = yf.download(ticker, start=start, end=end, group_by='ticker', auto_adjust=False)

    # Flatten columns if they are in MultiIndex format
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    data.reset_index(inplace=True)
    data.rename(columns={"Close": "Adj Close"}, inplace=True)  # Align column name to expected
    return data

def engineer_features(data):
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Price'] = data['Adj Close']
    for lag in range(1, 13):
        data[f'Lag_{lag}'] = data['Price'].shift(lag)
    data['Target'] = data['Price'].shift(-12)
    return data.dropna()

def train_model(data):
    features = [col for col in data.columns if 'Lag' in col]
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    all_tree_preds = np.stack([est.predict(X_test) for est in model.estimators_])
    pred_std = all_tree_preds.std(axis=0)
    return model, features, X.iloc[-1:], preds[-1], y_test.iloc[-1], r2_score(y_test, preds), pred_std[-1]

ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
if ticker:
    df = fetch_data(ticker)
    if df.empty:
        st.error("Failed to load data or ticker not valid.")
    else:
        st.write(f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}")
        st.line_chart(df.set_index("Date")["Adj Close"])

        df_feat = engineer_features(df)
        if len(df_feat) < 50:
            st.warning("Not enough data after feature engineering.")
        else:
            model, feat_cols, last_row, prediction, actual, r2, conf_std = train_model(df_feat)
            st.subheader("🔮 Forecast for 1 Year Ahead")
            st.metric("Predicted Price (in 12 months)", f"${prediction:.2f}", f"± ${conf_std:.2f}")
            st.metric("Last Actual Price Used", f"${last_row['Price'].values[0]:.2f}")
            st.metric("R² Score", f"{r2:.4f}")

            st.subheader("📊 Feature Importance")
            importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
            fig = px.bar(importances, orientation='h', labels={'value': 'Importance', 'index': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
