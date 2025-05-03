import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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

# App title and description
st.title("S&P 500 Stock Price Prediction")
st.markdown("Predict stock prices for the end of 2025 using historical data from January 2020 to December 2024.")

# Function to load and process data
def load_data(file_path='Data/qgzpz8q9nyapvqp1_csv.zip'):
    """
    Load stock data from zip file
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Get the CSV filename in the zip
            file_name = zip_ref.namelist()[0]
            # Extract the file to a temporary location
            zip_ref.extract(file_name, path='temp_data')
            # Read the extracted CSV file
            data = pd.read_csv(f'temp_data/{file_name}')
            
            # Display column names immediately for debugging
            st.write("Data loaded successfully!")
            st.write(f"Columns found in dataset: {data.columns.tolist()}")
            
            # Clean up the extracted file
            os.remove(f'temp_data/{file_name}')
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrame if error
        return pd.DataFrame()

# Main function
def main():
    # Create temporary directory if it doesn't exist
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
        
    # Load data
    data = load_data()
    
    # Check if data is empty
    if data.empty:
        st.error("No data loaded. Please check the file path.")
        return
    
    # Display data sample
    st.subheader("Sample Data")
    st.write(data.head())
    
    # Check if TICKER column exists
    ticker_col = None
    for col in data.columns:
        if col.lower() == 'ticker':
            ticker_col = col
            break
    
    if ticker_col is None:
        st.error("TICKER column not found in dataset")
        return
    
    # Filter for S&P500 tickers (common ones)
    common_sp500 = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "GOOG", "TSLA", "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "NVDA", "PYPL", "ADBE", "CMCSA", "NFLX", "XOM", "VZ", "KO", "CSCO", "PEP", "TMO", "ABT", "CVX", "ABBV", "AVGO", "WMT", "CRM", "MRK", "ACN", "INTC", "PFE", "ORCL"]
    
    # Get actual tickers in dataset
    available_tickers = data[ticker_col].unique().tolist()
    
    # Find intersection of common S&P500 and available tickers
    display_tickers = [t for t in available_tickers if t in common_sp500]
    
    if not display_tickers:
        st.warning("No common S&P500 tickers found. Using all available tickers.")
        display_tickers = available_tickers
    
    # Sidebar for ticker selection
    selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
    
    # Filter data for selected ticker
    ticker_data = data[data[ticker_col] == selected_ticker].copy()
    
    if ticker_data.empty:
        st.error(f"No data available for {selected_ticker}")
        return
    
    # Find date column
    date_col = None
    for col in data.columns:
        if col.lower() == 'date':
            date_col = col
            break
    
    if date_col is None:
        st.error("DATE column not found in dataset")
        return
    
    # Find price column
    price_col = None
    for col in data.columns:
        if col.lower() in ['prc', 'price']:
            price_col = col
            break
    
    if price_col is None:
        st.error("Price column (PRC/price) not found in dataset")
        return
    
    # Display raw data
    st.subheader(f"Raw Data for {selected_ticker}")
    st.write(ticker_data[[date_col, price_col]])
    
    # Convert date to datetime and extract month/year only
    try:
        # First try with pandas default format detection
        ticker_data[date_col] = pd.to_datetime(ticker_data[date_col])
    except:
        try:
            # Try with explicit month/day/year format
            ticker_data[date_col] = pd.to_datetime(ticker_data[date_col], format='%m/%d/%Y')
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            st.write("Sample date values:", ticker_data[date_col].head())
            return
    
    # Create month-year string for easier viewing
    ticker_data['Month-Year'] = ticker_data[date_col].dt.strftime('%b %Y')
    
    # Sort by date
    ticker_data = ticker_data.sort_values(date_col)
    
    # Display basic price chart
    st.subheader(f"Historical Price Data for {selected_ticker}")
    
    # Create price chart
    fig = px.line(
        ticker_data, 
        x=date_col, 
        y=price_col, 
        title=f"{selected_ticker} Stock Price History"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple price prediction - just for demonstration
    st.subheader("Simple Price Projection for End of 2025")
    
    # Get the last known price
    last_price = ticker_data[price_col].iloc[-1]
    last_date = ticker_data[date_col].iloc[-1]
    
    # Simple extrapolation - this is just for demonstration
    # In a real model, you would use more sophisticated prediction methods
    
    # Calculate average monthly return
    returns_col = None
    for col in data.columns:
        if col.lower() == 'ret':
            returns_col = col
            break
    
    if returns_col is not None:
        # Calculate average monthly return
        avg_monthly_return = ticker_data[returns_col].mean()
        
        # Project forward to end of 2025
        months_to_project = 12  # assuming last data point is end of 2024
        projected_price = last_price * (1 + avg_monthly_return) ** months_to_project
        
        # Display projected price
        st.metric(
            "Last Known Price", 
            f"${last_price:.2f}",
            f"Date: {last_date.strftime('%b %Y')}"
        )
        
        st.metric(
            "Projected Price (End of 2025)", 
            f"${projected_price:.2f}",
            f"{((projected_price / last_price) - 1) * 100:.2f}%"
        )
    else:
        st.warning("Returns column not found. Cannot calculate projection.")
        
    # Instructions for building a more sophisticated model
    st.subheader("Next Steps")
    st.markdown("""
    This is a simple demonstration of loading and visualizing stock data. For a full prediction model, you would:
    
    1. Engineer features from historical data (moving averages, volatility, etc.)
    2. Train machine learning models (Random Forest, XGBoost, etc.)
    3. Evaluate model performance and make predictions
    
    The full code includes these components, but they require proper data formatting to work correctly.
    """)

# Run the app
if __name__ == '__main__':
    main()
