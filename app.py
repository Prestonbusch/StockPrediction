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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
@st.cache_data
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
            # Display data info for debugging
            st.write("Data loaded successfully!")
            st.write(f"Number of records: {len(data)}")
            st.write(f"Columns: {data.columns.tolist()}")
            st.write("Sample of first few rows:")
            st.write(data.head())
            # Clean up the extracted file
            os.remove(f'temp_data/{file_name}')
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # If there's an error, display sample data for demonstration
        sample_data = """
        PERMNO,date,SICCD,TICKER,DIVAMT,FACPR,FACSHR,BIDLO,ASKHI,PRC,VOL,RET,SHROUT,ALTPRC,SPREAD,ALTPRCDT,RETX,vwretd,vwretx,ewretd,ewretx,sprtrn
        10104,1/31/2020,7372,ORCL,0.24,0,0,52.45,55.43,52.45,2005627,-0.00547,3207649,52.45,,1/31/2020,-0.01,-0.00173,-0.00285,-0.01333,-0.0142,-0.00163
        10104,2/28/2020,7372,ORCL,0,0,0,49.46,55.73,49.46,1935500,-0.05701,3207649,49.46,,2/28/2020,-0.05701,-0.07792,-0.07987,-0.06981,-0.07132,-0.08411
        10104,3/31/2020,7372,ORCL,0,0,0,39.8,50.9,48.33,5000331,-0.02285,3153584,48.33,,3/31/2020,-0.02285,-0.14173,-0.14369,-0.2075,-0.20977,-0.12512
        10104,4/30/2020,7372,ORCL,0.24,0,0,48.71,54.62,52.97,2615529,0.100972,3153584,52.97,,4/30/2020,0.096007,0.129674,0.128408,0.153868,0.152519,0.126844
        10104,5/29/2020,7372,ORCL,0,0,0,51.65,53.77,53.77,2138695,0.015103,3153584,53.77,,5/29/2020,0.015103,0.053739,0.051688,0.06407,0.062277,0.045282
        """
        return pd.read_csv(pd.StringIO(sample_data))

# Function to preprocess data
def preprocess_data(data, sp500_tickers):
    """
    Preprocess the data for analysis and modeling
    """
    # Check and standardize column names (convert to uppercase for consistency)
    data.columns = [col.upper() if col.lower() in ['ticker', 'prc', 'vol', 'ret', 'date', 'bidlo', 'askhi', 'shrout', 'divamt', 'sprtrn'] else col for col in data.columns]
    
    # Filter for S&P 500 stocks
    if 'TICKER' in data.columns:
        data = data[data['TICKER'].isin(sp500_tickers)]
    else:
        st.error(f"TICKER column not found. Available columns: {data.columns.tolist()}")
    
    # Convert date to datetime format - try multiple formats
    try:
        # First attempt with format detection
        data['date'] = pd.to_datetime(data['date'])
    except:
        try:
            # Second attempt with explicit format
            data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
        except:
            try:
                # Third attempt with another common format
                data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
            except Exception as e:
                st.error(f"Error converting dates: {e}")
                st.write("Sample of date values:", data['date'].head())
    
    # Check if PRC column exists and convert to absolute value
    if 'PRC' in data.columns:
        data['PRC'] = data['PRC'].abs()
    elif 'prc' in data.columns:
        # Rename to uppercase for consistency
        data.rename(columns={'prc': 'PRC'}, inplace=True)
        data['PRC'] = data['PRC'].abs()
    else:
        st.error("Price column (PRC/prc) not found in dataset")
    
    # Fill missing values
    data['DIVAMT'].fillna(0, inplace=True)
    data['SPREAD'].fillna(0, inplace=True)
    
    # Filter date range from Jan 2020 to Dec 2024
    data = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2024-12-31')]
    
    # Sort data by date
    data = data.sort_values(['TICKER', 'date'])
    
    return data

# Function for feature engineering
def engineer_features(data, ticker):
    """
    Create features for the prediction model
    """
    # Filter data for the selected ticker
    ticker_data = data[data['TICKER'] == ticker].copy()
    
    # Calculate moving averages
    ticker_data['MA_3'] = ticker_data['PRC'].rolling(window=3).mean()
    ticker_data['MA_6'] = ticker_data['PRC'].rolling(window=6).mean()
    ticker_data['MA_12'] = ticker_data['PRC'].rolling(window=12).mean()
    
    # Calculate price volatility
    ticker_data['Price_Volatility_3'] = ticker_data['PRC'].rolling(window=3).std()
    ticker_data['Price_Volatility_6'] = ticker_data['PRC'].rolling(window=6).std()
    ticker_data['Price_Volatility_12'] = ticker_data['PRC'].rolling(window=12).std()
    
    # Calculate return-based features
    ticker_data['Cum_Return_3'] = (1 + ticker_data['RET']).rolling(window=3).apply(lambda x: np.prod(x) - 1)
    ticker_data['Cum_Return_6'] = (1 + ticker_data['RET']).rolling(window=6).apply(lambda x: np.prod(x) - 1)
    ticker_data['Cum_Return_12'] = (1 + ticker_data['RET']).rolling(window=12).apply(lambda x: np.prod(x) - 1)
    
    # Calculate volume-based features
    ticker_data['Volume_MA_3'] = ticker_data['VOL'].rolling(window=3).mean()
    ticker_data['Volume_Ratio'] = ticker_data['VOL'] / ticker_data['Volume_MA_3']
    
    # Calculate price momentum
    ticker_data['Momentum_3'] = ticker_data['PRC'] / ticker_data['MA_3'] - 1
    ticker_data['Momentum_6'] = ticker_data['PRC'] / ticker_data['MA_6'] - 1
    ticker_data['Momentum_12'] = ticker_data['PRC'] / ticker_data['MA_12'] - 1
    
    # Calculate market relative performance
    ticker_data['Market_Rel_Return_3'] = ticker_data['RET'].rolling(window=3).mean() - ticker_data['sprtrn'].rolling(window=3).mean()
    ticker_data['Market_Rel_Return_6'] = ticker_data['RET'].rolling(window=6).mean() - ticker_data['sprtrn'].rolling(window=6).mean()
    
    # Extract year and month as features
    ticker_data['Year'] = ticker_data['date'].dt.year
    ticker_data['Month'] = ticker_data['date'].dt.month
    
    # Calculate price range and spread ratio
    ticker_data['Price_Range'] = (ticker_data['ASKHI'] - ticker_data['BIDLO']) / ticker_data['PRC']
    
    # Create lagged features (previous month's values)
    for lag in range(1, 4):
        ticker_data[f'PRC_Lag_{lag}'] = ticker_data['PRC'].shift(lag)
        ticker_data[f'RET_Lag_{lag}'] = ticker_data['RET'].shift(lag)
        ticker_data[f'VOL_Lag_{lag}'] = ticker_data['VOL'].shift(lag)
    
    # Drop rows with NaN values from the feature engineering
    ticker_data = ticker_data.dropna()
    
    return ticker_data

# Function to create yearly training data
def create_yearly_training_data(ticker_data):
    """
    Create yearly aggregated data for training long-term prediction models
    """
    yearly_data = []
    
    # Group data by year
    for year in range(2020, 2025):
        # Get data for the current year
        year_data = ticker_data[ticker_data['date'].dt.year == year]
        
        if not year_data.empty:
            # Get last record for the year (December)
            year_end_data = year_data.iloc[-1]
            
            # Create yearly features
            yearly_record = {
                'Year': year,
                'End_Price': year_end_data['PRC'],
                'Start_Price': year_data.iloc[0]['PRC'],
                'Avg_Price': year_data['PRC'].mean(),
                'Min_Price': year_data['PRC'].min(),
                'Max_Price': year_data['PRC'].max(),
                'Price_Volatility': year_data['PRC'].std(),
                'Avg_Volume': year_data['VOL'].mean(),
                'Annual_Return': (year_end_data['PRC'] / year_data.iloc[0]['PRC']) - 1,
                'Avg_Monthly_Return': year_data['RET'].mean(),
                'Total_Dividend': year_data['DIVAMT'].sum(),
                'Market_Correlation': year_data['RET'].corr(year_data['sprtrn']),
                'Avg_Market_Return': year_data['sprtrn'].mean()
            }
            
            # Get previous year's end price if available
            if year > 2020:
                prev_year_data = ticker_data[ticker_data['date'].dt.year == year-1]
                if not prev_year_data.empty:
                    prev_year_end = prev_year_data.iloc[-1]
                    yearly_record['Prev_Year_End_Price'] = prev_year_end['PRC']
                    yearly_record['YoY_Growth'] = (year_end_data['PRC'] / prev_year_end['PRC']) - 1
            
            yearly_data.append(yearly_record)
    
    # Convert to DataFrame
    yearly_df = pd.DataFrame(yearly_data)
    
    # Create target variable (next year's price)
    yearly_df['Next_Year_Price'] = yearly_df['End_Price'].shift(-1)
    
    return yearly_df

# Function to train and evaluate models
def train_yearly_model(yearly_data):
    """
    Train a model to predict next year's price based on yearly data
    """
    # Drop the last row since we don't have the target for it
    train_data = yearly_data.dropna(subset=['Next_Year_Price'])
    
    if len(train_data) < 3:
        return None, None, "Not enough yearly data points for training."
    
    # Define features and target
    features = [col for col in train_data.columns if col not in ['Year', 'Next_Year_Price']]
    X = train_data[features]
    y = train_data['Next_Year_Price']
    
    # If we have only a few years of data, we'll use all for training
    # In a real-world scenario with more data, you'd want to split into train/test
    
    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Train an XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Prepare data for 2025 prediction
    pred_data = yearly_data.iloc[-1:][features]
    
    # Make predictions
    rf_prediction = rf_model.predict(pred_data)[0]
    xgb_prediction = xgb_model.predict(pred_data)[0]
    
    # Ensemble prediction (average of both models)
    ensemble_prediction = (rf_prediction + xgb_prediction) / 2
    
    return ensemble_prediction, feature_importance, "Models trained successfully."

# Function to train time series model
def train_time_series_model(ticker_data):
    """
    Train a time series model to predict stock price
    """
    # Extract monthly prices
    monthly_prices = ticker_data[['date', 'PRC']].set_index('date')
    
    # Fit SARIMAX model
    try:
        # We'll start with simple parameters for this demonstration
        # In a real model, you would conduct parameter tuning
        model = SARIMAX(monthly_prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        
        # Forecast 12 months ahead (to end of 2025)
        forecast = results.forecast(steps=12)
        
        # Get the price at the end of 2025
        end_2025_price = forecast.iloc[-1]
        
        return end_2025_price, forecast, "Time series model trained successfully."
    except Exception as e:
        return None, None, f"Error training time series model: {e}"

# Main function
def main():
    # Sidebar for inputs
    st.sidebar.header("Model Configuration")
    
    # Load S&P 500 ticker list
    sp500_tickers = """AAPL MSFT NVDA AMZN META BRK.B GOOGL AVGO TSLA GOOG LLY JPM V NFLX XOM MA COST WMT PG UNH JNJ HD ABBV KO PM BAC CRM PLTR WFC CSCO MCD ORCL CVX ABT IBM GE LIN MRK T NOW ACN PEP VZ ISRG""".split()
    
    # Ticker selection
    selected_ticker = st.sidebar.selectbox("Select Stock", sp500_tickers)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Ensemble (RF + XGBoost)", "Time Series (SARIMAX)", "Both Models"]
    )
    
    # Load data
    data = load_data()
    
    # Check if directory exists
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    
    # Preprocess data
    processed_data = preprocess_data(data, sp500_tickers)
    
    # Check if selected ticker is in the data
    if selected_ticker not in processed_data['TICKER'].unique():
        st.error(f"No data available for {selected_ticker}")
        return
    
    # Engineer features
    ticker_data = engineer_features(processed_data, selected_ticker)
    
    # Create yearly data for training
    yearly_data = create_yearly_training_data(ticker_data)
    
    # Display basic information
    st.subheader(f"Analysis for {selected_ticker}")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Historical Data", "Prediction", "Model Insights"])
    
    with tab1:
        st.subheader("Historical Price Data (2020-2024)")
        
        # Plot historical prices
        fig = px.line(ticker_data, x='date', y='PRC', title=f"{selected_ticker} Stock Price History")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display yearly summary
        st.subheader("Yearly Summary")
        yearly_summary = yearly_data.drop('Next_Year_Price', axis=1, errors='ignore')
        st.dataframe(yearly_summary.style.format({
            'End_Price': '${:.2f}',
            'Start_Price': '${:.2f}',
            'Avg_Price': '${:.2f}',
            'Min_Price': '${:.2f}',
            'Max_Price': '${:.2f}',
            'Annual_Return': '{:.2%}',
            'Avg_Monthly_Return': '{:.2%}',
            'YoY_Growth': '{:.2%}',
            'Prev_Year_End_Price': '${:.2f}'
        }))
        
        # Plot key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot returns
            fig_returns = px.line(ticker_data, x='date', y='RET', title=f"{selected_ticker} Monthly Returns")
            fig_returns.update_layout(xaxis_title="Date", yaxis_title="Return")
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Plot volume
            fig_volume = px.bar(ticker_data, x='date', y='VOL', title=f"{selected_ticker} Trading Volume")
            fig_volume.update_layout(xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        st.subheader("Price Prediction for End of 2025")
        
        # Get the last known price
        last_price = ticker_data['PRC'].iloc[-1]
        
        # Container for model output
        prediction_container = st.container()
        
        with prediction_container:
            # Execute models based on selection
            if model_type in ["Ensemble (RF + XGBoost)", "Both Models"]:
                # Train ensemble model
                ensemble_prediction, feature_importance, ensemble_message = train_yearly_model(yearly_data)
                
                if ensemble_prediction is not None:
                    # Calculate price change
                    price_change = ensemble_prediction - last_price
                    price_change_pct = (price_change / last_price) * 100
                    
                    # Display ensemble prediction
                    st.subheader("Ensemble Model Prediction (RF + XGBoost)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Last Known Price (Dec 2024)", 
                            f"${last_price:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Predicted Price (Dec 2025)", 
                            f"${ensemble_prediction:.2f}", 
                            f"{price_change_pct:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Absolute Change", 
                            f"${price_change:.2f}"
                        )
                    
                    # Create forecast visualization
                    years = list(range(2020, 2026))
                    historical_prices = yearly_data['End_Price'].tolist()
                    forecasted_prices = historical_prices + [ensemble_prediction]
                    
                    forecast_df = pd.DataFrame({
                        'Year': years[:len(forecasted_prices)],
                        'Price': forecasted_prices,
                        'Type': ['Historical'] * len(historical_prices) + ['Forecasted'] * (len(forecasted_prices) - len(historical_prices))
                    })
                    
                    fig_forecast = px.line(
                        forecast_df, 
                        x='Year', 
                        y='Price', 
                        color='Type',
                        title=f"{selected_ticker} Year-End Price Forecast"
                    )
                    fig_forecast.update_layout(xaxis_title="Year", yaxis_title="Price ($)")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    st.warning(ensemble_message)
            
            if model_type in ["Time Series (SARIMAX)", "Both Models"]:
                # Train time series model
                ts_prediction, forecast, ts_message = train_time_series_model(ticker_data)
                
                if ts_prediction is not None:
                    # Calculate price change
                    price_change = ts_prediction - last_price
                    price_change_pct = (price_change / last_price) * 100
                    
                    # Display time series prediction
                    st.subheader("Time Series Model Prediction (SARIMAX)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Last Known Price (Dec 2024)", 
                            f"${last_price:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Predicted Price (Dec 2025)", 
                            f"${ts_prediction:.2f}", 
                            f"{price_change_pct:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Absolute Change", 
                            f"${price_change:.2f}"
                        )
                    
                    # Create forecast visualization
                    historical_dates = ticker_data['date']
                    historical_prices = ticker_data['PRC']
                    
                    forecast_dates = pd.date_range(
                        start=historical_dates.iloc[-1] + pd.DateOffset(months=1),
                        periods=12,
                        freq='M'
                    )
                    
                    # Combine historical and forecast data
                    dates_combined = pd.concat([historical_dates, pd.Series(forecast_dates)])
                    prices_combined = pd.concat([historical_prices, forecast])
                    types = ['Historical'] * len(historical_dates) + ['Forecasted'] * len(forecast_dates)
                    
                    forecast_df = pd.DataFrame({
                        'Date': dates_combined,
                        'Price': prices_combined,
                        'Type': types
                    })
                    
                    fig_ts_forecast = px.line(
                        forecast_df, 
                        x='Date', 
                        y='Price', 
                        color='Type',
                        title=f"{selected_ticker} Monthly Price Forecast"
                    )
                    fig_ts_forecast.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                    st.plotly_chart(fig_ts_forecast, use_container_width=True)
                else:
                    st.warning(ts_message)
    
    with tab3:
        st.subheader("Model Insights")
        
        if model_type in ["Ensemble (RF + XGBoost)", "Both Models"]:
            if 'feature_importance' in locals() and feature_importance is not None:
                st.subheader("Feature Importance")
                
                # Plot feature importance
                fig_importance = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Prediction Model"
                )
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.markdown("""
                ### Interpretation Guide
                
                - **End_Price**: The closing price at the end of each year
                - **Avg_Price**: The average price throughout the year
                - **Annual_Return**: The yearly return calculated from start to end price
                - **YoY_Growth**: Year-over-year growth compared to previous year
                - **Price_Volatility**: Standard deviation of prices within the year
                - **Avg_Volume**: Average trading volume during the year
                
                The model gives higher importance to features that most strongly correlate with next year's price.
                """)
        
        # Display general model information
        st.markdown("""
        ### Model Information
        
        **Ensemble Model (Random Forest + XGBoost)**
        
        This model combines Random Forest and XGBoost to leverage the strengths of both approaches:
        - Random Forest is less prone to overfitting and handles non-linear relationships well
        - XGBoost often provides better performance through gradient boosting techniques
        - The ensemble averages predictions from both models to improve stability
        
        **Time Series Model (SARIMAX)**
        
        The SARIMAX model (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) is specifically designed for time series forecasting:
        - Captures trend, seasonality, and autoregressive patterns
        - Accounts for monthly seasonality patterns in stock performance
        - Well-suited for long-term forecasting with temporal dependencies
        
        **Limitations**
        
        Stock price prediction has inherent limitations that users should be aware of:
        - Market conditions and external factors not in the historical data can significantly impact future prices
        - Longer prediction horizons (1 year) have greater uncertainty than short-term predictions
        - Past performance is not necessarily indicative of future results
        - The model doesn't account for unexpected events like economic crises, company-specific news, etc.
        """)

# Run the app
if __name__ == '__main__':
    main()
