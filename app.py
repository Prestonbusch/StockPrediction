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

def engineer_features(ticker_data, date_col, price_col, returns_col=None, volume_col=None):
    """
    Engineer features for prediction models
    """
    # Create a copy to avoid modifying the original
    df = ticker_data.copy()
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Extract year and month as features
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    
    # Create time-based features
    df['MonthsSinceStart'] = (df[date_col].dt.year - df[date_col].dt.year.min()) * 12 + df[date_col].dt.month - df[date_col].dt.month.min() + 1
    
    # Calculate moving averages for prices
    for window in [3, 6, 12]:
        df[f'MA_{window}'] = df[price_col].rolling(window=window).mean()
    
    # Calculate price volatility
    for window in [3, 6, 12]:
        df[f'Volatility_{window}'] = df[price_col].rolling(window=window).std()
    
    # If returns column exists, calculate return-based features
    if returns_col is not None:
        # Convert return column to numeric (handles strings from CSV)
        df[returns_col] = pd.to_numeric(df[returns_col], errors='coerce')

        # Calculate cumulative returns
        for window in [3, 6, 12]:
            df[f'Cum_Return_{window}'] = (1 + df[returns_col]).rolling(window=window).apply(lambda x: np.prod(x) - 1)

        # Calculate average returns
        for window in [3, 6, 12]:
            df[f'Avg_Return_{window}'] = df[returns_col].rolling(window=window).mean()
    
    # If volume column exists, calculate volume-based features
    if volume_col is not None:
        # Calculate volume moving averages
        for window in [3, 6, 12]:
            df[f'Volume_MA_{window}'] = df[volume_col].rolling(window=window).mean()
        
        # Calculate volume ratios
        df['Volume_Ratio'] = df[volume_col] / df[volume_col].rolling(window=3).mean()
    
    # Calculate price momentum
    for window in [3, 6, 12]:
        colname = f'MA_{window}'
        if colname in df.columns:
            df[f'Momentum_{window}'] = df[price_col] / df[colname] - 1
    
    # Create lagged features for price
    for lag in range(1, 13):
        df[f'Price_Lag_{lag}'] = df[price_col].shift(lag)
    
    # Calculate price differences
    for lag in range(1, 13):
        lag_col = f'Price_Lag_{lag}'
        if lag_col in df.columns:
            df[f'Price_Diff_{lag}'] = df[price_col] - df[lag_col]
    
    # Calculate price ratios
    for lag in range(1, 13):
        lag_col = f'Price_Lag_{lag}'
        if lag_col in df.columns:
            df[f'Price_Ratio_{lag}'] = df[price_col] / df[lag_col]
    
    # Create month of year cyclical features (sine and cosine transformation)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_yearly_data(ticker_data, date_col, price_col):
    """
    Create yearly aggregated data
    """
    # Group by year and get end-of-year prices
    yearly_data = ticker_data.groupby(ticker_data[date_col].dt.year)[price_col].last().reset_index()
    yearly_data.columns = ['Year', 'End_Price']
    
    # Add other yearly metrics
    yearly_stats = ticker_data.groupby(ticker_data[date_col].dt.year).agg({
        price_col: ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    yearly_stats.columns = ['Year', 'Mean_Price', 'Min_Price', 'Max_Price', 'Std_Price']
    
    # Merge with yearly_data
    yearly_data = pd.merge(yearly_data, yearly_stats, on='Year')
    
    # Calculate year-over-year growth
    yearly_data['YoY_Growth'] = yearly_data['End_Price'].pct_change()
    
    return yearly_data

def prepare_training_data(df, date_col, price_col):
    """
    Prepare data for training the prediction models
    """
    # Create a target value - next month's price
    df['Target_Next_Month'] = df[price_col].shift(-1)
    
    # Create a target value - price 12 months ahead
    df['Target_Next_Year'] = df[price_col].shift(-12)
    
    # For year-end 2025 prediction, we're interested in the last available data point
    # and using features to predict approximately 12 months ahead
    
    # Define features to use for prediction
    exclude_cols = [date_col, price_col, 'Target_Next_Month', 'Target_Next_Year', 'Month-Year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove rows with NaN in targets
    df_train = df.dropna(subset=['Target_Next_Year']).copy()
    
    # Split features and target
    X = df_train[feature_cols]
    y = df_train['Target_Next_Year']
    
    return X, y, feature_cols

def train_models(X, y):
    """
    Train RandomForest and XGBoost models
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    rf_pred = rf_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    
    # Calculate metrics
    rf_metrics = {
        'MAE': mean_absolute_error(y_val, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, rf_pred)),
        'R2': r2_score(y_val, rf_pred)
    }
    
    xgb_metrics = {
        'MAE': mean_absolute_error(y_val, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, xgb_pred)),
        'R2': r2_score(y_val, xgb_pred)
    }
    
    return rf_model, xgb_model, rf_metrics, xgb_metrics

def predict_future_price(model, most_recent_data, feature_cols):
    """
    Make a prediction for the future price
    """
    # Prepare the features for prediction
    prediction_features = most_recent_data[feature_cols].iloc[-1:].copy()
    
    # Make prediction
    prediction = model.predict(prediction_features)[0]
    
    return prediction

def ensemble_prediction(rf_model, xgb_model, most_recent_data, feature_cols, rf_weight=0.5):
    """
    Create an ensemble prediction by combining RandomForest and XGBoost
    """
    # Get individual predictions
    rf_pred = predict_future_price(rf_model, most_recent_data, feature_cols)
    xgb_pred = predict_future_price(xgb_model, most_recent_data, feature_cols)
    
    # Calculate weighted average
    ensemble_pred = rf_pred * rf_weight + xgb_pred * (1 - rf_weight)
    
    return ensemble_pred, rf_pred, xgb_pred

def get_feature_importance(rf_model, xgb_model, feature_cols):
    """
    Get feature importance from models
    """
    # Get RF feature importance
    rf_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Get XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return rf_importance, xgb_importance

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
    
    # Check if TICKER column exists
    ticker_col = None
    for col in data.columns:
        if col.lower() == 'ticker':
            ticker_col = col
            break
    
    if ticker_col is None:
        st.error("TICKER column not found in dataset")
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
    
    # Find returns column
    returns_col = None
    for col in data.columns:
        if col.lower() == 'ret':
            returns_col = col
            break
    
    # Find volume column
    volume_col = None
    for col in data.columns:
        if col.lower() == 'vol':
            volume_col = col
            break
    
    # Filter for S&P500 tickers (common ones)
    common_sp500 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "FB", "GOOG", "TSLA", "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "NVDA", "PYPL", "ADBE", "CMCSA", "NFLX", "XOM", "VZ", "KO", "CSCO", "PEP", "TMO", "ABT", "CVX", "ABBV", "AVGO", "WMT", "CRM", "MRK", "ACN", "INTC", "PFE", "ORCL"]
    
    # Get actual tickers in dataset
    available_tickers = data[ticker_col].unique().tolist()
    
    # Find intersection of common S&P500 and available tickers
    display_tickers = [t for t in available_tickers if t in common_sp500]
    
    if not display_tickers:
        st.warning("No common S&P500 tickers found. Using all available tickers.")
        display_tickers = available_tickers
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Ticker selection
    selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "Ensemble (RF + XGBoost)"]
    )
    
    # Ensemble weight slider (only shown if Ensemble is selected)
    rf_weight = 0.5
    if model_type == "Ensemble (RF + XGBoost)":
        rf_weight = st.sidebar.slider(
            "Random Forest Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight given to Random Forest prediction in the ensemble"
        )
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Show Debug Information", value=False)
    
    # Filter data for selected ticker
    ticker_data = data[data[ticker_col] == selected_ticker].copy()
    
    if ticker_data.empty:
        st.error(f"No data available for {selected_ticker}")
        return
    
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
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Historical Data", "Prediction", "Model Insights"])
    
    with tab1:
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
        
        # Create yearly data
        yearly_data = create_yearly_data(ticker_data, date_col, price_col)
        
        # Display yearly summary
        st.subheader("Yearly Summary")
        st.dataframe(yearly_data.style.format({
            'End_Price': '${:.2f}',
            'Mean_Price': '${:.2f}',
            'Min_Price': '${:.2f}',
            'Max_Price': '${:.2f}',
            'Std_Price': '${:.2f}',
            'YoY_Growth': '{:.2%}'
        }))
        
        # Create columns for additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot returns if available
            if returns_col is not None:
                fig_returns = px.line(
                    ticker_data, 
                    x=date_col, 
                    y=returns_col, 
                    title=f"{selected_ticker} Monthly Returns"
                )
                fig_returns.update_layout(xaxis_title="Date", yaxis_title="Return")
                st.plotly_chart(fig_returns, use_container_width=True)
            
        with col2:
            # Plot volume if available
            if volume_col is not None:
                fig_volume = px.bar(
                    ticker_data, 
                    x=date_col, 
                    y=volume_col, 
                    title=f"{selected_ticker} Trading Volume"
                )
                fig_volume.update_layout(xaxis_title="Date", yaxis_title="Volume")
                st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        st.subheader("Advanced Price Prediction for End of 2025")
        
        # Engineer features
        featured_data = engineer_features(ticker_data, date_col, price_col, returns_col, volume_col)
        
        # Show engineered data in debug mode
        if debug_mode:
            st.subheader("Engineered Features")
            st.write(featured_data.head())
        
        # Prepare training data
        X, y, feature_cols = prepare_training_data(featured_data, date_col, price_col)
        
        # Show training data info in debug mode
        if debug_mode:
            st.write(f"Training data shape: {X.shape}")
            st.write(f"Number of features: {len(feature_cols)}")
            st.write("First few feature values:")
            st.write(X.head())
        
        # Check if we have enough data for training
        if len(X) < 24:  # Require at least 2 years of data
            st.warning(f"Not enough historical data for {selected_ticker} to train a reliable model. Need at least 24 months, but only have {len(X)}.")
            
            # Display simple projection instead
            last_price = ticker_data[price_col].iloc[-1]
            last_date = ticker_data[date_col].iloc[-1]
            
            if returns_col is not None:
                avg_monthly_return = ticker_data[returns_col].mean()
                months_to_project = 12  # assuming last data point is end of 2024
                projected_price = last_price * (1 + avg_monthly_return) ** months_to_project
                
                st.subheader("Simple Projection (Insufficient Data for ML Model)")
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
            # Train models
            rf_model, xgb_model, rf_metrics, xgb_metrics = train_models(X, y)
            
            # Display model metrics
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Random Forest Metrics:")
                st.write(f"MAE: ${rf_metrics['MAE']:.2f}")
                st.write(f"RMSE: ${rf_metrics['RMSE']:.2f}")
                st.write(f"R²: {rf_metrics['R2']:.4f}")
            
            with col2:
                st.write("XGBoost Metrics:")
                st.write(f"MAE: ${xgb_metrics['MAE']:.2f}")
                st.write(f"RMSE: ${xgb_metrics['RMSE']:.2f}")
                st.write(f"R²: {xgb_metrics['R2']:.4f}")
            
            # Make predictions for end of 2025
            ensemble_pred, rf_pred, xgb_pred = ensemble_prediction(
                rf_model, xgb_model, featured_data, feature_cols, rf_weight
            )
            
            # Get the last known price
            last_price = ticker_data[price_col].iloc[-1]
            last_date = ticker_data[date_col].iloc[-1]
            
            # Calculate growth percentages
            rf_growth = (rf_pred / last_price - 1) * 100
            xgb_growth = (xgb_pred / last_price - 1) * 100
            ensemble_growth = (ensemble_pred / last_price - 1) * 100
            
            # Display predictions
            st.subheader("Price Predictions for End of 2025")
            
            # Create columns for different model predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Last Known Price", 
                    f"${last_price:.2f}",
                    f"Date: {last_date.strftime('%b %Y')}"
                )
            
            # Display the appropriate prediction based on model selection
            if model_type == "Random Forest":
                with col2:
                    st.metric(
                        "Random Forest Prediction", 
                        f"${rf_pred:.2f}",
                        f"{rf_growth:.2f}%"
                    )
            elif model_type == "XGBoost":
                with col2:
                    st.metric(
                        "XGBoost Prediction", 
                        f"${xgb_pred:.2f}",
                        f"{xgb_growth:.2f}%"
                    )
            else:  # Ensemble
                with col2:
                    st.metric(
                        "Ensemble Prediction", 
                        f"${ensemble_pred:.2f}",
                        f"{ensemble_growth:.2f}%"
                    )
                
                with col3:
                    st.write("Individual Model Predictions:")
                    st.write(f"Random Forest: ${rf_pred:.2f} ({rf_growth:.2f}%)")
                    st.write(f"XGBoost: ${xgb_pred:.2f} ({xgb_growth:.2f}%)")
            
            # Create visualizations for predictions
            # Get historical yearly prices
            yearly_prices = featured_data.groupby('Year')[price_col].last().reset_index()
            
            # Create future years for projection
            future_years = pd.DataFrame({'Year': [2025]})
            
            # Add predictions to future_years based on selected model
            if model_type == "Random Forest":
                future_years[price_col] = rf_pred
                prediction_label = "Random Forest Prediction"
            elif model_type == "XGBoost":
                future_years[price_col] = xgb_pred
                prediction_label = "XGBoost Prediction"
            else:  # Ensemble
                future_years[price_col] = ensemble_pred
                prediction_label = "Ensemble Prediction"
            
            # Combine historical and future data
            projection_data = pd.concat([yearly_prices, future_years])
            projection_data['Type'] = ['Historical'] * len(yearly_prices) + ['Predicted'] * len(future_years)
            
            # Create projection chart
            fig_projection = px.line(
                projection_data, 
                x='Year', 
                y=price_col, 
                color='Type',
                title=f"{selected_ticker} Year-End Price Projection",
                markers=True
            )
            fig_projection.update_layout(xaxis_title="Year", yaxis_title="Price ($)")
            st.plotly_chart(fig_projection, use_container_width=True)
    
    with tab3:
        st.subheader("Model Insights")
        
        # Engineer features if not already done
        if 'featured_data' not in locals():
            featured_data = engineer_features(ticker_data, date_col, price_col, returns_col, volume_col)
            X, y, feature_cols = prepare_training_data(featured_data, date_col, price_col)
        
        # Check if we have enough data
        if len(X) < 24:
            st.warning("Not enough data to provide model insights. Need at least 24 months of data.")
            return
        
        # Train models if not already trained
        if 'rf_model' not in locals() or 'xgb_model' not in locals():
            rf_model, xgb_model, rf_metrics, xgb_metrics = train_models(X, y)
        
        # Get feature importance
        rf_importance, xgb_importance = get_feature_importance(rf_model, xgb_model, feature_cols)
        
        # Display feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Random Forest Feature Importance")
            
            # Get top 10 features
            top_rf_features = rf_importance.head(10)
            
            # Create feature importance chart
            fig_rf_importance = px.bar(
                top_rf_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Features - Random Forest"
            )
            fig_rf_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_rf_importance, use_container_width=True)
        
        with col2:
            st.subheader("XGBoost Feature Importance")
            
            # Get top 10 features
            top_xgb_features = xgb_importance.head(10)
            
            # Create feature importance chart
            fig_xgb_importance = px.bar(
                top_xgb_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Features - XGBoost"
            )
            fig_xgb_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_xgb_importance, use_container_width=True)
        
        # Model explanation
        st.subheader("How the Models Work")
        
        # Random Forest explanation
        st.write("**Random Forest**")
        st.write("""
        Random Forest is an ensemble learning method that builds multiple decision trees during training. 
        Each tree is trained on a random subset of the data and features. For prediction, the model 
        averages the predictions of all trees, which helps reduce overfitting and improves accuracy.
        
        Key strengths:
        - Handles non-linear relationships well
        - Less prone to overfitting compared to a single decision tree
        - Can capture complex patterns in the data
        - Provides feature importance metrics
        """)
        
        # XGBoost explanation
        st.write("**XGBoost**")
        st.write("""
        XGBoost (eXtreme Gradient Boosting) is a gradient boosting framework that builds trees 
        sequentially, with each tree correcting the errors of the previous ones. It's known for its 
        performance and speed.
        
        Key strengths:
        - Often provides superior performance on structured data
        - Handles missing values internally
        - Includes regularization to prevent overfitting
        - Optimized for efficiency and speed
        """)
        
        # Ensemble explanation
        st.write("**Ensemble (RF + XGBoost)**")
        st.write("""
        The ensemble approach combines predictions from both Random Forest and XGBoost models. 
        By combining these diverse models, we can often achieve better performance than either 
        model alone, especially when the models make different types of errors.
        
        The slider in the sidebar allows you to adjust the weight given to the Random Forest model 
        in the ensemble. A weight of 0.5 means equal weighting between Random Forest and XGBoost.
        """)
        
        # Feature explanation
        st.subheader("Feature Descriptions")
        st.write("""
        The models use various features derived from historical price and volume data:
        
        - **Moving Averages (MA_3, MA_6, MA_12)**: Average price over 3, 6, and 12 months
        - **Price Volatility**: Standard deviation of price over different time windows
        - **Momentum**: Current price relative to moving averages
        - **Lagged Prices**: Previous months' prices
        - **Price Differences**: Changes in price compared to previous months
        - **Cumulative Returns**: Product of returns over different time periods
        - **Volume Features**: Moving averages and ratios of trading volume
        - **Seasonal Features**: Transformations of month to capture seasonality
        
        The feature importance charts above show which of these features have the most influence on 
        the models' predictions.
        """)

# Run the app
if __name__ == '__main__':
    main()
