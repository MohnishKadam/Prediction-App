import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Crypto Analysis & Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .important {
        color: #D81B60;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .buy {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
    }
    .sell {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid #F44336;
    }
    .hold {
        background-color: rgba(255, 152, 0, 0.2);
        border: 1px solid #FF9800;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Crypto Analysis & Trading Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("""
This application uses cryptocurrency data and machine learning to provide trading recommendations.
The system analyzes historical price patterns, technical indicators, and market sentiment to predict future price movements.
""")

# Add info about data sources
with st.expander("About the data sources"):
    st.markdown("""
    This application will attempt to fetch real-time data from CoinGecko API. 
    
    If real-time data cannot be retrieved (due to API rate limits or connection issues), 
    the app automatically falls back to generating realistic synthetic data for demonstration purposes.
    
    The synthetic data maintains the typical patterns and volatility characteristics of actual cryptocurrency markets,
    allowing you to test and explore the application's features even when live data is unavailable.
    """)

# Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    
    # Cryptocurrency selection
    crypto_options = {
        "Bitcoin": "BTC",
        "Ethereum": "ETH",
        "Binance Coin": "BNB",
        "Cardano": "ADA",
        "Solana": "SOL",
        "XRP": "XRP",
        "Dogecoin": "DOGE",
        "Polkadot": "DOT"
    }
    
    selected_crypto_name = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    selected_crypto = crypto_options[selected_crypto_name]
    
    # Time period selection
    time_period = st.selectbox(
        "Time Period", 
        ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
    )
    
    # Analysis type
    analysis_type = st.multiselect(
        "Analysis to Perform",
        ["Price Prediction", "Technical Indicators", "Sentiment Analysis", "Volatility Assessment"],
        default=["Price Prediction", "Technical Indicators"]
    )
    
    # Risk tolerance
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Very Low", "Low", "Moderate", "High", "Very High"],
        value="Moderate"
    )

    # Advanced options
    with st.expander("Advanced Options"):
        prediction_days = st.slider("Prediction Horizon (Days)", 1, 30, 7)
        confidence_threshold = st.slider("Recommendation Confidence Threshold (%)", 50, 95, 70)

# Function to fetch cryptocurrency data from API or use synthetic data if API fails
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_crypto_data(symbol, days):
    """Fetch historical cryptocurrency data from the CoinGecko API or generate synthetic data if API fails."""
    try:
        # First try to fetch from CoinGecko API
        url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 7 else "hourly"
        }
        
        # Add API key if you have one (CoinGecko Pro)
        # params["x_cg_pro_api_key"] = "YOUR_API_KEY"
        
        # Add user agent and increase timeout
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prices = data['prices']
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Add volume data if available
            if 'total_volumes' in data:
                volumes = data['total_volumes']
                df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df_volume['timestamp'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
                df_volume = df_volume.set_index('timestamp')
                df['volume'] = df_volume['volume']
            
            return df
        else:
            st.warning(f"API returned status code: {response.status_code}. Using synthetic data instead.")
            return generate_synthetic_crypto_data(symbol, days)
    except Exception as e:
        st.warning(f"Could not fetch live data: {e}. Using synthetic data instead.")
        return generate_synthetic_crypto_data(symbol, days)

# Function to generate synthetic cryptocurrency data for demonstration
def generate_synthetic_crypto_data(symbol, days):
    """Generate synthetic cryptocurrency data for demonstration purposes."""
    # Set random seed based on symbol for consistent behavior
    np.random.seed(hash(symbol) % 10000)
    
    # Generate timestamps
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=days*24 if days <= 7 else days)
    
    # Set base price based on cryptocurrency
    base_prices = {
        "bitcoin": 40000, 
        "ethereum": 2000, 
        "binancecoin": 300, 
        "cardano": 0.5, 
        "solana": 50, 
        "ripple": 0.5, 
        "dogecoin": 0.1, 
        "polkadot": 15
    }
    
    base_price = base_prices.get(symbol.lower(), 100)  # Default to 100 if not found
    
    # Generate synthetic price data with realistic patterns
    # Start with random walk
    random_walk = np.random.normal(0, 0.02, size=len(timestamps))
    cumulative_returns = np.cumsum(random_walk)
    
    # Add trend component
    trend = np.linspace(0, 0.2, len(timestamps))  # Slight upward trend
    
    # Add cyclical component (sine wave)
    cycle = 0.1 * np.sin(np.linspace(0, 3 * np.pi, len(timestamps)))
    
    # Combine components
    price_factor = 1 + cumulative_returns + trend + cycle
    prices = base_price * price_factor
    
    # Create DataFrame
    df = pd.DataFrame(index=timestamps)
    df['price'] = prices
    
    # Add volume data (correlated with price changes but with more randomness)
    price_changes = np.abs(df['price'].pct_change().fillna(0))
    base_volume = base_price * 1000  # Base volume depends on price
    df['volume'] = base_volume * (1 + 5 * price_changes + 0.5 * np.random.rand(len(df)))
    
    # Add some volatility clusters
    volatility_days = np.random.choice(len(df) - 3, 5)  # 5 volatility events
    for day in volatility_days:
        shock_factor = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)
        df.iloc[day:day+3, 0] *= (1 + shock_factor * np.array([1, 0.5, 0.25]))
        df.iloc[day:day+3, 1] *= 2  # Increase volume during volatility
    
    return df

# Map time period selection to days parameter
def get_days_from_period(period):
    mapping = {
        "1 Day": 1,
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    return mapping.get(period, 30)

# Calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate technical indicators for the given dataframe."""
    # Copy dataframe to avoid modifying the original
    df_indicators = df.copy()
    
    # Simple Moving Averages
    df_indicators['SMA20'] = df_indicators['price'].rolling(window=20).mean()
    df_indicators['SMA50'] = df_indicators['price'].rolling(window=50).mean()
    df_indicators['SMA200'] = df_indicators['price'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df_indicators['EMA12'] = df_indicators['price'].ewm(span=12, adjust=False).mean()
    df_indicators['EMA26'] = df_indicators['price'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df_indicators['MACD'] = df_indicators['EMA12'] - df_indicators['EMA26']
    df_indicators['MACD_signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_indicators['MACD_histogram'] = df_indicators['MACD'] - df_indicators['MACD_signal']
    
    # Relative Strength Index (RSI)
    delta = df_indicators['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df_indicators['BB_middle'] = df_indicators['price'].rolling(window=20).mean()
    df_indicators['BB_std'] = df_indicators['price'].rolling(window=20).std()
    df_indicators['BB_upper'] = df_indicators['BB_middle'] + (df_indicators['BB_std'] * 2)
    df_indicators['BB_lower'] = df_indicators['BB_middle'] - (df_indicators['BB_std'] * 2)
    
    # Average True Range (ATR) - Volatility indicator
    high_low = df_indicators['price'].rolling(2).max() - df_indicators['price'].rolling(2).min()
    high_close = np.abs(df_indicators['price'].rolling(2).max() - df_indicators['price'].shift())
    low_close = np.abs(df_indicators['price'].rolling(2).min() - df_indicators['price'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df_indicators['ATR'] = true_range.rolling(14).mean()
    
    return df_indicators

# Prepare data for LSTM model
def prepare_data_for_lstm(df, feature_column='price', prediction_days=60):
    """Prepare the data for LSTM model by creating sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_column].values.reshape(-1, 1))
    
    x_train, y_train = [], []
    
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

# Build and train LSTM model
def build_and_train_lstm_model(x_train, y_train):
    """Build and train the LSTM model for price prediction."""
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    return model

# Make predictions using the trained model
def predict_prices(model, data, scaler, prediction_days=60, future_days=7):
    """Make price predictions for the next few days."""
    last_sequence = data[-prediction_days:].reshape(-1, 1)
    predicted_prices = []
    
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        # Reshape for model input
        x_test = current_sequence.reshape(1, prediction_days, 1)
        
        # Predict next price
        predicted_price = model.predict(x_test, verbose=0)
        
        # Add to list
        predicted_prices.append(predicted_price[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], predicted_price)
        current_sequence = current_sequence.reshape(-1, 1)
    
    # Inverse transform to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    
    return predicted_prices

# Generate trading recommendation
def generate_recommendation(df, future_prices, risk_tolerance):
    """Generate trading recommendation based on analysis."""
    
    # Get current price
    current_price = df['price'].iloc[-1]
    
    # Get predicted prices
    predicted_next_day = future_prices[0][0]
    predicted_end_period = future_prices[-1][0]
    
    # Calculate predicted change
    short_term_change = (predicted_next_day - current_price) / current_price * 100
    long_term_change = (predicted_end_period - current_price) / current_price * 100
    
    # Get technical indicators
    latest_indicators = df.iloc[-1]
    
    # Define recommendation thresholds based on risk tolerance
    risk_multipliers = {
        "Very Low": 0.5,
        "Low": 0.75,
        "Moderate": 1.0,
        "High": 1.25,
        "Very High": 1.5
    }
    
    multiplier = risk_multipliers[risk_tolerance]
    
    # Set thresholds
    buy_threshold = 3.0 * multiplier
    strong_buy_threshold = 7.0 * multiplier
    sell_threshold = -3.0 * multiplier
    strong_sell_threshold = -7.0 * multiplier
    
    # Technical analysis signals
    signals = {
        'trend': 0,
        'momentum': 0,
        'volatility': 0
    }
    
    # Trend signals
    if latest_indicators['SMA20'] > latest_indicators['SMA50']:
        signals['trend'] += 1
    else:
        signals['trend'] -= 1
        
    if latest_indicators['price'] > latest_indicators['SMA200']:
        signals['trend'] += 1
    else:
        signals['trend'] -= 1
    
    # Momentum signals
    if latest_indicators['RSI'] < 30:  # Oversold
        signals['momentum'] += 2
    elif latest_indicators['RSI'] > 70:  # Overbought
        signals['momentum'] -= 2
    elif latest_indicators['RSI'] < 45:
        signals['momentum'] += 1
    elif latest_indicators['RSI'] > 55:
        signals['momentum'] -= 1
    
    if latest_indicators['MACD'] > latest_indicators['MACD_signal']:
        signals['momentum'] += 1
    else:
        signals['momentum'] -= 1
    
    # Volatility signals
    if latest_indicators['price'] > latest_indicators['BB_upper']:
        signals['volatility'] -= 1  # Potentially overbought
    elif latest_indicators['price'] < latest_indicators['BB_lower']:
        signals['volatility'] += 1  # Potentially oversold
    
    # Final score calculation
    ml_score = short_term_change * 0.4 + long_term_change * 0.6
    technical_score = (signals['trend'] + signals['momentum'] + signals['volatility']) * 1.5
    
    final_score = ml_score * 0.6 + technical_score * 0.4
    
    # Generate recommendation
    if final_score > strong_buy_threshold:
        recommendation = "Strong Buy"
        confidence = min(90 + (final_score - strong_buy_threshold) * 2, 99)
        explanation = f"The model strongly predicts an upward price movement of {long_term_change:.2f}% over the next {len(future_prices)} days. Technical indicators confirm bullish momentum."
    elif final_score > buy_threshold:
        recommendation = "Buy"
        confidence = 70 + (final_score - buy_threshold) * 5
        explanation = f"The model predicts a positive price movement of {long_term_change:.2f}% over the next {len(future_prices)} days. Technical indicators suggest a bullish trend."
    elif final_score < strong_sell_threshold:
        recommendation = "Strong Sell"
        confidence = min(90 + abs(final_score - strong_sell_threshold) * 2, 99)
        explanation = f"The model strongly predicts a downward price movement of {long_term_change:.2f}% over the next {len(future_prices)} days. Technical indicators confirm bearish momentum."
    elif final_score < sell_threshold:
        recommendation = "Sell"
        confidence = 70 + abs(final_score - sell_threshold) * 5
        explanation = f"The model predicts a negative price movement of {long_term_change:.2f}% over the next {len(future_prices)} days. Technical indicators suggest a bearish trend."
    else:
        recommendation = "Hold"
        confidence = 50 + abs(final_score) * 5
        explanation = f"The model predicts a modest price movement of {long_term_change:.2f}% over the next {len(future_prices)} days. Technical indicators are mixed."
    
    # Format confidence level
    confidence = min(99, max(50, confidence))
    
    return recommendation, confidence, explanation, signals

# Main analysis function
def run_crypto_analysis():
    days = get_days_from_period(time_period)
    
    # Show loading spinner
    with st.spinner(f"Fetching and analyzing {selected_crypto_name} data..."):
        # 1. Fetch data
        coin_id = selected_crypto.lower()
        if selected_crypto == "BTC":
            coin_id = "bitcoin"
        elif selected_crypto == "ETH":
            coin_id = "ethereum"
        elif selected_crypto == "BNB":
            coin_id = "binancecoin"
        elif selected_crypto == "ADA":
            coin_id = "cardano"
        elif selected_crypto == "SOL":
            coin_id = "solana"
        elif selected_crypto == "XRP":
            coin_id = "ripple"
        elif selected_crypto == "DOGE":
            coin_id = "dogecoin"
        elif selected_crypto == "DOT":
            coin_id = "polkadot"
            
        df = fetch_crypto_data(coin_id, days)
        
        # Additional fallback in case both API and synthetic data generation fail
        if df is None:
            st.error("Unable to process cryptocurrency data. Using emergency fallback data.")
            # Create emergency fallback data
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, periods=days)
            
            if selected_crypto == "BTC":
                base_price = 40000
            elif selected_crypto == "ETH":
                base_price = 2000
            else:
                base_price = 100
                
            prices = base_price * (1 + np.cumsum(np.random.normal(0, 0.02, size=len(dates))))
            volumes = base_price * 1000 * (1 + np.random.rand(len(dates)))
            
            df = pd.DataFrame({
                'price': prices,
                'volume': volumes
            }, index=dates)
        
        # Resample to daily data if we have hourly data
        if len(df) > days * 2:
            df = df.resample('D').agg({'price': 'last', 'volume': 'sum'})
        
        # Fill NaN values
        df = df.fillna(method='ffill')
        
        # 2. Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # 3. Build and train LSTM model for price prediction
        prediction_window = min(60, len(df) // 3)  # Use at most 1/3 of data length for sequence
        
        x_train, y_train, scaler, scaled_data = prepare_data_for_lstm(
            df, feature_column='price', prediction_days=prediction_window
        )
        
        model = build_and_train_lstm_model(x_train, y_train)
        
        # 4. Make predictions
        future_prices = predict_prices(
            model, scaled_data, scaler, 
            prediction_days=prediction_window, 
            future_days=prediction_days
        )
        
        # 5. Generate recommendation
        recommendation, confidence, explanation, signals = generate_recommendation(
            df_with_indicators, future_prices, risk_tolerance
        )

    # Display current price information
    current_price = df['price'].iloc[-1]
    price_change_24h = ((df['price'].iloc[-1] - df['price'].iloc[-2]) / df['price'].iloc[-2]) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{selected_crypto_name} Price (USD)", 
            value=f"${current_price:.2f}", 
            delta=f"{price_change_24h:.2f}%"
        )
    
    with col2:
        highest_price = df['price'].max()
        lowest_price = df['price'].min()
        st.metric(label="Highest Price", value=f"${highest_price:.2f}")
        st.metric(label="Lowest Price", value=f"${lowest_price:.2f}")
    
    with col3:
        if 'volume' in df.columns:
            latest_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].mean()
            volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
            st.metric(
                label="Trading Volume", 
                value=f"${latest_volume:.2f}", 
                delta=f"{volume_change:.2f}%"
            )
    
    # Display recommendation
    st.markdown("<h2 class='sub-header'>Trading Recommendation</h2>", unsafe_allow_html=True)
    
    recommendation_class = "hold"
    if recommendation.lower().find("buy") >= 0:
        recommendation_class = "buy"
    elif recommendation.lower().find("sell") >= 0:
        recommendation_class = "sell"
    
    st.markdown(f"""
        <div class="prediction-box {recommendation_class}">
            <h3>{recommendation} {selected_crypto_name} with {confidence:.1f}% confidence</h3>
            <p>{explanation}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display price chart
    st.markdown("<h2 class='sub-header'>Price Analysis</h2>", unsafe_allow_html=True)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['price'], 
            name="Price",
            line=dict(color='#1E88E5', width=2)
        ),
        secondary_y=False,
    )
    
    # Add volume bars if available
    if 'volume' in df.columns:
        # Normalize volume to fit on the same chart
        volume_norm = df['volume'] / df['volume'].max() * df['price'].max() * 0.3
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=volume_norm,
                name="Volume",
                marker=dict(color='rgba(192, 192, 192, 0.5)')
            ),
            secondary_y=False,
        )
    
    # Add moving averages
    if len(df) > 20:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['SMA20'],
                name="20-day MA",
                line=dict(color='#FF9800', width=1.5)
            ),
            secondary_y=False,
        )
    
    if len(df) > 50:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['SMA50'],
                name="50-day MA",
                line=dict(color='#4CAF50', width=1.5)
            ),
            secondary_y=False,
        )
    
    # Add Bollinger Bands
    if len(df) > 20:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['BB_upper'],
                name="Upper BB",
                line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['BB_lower'],
                name="Lower BB",
                line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(250, 0, 0, 0.05)'
            ),
            secondary_y=False,
        )
    
    # Add future predictions
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), 
        periods=len(future_prices), 
        freq='D'
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            name="Predicted",
            line=dict(color='#D81B60', width=2, dash='dash')
        ),
        secondary_y=False,
    )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_crypto_name} ({selected_crypto}) Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators tab
    if "Technical Indicators" in analysis_type:
        st.markdown("<h2 class='sub-header'>Technical Indicators</h2>", unsafe_allow_html=True)
        
        # RSI Chart
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(
            go.Scatter(
                x=df_with_indicators.index, 
                y=df_with_indicators['RSI'], 
                name="RSI",
                line=dict(color='#1E88E5', width=2)
            )
        )
        
        # Add RSI reference lines
        fig_rsi.add_shape(
            type="line",
            x0=df_with_indicators.index[0],
            y0=70,
            x1=df_with_indicators.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
        )
        
        fig_rsi.add_shape(
            type="line",
            x0=df_with_indicators.index[0],
            y0=30,
            x1=df_with_indicators.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
        )
        
        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # MACD Chart
        fig_macd = go.Figure()
        
        fig_macd.add_trace(
            go.Scatter(
                x=df_with_indicators.index, 
                y=df_with_indicators['MACD'], 
                name="MACD",
                line=dict(color='#1E88E5', width=2)
            )
        )
        
        fig_macd.add_trace(
            go.Scatter(
                x=df_with_indicators.index, 
                y=df_with_indicators['MACD_signal'], 
                name="Signal Line",
                line=dict(color='#FF9800', width=2)
            )
        )
        
        # Add MACD histogram
        colors = ['red' if val < 0 else 'green' for val in df_with_indicators['MACD_histogram'].values]
        
        fig_macd.add_trace(
            go.Bar(
                x=df_with_indicators.index,
                y=df_with_indicators['MACD_histogram'],
                name="Histogram",
                marker_color=colors
            )
        )
        
        fig_macd.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            xaxis_title="Date",
            yaxis_title="Value",
            height=300
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Display current indicator values
        st.markdown("<h3>Current Indicator Values</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_value = df_with_indicators['RSI'].iloc[-1]
            rsi_status = "Neutral"
            if rsi_value > 70:
                rsi_status = "Overbought"
            elif rsi_value < 30:
                rsi_status = "Oversold"
                
            st.metric(
                label="RSI", 
                value=f"{rsi_value:.2f}",
                delta=rsi_status
            )
            
            st.metric(
                label="MACD", 
                value=f"{df_with_indicators['MACD'].iloc[-1]:.4f}",
                delta=f"{df_with_indicators['MACD_histogram'].iloc[-1]:.4f}"
            )
        
        with col2:
            st.metric(
                label="20-day MA", 
                value=f"${df_with_indicators['SMA20'].iloc[-1]:.2f}"
            )
            
            st.metric(
                label="50-day MA", 
                value=f"${df_with_indicators['SMA50'].iloc[-1]:.2f}"
            )
        
        with col3:
            bb_position = (df_with_indicators['price'].iloc[-1] - df_with_indicators['BB_lower'].iloc[-1]) / (
                df_with_indicators['BB_upper'].iloc[-1] - df_with_indicators['BB_lower'].iloc[-1]
            ) * 100
            
            bb_status = "Neutral"
            if bb_position > 80:
                bb_status = "Upper Band"
            elif bb_position < 20:
                bb_status = "Lower Band"
                
            st.metric(
                label="Bollinger Position", 
                value=f"{bb_position:.2f}%",
                delta=bb_status
            )
        
        with col4:
            st.metric(
                label="ATR (Volatility)", 
                value=f"${df_with_indicators['ATR'].iloc[-1]:.2f}"
            )
    
    # Predictions tab
    if "Price Prediction" in analysis_type:
        st.markdown("<h2 class='sub-header'>Price Predictions</h2>", unsafe_allow_html=True)
        
        # Create prediction dataframe
        pred_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1), 
            periods=len(future_prices), 
            freq='D'
        )
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted Price': future_prices.flatten(),
            'Change (%)': [(price - current_price) / current_price * 100 for price in future_prices.flatten()]
        })
        
        # Display prediction table
        st.dataframe(pred_df.style.format({
            'Predicted Price': '${:.2f}',
            'Change (%)': '{:.2f}%'
        }))
        
        # Show price prediction chart
        fig_pred = go.Figure()
        
        # Historical data
        fig_pred.add_trace(
            go.Scatter(
                x=df.index[-30:],  # Show last 30 days
                y=df['price'].iloc[-30:],
                name="Historical Price",
                line=dict(color='#1E88E5', width=2)
            )
        )
        
        # Predicted data
        fig_pred.add_trace(
            go.Scatter(
                x=pred_dates,
                y=future_prices.flatten(),
                name="Predicted Price",
                line=dict(color='#D81B60', width=2, dash='dash')
            )
        )
        
        # Add confidence interval (simple estimation)
        upper_bound = future_prices.flatten() * 1.1  # 10% upper bound
        lower_bound = future_prices.flatten() * 0.9  # 10% lower bound
        
        fig_pred.add_trace(
            go.Scatter(
                x=pred_dates,
                y=upper_bound,
                name="Upper Bound",
                line=dict(color='rgba(216, 27, 96, 0.2)', width=0),
                showlegend=False
            )
        )
        
        fig_pred.add_trace(
            go.Scatter(
                x=pred_dates,
                y=lower_bound,
                name="Lower Bound",
                line=dict(color='rgba(216, 27, 96, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(216, 27, 96, 0.2)',
                showlegend=False
            )
        )
        
        fig_pred.update_layout(
            title=f"{selected_crypto_name} Price Prediction for Next {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=400
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
    # Sentiment Analysis tab
    if "Sentiment Analysis" in analysis_type:
        st.markdown("<h2 class='sub-header'>Market Sentiment Analysis</h2>", unsafe_allow_html=True)
        
        st.info("Note: In a production environment, this section would integrate with news APIs, social media sentiment analysis, and market fear/greed indices.")
        
        # Placeholder for sentiment analysis
        # In a real application, this would call external APIs or use NLP models
        
        # Mock sentiment data
        sentiment_data = {
            "News Sentiment": 65,
            "Social Media Buzz": 78,
            "Reddit Mentions": 120,
            "Twitter Sentiment": 58,
            "Fear & Greed Index": 72
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create sentiment gauge chart
            fig_sentiment = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_data["News Sentiment"],
                title = {'text': "News Sentiment Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#EF5350"},
                        {'range': [30, 70], 'color': "#FFCA28"},
                        {'range': [70, 100], 'color': "#66BB6A"}
                    ]
                }
            ))
            
            fig_sentiment.update_layout(height=250)
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Twitter/social sentiment
            fig_twitter = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_data["Twitter Sentiment"],
                title = {'text': "Social Media Sentiment"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#EF5350"},
                        {'range': [30, 70], 'color': "#FFCA28"},
                        {'range': [70, 100], 'color': "#66BB6A"}
                    ]
                }
            ))
            
            fig_twitter.update_layout(height=250)
            st.plotly_chart(fig_twitter, use_container_width=True)
        
        with col2:
            # Fear & Greed gauge
            fig_fear = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_data["Fear & Greed Index"],
                title = {'text': "Market Fear & Greed Index"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 25], 'color': "#B71C1C"},  # Extreme Fear
                        {'range': [25, 45], 'color': "#EF5350"},  # Fear
                        {'range': [45, 55], 'color': "#FFCA28"},  # Neutral
                        {'range': [55, 75], 'color': "#66BB6A"},  # Greed
                        {'range': [75, 100], 'color': "#2E7D32"}  # Extreme Greed
                    ]
                }
            ))
            
            fig_fear.update_layout(height=250)
            st.plotly_chart(fig_fear, use_container_width=True)
            
            # Social Media Mentions
            fig_mentions = go.Figure()
            
            # Mock data - in real app would come from API
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            mentions = [80, 92, 105, 131, 120, 87, 120]
            
            fig_mentions.add_trace(go.Bar(
                x=days,
                y=mentions,
                marker_color='#1E88E5'
            ))
            
            fig_mentions.update_layout(
                title="Social Media Mentions (Last 7 Days)",
                height=250
            )
            
            st.plotly_chart(fig_mentions, use_container_width=True)
    
    # Volatility Assessment tab
    if "Volatility Assessment" in analysis_type:
        st.markdown("<h2 class='sub-header'>Volatility Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate daily returns
        df['daily_return'] = df['price'].pct_change() * 100
        
        # Calculate volatility metrics
        volatility_daily = df['daily_return'].std()
        volatility_weekly = df['daily_return'].rolling(window=7).std().iloc[-1]
        volatility_monthly = df['daily_return'].rolling(window=30).std().iloc[-1]
        
        # Create volatility chart
        fig_vol = go.Figure()
        
        fig_vol.add_trace(
            go.Scatter(
                x=df.index,
                y=df['daily_return'].rolling(window=20).std(),
                name="20-day Volatility",
                line=dict(color='#D81B60', width=2)
            )
        )
        
        if len(df) > 30:
            fig_vol.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['daily_return'].rolling(window=60).std(),
                    name="60-day Volatility",
                    line=dict(color='#1E88E5', width=2)
                )
            )
        
        fig_vol.update_layout(
            title="Historical Volatility (Standard Deviation of Returns)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Display volatility metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Daily Volatility", 
                value=f"{volatility_daily:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Weekly Volatility", 
                value=f"{volatility_weekly:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Monthly Volatility", 
                value=f"{volatility_monthly:.2f}%"
            )
        
        # Returns distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(
            go.Histogram(
                x=df['daily_return'].dropna(),
                nbinsx=30,
                marker_color='#1E88E5'
            )
        )
        
        fig_dist.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Risk assessment based on volatility
        vol_risk = "Medium"
        if volatility_daily > 5:
            vol_risk = "Very High"
        elif volatility_daily > 3:
            vol_risk = "High"
        elif volatility_daily < 1:
            vol_risk = "Low"
        elif volatility_daily < 0.5:
            vol_risk = "Very Low"
            
        st.info(f"Risk Assessment: {selected_crypto_name} currently has a **{vol_risk}** risk level based on price volatility.")
        
        max_drawdown = ((df['price'].cummax() - df['price']) / df['price'].cummax()).max() * 100
        
        st.warning(f"Maximum Drawdown: {max_drawdown:.2f}% (maximum percentage decline from a peak to a trough)")
    
    # Additional Information
    with st.expander("About this Analysis"):
        st.write("""
        This analysis combines multiple approaches to generate trading recommendations:
        
        1. **Machine Learning Prediction**: An LSTM (Long Short-Term Memory) neural network model is used to predict future prices based on historical patterns.
        
        2. **Technical Analysis**: Multiple technical indicators are calculated and analyzed, including:
           - Moving Averages (simple and exponential)
           - Relative Strength Index (RSI)
           - Moving Average Convergence Divergence (MACD)
           - Bollinger Bands
           - Average True Range (ATR)
           
        3. **Risk Assessment**: The recommendation is adjusted based on your selected risk tolerance level.
        
        **Disclaimer**: Cryptocurrency investments are subject to high market risk. This tool is for informational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.
        """)

# Run the analysis when the user clicks the button
if st.button("Analyze", type="primary"):
    run_crypto_analysis()

# Footer
st.markdown("---")
st.markdown(
    "Cryptocurrency Analysis & Trading Recommendation System | Data source: CoinGecko API", 
    unsafe_allow_html=True
)