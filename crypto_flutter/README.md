# Crypto Flutter Dashboard

A cryptocurrency dashboard built with Flutter that provides real-time market data, AI-powered predictions, and portfolio management features.

## Features

- **Real-time Market Data**: Live cryptocurrency prices and market information
- **AI Price Predictions**: Machine learning-powered price movement predictions
- **Portfolio Management**: Buy/sell cryptocurrencies and track your holdings
- **Market Analysis**: Trending coins, global market statistics, and search functionality
- **Interactive Charts**: Real-time price charts with technical indicators

## Backend Integration

This Flutter app connects to a Python Flask backend (`app_combined.py`) that provides:
- LSTM-based price prediction model
- Wallet management (buy/sell operations)
- Real-time market data aggregation

## Setup Instructions

### 1. Backend Setup (Python Flask)

1. Install Python dependencies:
   ```bash
   pip install flask flask-cors requests numpy tensorflow scikit-learn joblib
   ```

2. Train the ML model (optional - pre-trained model included):
   ```bash
   python train_lstm_multi.py
   ```

3. Start the Flask backend:
   ```bash
   python app_combined.py
   ```
   The backend will run on `http://localhost:5000`

### 2. Flutter Frontend Setup

1. Install Flutter dependencies:
   ```bash
   flutter pub get
   ```

2. Update the backend URL in `lib/services/api_service.dart`:
   - For local development: `http://localhost:5000`
   - For Android emulator: `http://10.0.2.2:5000`
   - For physical device: `http://YOUR_COMPUTER_IP:5000`

3. Run the Flutter app:
   ```bash
   flutter run
   ```

### 3. Network Configuration

**For Android Emulator:**
- The backend URL should be `http://10.0.2.2:5000`

**For Physical Device:**
- Find your computer's IP address:
  - Windows: `ipconfig`
  - macOS/Linux: `ifconfig`
- Update the backend URL to `http://YOUR_IP:5000`
- Ensure your firewall allows connections on port 5000

**For iOS Simulator:**
- Use `http://localhost:5000` or your computer's IP address

## API Endpoints

The Flutter app connects to these Flask backend endpoints:

- `GET /wallet` - Get wallet balance and holdings
- `POST /buy` - Buy cryptocurrency
- `POST /sell` - Sell cryptocurrency
- `POST /predict` - Get AI price prediction (historical data)
- `POST /fetch-and-predict` - Get quick AI price prediction (current data)

## Features Overview

### Market Data Tab
- View top cryptocurrencies by market cap
- Real-time price updates
- 24h price change indicators
- Volume information

### Search Tab
- Search for any cryptocurrency
- View detailed coin information
- Market cap rankings

### Trending Tab
- See currently trending cryptocurrencies
- Trending scores and rankings

### Global Stats Tab
- Global market capitalization
- Total trading volume
- Bitcoin and Ethereum dominance

### Wallet Tab
- View current balance and holdings
- Buy and sell cryptocurrencies
- Real-time profit/loss calculations

### Prediction Tab
- AI-powered price predictions
- Confidence scores
- Buy/sell recommendations
- Uses advanced LSTM neural network

## Machine Learning Model

The app uses an advanced LSTM (Long Short-Term Memory) neural network that analyzes:
- Historical price data
- Trading volume
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market momentum
- Volatility patterns

The model provides binary predictions (UP/DOWN) with confidence scores.

## Development Notes

- The app uses the CoinGecko API for market data
- All trading is simulated (no real money involved)
- The ML model is trained on historical data
- Predictions are for educational purposes only

## Troubleshooting

1. **Connection Issues:**
   - Ensure the Flask backend is running
   - Check the correct IP address/URL configuration
   - Verify firewall settings

2. **Android Network Issues:**
   - Add `android:usesCleartextTraffic="true"` to AndroidManifest.xml for HTTP connections
   - Use `http://10.0.2.2:5000` for emulator

3. **iOS Network Issues:**
   - Add network permissions in Info.plist if needed
   - Use `http://localhost:5000` or computer's IP

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)