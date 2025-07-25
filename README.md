# Crypto Prediction Project

A comprehensive cryptocurrency dashboard application with price prediction capabilities, featuring a Flutter mobile frontend and Python Flask backend with machine learning integration.
This is a real-time cryptocurrency prediction app designed to help users make better investment decisions using AI and machine learning. Built using Flutter for a smooth and responsive UI and Python (with TensorFlow and Flask) for the backend, the app offers live market data along with predictive insights for popular cryptocurrencies.

💡 What the App Does:
Users can input any coin ID (e.g., bitcoin, ripple, etc.).

The app sends this input to a Python backend running a trained LSTM machine learning model.

It then predicts whether the price will go UP or DOWN, along with a confidence score.

Based on the prediction, it suggests an action (BUY or SELL).

The app also displays a wallet balance, giving users a complete snapshot of their portfolio.

## 🚀 Features

### Frontend (Flutter)
- **Real-time Crypto Data**: Live cryptocurrency prices and market data
- **Interactive Dashboard**: Multiple tabs for different data views
- **Market Overview**: Top cryptocurrencies with price changes
- **Search Functionality**: Find specific cryptocurrencies
- **Trending Coins**: Popular and trending cryptocurrencies
- **Global Market Stats**: Overall market statistics
- **Digital Wallet**: Portfolio management capabilities
- **Price Prediction**: AI-powered cryptocurrency price predictions
- **Dark/Light Theme**: Adaptive theme support
- **Custom Charts**: Beautiful data visualization with FL Chart

### Backend (Python Flask)
- **REST API**: RESTful endpoints for cryptocurrency data
- **Machine Learning**: LSTM neural network for price prediction
- **Data Caching**: Efficient data caching system
- **CoinGecko Integration**: Real-time data from CoinGecko API
- **Advanced Features**: 15-feature extraction for ML predictions
- **Rate Limiting**: Smart API rate limiting and retry mechanisms


### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Crypto-main
```

2. **Install Python dependencies**
```bash
pip install flask flask-cors numpy tensorflow scikit-learn pandas requests joblib
```

3. **Run the Flask backend**
```bash
python app_combined.py
```

The backend will start on `http://127.0.0.1:5000`

### Frontend Setup

1. **Navigate to Flutter directory**
```bash
cd crypto_flutter
```

2. **Install Flutter dependencies**
```bash
flutter pub get
```

3. **Run the Flutter app**
```bash
flutter run
```

## 🏗️ Project Structure

```
Crypto-main/
├── crypto_flutter/                 # Flutter frontend
│   ├── lib/
│   │   ├── main.dart              # App entry point
│   │   ├── providers/             # State management
│   │   │   └── crypto_provider.dart
│   │   ├── screens/               # UI screens
│   │   │   ├── crypto_dashboard.dart
│   │   │   └── home_screen.dart
│   │   ├── models/                # Data models
│   │   ├── services/              # API services
│   │   └── utils/                 # Utilities
│   ├── assets/                    # Images, fonts, icons
│   └── pubspec.yaml              # Flutter dependencies
├── app_combined.py               # Flask backend server
├── train_lstm_multi.py          # ML model training
├── crypto_data_cache/           # Cached cryptocurrency data
├── advanced_lstm_price_predictor.h5  # Trained ML model
├── lstm_scalers.pkl             # ML model scalers
└── wallet.json                  # Wallet data storage
```

## 🔧 Configuration

### Backend Configuration
- Update API endpoints in `app_combined.py`
- Configure CoinGecko API settings
- Modify machine learning model parameters

### Frontend Configuration
- Update backend URL in `crypto_provider.dart`
- Customize theme colors in `main.dart`
- Configure API endpoints

## 📱 Usage

### Running the Application

1. **Start the Backend**
```bash
python app_combined.py
```

2. **Launch Flutter App**
```bash
cd crypto_flutter
flutter run
```

### Key Features Usage

#### Market Tab
- View top cryptocurrencies
- See real-time price changes
- Monitor market cap and volume

#### Search Tab
- Search for specific cryptocurrencies
- Get detailed coin information
- View historical data

#### Prediction Tab
- Enter a cryptocurrency symbol
- Get AI-powered price predictions
- View prediction confidence levels

#### Wallet Tab
- Add cryptocurrencies to portfolio
- Track portfolio performance
- View total portfolio value

## 🤖 Machine Learning

### Price Prediction Model
- **Algorithm**: LSTM (Long Short-Term Memory) Neural Network
- **Features**: 15-feature extraction including:
  - Current price and volume
  - Moving averages (SMA, EMA)
  - Technical indicators
  - Market cap ratios
  - Price volatility metrics

### Model Training
```bash
python train_lstm_multi.py
```

This will:
- Fetch historical cryptocurrency data
- Process and clean the data
- Train the LSTM model
- Save the trained model and scalers

## 🔗 API Endpoints

### Backend Endpoints
- `GET /api/global` - Global market statistics
- `GET /api/trending` - Trending cryptocurrencies
- `GET /api/search` - Search cryptocurrencies
- `GET /api/markets` - Market data
- `POST /api/predict` - Price prediction
- `GET /api/wallet` - Wallet data
- `POST /api/wallet` - Update wallet

## 📊 Dependencies

### Flutter Dependencies
- `provider: ^6.0.5` - State management
- `http: ^1.1.0` - HTTP requests
- `fl_chart: ^0.63.0` - Charts and graphs
- `intl: ^0.18.1` - Internationalization
- `shared_preferences: ^2.2.0` - Local storage


## 🎨 Customization

### Themes
- Light and dark theme support
- Custom color schemes in `main.dart`
- Material Design 3 support

### Fonts
- Poppins font family included
- Multiple font weights available
- Easy font customization

## 🚨 Troubleshooting

### Common Issues

1. **Backend not connecting**
   - Ensure Flask server is running on port 5000
   - Check firewall settings
   - Verify IP address in Flutter app

2. **ML model errors**
   - Ensure TensorFlow is properly installed
   - Check model file permissions
   - Verify Python version compatibility

3. **Flutter build issues**
   - Run `flutter clean` and `flutter pub get`
   - Check Flutter and Dart SDK versions
   - Verify device compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🔮 Future Enhancements

- Real-time notifications
- Advanced portfolio analytics
- Social trading features
- More ML models
- Multi-language support
- Desktop application version

---

**Happy Trading! 📈**
