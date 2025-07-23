# 🚀 How to Run Your Crypto Dashboard App

## Quick Start Guide

### Step 1: Start the Python Backend

**Option A: Using the run script (Recommended)**
```bash
python run_app.py
```

**Option B: Direct command**
```bash
python app_combined.py
```

The backend will start on: **http://localhost:5000**

### Step 2: Start the Flutter Frontend

**For Windows:**
```bash
cd crypto_flutter
run_flutter.bat
```

**For Mac/Linux:**
```bash
cd crypto_flutter
chmod +x run_flutter.sh
./run_flutter.sh
```

**Manual Flutter commands:**
```bash
cd crypto_flutter
flutter pub get
flutter run
```

## Where You Can See the App Running

### 1. 📱 **Mobile Device/Emulator**
- **Android Emulator**: Install Android Studio and create a virtual device
- **iOS Simulator**: Available on macOS with Xcode installed
- **Physical Device**: Connect via USB with developer mode enabled

### 2. 🌐 **Web Browser**
```bash
cd crypto_flutter
flutter run -d web-server --web-port 8080
```
Then open: **http://localhost:8080**

### 3. 💻 **Desktop App**
```bash
cd crypto_flutter
flutter run -d windows  # Windows
flutter run -d macos    # macOS
flutter run -d linux    # Linux
```

## Network Configuration

### For Different Devices:

1. **Android Emulator**: 
   - Backend URL: `http://10.0.2.2:5000`

2. **Physical Device**: 
   - Find your computer's IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
   - Backend URL: `http://YOUR_IP:5000`
   - Example: `http://192.168.1.100:5000`

3. **iOS Simulator**: 
   - Backend URL: `http://localhost:5000`

4. **Web Browser**: 
   - Backend URL: `http://localhost:5000`

### Update Backend URL:
Edit `crypto_flutter/lib/services/api_service.dart`:
```dart
static const String _flaskBaseUrl = "http://YOUR_IP:5000";
```

## Checking Connected Devices

```bash
flutter devices
```

This shows all available devices where you can run the app.

## Troubleshooting

### Backend Issues:
- ✅ Python backend running on port 5000
- ✅ Required packages installed
- ✅ Model files present (for predictions)

### Frontend Issues:
- ✅ Flutter installed and in PATH
- ✅ Device/emulator connected
- ✅ Correct backend URL configured

### Network Issues:
- ✅ Firewall allows port 5000
- ✅ Both devices on same network (for physical devices)
- ✅ Correct IP address used

## App Features You'll See:

1. **Market Data Tab**: Live crypto prices
2. **Search Tab**: Find any cryptocurrency
3. **Trending Tab**: Hot cryptocurrencies
4. **Global Stats Tab**: Market overview
5. **Wallet Tab**: Buy/sell crypto (simulated)
6. **Prediction Tab**: AI price predictions

## URLs Summary:

- **Backend API**: http://localhost:5000
- **Flutter Web**: http://localhost:8080 (if running on web)
- **Mobile/Desktop**: Native app interface

Start with the backend first, then the Flutter app, and you'll see your crypto dashboard running! 🎉