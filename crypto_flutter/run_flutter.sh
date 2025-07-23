#!/bin/bash

echo "🚀 Starting Crypto Flutter App"
echo "================================"

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter is not installed or not in PATH"
    echo "📱 Install Flutter from: https://flutter.dev/docs/get-started/install"
    exit 1
fi

# Check Flutter doctor
echo "🔍 Checking Flutter setup..."
flutter doctor

echo ""
echo "📦 Getting Flutter dependencies..."
flutter pub get

echo ""
echo "🏗️  Building Flutter app..."
echo ""
echo "Available options:"
echo "1. Run on connected device/emulator"
echo "2. Run on web browser"
echo "3. Show connected devices"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "📱 Running on device/emulator..."
        flutter run
        ;;
    2)
        echo "🌐 Running on web browser..."
        flutter run -d web-server --web-port 8080
        ;;
    3)
        echo "📱 Connected devices:"
        flutter devices
        echo ""
        echo "Run 'flutter run' to start the app"
        ;;
    *)
        echo "📱 Running on default device..."
        flutter run
        ;;
esac