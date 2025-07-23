@echo off
echo 🚀 Starting Crypto Flutter App
echo ================================

REM Check if Flutter is installed
flutter --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Flutter is not installed or not in PATH
    echo 📱 Install Flutter from: https://flutter.dev/docs/get-started/install
    pause
    exit /b 1
)

REM Check Flutter doctor
echo 🔍 Checking Flutter setup...
flutter doctor

echo.
echo 📦 Getting Flutter dependencies...
flutter pub get

echo.
echo 🏗️ Building Flutter app...
echo.
echo Available options:
echo 1. Run on connected device/emulator
echo 2. Run on web browser
echo 3. Show connected devices
echo.

set /p choice="Choose option (1-3): "

if "%choice%"=="1" (
    echo 📱 Running on device/emulator...
    flutter run
) else if "%choice%"=="2" (
    echo 🌐 Running on web browser...
    flutter run -d web-server --web-port 8080
) else if "%choice%"=="3" (
    echo 📱 Connected devices:
    flutter devices
    echo.
    echo Run 'flutter run' to start the app
    pause
) else (
    echo 📱 Running on default device...
    flutter run
)

pause