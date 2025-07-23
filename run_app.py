#!/usr/bin/env python3
"""
Script to run the crypto app backend
"""
import subprocess
import sys
import os

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask', 'flask-cors', 'requests', 'numpy', 
        'tensorflow', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model_files():
    """Check if ML model files exist"""
    required_files = [
        'advanced_lstm_price_predictor.h5',
        'lstm_scalers.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🤖 Train the model first with:")
        print("python train_lstm_multi.py")
        return False
    
    return True

def main():
    print("🚀 Starting Crypto Dashboard Backend")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        print("\n⚠️  Model files missing, but you can still run the app")
        print("   (Predictions will not work until you train the model)")
    
    print("\n✅ All checks passed!")
    print("\n🌐 Starting Flask backend on http://localhost:5000")
    print("📱 Your Flutter app should connect to this URL")
    print("\n" + "=" * 50)
    
    # Run the Flask app
    try:
        subprocess.run([sys.executable, 'app_combined.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Backend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running backend: {e}")

if __name__ == "__main__":
    main()