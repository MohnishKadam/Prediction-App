import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import time
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

API_BASE = "https://api.coingecko.com/api/v3"

# Cache directory for storing fetched data
CACHE_DIR = "crypto_data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def fetch_top_10_coins():
    """Fetch top 10 cryptocurrencies by market cap with caching"""
    cache_file = os.path.join(CACHE_DIR, "top_coins.txt")
    
    # Check if we have cached data (valid for 24 hours)
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=24):
            print("📋 Using cached coin list...")
            with open(cache_file, 'r') as f:
                return f.read().strip().split(',')
    
    # Fetch fresh data
    url = f"{API_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1"
    try:
        print("📥 Fetching fresh coin list...")
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        coin_ids = [coin['id'] for coin in res.json()]
        
        # Cache the result
        with open(cache_file, 'w') as f:
            f.write(','.join(coin_ids))
        
        return coin_ids
    except Exception as e:
        print(f"❌ Error fetching coin list: {e}")
        # Fallback to hardcoded list
        return ['bitcoin', 'ethereum', 'tether', 'bnb', 'solana', 'xrp', 'steth', 'usdc', 'cardano', 'dogecoin']

def fetch_batch_price_data(coin_ids, days=365):
    """Fetch data for multiple coins in a single API call using batch endpoint"""
    # CoinGecko allows fetching multiple coins in one call
    coins_str = ','.join(coin_ids)
    url = f"{API_BASE}/coins/markets?ids={coins_str}&vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    
    try:
        print(f"📥 Fetching batch market data for {len(coin_ids)} coins...")
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"❌ Error fetching batch data: {e}")
        return []

def calculate_technical_indicators(prices, volumes):
    """Calculate advanced technical indicators"""
    # Simple Moving Averages
    sma_5 = np.convolve(prices, np.ones(5)/5, mode='same')
    sma_10 = np.convolve(prices, np.ones(10)/10, mode='same')
    
    # Exponential Moving Average
    ema_12 = np.zeros_like(prices)
    alpha = 2 / (12 + 1)
    ema_12[0] = prices[0]
    for i in range(1, len(prices)):
        ema_12[i] = alpha * prices[i] + (1 - alpha) * ema_12[i-1]
    
    # RSI (Relative Strength Index)
    rsi = np.zeros_like(prices)
    window = 14
    for i in range(window, len(prices)):
        price_changes = np.diff(prices[i-window:i+1])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # MACD
    ema_12_short = np.zeros_like(prices)
    ema_26_long = np.zeros_like(prices)
    alpha_12 = 2 / (12 + 1)
    alpha_26 = 2 / (26 + 1)
    
    ema_12_short[0] = ema_26_long[0] = prices[0]
    for i in range(1, len(prices)):
        ema_12_short[i] = alpha_12 * prices[i] + (1 - alpha_12) * ema_12_short[i-1]
        ema_26_long[i] = alpha_26 * prices[i] + (1 - alpha_26) * ema_26_long[i-1]
    
    macd = ema_12_short - ema_26_long
    
    # Bollinger Bands
    bb_upper = np.zeros_like(prices)
    bb_lower = np.zeros_like(prices)
    bb_ratio = np.zeros_like(prices)
    
    window = 20
    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        mean_price = np.mean(window_prices)
        std_price = np.std(window_prices)
        bb_upper[i] = mean_price + (2 * std_price)
        bb_lower[i] = mean_price - (2 * std_price)
        bb_ratio[i] = (prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if bb_upper[i] != bb_lower[i] else 0.5
    
    return sma_5, sma_10, ema_12, rsi, macd, bb_ratio

def fetch_enhanced_price_history_cached(coin_id, days=365):
    """Fetch comprehensive price and volume data with caching"""
    cache_file = os.path.join(CACHE_DIR, f"{coin_id}_{days}d.pkl")
    
    # Check if we have cached data (valid for 6 hours)
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=6):
            print(f"   📋 Using cached data for {coin_id}...")
            try:
                return joblib.load(cache_file)
            except:
                print(f"   ⚠️ Cache corrupted for {coin_id}, fetching fresh data...")
    
    # Fetch fresh data
    url = f"{API_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    try:
        print(f"   📥 Fetching fresh data for {coin_id}...")
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if 'prices' not in data or 'total_volumes' not in data:
            raise ValueError("Required data missing")
        
        # Extract price and volume data
        prices = np.array([item[1] for item in data['prices']])
        volumes = np.array([item[1] for item in data['total_volumes']])
        
        # Remove data points with zero prices or volumes
        valid_indices = (prices > 0) & (volumes > 0)
        prices = prices[valid_indices]
        volumes = volumes[valid_indices]
        
        if len(prices) < 50:
            raise ValueError("Insufficient valid data points")
        
        # Calculate price-based features
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Log returns (more stable)
        log_returns = np.zeros_like(prices)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])
        
        # Rolling volatility (14-day window)
        volatility = np.zeros_like(prices)
        window = 14
        for i in range(window, len(prices)):
            volatility[i] = np.std(returns[i-window:i])
        
        # Volume-based features
        volume_sma = np.convolve(volumes, np.ones(10)/10, mode='same')
        volume_ratio = volumes / (volume_sma + 1e-8)
        
        # Price momentum (5-day and 10-day)
        momentum_5 = np.zeros_like(prices)
        momentum_10 = np.zeros_like(prices)
        momentum_5[5:] = (prices[5:] - prices[:-5]) / prices[:-5]
        momentum_10[10:] = (prices[10:] - prices[:-10]) / prices[:-10]
        
        # Technical indicators
        sma_5, sma_10, ema_12, rsi, macd, bb_ratio = calculate_technical_indicators(prices, volumes)
        
        # Price relative to moving averages
        price_to_sma5 = prices / (sma_5 + 1e-8)
        price_to_sma10 = prices / (sma_10 + 1e-8)
        
        # Combine all features (15 features total)
        features = np.column_stack([
            prices,              # 0: Raw price
            volumes,             # 1: Volume
            returns,             # 2: Price returns
            log_returns,         # 3: Log returns
            volatility,          # 4: Volatility
            volume_ratio,        # 5: Volume ratio
            momentum_5,          # 6: 5-day momentum
            momentum_10,         # 7: 10-day momentum
            rsi,                # 8: RSI
            macd,               # 9: MACD
            bb_ratio,           # 10: Bollinger Band ratio
            price_to_sma5,      # 11: Price to SMA5 ratio
            price_to_sma10,     # 12: Price to SMA10 ratio
            sma_5,              # 13: SMA5
            ema_12              # 14: EMA12
        ])
        
        # Remove any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Cache the result
        joblib.dump(features, cache_file)
        
        return features
        
    except Exception as e:
        print(f"   ❌ Error fetching data for {coin_id}: {e}")
        return None

def parallel_fetch_with_retry(coin_ids, max_retries=3, delay_between_batches=45):
    """Fetch data for multiple coins with intelligent batching and retry logic"""
    successful_data = {}
    failed_coins = []
    
    # Process coins in batches of 3 to respect rate limits
    batch_size = 3
    total_batches = (len(coin_ids) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(coin_ids))
        batch_coins = coin_ids[start_idx:end_idx]
        
        print(f"\n📦 Processing batch {batch_idx + 1}/{total_batches}: {batch_coins}")
        
        for coin in batch_coins:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    features = fetch_enhanced_price_history_cached(coin, days=365)
                    if features is not None:
                        successful_data[coin] = features
                        print(f"   ✅ Successfully fetched data for {coin}")
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"   ⚠️ Retry {retry_count}/{max_retries} for {coin}")
                            time.sleep(2)
                except Exception as e:
                    retry_count += 1
                    print(f"   ❌ Error fetching {coin} (attempt {retry_count}): {e}")
                    if retry_count < max_retries:
                        time.sleep(2)
            
            if coin not in successful_data:
                failed_coins.append(coin)
                print(f"   ❌ Failed to fetch data for {coin} after {max_retries} attempts")
            
            # Rate limiting between coins
            time.sleep(2)
        
        # Longer delay between batches
        if batch_idx < total_batches - 1:
            print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    return successful_data, failed_coins

def create_enhanced_dataset(features, look_back=20, prediction_threshold=0.01):
    """Create dataset with balanced labeling strategy"""
    X, y = [], []
    
    # Calculate all price changes first
    price_changes = []
    for i in range(look_back, len(features) - 1):
        current_price = features[i, 0]
        next_price = features[i+1, 0]
        price_change = (next_price - current_price) / current_price
        price_changes.append(price_change)
    
    # Use percentile-based thresholds for more balanced classes
    price_changes = np.array(price_changes)
    upper_threshold = np.percentile(price_changes, 60)  # Top 40% as UP
    lower_threshold = np.percentile(price_changes, 40)  # Bottom 40% as DOWN
    
    # Ensure minimum threshold
    upper_threshold = max(upper_threshold, prediction_threshold)
    lower_threshold = min(lower_threshold, -prediction_threshold)
    
    print(f"   📊 Thresholds - Up: {upper_threshold:.4f}, Down: {lower_threshold:.4f}")
    
    for i, price_change in enumerate(price_changes):
        # Use features from original index
        actual_idx = i + look_back
        X.append(features[actual_idx-look_back:actual_idx])
        
        # Create balanced labels
        if price_change > upper_threshold:
            y.append(1)  # Up movement
        else:
            y.append(0)  # Down or neutral movement
    
    return np.array(X), np.array(y)

def create_advanced_lstm_model(input_shape):
    """Create advanced bidirectional LSTM model with better architecture"""
    model = Sequential([
        # First bidirectional LSTM layer
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1), 
                     input_shape=input_shape),
        BatchNormalization(),
        
        # Second bidirectional LSTM layer
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)),
        BatchNormalization(),
        
        # Final LSTM layer
        LSTM(16, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),
        BatchNormalization(),
        
        # Dense layers with moderate regularization
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    # Optimizer with moderate learning rate
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def safe_f1_score(precision, recall):
    """Calculate F1-score with zero division protection"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def train_enhanced_model(X, y):
    """Train model with advanced techniques and better error handling"""
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"📊 Original class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Ensure we have both classes
    if len(unique_classes) < 2:
        print("❌ Error: Only one class present in data. Cannot train binary classifier.")
        return None, None
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"📊 Class weights: {class_weight_dict}")
    
    # Split data with stratification
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    except ValueError as e:
        print(f"❌ Error in stratified split: {e}")
        # Fall back to regular split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"📊 Train samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"📊 Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"📊 Val class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"📊 Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Create model
    model = create_advanced_lstm_model((X.shape[1], X.shape[2]))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
        mode='min'
    )
    
    # Train model
    print("🚀 Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Safe F1-score calculation
    f1 = safe_f1_score(test_precision, test_recall)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Get predictions for detailed analysis
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Alternative F1-score calculation using sklearn
    sklearn_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Sklearn F1-Score: {sklearn_f1:.4f}")
    
    # Classification report
    print("\n📊 Classification Report:")
    try:
        print(classification_report(y_test, y_pred, target_names=['Down/Neutral', 'Up'], zero_division=0))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Confusion matrix
    print("\n📊 Confusion Matrix:")
    try:
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"   [[TN: {cm[0,0]}, FP: {cm[0,1]}],")
        print(f"    [FN: {cm[1,0]}, TP: {cm[1,1]}]]")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    # Prediction distribution
    print(f"\n📊 Prediction distribution:")
    print(f"   Predicted probabilities - Mean: {np.mean(y_pred_prob):.4f}, Std: {np.std(y_pred_prob):.4f}")
    print(f"   Predicted classes: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    
    return model, history

def main():
    """Enhanced main training pipeline with optimized API usage"""
    print("🚀 Starting Advanced LSTM Crypto Price Predictor Training")
    print("🔧 Using optimized API fetching with caching and batching")
    print("="*70)
    
    # Fetch top coins
    coin_ids = fetch_top_10_coins()
    if not coin_ids:
        print("❌ Failed to fetch coin list. Exiting.")
        return
    
    print(f"📋 Training on coins: {', '.join(coin_ids)}")
    
    # Use optimized parallel fetching
    print(f"\n🔄 Starting optimized data fetching...")
    successful_data, failed_coins = parallel_fetch_with_retry(coin_ids)
    
    if failed_coins:
        print(f"⚠️ Failed to fetch data for: {', '.join(failed_coins)}")
    
    if not successful_data:
        print("❌ No data fetched successfully. Exiting.")
        return
    
    print(f"✅ Successfully fetched data for {len(successful_data)} coins")
    
    all_X, all_y = [], []
    coin_scalers = {}
    successful_coins = []
    
    # Process each coin's data
    for coin, features in successful_data.items():
        print(f"\n📊 Processing {coin.upper()}...")
        
        try:
            print(f"   📊 Raw data shape: {features.shape}")
            
            # Use RobustScaler
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Create dataset with balanced labeling
            X, y = create_enhanced_dataset(scaled_features, look_back=20, prediction_threshold=0.01)
            
            if len(X) < 50:
                print(f"   ⚠️ Insufficient samples for {coin}: {len(X)}")
                continue
            
            # Check class distribution for this coin
            unique_classes, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            print(f"   ✅ Dataset shape: {X.shape}, Class distribution: {class_dist}")
            
            # Only add if we have both classes
            if len(unique_classes) >= 2:
                all_X.append(X)
                all_y.append(y)
                coin_scalers[coin] = scaler
                successful_coins.append(coin)
            else:
                print(f"   ⚠️ Skipping {coin} - only one class present")
                
        except Exception as e:
            print(f"   ❌ Error processing {coin}: {e}")
            continue
    
    # Check if we have enough data
    if not all_X:
        print("❌ No training data collected. Exiting.")
        return
    
    print(f"\n✅ Successfully processed {len(successful_coins)} coins: {', '.join(successful_coins)}")
    
    # Combine all data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"📊 Final dataset shape: {X_combined.shape}")
    final_class_dist = dict(zip(*np.unique(y_combined, return_counts=True)))
    print(f"📊 Final class distribution: {final_class_dist}")
    
    # Check class balance
    if len(final_class_dist) < 2:
        print("❌ Error: Final dataset has only one class. Cannot proceed.")
        return
    
    class_counts = np.array(list(final_class_dist.values()))
    class_ratio = min(class_counts) / max(class_counts)
    print(f"📊 Class balance ratio: {class_ratio:.3f}")
    
    # Train model
    print(f"\n🎯 Training Advanced LSTM Model...")
    print("="*50)
    
    result = train_enhanced_model(X_combined, y_combined)
    if result[0] is None:
        print("❌ Training failed. Exiting.")
        return
    
    model, history = result
    
    # Save model and scalers
    print(f"\n💾 Saving model and scalers...")
    model.save("advanced_lstm_price_predictor.h5")
    joblib.dump(coin_scalers, "lstm_scalers.pkl")
    
    # Save training history
    if history is not None:
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }
        joblib.dump(history_dict, "training_history.pkl")
    
    print("✅ Files saved:")
    print("   - advanced_lstm_price_predictor.h5 (model)")
    print("   - lstm_scalers.pkl (scalers for each coin)")
    print("   - training_history.pkl (training metrics)")
    print(f"   - {CACHE_DIR}/ (cached data for future runs)")
    
    # Model summary
    print(f"\n📋 Model Summary:")
    print(f"   - Architecture: Bidirectional LSTM with regularization")
    print(f"   - Features: 15 (price, volume, technical indicators)")
    print(f"   - Lookback window: 20 days")
    print(f"   - Total parameters: {model.count_params():,}")
    print(f"   - Training samples: {len(X_combined):,}")
    print(f"   - Successful coins: {len(successful_coins)}")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"💡 Next runs will be faster due to cached data!")

if __name__ == "__main__":
    main()