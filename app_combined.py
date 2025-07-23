
from flask import Flask, request, jsonify, send_from_directory
import json, os, requests
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import joblib
import time
from functools import lru_cache
import random
from train_lstm_multi import fetch_enhanced_price_history_cached
from tensorflow.keras.initializers import Orthogonal


@lru_cache(maxsize=50)
def fetch_coin_data_cached(coin_id):
    print(f"📦 Fetching live data for: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"

    for attempt in range(3):  # max 3 retries
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 429:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"⏳ Rate limited. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            print(f"❌ Error fetching {coin_id} (attempt {attempt+1}):", e)
            time.sleep(2)
    return None



model = load_model("advanced_lstm_price_predictor.h5", custom_objects={'Orthogonal': Orthogonal})
scalers = joblib.load("lstm_scalers.pkl")

def extract_15_features(coin_data):
    try:
        mkt = coin_data['market_data']
        price = mkt['current_price']['usd']
        volume = mkt['total_volume']['usd']
        ath = mkt['ath']['usd']
        sma5 = price * 0.97  # dummy placeholder
        sma10 = price * 0.96
        ema12 = price * 0.98

        # Derived features
        returns = 0.02  # placeholder
        log_returns = 0.019
        volatility = abs(mkt['price_change_percentage_24h']) / 10
        volume_ratio = 1.2  # placeholder
        momentum_5 = 0.01
        momentum_10 = 0.008
        rsi = 55
        macd = 1.1
        bb_ratio = 0.6

        price_to_sma5 = price / (sma5 + 1e-8)
        price_to_sma10 = price / (sma10 + 1e-8)

        # Match training feature order:
        features = np.array([[price, volume, returns, log_returns, volatility,
                              volume_ratio, momentum_5, momentum_10, rsi, macd,
                              bb_ratio, price_to_sma5, price_to_sma10, sma5, ema12]])

        return features
    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None


app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, origins=["*"])  # Allow all origins for development

WALLET_FILE = 'wallet.json'
DEFAULT_BALANCE = 10000

@app.route('/')
def serve_html():
    return send_from_directory('.', 'cc_with_prediction_tab.html') 

def load_wallet():
    if not os.path.exists(WALLET_FILE):
        return {"balance": DEFAULT_BALANCE, "holdings": {}}
    with open(WALLET_FILE, 'r') as f:
        return json.load(f)


def save_wallet(data):
    with open(WALLET_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_current_price(coin):
    url = f'https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd'
    res = requests.get(url)
    data = res.json()
    return data.get(coin, {}).get('usd', None)


@app.route('/wallet', methods=['GET'])
def get_wallet():
    return jsonify(load_wallet())
@app.route('/fetch-and-predict', methods=['POST'])
def fetch_and_predict():
    data = request.get_json()
    coin_id = data.get("id")
    if not coin_id:
        return jsonify({"error": "Coin ID is required"}), 400

    coin_data = fetch_coin_data_cached(coin_id)
    if not coin_data:
        return jsonify({"error": "Failed to fetch coin data."}), 500

    features = extract_15_features(coin_data)
    if features is None:
        return jsonify({"error": "Failed to extract features."}), 500

    # Get scaler for the coin
    scaler = scalers.get(coin_id)
    if scaler is None:
        return jsonify({"error": f"No scaler found for coin '{coin_id}'"}), 400

    try:
        scaled = scaler.transform(features)
        input_scaled = scaled.reshape((1, 1, scaled.shape[1]))
        pred = model.predict(input_scaled)[0][0]

        label = "up" if pred >= 0.5 else "down"
        conf = float(pred) if pred >= 0.5 else 1 - float(pred)

        return jsonify({"prediction": label, "confidence": round(conf, 2)})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/buy', methods=['POST'])
def buy():
    data = request.json
    coin = data.get('coin')
    qty = float(data.get('qty', 0))
    if not coin or qty <= 0:
        return jsonify({"error": "Invalid coin or quantity"}), 400

    price = get_current_price(coin)
    if price is None:
        return jsonify({"error": f"Unable to fetch price for {coin}"}), 400

    wallet = load_wallet()
    cost = qty * price
    if cost > wallet['balance']:
        return jsonify({"error": "Insufficient balance"}), 400

    wallet['balance'] -= cost

    if coin not in wallet['holdings']:
        wallet['holdings'][coin] = {"qty": 0, "buyPrice": 0}

    old_qty = wallet['holdings'][coin]['qty']
    old_price = wallet['holdings'][coin]['buyPrice']
    new_qty = old_qty + qty
    weighted_price = ((old_qty * old_price) + (qty * price)) / new_qty

    wallet['holdings'][coin]['qty'] = new_qty
    wallet['holdings'][coin]['buyPrice'] = weighted_price

    save_wallet(wallet)
    return jsonify(wallet)


@app.route('/sell', methods=['POST'])
def sell():
    data = request.json
    coin = data.get('coin')
    qty = float(data.get('qty', 0))
    if not coin or qty <= 0:
        return jsonify({"error": "Invalid coin or quantity"}), 400

    wallet = load_wallet()
    if coin not in wallet['holdings'] or wallet['holdings'][coin]['qty'] < qty:
        return jsonify({"error": "Insufficient holdings"}), 400

    price = get_current_price(coin)
    if price is None:
        return jsonify({"error": f"Unable to fetch price for {coin}"}), 400

    gain = qty * price
    wallet['balance'] += gain
    wallet['holdings'][coin]['qty'] -= qty

    if wallet['holdings'][coin]['qty'] <= 0:
        del wallet['holdings'][coin]

    save_wallet(wallet)
    return jsonify(wallet)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    coin_id = data.get("id", "bitcoin")

    try:
        # ✅ Load full feature history (like in training)
        from train_lstm_multi import fetch_enhanced_price_history_cached  # Reuse from training
        features = fetch_enhanced_price_history_cached(coin_id, days=365)
        if features is None or features.shape[0] < 40:
            return jsonify({"error": "Not enough historical data"}), 400

        # ✅ Extract the last 20 timesteps
        latest_sequence = features[-20:]  # Shape: (20, 15)

        # ✅ Load scaler and apply it
        scaler = scalers.get(coin_id)
        if scaler is None:
            return jsonify({"error": f"No scaler found for coin '{coin_id}'"}), 400

        scaled_seq = scaler.transform(latest_sequence)  # shape: (20, 15)
        input_seq = scaled_seq.reshape((1, 20, 15))  # model expects 3D input

        # ✅ Run prediction
        pred = model.predict(input_seq)[0][0]
        label = "up" if pred >= 0.5 else "down"
        confidence = float(pred) if pred >= 0.5 else 1 - float(pred)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Allow external connections


