
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

API_BASE = "https://api.coingecko.com/api/v3"

def fetch_top_coins(limit=20):
    url = f"{API_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1"
    res = requests.get(url , timeout=15)
    return [coin['id'] for coin in res.json()]

def fetch_market_chart(coin_id):
    url = f"{API_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days=90&interval=daily"
    res = requests.get(url)
    data = res.json()
    if 'prices' not in data or 'total_volumes' not in data:
        raise ValueError("Missing price or volume data")
    prices = data['prices']
    volumes = data['total_volumes']
    return prices, volumes

def build_dataset():
    coins = fetch_top_coins()
    all_data = []

    for coin in coins:
        try:
            prices, volumes = fetch_market_chart(coin)
            prices = pd.DataFrame(prices, columns=["timestamp", "price"])
            volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            df = pd.merge(prices, volumes, on="timestamp")
            df["price"] = df["price"].astype(float)
            df["volume"] = df["volume"].astype(float)

            df["3_day_ma"] = df["price"].rolling(window=3).mean()
            df["volatility"] = df["price"].rolling(window=3).std()
            df["rel_volume"] = df["volume"] / df["volume"].rolling(window=3).mean()

            for i in range(len(df) - 3):
                row = df.iloc[i]
                row_next = df.iloc[i+1]
                row_future = df.iloc[i+2]

                if pd.isna(row["3_day_ma"]) or pd.isna(row["volatility"]) or pd.isna(row["rel_volume"]):
                    continue

                pct_change = (row_next["price"] - row["price"]) / row["price"] * 100
                ath_deviation = (row["price"] - df["price"].max()) / df["price"].max() * 100
                label = int(row_future["price"] > row_next["price"])

                all_data.append({
                    "coin": coin,
                    "price": row["price"],
                    "24h_pct_change": pct_change,
                    "3_day_ma": row["3_day_ma"],
                    "volatility": row["volatility"],
                    "rel_volume": row["rel_volume"],
                    "ath_deviation": ath_deviation,
                    "label": label
                })
            time.sleep(1.2)
        except Exception as e:
            print(f"Skipping {coin}: {e}")
    return pd.DataFrame(all_data)

def train_and_save_model(df):
    features = ["price", "24h_pct_change", "3_day_ma", "volatility", "rel_volume", "ath_deviation"]
    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, "price_predictor.pkl")
    print("✅ Model saved as price_predictor.pkl")

if __name__ == "__main__":
    df = build_dataset()
    print("✅ Dataset size:", len(df))
    if not df.empty:
        train_and_save_model(df)
    else:
        print("⚠️ No data collected.")
