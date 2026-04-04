import streamlit as st
import pandas as pd
import requests
import ta
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# === TELEGRAM ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_signal(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})
    except:
        pass

# === DATA ===
def get_data(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=300"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            return pd.DataFrame()

        data = res.json()

        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "ct","qv","n","tbb","tbq","ignore"
        ])

        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)

        return df
    except:
        return pd.DataFrame()

# === ML ===
def prepare(df):
    if df.empty or len(df) < 50:
        return None, None

    df["ema"] = ta.trend.ema_indicator(df["close"], window=10)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna()

    if df.empty:
        return None, None

    return df[["ema","rsi","atr"]], df["target"]

def train(X, y):
    model = XGBClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# === MAIN ===
st.title("🤖 Signal Bot")

SYMBOL = st.selectbox("Пара", ["BTCUSDT", "ETHUSDT"])

df = get_data(SYMBOL)

if df.empty:
    st.error("Нет данных")
    st.stop()

X, y = prepare(df)

if X is None:
    st.warning("Недостаточно данных")
    st.stop()

model = train(X, y)

last = X.iloc[-1:]
pred = model.predict(last)[0]
prob = model.predict_proba(last)[0][pred]

direction = "BUY" if pred else "SELL"
confidence = int(prob * 100)
price = df["close"].iloc[-1]

st.metric("Сигнал", direction)
st.metric("Уверенность", f"{confidence}%")

# === СИГНАЛ ===
if confidence >= 75:
    msg = f"""
🚨 СИЛЬНЫЙ СИГНАЛ

📊 {direction}
💰 Цена: {price}
🧠 Уверенность: {confidence}%
📈 {SYMBOL}
    """
    send_signal(msg)
    st.success("Сигнал отправлен в Telegram")

else:
    st.info("Сигнал слабый — пропуск")
