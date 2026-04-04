import streamlit as st
import pandas as pd
import requests
import ta
import os
import time
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

# === STATE ===
if "last_signal_time" not in st.session_state:
    st.session_state.last_signal_time = 0

# === DATA ===
def get_data(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=300"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            return pd.DataFrame()

        df = pd.DataFrame(res.json(), columns=[
            "time","open","high","low","close","volume",
            "ct","qv","n","tbb","tbq","ignore"
        ])

        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)

        return df
    except:
        return pd.DataFrame()

# === PREPARE ===
def prepare(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], 9)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], 21)
    df["rsi"] = ta.momentum.rsi(df["close"], 14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna()

    return df

# === MODEL ===
def train(df):
    X = df[["ema_fast","ema_slow","rsi","atr"]]
    y = df["target"]

    model = XGBClassifier(n_estimators=120)
    model.fit(X, y)

    return model

# === FILTERS ===
def market_filter(df):
    # тренд
    trend = df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]

    # волатильность
    atr = df["atr"].iloc[-1]
    low_volatility = atr < df["atr"].mean()

    if low_volatility:
        return False

    return True

# === MAIN ===
st.title("🚀 PRO Signal Bot")

SYMBOL = st.selectbox("Пара", ["BTCUSDT","ETHUSDT"])

df = get_data(SYMBOL)

if df.empty:
    st.stop()

df = prepare(df)

model = train(df)

last = df.iloc[-1:]
X_last = last[["ema_fast","ema_slow","rsi","atr"]]

pred = model.predict(X_last)[0]
prob = model.predict_proba(X_last)[0][pred]

direction = "BUY" if pred else "SELL"
confidence = int(prob * 100)
price = df["close"].iloc[-1]

st.metric("Сигнал", direction)
st.metric("Уверенность", f"{confidence}%")

# === SMART SIGNAL SYSTEM ===
current_time = time.time()

if (
    confidence >= 75 and
    market_filter(df) and
    current_time - st.session_state.last_signal_time > 300
):
    msg = f"""
🚨 PRO SIGNAL

📊 {direction}
💰 Цена: {price}
🧠 Уверенность: {confidence}%
📈 {SYMBOL}

⏱ Время: {time.strftime('%H:%M:%S')}
    """

    send_signal(msg)
    st.session_state.last_signal_time = current_time
    st.success("Сильный сигнал отправлен 🚀")

else:
    st.info("Нет подходящего сигнала")
