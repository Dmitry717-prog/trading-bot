import streamlit as st
import pandas as pd
import requests
import ta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Super ML Bot", layout="wide")

st.title("🚀 Super ML Trading Bot")

# === STATE ===
if "balance" not in st.session_state:
    st.session_state.balance = 1000
    st.session_state.history = []
    st.session_state.model = None

# === SETTINGS ===
SYMBOL = st.sidebar.selectbox("Пара", ["BTCUSDT", "ETHUSDT"])
BET = st.sidebar.number_input("Ставка", value=10)

# === DATA ===
def get_data(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=300"
    data = requests.get(url).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "ct","qv","n","tbb","tbq","ignore"
    ])

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    return df

# === FEATURES ===
def prepare(df):
    df["ema"] = ta.trend.ema_indicator(df["close"], window=10)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    df = df.dropna()

    X = df[["ema","rsi","atr"]]
    y = df["target"]

    return X, y

# === TRAIN ===
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# === LOAD ===
df = get_data(SYMBOL)
X, y = prepare(df)

if st.button("🧠 Обучить AI"):
    model, acc = train(X, y)
    st.session_state.model = model
    st.success(f"Точность: {round(acc*100,2)}%")

# === PREDICT ===
if st.session_state.model:
    last = X.iloc[-1:]
    pred = st.session_state.model.predict(last)[0]
    prob = st.session_state.model.predict_proba(last)[0][pred]

    direction = "BUY" if pred else "SELL"
    confidence = int(prob * 100)
else:
    direction = "-"
    confidence = 0

price = df["close"].iloc[-1]

# === UI ===
col1, col2, col3 = st.columns(3)
col1.metric("Баланс", f"{st.session_state.balance}$")
col2.metric("Сигнал", direction)
col3.metric("Уверенность", f"{confidence}%")

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])
st.plotly_chart(fig, use_container_width=True)
