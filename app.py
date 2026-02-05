import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from twelvedata import TDClient
import google.generativeai as genai
import json

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SYMBOL = "XAU/USD"
INTERVAL = "5min"
CANDLE_LIMIT = 300

EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14

MODE = st.sidebar.selectbox("Mode", ["PROP / FUNDED", "GROWTH"])

TD_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ─────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────
td = TDClient(apikey=TD_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────
# HARD RESET (NUKE CACHE + STATE)
# ─────────────────────────────────────────────────────────────
if st.sidebar.button("🔄 FORCE FULL RESET"):
    st.cache_data.clear()
    st.session_state.clear()
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# MARKET DATA (TTL CACHE ONLY)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_market_data():
    ts = td.time_series(
        symbol=SYMBOL,
        interval=INTERVAL,
        outputsize=CANDLE_LIMIT
    ).as_pandas()

    ts = ts.sort_index()
    ts = ts.astype(float)
    ts["timestamp"] = ts.index
    return ts

df = fetch_market_data()

# ─────────────────────────────────────────────────────────────
# INDICATORS (SINGLE SOURCE OF TRUTH)
# ─────────────────────────────────────────────────────────────
df["ema_50"] = df["close"].ewm(span=EMA_FAST).mean()
df["ema_200"] = df["close"].ewm(span=EMA_SLOW).mean()

high = df["high"]
low = df["low"]
close = df["close"]

tr = pd.concat([
    high - low,
    (high - close.shift()).abs(),
    (low - close.shift()).abs()
], axis=1).max(axis=1)

df["atr"] = tr.rolling(ATR_PERIOD).mean()

latest = df.iloc[-1]

# ─────────────────────────────────────────────────────────────
# FROZEN MARKET SNAPSHOT
# ─────────────────────────────────────────────────────────────
market_snapshot = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "price": round(float(latest.close), 2),
    "ema_50": round(float(latest.ema_50), 2),
    "ema_200": round(float(latest.ema_200), 2),
    "atr": round(float(latest.atr), 2),
    "trend": (
        "BULLISH" if latest.ema_50 > latest.ema_200 else
        "BEARISH" if latest.ema
