import streamlit as st
import pandas as pd
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
# HARD RESET
# ─────────────────────────────────────────────────────────────
if st.sidebar.button("🔄 FORCE FULL RESET"):
    st.cache_data.clear()
    st.session_state.clear()
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_market_data():
    df = td.time_series(
        symbol=SYMBOL,
        interval=INTERVAL,
        outputsize=CANDLE_LIMIT
    ).as_pandas()

    df = df.sort_index()
    df = df.astype(float)
    return df

df = fetch_market_data()

# ─────────────────────────────────────────────────────────────
# INDICATORS
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
# TREND (NO TERNARY, NO PARENS)
# ─────────────────────────────────────────────────────────────
if latest.ema_50 > latest.ema_200:
    trend = "BULLISH"
elif latest.ema_50 < latest.ema_200:
    trend = "BEARISH"
else:
    trend = "RANGE"

# ─────────────────────────────────────────────────────────────
# MARKET SNAPSHOT
# ─────────────────────────────────────────────────────────────
market_snapshot = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "price": round(float(latest.close), 2),
    "ema_50": round(float(latest.ema_50), 2),
    "ema_200": round(float(latest.ema_200), 2),
    "atr": round(float(latest.atr), 2),
    "trend": trend
}

st.subheader("📊 Market Snapshot")
st.json(market_snapshot)

# ─────────────────────────────────────────────────────────────
# AI (DETERMINISTIC)
# ─────────────────────────────────────────────────────────────
def run_ai(name, role):
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-pro",
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1
        }
    )

    prompt = f"""
You are {name}, acting strictly as a {role}.

You are given a frozen market snapshot.
You MUST NOT assume future candles.
You MUST NOT invent data.
You MUST return VALID JSON ONLY.

SNAPSHOT:
{json.dumps(market_snapshot, indent=2)}

Return EXACTLY this schema:

{{
  "bias": "LONG | SHORT | NO_TRADE",
  "confidence": 0-100,
  "reason": "one concise sentence"
}}
"""

    response = model.generate_content(prompt)
    return json.loads(response.text)

# ─────────────────────────────────────────────────────────────
# MULTI-AI
# ─────────────────────────────────────────────────────────────
ai_outputs = {
    "Structure_AI": run_ai("Structure AI", "market structure analyst"),
    "Trend_AI": run_ai("Trend AI", "EMA trend analyst"),
    "Risk_AI": run_ai("Risk AI", "risk control analyst")
}

st.subheader("🤖 AI Outputs")
st.json(ai_outputs)

# ─────────────────────────────────────────────────────────────
# CONSENSUS
# ─────────────────────────────────────────────────────────────
bias_votes = [v["bias"] for v in ai_outputs.values()]
final_bias = max(set(bias_votes), key=bias_votes.count)

st.subheader("✅ Final Bias")
st.success(final_bias)

# ─────────────────────────────────────────────────────────────
# MODE INFO
# ─────────────────────────────────────────────────────────────
if MODE == "PROP / FUNDED":
    st.info("Prop / Funded mode active: conservative risk, trailing SL enforced.")
else:
    st.info("Growth mode active: relaxed trailing, scaling allowed.")
