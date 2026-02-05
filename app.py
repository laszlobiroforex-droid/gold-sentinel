import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timezone
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import threading
import schedule
import json
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SPREAD_BUFFER_POINTS = 0.30
SLIPPAGE_BUFFER_POINTS = 0.20
COMMISSION_PER_LOT_RT = 1.00
PIP_VALUE = 100

ALERT_COOLDOWN_MIN = 30          # don't spam same setup
MIN_CONVICTION_FOR_ALERT = 2     # how many AIs must say HIGH/ELITE

# ─── RETRY HELPER ────────────────────────────────────────────────────────────
def retry_api(max_attempts=3, backoff=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(backoff * (attempt + 1))
            return None
        return wrapper
    return decorator

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    grok_client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

except Exception as e:
    st.error(f"API setup failed: {e}\nCheck secrets.")
    st.stop()

# ─── TELEGRAM SENDER ─────────────────────────────────────────────────────────
def send_telegram(message: str, priority: str = "normal"):
    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.warning("Telegram not configured — add TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID to secrets")
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code != 200:
            st.error(f"Telegram send failed: {resp.text}")
    except Exception as e:
        st.warning(f"Telegram error: {e}")

# ─── FRACTAL LEVELS ──────────────────────────────────────────────────────────
def get_fractal_levels(df, window=5, min_dist_factor=0.4):
    levels = []
    atr_approx = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            price = round(df['high'].iloc[i], 2)
            if abs(price - df['close'].iloc[-1]) > min_dist_factor * atr_approx:
                levels.append(('RES', price))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            price = round(df['low'].iloc[i], 2)
            if abs(price - df['close'].iloc[-1]) > min_dist_factor * atr_approx:
                levels.append(('SUP', price))
    return sorted(levels, key=lambda x: x[1], reverse=True)

# ─── CACHED DATA ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
@retry_api()
def fetch_15m_data():
    return td.time_series(**{
        "symbol": "XAU/USD",
        "interval": "15min",
        "outputsize": 500
    }).with_rsi({}).with_ema({"time_period": 200}).with_ema({"time_period": 50}).with_atr({"time_period": 14}).as_pandas()

@st.cache_data(ttl=300)
@retry_api()
def fetch_htf_data(interval):
    out_size = 500 if interval == "5min" else 200
    return td.time_series(**{
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": out_size
    }).with_ema({"time_period": 200}).as_pandas()

@st.cache_data(ttl=60)
@retry_api()
def get_current_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

# ─── AI AUDIT (same as before) ───────────────────────────────────────────────
@retry_api()
def get_ai_advice(market, setup, levels, buffer, mode):
    # ... (keep your original get_ai_advice function here unchanged)
    # For brevity I'm not pasting the whole prompt again — copy it from your current script
    pass  # ← replace with your full function body

# ─── PARSE AI OUTPUT (same) ──────────────────────────────────────────────────
def parse_ai_output(text):
    # ... (keep your original parse_ai_output unchanged)
    pass  # ← replace with your full function

# ─── CORE ALERT CHECK FUNCTION ───────────────────────────────────────────────
def check_for_high_conviction_setup():
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0
    if "last_alerted_entry_sl" not in st.session_state:
        st.session_state.last_alerted_entry_sl = None

    now = time.time()
    if now - st.session_state.last_alert_time < ALERT_COOLDOWN_MIN * 60:
        return  # too soon

    try:
        live_price = get_current_price()
        ts_15m = fetch_15m_data()
        mode = st.session_state.get("mode", "Standard")
        interval = "1h" if mode.startswith("Standard") else "5min"
        ts_htf = fetch_htf_data(interval)

        latest_15m = ts_15m.iloc[-1]
        rsi = latest_15m.get('rsi', 50.0)
        atr = latest_15m.get('atr', 10.0)

        # ... (keep the rest of your market analysis logic here: ema, bias, levels, original setup, etc.)
        # For space reasons, assume you copy-paste your full analysis block up to getting g_parsed, k_parsed, c_parsed

        # Example placeholder — replace with your actual variables
        parsed_verdicts = [g_parsed["verdict"], k_parsed["verdict"], c_parsed["verdict"]]  # ← from your code
        high_count = sum(1 for v in parsed_verdicts if v in ["ELITE", "HIGH_CONV"])

        if high_count >= MIN_CONVICTION_FOR_ALERT:
            # Build nice message
            direction = g_parsed.get("direction") or "UNKNOWN"
            entry = g_parsed.get("entry") or final_entry  # use your final vars
            sl = g_parsed.get("sl") or final_sl
            tp = g_parsed.get("tp") or final_tp
            rr = g_parsed.get("rr") or actual_rr

            key = f"{direction} {entry:.2f}/{sl:.2f}"
            if st.session_state.last_alerted_entry_sl == key:
                return  # duplicate

            msg = (
                f"**High Conviction Setup Detected!** ({high_count}/3)\n"
                f"Direction: {direction}\n"
                f"Entry: ${entry:.2f}\n"
                f"SL: ${sl:.2f}\n"
                f"TP: ${tp:.2f}\n"
                f"R:R ≈ 1:{rr:.1f}\n"
                f"Current: ${live_price:.2f} | RSI {rsi:.1f}"
            )

            priority = "high" if high_count == 3 else "normal"
            send_telegram(msg, priority)

            st.session_state.last_alert_time = now
            st.session_state.last_alerted_entry_sl = key

            # Optional: try to persist to file (helps across restarts)
            try:
                with open("last_alert.json", "w") as f:
                    json.dump({"entry_sl": key, "time": now}, f)
            except:
                pass

    except Exception as e:
        st.error(f"Alert check failed: {e}")

# ─── BACKGROUND SCHEDULER ────────────────────────────────────────────────────
def run_background_checker():
    schedule.every(5).minutes.do(check_for_high_conviction_setup)
    while True:
        schedule.run_pending()
        time.sleep(1)

if "checker_started" not in st.session_state:
    st.session_state.checker_started = True
    thread = threading.Thread(target=run_background_checker, daemon=True)
    thread.start()

# ─── YOUR EXISTING STREAMLIT UI CODE BELOW ───────────────────────────────────
# (keep everything from st.set_page_config(...) down to the end unchanged)
# Just make sure to call check_for_high_conviction_setup() also when user clicks "Analyze" if you want instant check

st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
# ... rest of your UI ...
