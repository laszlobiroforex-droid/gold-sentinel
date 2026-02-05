import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import json
from datetime import datetime, timezone, timedelta
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import os
from bs4 import BeautifulSoup

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SPREAD_BUFFER_POINTS = 0.30
SLIPPAGE_BUFFER_POINTS = 0.20
PIP_VALUE = 100

ALERT_COOLDOWN_MIN = 30
MIN_CONVICTION_FOR_ALERT = 2
MAX_SL_ATR_MULT = 2.2
OPPOSING_FRACTAL_ATR_MULT = 0.8

DEFAULT_BALANCE = 5000.0
DEFAULT_DAILY_LIMIT = 250.0
DEFAULT_FLOOR = 4500.0
DEFAULT_RISK_PCT = 25

LAST_ALERT_FILE = "last_alert.json"

# ─── SAFE TD CALL (FAIL FAST ON LIMITS) ───────────────────────────────────────
def safe_td(fn):
    try:
        return fn()
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ["429", "quota", "limit", "credit", "rate"]):
            st.warning("Twelve Data rate limit hit — skipping this cycle.")
            return None
        raise

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    grok_client = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

except Exception as e:
    st.error(f"API init failed: {e}")
    st.stop()

# ─── SINGLE PRICE SOURCE (CACHED) ────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_live_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

# ─── MARKET DATA (CACHED) ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_15m_data():
    ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200)
    ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
    return ts.as_pandas()

@st.cache_data(ttl=300)
def fetch_1h_data():
    ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=100)
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

@st.cache_data(ttl=300)
def fetch_5m_data():
    ts = td.time_series(symbol="XAU/USD", interval="5min", outputsize=200)
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

# ─── FRACTALS & INVALIDATION ─────────────────────────────────────────────────
def get_fractal_levels(df, window=5, min_dist_factor=0.4):
    if df.empty:
        return []
    levels = []
    atr = df.get("atr", pd.Series([10.0])).iloc[-1]
    last_price = df["close"].iloc[-1]

    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i-window:i+window].max():
            p = round(df["high"].iloc[i], 2)
            if abs(p - last_price) > min_dist_factor * atr:
                levels.append(("RES", p))
        if df["low"].iloc[i] == df["low"].iloc[i-window:i+window].min():
            p = round(df["low"].iloc[i], 2)
            if abs(p - last_price) > min_dist_factor * atr:
                levels.append(("SUP", p))

    return sorted(levels, key=lambda x: x[1], reverse=True)

def check_opposing_invalidation(levels, direction, entry, atr):
    thresh = atr * OPPOSING_FRACTAL_ATR_MULT
    if "BULL" in direction:
        return any(t == "RES" and entry < p < entry + thresh for t, p in levels)
    if "BEAR" in direction:
        return any(t == "SUP" and entry - thresh < p < entry for t, p in levels)
    return False

# ─── LAST ALERT STORAGE ──────────────────────────────────────────────────────
def load_last_alert():
    if os.path.exists(LAST_ALERT_FILE):
        try:
            with open(LAST_ALERT_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"time": 0, "key": None, "proposal": {}}

def save_last_alert(ts, key, proposal):
    with open(LAST_ALERT_FILE, "w") as f:
        json.dump({"time": ts, "key": key, "proposal": proposal}, f)

# ─── CORE CHECK ──────────────────────────────────────────────────────────────
def check_for_high_conviction_setup():
    last = load_last_alert()
    now_ts = time.time()
    if now_ts - last["time"] < ALERT_COOLDOWN_MIN * 60:
        return

    price = safe_td(get_live_price)
    if price is None:
        return

    ts_15m = safe_td(fetch_15m_data)
    if ts_15m is None:
        return
    time.sleep(15)

    ts_1h = safe_td(fetch_1h_data)
    if ts_1h is None:
        return
    time.sleep(15)

    ts_5m = safe_td(fetch_5m_data)
    if ts_5m is None:
        return

    latest = ts_15m.iloc[-1]
    atr = latest.get("atr", 10.0)
    rsi = latest.get("rsi", 50.0)

    ema200_15m = latest.get("ema_200", price)
    ema50_15m  = latest.get("ema_50", price)
    ema200_1h  = ts_1h.iloc[-1].get("ema_200", price) if not ts_1h.empty else price
    ema200_5m  = ts_5m.iloc[-1].get("ema_200", price) if not ts_5m.empty else price

    aligned_1h = (price > ema200_15m) == (price > ema200_1h)
    aligned_5m = (price > ema200_15m) == (price > ema200_5m)

    levels = get_fractal_levels(ts_15m)

    # ─── REAL AI CALLS (using pre-fetched data only) ──────────────────────────
    market = {"price": price, "rsi": rsi}
    setup = {"entry": price, "atr": atr}
    buffer = st.session_state.balance - st.session_state.floor

    base_prompt = f"""
You are a high-conviction gold trading auditor.
Use ONLY the provided data — do NOT hallucinate prices, levels, or any facts.

Current UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Market: ${price:.2f}, RSI {rsi:.1f}
Trend alignment: 15m vs 1H EMA200: {'agree' if aligned_1h else 'disagree'}, 15m vs 5M EMA200: {'agree' if aligned_5m else 'disagree'}
Fractals: {', '.join([f"{t}@{p}" for t,p in levels[:6]]) or "None"}

Setup context: entry \~${setup['entry']:.2f}, ATR ${setup['atr']:.2f}, risk buffer ${buffer:.2f}

Respond **ONLY** with valid JSON. No explanations, no fences, no markdown, no extra text before or after.

{{
  "verdict": "ELITE" | "HIGH_CONV" | "LOW_EDGE" | "GAMBLE" | "SKIP",
  "reason": "short explanation",
  "entry": number or null,
  "sl": number or null,
  "tp": number or null,
  "rr": number or null,
  "style": "SCALP" | "SWING" | "BREAKOUT" | "REVERSAL" | "RANGE" | "NONE",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "1-2 sentences"
}}
"""

    # Gemini
    try:
        g_out = gemini_model.generate_content(base_prompt).text.strip()
    except Exception as ex:
        g_out = f'{{"verdict":"SKIP","reason":"Gemini error: {str(ex)}"}}'

    # Grok
    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": base_prompt + "\nBe concise and factual."}],
            max_tokens=300,
            temperature=0.4
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as ex:
        k_out = f'{{"verdict":"SKIP","reason":"Grok error: {str(ex)}"}}'

    # OpenAI
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": base_prompt + "\nOutput strict JSON only, no extra text."}],
            max_tokens=300,
            temperature=0.5
        )
        c_out = response.choices[0].message.content.strip()
    except Exception as ex:
        c_out = f'{{"verdict":"SKIP","reason":"OpenAI error: {str(ex)}"}}'

    g_p = parse_ai_output(g_out)
    k_p = parse_ai_output(k_out)
    c_p = parse_ai_output(c_out)

    high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

    if high_count < MIN_CONVICTION_FOR_ALERT:
        return

    p = g_p  # prefer Gemini
    direction = p.get("direction", "NEUTRAL")
    style = p.get("style", "NONE")
    entry = p.get("entry") or price
    sl = p.get("sl") or (entry - atr*1.5 if "BULL" in direction else entry + atr*1.5)
    tp = p.get("tp") or (entry + atr*3 if "BULL" in direction else entry - atr*3)
    rr = p.get("rr") or 2.0

    if check_opposing_invalidation(levels, direction, entry, atr):
        return

    cash_risk = (st.session_state.balance - st.session_state.floor) * (st.session_state.risk_pct / 100)
    risk_per_lot = (abs(entry - sl) + SPREAD_BUFFER_POINTS + SLIPPAGE_BUFFER_POINTS) * PIP_VALUE
    lots = max(0.01, round((cash_risk / risk_per_lot) / 0.01) * 0.01)

    key = f"{direction}_{entry:.2f}_{sl:.2f}"
    if key == last["key"]:
        return

    save_last_alert(now_ts, key, {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "lots": lots
    })

    st.success("High conviction setup detected — check Telegram.")

# ─── PARSE AI OUTPUT (HARDENED) ──────────────────────────────────────────────
def parse_ai_output(text):
    if not text or not isinstance(text, str):
        return {"verdict": "UNKNOWN", "reason": "No output from AI"}

    # Aggressive cleaning
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'^Here is the JSON:?\s*', '', text, flags=re.IGNORECASE)
    text = text.strip()

    try:
        data = json.loads(text)
        return {
            "verdict": data.get("verdict", "UNKNOWN").upper(),
            "reason": data.get("reason", ""),
            "entry": data.get("entry"),
            "sl": data.get("sl"),
            "tp": data.get("tp"),
            "rr": data.get("rr"),
            "style": data.get("style", "NONE").upper(),
            "direction": data.get("direction", "NEUTRAL").upper(),
            "reasoning": data.get("reasoning", "")
        }
    except json.JSONDecodeError as e:
        return {
            "verdict": "PARSE_ERROR",
            "reason": f"JSON parse failed: {str(e)}. Raw: {text[:200]}..."
        }

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
st.set_page_config("Gold Sentinel Pro", "🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")

if "balance" not in st.session_state:
    st.session_state.balance = DEFAULT_BALANCE
    st.session_state.daily_limit = DEFAULT_DAILY_LIMIT
    st.session_state.floor = DEFAULT_FLOOR
    st.session_state.risk_pct = DEFAULT_RISK_PCT

st.header("Account Settings")
st.session_state.balance = st.number_input("Balance ($)", value=st.session_state.balance)
st.session_state.floor = st.number_input("Floor ($)", value=st.session_state.floor)
st.session_state.risk_pct = st.slider("Risk %", 5, 50, st.session_state.risk_pct, step=5)

if st.button("📡 Manual Check (\~4 credits)"):
    with st.spinner("Running check…"):
        check_for_high_conviction_setup()

# Auto-check toggle (optional, for browser keep-alive)
auto_enabled = st.checkbox("Enable auto-check while page open", value=False)
auto_interval_min = st.slider("Check every (minutes)", 10, 60, 15, step=5)

# ─── SAFE AUTO-CHECK TIMER (runs on every script execution) ───────────────────
CHECK_INTERVAL_SEC = auto_interval_min * 60

if auto_enabled:
    if "last_check_time" not in st.session_state:
        st.session_state.last_check_time = 0

    now = time.time()
    time_since_last = now - st.session_state.last_check_time

    if time_since_last >= CHECK_INTERVAL_SEC:
        st.session_state.last_check_time = now
        st.info(f"Auto-check triggered (last was {time_since_last/60:.1f} min ago)")
        with st.status("Running auto-check...", expanded=True) as status:
            check_for_high_conviction_setup()
            status.update(label=f"Auto-check complete at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}", state="complete")
        st.rerun()
    else:
        st.caption(f"Next auto-check in \~{int(CHECK_INTERVAL_SEC - time_since_last)} seconds")
else:
    st.info("Auto-check paused. Use manual button above.")
