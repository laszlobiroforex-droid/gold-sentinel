import streamlit as st
import json
import time
from datetime import datetime, timezone
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────
CHECK_INTERVAL_MIN = 15

# ─── API INIT (UNCHANGED KEYS) ────────────────────────────────────────────
td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
genai.configure(api_key=st.secrets["GEMINI_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

grok_client   = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── TELEGRAM ─────────────────────────────────────────────────────────────
def send_telegram(message, priority="normal"):
    token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=8)

# ─── DATA (NO CACHE — YOU WERE RIGHT) ──────────────────────────────────────
def get_live_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

def fetch_15m():
    ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=120)
    ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
    return ts.as_pandas()

def fetch_1h():
    ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=60)
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

# ─── PARSER (UNCHANGED) ────────────────────────────────────────────────────
def parse_ai_output(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return data
    except:
        return None

# ─── MAIN ──────────────────────────────────────────────────────────────────
def run_check():
    price = get_live_price()
    ts_15m = fetch_15m()
    ts_1h  = fetch_1h()

    latest = ts_15m.iloc[-1]

    snapshot_id = f"{price:.2f}_{datetime.now(timezone.utc).isoformat()}"

    prompt = f"""
You are a professional gold trading auditor.

IMPORTANT RULES:
- Use ONLY the data provided.
- DO NOT reuse any previous answers.
- DO NOT hallucinate levels or prices.
- Evaluate ONLY this snapshot.

SNAPSHOT_ID: {snapshot_id}

Market data:
Price: {price}
RSI (15m): {latest['rsi']}
ATR (15m): {latest['atr']}
EMA50 (15m): {latest['ema_50']}
EMA200 (15m): {latest['ema_200']}
EMA200 (1h): {ts_1h.iloc[-1]['ema_200']}

Respond ONLY with valid JSON:

{{
  "verdict": "ELITE|HIGH_CONV|MODERATE|LOW_EDGE|GAMBLE",
  "direction": "BULLISH|BEARISH",
  "entry": number,
  "sl": number,
  "tp": number,
  "rr": number,
  "reason": "short explanation"
}}
"""

    outputs = []

    for name, call in [
        ("Gemini", lambda: gemini_model.generate_content(prompt).text),
        ("Grok", lambda: grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        ).choices[0].message.content),
        ("ChatGPT", lambda: openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        ).choices[0].message.content)
    ]:
        raw = call()
        parsed = parse_ai_output(raw)
        outputs.append((name, parsed))

    st.subheader("AI Verdicts")
    for name, data in outputs:
        st.write(name, data)

# ─── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel")

if st.button("📡 Run Analysis Now"):
    run_check()
