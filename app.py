import streamlit as st
import json
from datetime import datetime, timezone, timedelta
import time
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import numpy as np
import pandas as pd
import io
import os

CHECK_INTERVAL_MIN = 30
CSV_1D_PATH = "longterm_history_1D.csv"
CSV_4H_PATH = "longterm_history_4H.csv"

td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
genai.configure(api_key=st.secrets["GEMINI_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

grok_client   = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def send_telegram(message, priority="normal"):
    token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except:
        pass

def get_live_price():
    try:
        return float(td.price(symbol="XAU/USD").as_json()["price"])
    except:
        return None

def fetch_recent_15m(outputsize=800):
    try:
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=outputsize, timezone="UTC")
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_atr(time_period=14)
        df = ts.as_pandas()
        df = df.sort_index(ascending=True)
        return df
    except Exception as e:
        st.warning(f"15m data fetch failed: {str(e)}")
        return None

def fetch_recent_1h(outputsize=200):
    try:
        ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=outputsize, timezone="UTC")
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_atr(time_period=14)
        df = ts.as_pandas()
        df = df.sort_index(ascending=True)
        return df
    except Exception as e:
        st.warning(f"1h data fetch failed: {str(e)}")
        return None

def download_4h_data():
    try:
        ts = td.time_series(symbol="XAU/USD", interval="4h", outputsize=360, timezone="UTC")
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_rsi(time_period=14)
        ts = ts.with_atr(time_period=14)
        return ts.as_pandas()
    except Exception as e:
        st.error(f"4H download failed: {str(e)}")
        return None

def download_1d_data():
    try:
        ts = td.time_series(symbol="XAU/USD", interval="1day", outputsize=60, timezone="UTC")
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_rsi(time_period=14)
        ts = ts.with_atr(time_period=14)
        return ts.as_pandas()
    except Exception as e:
        st.error(f"1D download failed: {str(e)}")
        return None

@st.cache_data
def load_long_term_1d():
    if os.path.exists(CSV_1D_PATH):
        try:
            df = pd.read_csv(CSV_1D_PATH, parse_dates=['datetime'], index_col='datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            return df
        except:
            return None
    return None

@st.cache_data
def load_long_term_4h():
    if os.path.exists(CSV_4H_PATH):
        try:
            df = pd.read_csv(CSV_4H_PATH, parse_dates=['datetime'], index_col='datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            return df
        except:
            return None
    return None

def calculate_strict_lot_size(entry, sl, max_risk_dollars, current_price=None):
    if not all(v is not None for v in [entry, sl, max_risk_dollars]):
        return 0.01, "Missing data — min lot used"

    try:
        entry = float(entry)
        sl = float(sl)
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return 0.01, "Invalid SL — min lot used"

        lot_size = max_risk_dollars / (price_diff * 100)
        lot_size_rounded = max(round(lot_size, 2), 0.01)

        actual_risk = lot_size_rounded * price_diff * 100
        note = f"Adjusted to fit ${max_risk_dollars:.2f} max risk"
        if actual_risk > max_risk_dollars * 1.05:
            note += " — still slightly over (wide SL)"
        return lot_size_rounded, note
    except:
        return 0.01, "Calc error — min lot used"

def run_check():
    with st.spinner("Fetching recent data..."):
        ts_15m = fetch_recent_15m()
        time.sleep(2)
        ts_1h  = fetch_recent_1h()
        if ts_15m is None or ts_15m.empty:
            st.error("No recent 15m data available")
            return
        price = ts_15m['close'].iloc[-1]
        current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    df_1d = load_long_term_1d()
    df_4h = load_long_term_4h()

    long_term_summary = ""
    if df_1d is not None and not df_1d.empty:
        ema200_1d = df_1d.get('ema_200') or df_1d.get('ema (200)') or df_1d.get('ema')
        if ema200_1d is not None:
            lt_price = df_1d['close'].iloc[-1]
            lt_trend = "bullish" if lt_price > ema200_1d.iloc[-1] else "bearish"
            long_term_summary += f"\nDaily trend: {lt_trend} (price vs EMA200 1D)"
        else:
            long_term_summary += "\nDaily EMA200 column not found in CSV"

    if df_4h is not None and not df_4h.empty:
        lt_price_4h = df_4h['close'].iloc[-1]
        long_term_summary += f"\n4H current price: {lt_price_4h:.2f}"

    latest_15m = ts_15m.iloc[-1]
    rsi = latest_15m.get('rsi', 50.0)
    atr = latest_15m.get('atr', 10.0)
    ema200_15m = latest_15m.get('ema_200', price)
    ema50_15m  = latest_15m.get('ema_50', price)

    ema200_1h = ts_1h.iloc[-1].get('ema_200', price) if ts_1h is not None and not ts_1h.empty else price

    levels = []
    for i in range(5, min(40, len(ts_15m)-5)):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    dd_limit = st.session_state.get("dd_limit")
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct", 25.0)
    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit else 50.0

    st.write(f"DD limit: ${dd_limit:.2f}")
    st.write(f"Risk % of DD: {risk_of_dd_pct}%")
    st.write(f"Max risk $: ${max_risk_dollars:.2f}")

    prompt = f"""
Current UTC: {current_time_str}

Recent market data (15m):
Price: ${price:.2f}
RSI (15m): {rsi:.1f}
ATR (15m): {atr:.2f}
EMA50 / EMA200 (15m): {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 (1h): {ema200_1h:.2f}

Recent support/resistance fractals: {', '.join([f"{t}@{p}" for t,p in levels[-8:]]) or 'None'}

{long_term_summary}

Account risk preference: max ${max_risk_dollars:.2f} loss per trade

Analyze the current market state using the provided data.
Be conservative — only mention setups you consider strong.

Output in this exact labeled format (no JSON, no extra text):

Bias: bullish / bearish / neutral
Confidence: high / medium / low / no edge
Key levels: support/resistance zones (e.g. support 4800–4820)
Suggested setup (if any):
Entry zone: approximate range (e.g. 4820–4835)
Stop zone: approximate range (e.g. below 4790)
Target zone: approximate range (e.g. 4880–4900)
Reasoning: your full explanation

If no strong setup, write only:
Bias: neutral
Confidence: no edge
Key levels: none
Suggested setup: none
Reasoning: market lacks clear edge at this time

Important: Use the 1D/4H context and recent data to suggest realistic zones based on structure, not arbitrary numbers.
"""

    with st.spinner("Consulting AIs..."):
        g_raw = "Bias: neutral\nConfidence: no edge\nKey levels: none\nSuggested setup: none\nReasoning: Gemini offline"
        try:
            g_raw = gemini_model.generate_content(prompt, generation_config={"temperature": 0.2}).text.strip()
        except:
            pass

        k_raw = "Bias: neutral\nConfidence: no edge\nKey levels: none\nSuggested setup: none\nReasoning: Grok offline"
        try:
            r = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4
            )
            k_raw = r.choices[0].message.content.strip()
        except:
            pass

        c_raw = "Bias: neutral\nConfidence: no edge\nKey levels: none\nSuggested setup: none\nReasoning: ChatGPT offline"
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5
            )
            c_raw = resp.choices[0].message.content.strip()
        except:
            pass

    ai_responses = [
        {"name": "Gemini", "text": g_raw},
        {"name": "Grok", "text": k_raw},
        {"name": "ChatGPT", "text": c_raw}
    ]

    biases = []
    confidences = []
    entry_zones = []
    stop_zones = []
    target_zones = []
    for resp in ai_responses:
        text = resp["text"]
        bias_line = [line for line in text.split('\n') if line.strip().startswith("Bias:")]
        bias = bias_line[0].split(":", 1)[1].strip().upper() if bias_line else "NEUTRAL"
        biases.append(bias)

        conf_line = [line for line in text.split('\n') if line.strip().startswith("Confidence:")]
        conf = conf_line[0].split(":", 1)[1].strip().lower() if conf_line else "no edge"
        confidences.append(conf)

        entry_line = [line for line in text.split('\n') if line.strip().startswith("Entry zone:")]
        if entry_line:
            entry_zones.append(entry_line[0].split(":", 1)[1].strip())

        stop_line = [line for line in text.split('\n') if line.strip().startswith("Stop zone:")]
        if stop_line:
            stop_zones.append(stop_line[0].split(":", 1)[1].strip())

        target_line = [line for line in text.split('\n') if line.strip().startswith("Target zone:")]
        if target_line:
            target_zones.append(target_line[0].split(":", 1)[1].strip())

    consensus_bias = max(set(biases), key=biases.count) if biases else "NEUTRAL"
    agreement_count = biases.count(consensus_bias)

    high_conf_count = sum(1 for c in confidences if c == "high")
    med_conf_count = sum(1 for c in confidences if c in ["high", "medium"])

    if agreement_count == 3 and high_conf_count >= 2:
        verdict = "ELITE"
    elif agreement_count >= 2 and med_conf_count >= 2:
        verdict = "HIGH_CONVICTION"
    else:
        verdict = "NO_CONVICTION"

    lot_size = 0.01
    lot_note = "Min lot — no strong consensus or valid setup"
    risk_warning = ""
    telegram_zones = ""

    if verdict in ["ELITE", "HIGH_CONVICTION"] and consensus_bias != "NEUTRAL" and len(entry_zones) >= 2 and len(stop_zones) >= 2:
        try:
            entry_lows = []
            entry_highs = []
            stop_lows = []
            stop_highs = []
            for z in entry_zones:
                if '–' in z:
                    parts = z.split('–')
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    entry_lows.append(low)
                    entry_highs.append(high)
            for z in stop_zones:
                if '–' in z or 'below' in z or 'above' in z:
                    z_clean = z.replace('below', '').replace('above', '').strip()
                    if '–' in z_clean:
                        parts = z_clean.split('–')
                        low = float(parts[0].strip())
                        high = float(parts[1].strip())
                        stop_lows.append(low)
                        stop_highs.append(high)

            if entry_lows and entry_highs and stop_lows:
                consensus_entry_low = min(entry_lows)
                consensus_entry_high = max(entry_highs)
                consensus_stop = min(stop_lows) if consensus_bias == "BULLISH" else max(stop_highs)

                avg_entry = (consensus_entry_low + consensus_entry_high) / 2
                theoretical_risk = abs(avg_entry - consensus_stop) * 100

                if theoretical_risk > max_risk_dollars:
                    risk_warning = f"**RISK WARNING**: Consensus stop zone implies \~${theoretical_risk:.2f} risk — exceeds your ${max_risk_dollars:.2f} limit. Review or skip."
                else:
                    lot_size, lot_note = calculate_strict_lot_size(avg_entry, consensus_stop, max_risk_dollars, price)
                    telegram_zones = f"Consensus entry zone: {consensus_entry_low:.0f}–{consensus_entry_high:.0f}\nTightest stop zone: {'below' if consensus_bias == 'BULLISH' else 'above'} {consensus_stop:.0f}"
        except:
            lot_note = "Zone parsing failed — min lot used"

    st.divider()
    st.subheader("AI Consensus & Setup Review")

    st.markdown(f"**Consensus Bias:** {consensus_bias}")
    st.markdown(f"**Agreement Level:** {agreement_count}/3 AIs")
    st.markdown(f"**Verdict:** {verdict}")
    st.markdown(f"**Lot Size:** {lot_size:.2f} ({lot_note})")
    if risk_warning:
        st.error(risk_warning)

    st.markdown("**Important Disclaimer**")
    st.markdown("Agreement among AIs reflects narrative clarity and obviousness, NOT statistical edge. Obvious setups are often crowded and prone to chop/fakeouts in Gold. Use only as a second opinion — never as a signal. Always verify independently.")

    for resp in ai_responses:
        st.markdown(f"**{resp['name']} Response:**\n\n{resp['text']}")

    if verdict in ["ELITE", "HIGH_CONVICTION"] and consensus_bias != "NEUTRAL" and not risk_warning:
        msg = f"**{verdict} on {consensus_bias}** ({agreement_count}/3 AIs)\n{telegram_zones}\nLot: {lot_size:.2f} ({lot_note})\n{long_term_summary.strip()}"
        send_telegram(msg, priority="high" if verdict == "ELITE" else "normal")

st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Gemini • Grok • ChatGPT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.expander("Risk Settings", expanded=True):
    default_dd = st.session_state.get("dd_limit", 251.45)
    default_pct = st.session_state.get("risk_of_dd_pct", 25.0)

    col1, col2 = st.columns(2)
    with col1:
        dd_limit = st.number_input("Daily Drawdown Limit ($)", min_value=0.0, value=float(default_dd), format="%.2f", step=1.0)
    with col2:
        risk_pct = st.number_input("Risk % of DD", min_value=1, max_value=100, value=int(default_pct), format="%d", step=1)

    if st.button("Save Settings"):
        st.session_state.dd_limit = dd_limit
        st.session_state.risk_of_dd_pct = risk_pct
        st.success("Settings saved")

with st.expander("Update Long-Term History (weekly)", expanded=False):
    st.info("Download fresh context files for 1D and 4H (combine or upload separately). Update once a week when market is closed.")
    
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Download 4H (60 days)"):
            df = download_4h_data()
            if df is not None:
                st.download_button("Save 4H CSV", df.to_csv(), "longterm_history_4H.csv", "text/csv")

    with col_b:
        if st.button("Download 1D (60 days)"):
            df = download_1d_data()
            if df is not None:
                st.download_button("Save 1D CSV", df.to_csv(), "longterm_history_1D.csv", "text/csv")

if st.button("Run Live Analysis", type="primary", use_container_width=True):
    run_check()
