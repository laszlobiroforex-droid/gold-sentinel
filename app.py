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

# ─── CONFIG ────────────────────────────────────────────────────────────────
CHECK_INTERVAL_MIN = 30
LONG_TERM_CSV_PATH = "history_longterm.csv"

# ─── API INIT ──────────────────────────────────────────────────────────────
td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
genai.configure(api_key=st.secrets["GEMINI_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

grok_client   = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── TELEGRAM ──────────────────────────────────────────────────────────────
def send_telegram(message, priority="normal"):
    token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.warning("Telegram not configured")
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# ─── DATA FETCH ────────────────────────────────────────────────────────────
def get_live_price():
    try:
        return float(td.price(symbol="XAU/USD").as_json()["price"])
    except:
        return None

def fetch_recent_15m(outputsize=800):
    try:
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=outputsize, timezone="UTC")
        ts = ts.with_rsi(time_period=14)
        ts = ts.with_ema(time_period=21)
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_macd(fast_period=12, slow_period=26, signal_period=9)
        ts = ts.with_adx(time_period=14)
        ts = ts.with_bbands(time_period=20, std_dev_up=2, std_dev_down=2)
        ts = ts.with_atr(time_period=14)
        df = ts.as_pandas()
        df = df.rename(columns={
            'ema_21': 'ema_21', 'ema_50': 'ema_50', 'ema_200': 'ema_200',
            'macd_macd': 'macd', 'macd_signal': 'macd_signal', 'macd_hist': 'macd_hist',
            'bbands_upper': 'bb_upper', 'bbands_middle': 'bb_middle', 'bbands_lower': 'bb_lower'
        })
        df = df.sort_index(ascending=True)
        return df
    except Exception as e:
        st.warning(f"Recent 15m fetch failed: {str(e)}")
        return None

def fetch_recent_1h(outputsize=200):
    try:
        ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=outputsize, timezone="UTC")
        ts = ts.with_rsi(time_period=14)
        ts = ts.with_ema(time_period=21)
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_macd(fast_period=12, slow_period=26, signal_period=9)
        ts = ts.with_adx(time_period=14)
        ts = ts.with_bbands(time_period=20, std_dev_up=2, std_dev_down=2)
        ts = ts.with_atr(time_period=14)
        df = ts.as_pandas()
        df = df.rename(columns={
            'ema_21': 'ema_21', 'ema_50': 'ema_50', 'ema_200': 'ema_200',
            'macd_macd': 'macd', 'macd_signal': 'macd_signal', 'macd_hist': 'macd_hist',
            'bbands_upper': 'bb_upper', 'bbands_middle': 'bb_middle', 'bbands_lower': 'bb_lower'
        })
        df = df.sort_index(ascending=True)
        return df
    except Exception as e:
        st.warning(f"Recent 1h fetch failed: {str(e)}")
        return None

def fetch_longterm_4h_1d():
    try:
        ts_4h = td.time_series(symbol="XAU/USD", interval="4h", outputsize=360, timezone="UTC")
        ts_4h = ts_4h.with_rsi(time_period=14)
        ts_4h = ts_4h.with_ema(time_period=21)
        ts_4h = ts_4h.with_ema(time_period=50)
        ts_4h = ts_4h.with_ema(time_period=200)
        ts_4h = ts_4h.with_macd(fast_period=12, slow_period=26, signal_period=9)
        ts_4h = ts_4h.with_adx(time_period=14)
        ts_4h = ts_4h.with_bbands(time_period=20, std_dev_up=2, std_dev_down=2)
        ts_4h = ts_4h.with_atr(time_period=14)

        ts_1d = td.time_series(symbol="XAU/USD", interval="1day", outputsize=60, timezone="UTC")
        ts_1d = ts_1d.with_rsi(time_period=14)
        ts_1d = ts_1d.with_ema(time_period=21)
        ts_1d = ts_1d.with_ema(time_period=50)
        ts_1d = ts_1d.with_ema(time_period=200)
        ts_1d = ts_1d.with_macd(fast_period=12, slow_period=26, signal_period=9)
        ts_1d = ts_1d.with_adx(time_period=14)
        ts_1d = ts_1d.with_bbands(time_period=20, std_dev_up=2, std_dev_down=2)
        ts_1d = ts_1d.with_atr(time_period=14)

        df_4h = ts_4h.as_pandas()
        df_1d = ts_1d.as_pandas()

        df_4h = df_4h.rename(columns={
            'macd_macd': 'macd', 'macd_signal': 'macd_signal', 'macd_hist': 'macd_hist',
            'bbands_upper': 'bb_upper', 'bbands_middle': 'bb_middle', 'bbands_lower': 'bb_lower'
        })
        df_1d = df_1d.rename(columns={
            'macd_macd': 'macd', 'macd_signal': 'macd_signal', 'macd_hist': 'macd_hist',
            'bbands_upper': 'bb_upper', 'bbands_middle': 'bb_middle', 'bbands_lower': 'bb_lower'
        })

        combined = pd.concat([df_4h, df_1d]).sort_index(ascending=True)
        return combined
    except Exception as e:
        st.error(f"Long-term 4H/1D fetch failed: {str(e)}")
        return None

# ─── LOAD LONG-TERM HISTORY ────────────────────────────────────────────────
@st.cache_data
def load_long_term_history():
    if os.path.exists(LONG_TERM_CSV_PATH):
        try:
            df = pd.read_csv(LONG_TERM_CSV_PATH, parse_dates=['datetime'], index_col='datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            st.success(f"Loaded long-term history ({len(df)} rows)")
            return df
        except Exception as e:
            st.error(f"Failed to load long-term CSV: {str(e)}")
    return None

# ─── MAIN ANALYSIS FUNCTION ────────────────────────────────────────────────
def run_check(historical_end_time=None, test_dd_limit=None, test_risk_pct=None):
    is_historical = historical_end_time is not None

    with st.spinner("Fetching recent data..."):
        ts_15m = fetch_recent_15m()
        ts_1h  = fetch_recent_1h()
        if ts_15m is None or ts_15m.empty:
            st.error("No recent 15m data available")
            return
        price = ts_15m['close'].iloc[-1]
        current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if not is_historical else historical_end_time.strftime('%Y-%m-%d %H:%M UTC')

        if is_historical:
            st.info(f"Historical test mode: simulating market at {current_time_str}")
            st.write(f"Simulated current price: ${price:.2f}")
            debug_cols = ['close', 'rsi', 'ema_21', 'ema_50', 'ema_200', 'atr', 'adx']  # Add more if present
            available_cols = [col for col in debug_cols if col in ts_15m.columns]
            st.write(f"Recent 15m candles: {len(ts_15m)}")
            st.write("Last timestamp:", ts_15m.index[-1])
            st.write("Last 5 recent candles:", ts_15m.tail(5)[available_cols])

    # Long-term history (for summary)
    long_term_df = load_long_term_history()
    long_term_summary = ""
    if long_term_df is not None and not long_term_df.empty:
        lt_price = long_term_df['close'].iloc[-1]
        lt_trend = "bullish" if lt_price > long_term_df['ema_200'].iloc[-1] else "bearish"
        lt_adx = long_term_df['adx'].iloc[-1] if 'adx' in long_term_df.columns else "N/A"
        lt_rsi_avg = long_term_df['rsi'].iloc[-10:].mean() if 'rsi' in long_term_df.columns else "N/A"
        long_term_summary = f"""
Long-term context (4H/1D last 60 days):
- Trend: {lt_trend} (price vs EMA200)
- ADX: {lt_adx} → {'strong trend' if isinstance(lt_adx, (int,float)) and lt_adx > 25 else 'ranging'}
- RSI 1D avg last 10 days: {lt_rsi_avg:.1f}
"""

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

    if is_historical:
        dd_limit = test_dd_limit
        risk_of_dd_pct = test_risk_pct
    else:
        dd_limit = st.session_state.get("dd_limit")
        risk_of_dd_pct = st.session_state.get("risk_of_dd_pct", 25.0)

    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit else 50.0

    if is_historical:
        st.write(f"Test DD limit: ${dd_limit:.2f}")
        st.write(f"Test risk %: {risk_of_dd_pct}%")
        st.write(f"Max risk $: ${max_risk_dollars:.2f}")

    prompt = f"""
Current UTC: {current_time_str}

Recent market data (15m):
Price: ${price:.2f}
RSI (15m): {rsi:.1f}
ATR (15m): {atr:.2f}
EMA21 / EMA50 / EMA200 (15m): {latest_15m.get('ema_21', 'N/A'):.2f} / {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 (1h): {ema200_1h:.2f}

Recent support/resistance fractals: {', '.join([f"{t}@{p}" for t,p in levels[-8:]]) or 'None'}

{long_term_summary}

Account risk limit: max ${max_risk_dollars:.2f} loss per trade (preferred)

Verdict rules (strict):
- ELITE: win prob ≥80%, exceptional setup
- HIGH_CONV: win prob ≥65%, strong edge
- MODERATE: win prob 50–65%
- LOW_EDGE: win prob <50%
- NO_EDGE: no valid setup or prob too low
Always base verdict on your estimated_win_prob — do not assign HIGH_CONV or ELITE to low-prob setups.

Propose high-probability setups with confirmation entries only.
Respond **ONLY** with valid JSON:

{{
  "verdict": "ELITE" | "HIGH_CONV" | "MODERATE" | "LOW_EDGE" | "NO_EDGE",
  "reason": "short explanation",
  "entry_type": "LIMIT" | "STOP" | "MARKET" | null,
  "entry_price": number or null,
  "sl": number or null,
  "tp": number or null,
  "rr": number or null,
  "estimated_win_prob": number or null,
  "style": "SCALP" | "SWING" | "BREAKOUT" | "REVERSAL" | "RANGE" | "NONE",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "1-3 sentences"
}}
"""

    # (rest of the AI consultation, parsing, display, consensus code remains the same as before)
    # ... paste the rest of run_check() from your current version ...

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Gemini • Grok • ChatGPT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.expander("Prop Challenge / Risk Settings (required for lot sizing)", expanded=True):
    # (your existing risk inputs code)

# ─── HISTORICAL TEST MODE ──────────────────────────────────────────────────
with st.expander("Historical Test Mode (optional backtesting)", expanded=False):
    st.info("Select a PAST date/time to simulate market state exactly then.")

    col1, col2 = st.columns(2)
    with col1:
        test_date = st.date_input("Test Date", value=datetime.now(timezone.utc).date() - timedelta(days=1))
    with col2:
        test_time = st.time_input("Test Time (UTC)", value=datetime.strptime("14:30", "%H:%M").time())

    test_dd_limit = st.number_input("Daily Drawdown Limit ($) for this test", min_value=0.0, step=1.0, value=st.session_state.get("dd_limit", 251.45), format="%.2f")
    test_risk_pct = st.number_input("Risk % of Daily DD for this test", min_value=1.0, max_value=100.0, value=st.session_state.get("risk_of_dd_pct", 25.0), step=1.0, format="%.0f")

    test_datetime = datetime.combine(test_date, test_time, tzinfo=timezone.utc)

    if st.button("Run Historical Test"):
        if test_datetime >= datetime.now(timezone.utc):
            st.error("Cannot test future or current time — pick a past moment.")
        else:
            run_check(historical_end_time=test_datetime, test_dd_limit=test_dd_limit, test_risk_pct=test_risk_pct)

    # Long-term history update button
    if st.button("Download fresh 60-day 4H & 1D data for manual update"):
        df_long = fetch_longterm_4h_1d()
        if df_long is not None:
            csv_buffer = io.StringIO()
            df_long.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="Download long-term history CSV",
                data=csv_data,
                file_name="history_longterm_update.csv",
                mime="text/csv",
                help="Rename to history_longterm.csv and upload to app root"
            )
        else:
            st.error("Failed to fetch long-term data — check API key or network")

    # (rest of expander: recent snapshot download)

# (rest of UI: auto mode, live button, etc.)
