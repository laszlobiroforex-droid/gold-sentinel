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
CSV_1D_PATH = "longterm_history_1D.csv"
CSV_4H_PATH = "longterm_history_4H.csv"

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
        ts = ts.with_ema(time_period=50)
        ts = ts.with_ema(time_period=200)
        ts = ts.with_atr(time_period=14)
        df = ts.as_pandas()
        df = df.sort_index(ascending=True)
        return df
    except Exception as e:
        st.warning(f"Recent 15m fetch failed: {str(e)}")
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
        st.warning(f"Recent 1h fetch failed: {str(e)}")
        return None

def download_4h_data():
    try:
        ts_4h = td.time_series(symbol="XAU/USD", interval="4h", outputsize=360, timezone="UTC")
        ts_4h = ts_4h.with_ema(time_period=50)
        ts_4h = ts_4h.with_ema(time_period=200)
        ts_4h = ts_4h.with_rsi(time_period=14)
        ts_4h = ts_4h.with_atr(time_period=14)
        df = ts_4h.as_pandas()
        return df
    except Exception as e:
        st.error(f"4H download failed: {str(e)}")
        return None

def download_1d_data():
    try:
        ts_1d = td.time_series(symbol="XAU/USD", interval="1day", outputsize=60, timezone="UTC")
        ts_1d = ts_1d.with_ema(time_period=50)
        ts_1d = ts_1d.with_ema(time_period=200)
        ts_1d = ts_1d.with_rsi(time_period=14)
        ts_1d = ts_1d.with_atr(time_period=14)
        df = ts_1d.as_pandas()
        return df
    except Exception as e:
        st.error(f"1D download failed: {str(e)}")
        return None

# ─── LOAD LONG-TERM HISTORY ────────────────────────────────────────────────
@st.cache_data
def load_long_term_1d():
    if os.path.exists(CSV_1D_PATH):
        try:
            df = pd.read_csv(CSV_1D_PATH, parse_dates=['datetime'], index_col='datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            st.success(f"Loaded 1D history ({len(df)} rows)")
            return df
        except Exception as e:
            st.error(f"Failed to load 1D CSV: {str(e)}")
    return None

@st.cache_data
def load_long_term_4h():
    if os.path.exists(CSV_4H_PATH):
        try:
            df = pd.read_csv(CSV_4H_PATH, parse_dates=['datetime'], index_col='datetime')
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            st.success(f"Loaded 4H history ({len(df)} rows)")
            return df
        except Exception as e:
            st.error(f"Failed to load 4H CSV: {str(e)}")
    return None

# ─── MAIN ANALYSIS FUNCTION ────────────────────────────────────────────────
def run_check(historical_end_time=None, test_dd_limit=None, test_risk_pct=None):
    is_historical = historical_end_time is not None

    with st.spinner("Fetching recent data..."):
        ts_15m = fetch_recent_15m()
        time.sleep(2)  # small delay to help with rate limit
        ts_1h  = fetch_recent_1h()
        if ts_15m is None or ts_15m.empty:
            st.error("No recent 15m data available")
            return
        price = ts_15m['close'].iloc[-1]
        current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if not is_historical else historical_end_time.strftime('%Y-%m-%d %H:%M UTC')

        if is_historical:
            st.info(f"Historical test mode: simulating market at {current_time_str}")
            st.write(f"Simulated current price: ${price:.2f}")
            debug_cols = ['close', 'ema_50', 'ema_200', 'atr']
            available_cols = [col for col in debug_cols if col in ts_15m.columns]
            st.write(f"Recent 15m candles: {len(ts_15m)}")
            st.write("Last timestamp:", ts_15m.index[-1])
            st.write("Last 5 recent candles:", ts_15m.tail(5)[available_cols])

    # Load long-term files
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
            long_term_summary += "\nDaily EMA200 column not found"

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
EMA50 / EMA200 (15m): {ema50_15m:.2f} / {ema200_15m:.2f}
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

    with st.spinner("Consulting AIs..."):
        g_raw = k_raw = c_raw = '{"verdict":"ERROR","reason":"AI offline"}'

        try:
            g_raw = gemini_model.generate_content(prompt, generation_config={"temperature": 0.2}).text.strip()
        except Exception as e:
            st.warning(f"Gemini failed: {str(e)}")

        try:
            r = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4
            )
            k_raw = r.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Grok failed: {str(e)}")

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )
            c_raw = resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"ChatGPT failed: {str(e)}")

    g_p = parse_ai_output(g_raw)
    k_p = parse_ai_output(k_raw)
    c_p = parse_ai_output(c_raw)

    # Strict lot calculation
    best_entry = best_sl = None
    for p in [g_p, k_p, c_p]:
        e = p.get("entry_price") or p.get("entry")
        s = p.get("sl")
        if isinstance(e, (int, float)) and isinstance(s, (int, float)):
            best_entry = e
            best_sl = s
            break

    lot_size = 0.01
    lot_note = "Min lot (no valid entry/SL)"
    if best_entry and best_sl:
        lot_size, lot_note = calculate_strict_lot_size(best_entry, best_sl, max_risk_dollars, price)

    # ─── DISPLAY VERDICTS ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("AI Verdicts")

    def format_verdict(p, ai_name, lot=None, note=""):
        if p["verdict"] == "PARSE_ERROR":
            return f"**{ai_name}** — Parse error: {p['reason']}"

        verdict = p["verdict"]
        colors = {
            "ELITE": "#2ecc71", "HIGH_CONV": "#3498db",
            "MODERATE": "#f1c40f", "LOW_EDGE": "#e67e22",
            "NO_EDGE": "#95a5a6", "PARSE_ERROR": "#7f8c8d"
        }
        color = colors.get(verdict, "#7f8c8d")

        entry_type = p.get('entry_type', '—')
        entry_price_val = p.get('entry_price')
        if isinstance(entry_price_val, (int, float)):
            entry_price_str = f"${entry_price_val:.2f}"
        else:
            entry_price_str = "—"
        entry_str = f"{entry_type} @ {entry_price_str}"

        sl_val = p.get('sl')
        sl_str = f"${sl_val:.2f}" if isinstance(sl_val, (int, float)) else "—"

        tp_val = p.get('tp')
        tp_str = f"${tp_val:.2f}" if isinstance(tp_val, (int, float)) else "—"

        rr_val = p.get('rr')
        rr_str = f"1:{rr_val:.1f}" if isinstance(rr_val, (int, float)) else "—"

        prob_val = p.get('estimated_win_prob')
        prob_str = f"{prob_val}%" if isinstance(prob_val, (int, float)) else "—"

        risk_dollars_val = p.get('risk_dollars')
        risk_pct_val     = p.get('risk_pct_of_dd')
        exceed_flag      = p.get('exceeds_preferred_risk', False)

        if isinstance(risk_dollars_val, (int, float)):
            dollars_part = f"${risk_dollars_val:.2f}"
        else:
            dollars_part = "—"

        if isinstance(risk_pct_val, (int, float)):
            pct_part = f"{risk_pct_val:.1f}%"
        else:
            pct_part = "—"

        risk_str = f"{dollars_part} ({pct_part} of DD)"
        exceed_warning = '<span style="color:#e74c3c; font-weight:bold;">EXCEEDS preferred risk!</span>' if exceed_flag else ""

        if isinstance(entry_price_val, (int, float)) and price and abs(entry_price_val - price) / price > 0.10:
            entry_str += ' <span style="color:#e74c3c;">(unrealistic distance!)</span>'

        lot_str = f"{lot:.2f}" if lot is not None else "—"

        html = f"""
        <div style="background:{color}11; border-left:5px solid {color}; padding:16px 20px; margin:16px 0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="font-size:1.3em; font-weight:bold; color:{color}; margin-bottom:8px;">
                {ai_name} — {verdict}
            </div>
            <div style="margin-bottom:12px; color:#555;">{p['reason']}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0;">
                <div><strong>Direction:</strong> {p['direction']}</div>
                <div><strong>Style:</strong> {p['style']}</div>
                <div><strong>Entry:</strong> {entry_str}</div>
                <div><strong>SL:</strong> {sl_str}</div>
                <div><strong>TP:</strong> {tp_str}</div>
                <div><strong>RR:</strong> {rr_str}</div>
                <div><strong>Est. Win Prob:</strong> {prob_str}</div>
                <div><strong>Risk:</strong> {risk_str} {exceed_warning}</div>
                <div><strong>Lot size:</strong> {lot_str}</div>
            </div>
            <div style="color:#555; font-size:0.9em; margin:8px 0;">{note or ''}</div>
            <div style="font-style:italic; color:#444; margin-top:12px;">{p['reasoning']}</div>
        </div>
        """
        return html

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(format_verdict(g_p, "Gemini",   lot_size, lot_note), unsafe_allow_html=True)
    with col2: st.markdown(format_verdict(k_p, "Grok",     lot_size, lot_note), unsafe_allow_html=True)
    with col3: st.markdown(format_verdict(c_p, "ChatGPT",  lot_size, lot_note), unsafe_allow_html=True)

    # ─── CONSENSUS & TELEGRAM ──────────────────────────────────────────────────
    high_verdicts = [p for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"] and p.get("estimated_win_prob", 0) >= 60]
    if len(high_verdicts) >= 2:
        directions = [p["direction"] for p in high_verdicts if p["direction"] != "NEUTRAL"]
        if len(set(directions)) == 1 and directions:
            direction = directions[0]

            valid_entries = []
            for p in high_verdicts:
                e = p.get("entry_price") or p.get("entry")
                if isinstance(e, (int, float)) and abs(e - price) / price < 0.05:
                    valid_entries.append({
                        "entry": e,
                        "sl": p.get("sl"),
                        "tp": p.get("tp"),
                        "source": p
                    })

            if len(valid_entries) >= 2:
                entries_sorted = sorted(valid_entries, key=lambda x: x["entry"])
                consensus_entry = entries_sorted[0]["entry"]
                tps = [v["tp"] for v in valid_entries if isinstance(v["tp"], (int, float))]
                consensus_tp = np.median(tps) if tps else "—"
                sls = [v["sl"] for v in valid_entries if isinstance(v["sl"], (int, float))]
                consensus_sl = min(sls) if sls else "—"

                msg = (
                    f"**Consensus High Conviction ({len(high_verdicts)} AIs)**\n"
                    f"Direction: {direction}\n"
                    f"Entry (lowest/safest): LIMIT @ ${consensus_entry:.2f}\n"
                    f"SL (tightest): ${consensus_sl:.2f}\n"
                    f"TP (median): ${consensus_tp if isinstance(consensus_tp, (int, float)) else consensus_tp:.2f}\n"
                    f"Lot size: {lot_size:.2f} ({lot_note})\n"
                )
                send_telegram(msg, priority="high" if len(high_verdicts) == 3 else "normal")
                st.success("Consensus alert sent!")
            else:
                st.info("No clustered valid entries — no alert")
        else:
            st.info("Direction mismatch — no alert")
    else:
        st.info("No strong consensus — no alert")

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Gemini • Grok • ChatGPT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.expander("Prop Challenge / Risk Settings (required for lot sizing)", expanded=True):
    default_balance = st.session_state.get("balance", 5029.00)
    default_dd = st.session_state.get("dd_limit", 251.45)
    default_risk_pct = st.session_state.get("risk_of_dd_pct", 25.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        balance_input = st.number_input("Current Balance ($)", min_value=0.0, step=0.01, value=default_balance, format="%.2f")
    with col2:
        dd_input = st.number_input("Daily Drawdown Limit ($)", min_value=0.0, step=1.0, value=default_dd, format="%.2f")
    with col3:
        risk_input = st.number_input("Preferred risk % of Daily DD", min_value=1.0, max_value=100.0, value=default_risk_pct, step=1.0, format="%.0f")

    if st.button("Save & Apply Settings", type="primary", use_container_width=True):
        st.session_state.balance = balance_input
        st.session_state.dd_limit = dd_input
        st.session_state.risk_of_dd_pct = risk_input
        st.success("Settings saved!")
        st.rerun()

# ─── HISTORICAL TEST MODE ──────────────────────────────────────────────────
with st.expander("Historical Test Mode (optional backtesting)", expanded=False):
    st.info("Select a PAST date/time to simulate market state exactly then. Uses recent API data + optional long-term CSV files.")
    
    col1, col2 = st.columns(2)
    with col1:
        test_date = st.date_input("Test Date", value=datetime.now(timezone.utc).date() - timedelta(days=1))
    with col2:
        test_time = st.time_input("Test Time (UTC)", value=datetime.strptime("14:30", "%H:%M").time())

    test_dd_limit = st.number_input("Daily Drawdown Limit ($) for this test", 
                                    min_value=0.0, step=1.0, 
                                    value=st.session_state.get("dd_limit", 251.45), 
                                    format="%.2f")
    test_risk_pct = st.number_input("Risk % of Daily DD for this test", 
                                    min_value=1.0, max_value=100.0, 
                                    value=st.session_state.get("risk_of_dd_pct", 25.0), 
                                    step=1.0, format="%.0f")

    test_datetime = datetime.combine(test_date, test_time, tzinfo=timezone.utc)

    if st.button("Run Historical Test"):
        if test_datetime >= datetime.now(timezone.utc):
            st.error("Cannot test future or current time — pick a past moment.")
        else:
            run_check(historical_end_time=test_datetime, 
                      test_dd_limit=test_dd_limit, 
                      test_risk_pct=test_risk_pct)

    # Long-term history split downloads
    st.markdown("**Download long-term history in two parts (combine later if needed)**")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Download 4H data (60 days)"):
            df_4h = download_4h_data()
            if df_4h is not None:
                st.download_button("Download 4H CSV", df_4h.to_csv(), "longterm_history_4H.csv", "text/csv")

    with col_b:
        if st.button("Download 1D data (60 days)"):
            df_1d = download_1d_data()
            if df_1d is not None:
                st.download_button("Download 1D CSV", df_1d.to_csv(), "longterm_history_1D.csv", "text/csv")

    # Recent snapshot download
    if 'last_ts_15m' in st.session_state and st.session_state['last_ts_15m'] is not None:
        df = st.session_state['last_ts_15m']
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()

        filename = f"gold_15m_snapshot_{st.session_state.get('last_test_datetime', 'unknown').strftime('%Y%m%d_%H%M')}.csv" if st.session_state.get('last_test_datetime') else "gold_15m_snapshot.csv"

        st.download_button(
            label="Download Recent 15m Snapshot (CSV)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Exact recent data slice used for this test"
        )
    else:
        st.info("Run a test first to enable recent snapshot download.")

# ─── AUTO / MANUAL CONTROLS ────────────────────────────────────────────────
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

auto_enabled = st.checkbox(f"Auto-run every {CHECK_INTERVAL_MIN} minutes (keep tab open)", value=False)

if auto_enabled:
    now = time.time()
    time_since = now - st.session_state.last_check_time
    if time_since >= CHECK_INTERVAL_MIN * 60:
        st.session_state.last_check_time = now
        run_check()
    else:
        remaining = int(CHECK_INTERVAL_MIN * 60 - time_since)
        st.caption(f"Next auto-run in {remaining // 60} min {remaining % 60} sec")
else:
    st.info("Auto mode off — use the button below")

if st.button("📡 Run Analysis Now (Live)", type="primary", use_container_width=True):
    run_check()
