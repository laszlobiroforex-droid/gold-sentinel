import streamlit as st
import json
from datetime import datetime, timezone, timedelta
import time
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
CHECK_INTERVAL_MIN = 30

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

def fetch_15m(end_time=None, outputsize=120):
    try:
        params = {"symbol": "XAU/USD", "interval": "15min", "outputsize": outputsize}
        if end_time:
            # Buffer end_time to include last closed candle
            params["end"] = (end_time + timedelta(minutes=15)).isoformat()
            params["start"] = (end_time - timedelta(hours=6)).isoformat()  # enough for indicators
        ts = td.time_series(**params)
        ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
        return ts.as_pandas()
    except Exception as e:
        st.warning(f"15m fetch failed: {str(e)}")
        return None

def fetch_1h(end_time=None, outputsize=60):
    try:
        params = {"symbol": "XAU/USD", "interval": "1h", "outputsize": outputsize}
        if end_time:
            params["end"] = (end_time + timedelta(hours=1)).isoformat()
            params["start"] = (end_time - timedelta(days=2)).isoformat()
        ts = td.time_series(**params)
        ts = ts.with_ema(time_period=200)
        return ts.as_pandas()
    except:
        return None

# ─── PARSE AI OUTPUT ───────────────────────────────────────────────────────
def parse_ai_output(text):
    if not text or not isinstance(text, str):
        return {"verdict": "UNKNOWN", "reason": "No output"}

    start = text.find('{')
    end   = text.rfind('}') + 1
    if start == -1 or end == 0:
        return {"verdict": "PARSE_ERROR", "reason": "No JSON found"}

    json_str = text[start:end].strip().replace('```json', '').replace('```', '').strip()

    try:
        data = json.loads(json_str)
        return {
            "verdict": data.get("verdict", "UNKNOWN").upper(),
            "reason": data.get("reason", "No reason"),
            "entry_type": data.get("entry_type"),
            "entry_price": data.get("entry_price"),
            "sl": data.get("sl"),
            "tp": data.get("tp"),
            "rr": data.get("rr"),
            "estimated_win_prob": data.get("estimated_win_prob"),
            "style": data.get("style", "NONE").upper(),
            "direction": data.get("direction", "NEUTRAL").upper(),
            "reasoning": data.get("reasoning", "")
        }
    except Exception as e:
        return {
            "verdict": "PARSE_ERROR",
            "reason": f"Invalid JSON: {str(e)}. Raw: {text[:150]}..."
        }

# ─── STRICT LOT SIZE CALCULATION ───────────────────────────────────────────
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

# ─── MAIN ANALYSIS FUNCTION ────────────────────────────────────────────────
def run_check(historical_end_time=None):
    is_historical = historical_end_time is not None

    with st.spinner("Fetching data..."):
        if is_historical:
            ts_15m = fetch_15m(end_time=historical_end_time)
            ts_1h  = fetch_1h(end_time=historical_end_time)
            if ts_15m is None or ts_15m.empty:
                st.error("No historical 15m data returned for selected time")
                return
            price = ts_15m['close'].iloc[-1]
            current_time_str = historical_end_time.strftime('%Y-%m-%d %H:%M UTC')
            st.info(f"Historical mode: simulating {current_time_str} | Last price used: ${price:.2f}")
            st.write("Last 3 candles:", ts_15m.tail(3)[['close', 'rsi', 'ema_50', 'ema_200', 'atr']])
        else:
            price = get_live_price()
            ts_15m = fetch_15m()
            ts_1h  = fetch_1h()
            if price is None:
                st.error("Failed to fetch live price")
                return
            current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # ─── INDICATORS & LEVELS ───────────────────────────────────────────────────
    latest_15m = ts_15m.iloc[-1]
    rsi = latest_15m.get('rsi', 50.0)
    atr = latest_15m.get('atr', 10.0)
    ema200_15m = latest_15m.get('ema_200', price)
    ema50_15m  = latest_15m.get('ema_50',  price)

    ema200_1h = ts_1h.iloc[-1].get('ema_200', price) if ts_1h is not None and not ts_1h.empty else price

    levels = []
    for i in range(5, min(40, len(ts_15m)-5)):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    balance        = st.session_state.get("balance")
    dd_limit       = st.session_state.get("dd_limit")
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct", 25.0)

    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit else 50.0

    prompt = f"""
Current UTC: {current_time_str}

Market data:
Price: ${price:.2f}
RSI (15m): {rsi:.1f}
ATR (15m): {atr:.2f}
EMA50 / EMA200 (15m): {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 (1h): {ema200_1h:.2f}

Recent support/resistance fractals: {', '.join([f"{t}@{p}" for t,p in levels[-8:]]) or 'None'}

Account risk limit: max ${max_risk_dollars:.2f} loss per trade (preferred)

Propose high-prob setups with confirmation entries only.
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

    # Strict lot calculation (Python overrides AI)
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

    # ─── DISPLAY VERDICTS (use previous robust format_verdict) ──────────────
    # ... paste your existing format_verdict function here (the one with isinstance checks)

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(format_verdict(g_p, "Gemini",   lot_size, lot_note), unsafe_allow_html=True)
    with col2: st.markdown(format_verdict(k_p, "Grok",     lot_size, lot_note), unsafe_allow_html=True)
    with col3: st.markdown(format_verdict(c_p, "ChatGPT",  lot_size, lot_note), unsafe_allow_html=True)

    # ─── CONSENSUS & TELEGRAM (conservative picking) ─────────────────────────
    high_verdicts = [p for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"]]
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
                consensus_entry = entries_sorted[0]["entry"]  # lowest/safest
                tps = [v["tp"] for v in valid_entries if isinstance(v["tp"], (int, float))]
                consensus_tp = np.median(tps) if tps else "—"
                sls = [v["sl"] for v in valid_entries if isinstance(v["sl"], (int, float))]
                consensus_sl = min(sls) if sls else "—"

                msg = (
                    f"**Consensus High Conviction ({len(high_verdicts)} AIs)**\n"
                    f"Direction: {direction}\n"
                    f"Entry (lowest): LIMIT @ ${consensus_entry:.2f}\n"
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

# Risk settings expander (unchanged)
with st.expander("Prop Challenge / Risk Settings (required for lot sizing)", expanded=True):
    # ... your existing code for balance, dd_limit, risk_pct ...

# ─── HISTORICAL TEST MODE (optional) ───────────────────────────────────────
with st.expander("Historical Test Mode (optional backtesting)", expanded=False):
    st.info("Select a PAST date/time to simulate market state exactly then. Uses only data available up to that moment.")
    
    col1, col2 = st.columns(2)
    with col1:
        test_date = st.date_input("Test Date", value=datetime.now(timezone.utc).date() - timedelta(days=1))
    with col2:
        test_time = st.time_input("Test Time (UTC)", value=datetime.strptime("05:27", "%H:%M").time())

    test_datetime = datetime.combine(test_date, test_time, tzinfo=timezone.utc)

    if st.button("Run Historical Test"):
        if test_datetime >= datetime.now(timezone.utc):
            st.error("Cannot test future or current time — pick a past moment.")
        else:
            run_check(historical_end_time=test_datetime)

# ─── AUTO / MANUAL CONTROLS ────────────────────────────────────────────────
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

auto_enabled = st.checkbox(f"Auto-run every {CHECK_INTERVAL_MIN} minutes (keep tab open)", value=False)

if auto_enabled:
    now = time.time()
    time_since = now - st.session_state.last_check_time
    if time_since >= CHECK_INTERVAL_MIN * 60:
        st.session_state.last_check_time = now
        run_check()  # live mode
    else:
        remaining = int(CHECK_INTERVAL_MIN * 60 - time_since)
        st.caption(f"Next auto-run in {remaining // 60} min {remaining % 60} sec")
else:
    st.info("Auto mode off — use the button below")

if st.button("📡 Run Analysis Now (Live)", type="primary", use_container_width=True):
    run_check()  # live mode
