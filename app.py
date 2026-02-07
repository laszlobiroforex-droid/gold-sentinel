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
            params["end"] = end_time.isoformat()
        ts = td.time_series(**params)
        ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
        return ts.as_pandas()
    except:
        return None

def fetch_1h(end_time=None, outputsize=60):
    try:
        params = {"symbol": "XAU/USD", "interval": "1h", "outputsize": outputsize}
        if end_time:
            params["end"] = end_time.isoformat()
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

# ─── STRICT LOT SIZE CALCULATION (always Python-side) ──────────────────────
def calculate_strict_lot_size(entry, sl, max_risk_dollars, current_price=None):
    if not all(v is not None for v in [entry, sl, max_risk_dollars]):
        return 0.01, "Missing data — min lot used"

    try:
        entry = float(entry)
        sl = float(sl)
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return 0.01, "Invalid SL — min lot used"

        # Gold: 1 lot = $100 per $1 move → 0.01 lot = $1 per $1 move
        lot_size = max_risk_dollars / (price_diff * 100)
        lot_size_rounded = max(round(lot_size, 2), 0.01)  # never below 0.01

        actual_risk = lot_size_rounded * price_diff * 100
        note = f"Adjusted to fit ${max_risk_dollars:.2f} max risk"
        if actual_risk > max_risk_dollars * 1.05:  # slight tolerance
            note += " — still slightly over (wide SL)"
        return lot_size_rounded, note
    except:
        return 0.01, "Calc error — min lot used"

# ─── MAIN CHECK (supports historical mode) ─────────────────────────────────
def run_check(historical_end_time=None, fake_price=None, fake_utc=None):
    is_historical = historical_end_time is not None

    with st.spinner("Fetching market data..."):
        if is_historical:
            ts_15m = fetch_15m(end_time=historical_end_time, outputsize=80)
            ts_1h  = fetch_1h(end_time=historical_end_time, outputsize=40)
            price = fake_price or (ts_15m['close'].iloc[-1] if ts_15m is not None else None)
            current_time_str = fake_utc or historical_end_time.strftime('%Y-%m-%d %H:%M UTC')
        else:
            price = get_live_price()
            ts_15m = fetch_15m()
            ts_1h  = fetch_1h()
            current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    if price is None:
        st.error("Failed to fetch price")
        return
    if ts_15m is None or ts_15m.empty:
        st.error("No 15m data received")
        return

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

    balance        = st.session_state.get("balance")
    dd_limit       = st.session_state.get("dd_limit")
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct")

    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit and risk_of_dd_pct else 50.0

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

You are a disciplined gold trader. Propose only high-probability setups with confirmation entries.

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

    # ─── STRICT LOT SIZE OVERRIDE ──────────────────────────────────────────────
    best_entry = best_sl = None
    for p in [g_p, k_p, c_p]:
        e = p.get("entry_price") or p.get("entry")
        s = p.get("sl")
        if isinstance(e, (int, float)) and isinstance(s, (int, float)):
            best_entry = e
            best_sl = s
            break  # take first valid

    if best_entry and best_sl:
        lot_size, lot_note = calculate_strict_lot_size(best_entry, best_sl, max_risk_dollars, price)
    else:
        lot_size, lot_note = 0.01, "No valid entry/SL — min lot"

    # ─── CONSENSUS LOGIC ───────────────────────────────────────────────────────
    high_verdicts = [p for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"]]
    if len(high_verdicts) >= 2:
        directions = [p["direction"] for p in high_verdicts if p["direction"] != "NEUTRAL"]
        if len(set(directions)) == 1 and directions:
            direction = directions[0]

            # Collect valid entries
            valid_entries = []
            for p in high_verdicts:
                e = p.get("entry_price") or p.get("entry")
                if isinstance(e, (int, float)) and abs(e - price) / price < 0.05:  # within 5%
                    valid_entries.append({
                        "entry": e,
                        "sl": p.get("sl"),
                        "tp": p.get("tp"),
                        "prob": p.get("estimated_win_prob", 50),
                        "source": p
                    })

            if len(valid_entries) >= 2:
                # Conservative consensus
                entries_sorted = sorted(valid_entries, key=lambda x: x["entry"])
                consensus_entry = entries_sorted[0]["entry"]     # lowest (safest pullback)
                tps = [v["tp"] for v in valid_entries if v["tp"]]
                consensus_tp = np.median(tps) if tps else "—"
                sls = [v["sl"] for v in valid_entries if v["sl"]]
                consensus_sl = min(sls) if sls else "—"          # tightest SL

                msg = (
                    f"**Consensus High Conviction ({len(high_verdicts)}/{len([g_p,k_p,c_p])})**\n"
                    f"Direction: {direction}\n"
                    f"Entry (lowest): LIMIT @ ${consensus_entry:.2f}\n"
                    f"SL (tightest): ${consensus_sl:.2f}\n"
                    f"TP (median): ${consensus_tp:.2f}\n"
                    f"Lot size: {lot_size:.2f} ({lot_note})\n"
                    f"Based on clustered high-prob setups."
                )
                send_telegram(msg, priority="high" if len(high_verdicts) == 3 else "normal")
                st.success("Consensus alert sent!")
            else:
                st.info("No valid clustered entries — no alert")
        else:
            st.info("Direction mismatch — no alert")
    else:
        st.info("No strong consensus — no alert")

    # Display verdicts (same as before, using format_verdict from previous full code)

# ─── HISTORICAL TEST MODE ──────────────────────────────────────────────────
with st.expander("Historical Test Mode (for backtesting past moments)", expanded=False):
    test_date = st.date_input("Test Date", value=datetime.now(timezone.utc).date() - timedelta(days=1))
    test_time = st.time_input("Test Time (UTC)", value=datetime(2025, 2, 6, 5, 27).time())
    test_datetime = datetime.combine(test_date, test_time, tzinfo=timezone.utc)

    if st.button("Run Historical Test at selected time"):
        run_check(historical_end_time=test_datetime)

# Rest of UI (risk settings, auto, manual button) remains the same as in previous full code

# ... (paste the rest of the UI code from your last working version: risk expander, last_check_time, auto checkbox, manual button)
