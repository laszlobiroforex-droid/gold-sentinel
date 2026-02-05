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
COMMISSION_PER_LOT_RT = 1.00
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

# ─── RETRY HELPER ────────────────────────────────────────────────────────────
def retry_api(max_attempts=3, backoff=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    st.error(f"Retry {attempt+1}/{max_attempts} failed: {str(e)}")
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(backoff * (attempt + 1))
            return None
        return wrapper
    return decorator

# ─── QUOTA-SAFE TD CALL WRAPPER ──────────────────────────────────────────────
def safe_td_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "credit" in err_str or "quota" in err_str or "limit" in err_str or "rate" in err_str:
            st.warning("Minutely rate limit hit (8 credits/min). Waiting 60s to reset...")
            time.sleep(60)
            try:
                return func(*args, **kwargs)  # One retry after wait
            except:
                st.error("Still limited after wait. Try again in a minute or upgrade plan.")
                return None
        raise e  # Other errors get normal retry

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    grok_client = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

except Exception as e:
    st.error(f"API setup failed: {e}")
    st.stop()

# ─── TELEGRAM SENDER ─────────────────────────────────────────────────────────
def send_telegram(message: str, priority: str = "normal"):
    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# ─── ECONOMIC CALENDAR SCRAPE ────────────────────────────────────────────────
@retry_api()
@st.cache_data(ttl=900)
def fetch_upcoming_events():
    url = "https://www.investing.com/economic-calendar/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        events = []
        rows = soup.find_all("tr", class_="js-event-item")
        now_utc = datetime.now(timezone.utc)

        for row in rows:
            time_str = row.find("td", class_="time").get_text(strip=True)
            if "All Day" in time_str or not time_str:
                continue
            try:
                event_time = datetime.strptime(time_str, "%b %d, %Y %H:%M").replace(tzinfo=timezone.utc)
                if event_time < now_utc - timedelta(days=1):
                    event_time += timedelta(days=365)
            except:
                continue

            currency = row.find("td", class_="flagCur").get_text(strip=True)
            impact_icons = row.find("td", class_="sentiment").find_all("i", class_="grayFull")
            impact_level = 3 - len([i for i in impact_icons if "gray" in i["class"]])

            event_name = row.find("td", class_="event").get_text(strip=True)

            if impact_level >= 2 and (currency in ["USD", "XAU", "ALL"] or 
                                      any(kw in event_name.lower() for kw in ["fed", "cpi", "nfp", "payroll", "fomc", "rate", "inflation", "geopol"])):
                events.append({
                    "time": event_time,
                    "name": event_name,
                    "currency": currency,
                    "impact": "High" if impact_level == 3 else "Medium",
                    "minutes_away": (event_time - now_utc).total_seconds() / 60
                })

        return sorted(events, key=lambda x: x["time"])[:8]
    except Exception as e:
        st.warning(f"Calendar scrape failed: {e}. No event data.")
        return []

def get_relevant_event_warning():
    events = fetch_upcoming_events()
    if not events:
        return "", None

    now = datetime.now(timezone.utc)
    reminder_event = None
    warning_text = ""

    for ev in events:
        mins = ev["minutes_away"]
        if 15 <= mins <= 45:
            reminder_event = ev
        if 0 < mins <= 120:
            warning_text += f"⚠️ {ev['impact']} event in \~{int(mins)} min: {ev['name']} ({ev['currency']})\n"

    return warning_text.strip(), reminder_event

def send_event_reminder_if_needed():
    _, reminder_ev = get_relevant_event_warning()
    if reminder_ev:
        msg = f"🚨 REMINDER: High-impact event approaching in \~{int(reminder_ev['minutes_away'])} min!\n{reminder_ev['name']} ({reminder_ev['currency']})\nConsider closing Gold positions / avoiding new entries."
        send_telegram(msg, priority="high")

# ─── FRACTALS & INVALIDATION ─────────────────────────────────────────────────
def get_fractal_levels(df, window=5, min_dist_factor=0.4):
    if df.empty:
        return []
    levels = []
    atr_approx = (df['high'] - df['low']).rolling(14).mean().iloc[-1] or 10.0
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

def check_opposing_invalidation(levels, direction, entry, atr):
    threshold = atr * OPPOSING_FRACTAL_ATR_MULT
    if "BULL" in direction:
        close_res = [p for t, p in levels if t == 'RES' and entry < p < entry + threshold]
        return bool(close_res)
    elif "BEAR" in direction:
        close_sup = [p for t, p in levels if t == 'SUP' and entry - threshold < p < entry]
        return bool(close_sup)
    return False

# ─── DATA FETCH ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
@retry_api()
def fetch_15m_data():
    ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200)
    ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
    return ts.as_pandas()

@st.cache_data(ttl=300)
@retry_api()
def fetch_htf_data(interval):
    out_size = 200 if interval == "5min" else 100
    ts = td.time_series(symbol="XAU/USD", interval=interval, outputsize=out_size)
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

@st.cache_data(ttl=60)
@retry_api()
def get_current_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

# ─── TD USAGE CHECK ──────────────────────────────────────────────────────────
def get_td_usage():
    try:
        resp = requests.get(f"https://api.twelvedata.com/api_usage?apikey={st.secrets['TWELVE_DATA_KEY']}")
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# ─── AI ADVICE ───────────────────────────────────────────────────────────────
@retry_api()
def get_ai_advice(market, setup, levels, buffer, aligned_1h, aligned_5m):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[:6]]) if levels else "No clear levels"
    current_price = market['price']

    base_prompt = f"""
You are a high-conviction gold trading auditor.
Evaluate ANY high-edge pattern: continuation pullbacks (scalp or swing), momentum breakouts, reversals at key levels, range mean-reversion, etc.
Prioritize RR ≥ 1.8, strong confluence, clean risk.
Skip if low quality, chasing, or gamble.

Current UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} (watch whipsaw after 21:30 UTC)
Market: ${current_price:.2f}, RSI {market['rsi']:.1f}
Trend: 15m vs 1H EMA200: {'agree' if aligned_1h else 'disagree'}, 15m vs 5M EMA200: {'agree' if aligned_5m else 'disagree'}
Fractals: {levels_str}
Setup context: entry \~${setup['entry']:.2f}, ATR ${setup['atr']:.2f}, risk buffer ${buffer:.2f}

Respond ONLY with valid JSON:
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

    try:
        g_out = gemini_model.generate_content(base_prompt).text.strip()
    except:
        g_out = '{"verdict":"SKIP","reason":"Gemini offline"}'

    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": base_prompt + "\nBe concise and factual."}],
            max_tokens=300,
            temperature=0.4
        )
        k_out = r.choices[0].message.content.strip()
    except:
        k_out = '{"verdict":"SKIP","reason":"Grok error"}'

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": base_prompt + "\nOutput strict JSON only, no extra text."}],
            max_tokens=300,
            temperature=0.5
        )
        c_out = response.choices[0].message.content.strip()
    except:
        c_out = '{"verdict":"SKIP","reason":"ChatGPT error"}'

    return g_out, k_out, c_out

# ─── PARSE AI OUTPUT ─────────────────────────────────────────────────────────
def parse_ai_output(text):
    try:
        cleaned = text.strip('```json\n').strip('```').strip()
        data = json.loads(cleaned)
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
    except:
        return {"verdict": "UNKNOWN", "reason": "Parsing failed"}

# ─── LAST ALERT PERSISTENCE ──────────────────────────────────────────────────
def load_last_alert():
    if os.path.exists(LAST_ALERT_FILE):
        try:
            with open(LAST_ALERT_FILE, "r") as f:
                return json.load(f)
        except:
            return {"time": 0, "key": None, "proposal": {}}
    return {"time": 0, "key": None, "proposal": {}}

def save_last_alert(time_val, key, proposal):
    try:
        with open(LAST_ALERT_FILE, "w") as f:
            json.dump({"time": time_val, "key": key, "proposal": proposal}, f)
    except:
        pass

# ─── CORE CHECK FUNCTION ─────────────────────────────────────────────────────
def check_for_high_conviction_setup():
    last = load_last_alert()
    now_ts = time.time()
    if now_ts - last["time"] < ALERT_COOLDOWN_MIN * 60:
        return

    try:
        warning_text, reminder_ev = get_relevant_event_warning()
        if reminder_ev:
            send_event_reminder_if_needed()

        events = fetch_upcoming_events()
        imminent = any(0 < ev["minutes_away"] < 60 for ev in events if ev["impact"] == "High")
        if imminent:
            return

        live_price = safe_td_call(lambda: float(td.price(symbol="XAU/USD").as_json()["price"]))
        if live_price is None:
            return
        time.sleep(15)

        ts_15m = safe_td_call(lambda: td.time_series(symbol="XAU/USD", interval="15min", outputsize=200)
                              .with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
                              .as_pandas())
        if ts_15m is None or ts_15m.empty:
            return
        time.sleep(15)

        latest_15m = ts_15m.iloc[-1]
        rsi = latest_15m.get('rsi', 50.0)
        atr = latest_15m.get('atr', 10.0)

        market = {"price": live_price, "rsi": rsi}
        setup = {"entry": live_price, "sl_distance": atr*1.5, "atr": atr, "risk_pct": DEFAULT_RISK_PCT}
        levels = get_fractal_levels(ts_15m)
        buffer = DEFAULT_BALANCE - DEFAULT_FLOOR

        try:
            balance = st.session_state.get("balance", DEFAULT_BALANCE)
            floor = st.session_state.get("floor", DEFAULT_FLOOR)
            daily_limit = st.session_state.get("daily_limit", None)
            risk_pct = st.session_state.get("risk_pct", DEFAULT_RISK_PCT)
            buffer = balance - floor
        except:
            pass

        ts_1h = safe_td_call(lambda: td.time_series(symbol="XAU/USD", interval="1h", outputsize=100)
                             .with_ema(time_period=200).as_pandas())
        if ts_1h is None:
            return
        time.sleep(15)

        ts_5m = safe_td_call(lambda: td.time_series(symbol="XAU/USD", interval="5min", outputsize=200)
                             .with_ema(time_period=200).as_pandas())
        if ts_5m is None:
            return

        ema200_15m = latest_15m.get('ema_200', live_price)
        ema200_1h = ts_1h.iloc[-1].get('ema_200', live_price) if not ts_1h.empty else live_price
        ema200_5m = ts_5m.iloc[-1].get('ema_200', live_price) if not ts_5m.empty else live_price

        aligned_1h = (live_price > ema200_15m) == (live_price > ema200_1h)
        aligned_5m = (live_price > ema200_15m) == (live_price > ema200_5m)

        g_raw, k_raw, c_raw = get_ai_advice(market, setup, levels, buffer, aligned_1h, aligned_5m)
        g_p = parse_ai_output(g_raw)
        k_p = parse_ai_output(k_raw)
        c_p = parse_ai_output(c_raw)

        high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

        if high_count >= MIN_CONVICTION_FOR_ALERT:
            p = g_p
            direction = p.get("direction", "NEUTRAL")
            style = p.get("style", "NONE")
            entry = p.get("entry") or live_price
            proposed_sl = p.get("sl")
            proposed_tp = p.get("tp")
            rr = p.get("rr") or 2.0

            default_sl_dist = atr * 1.5
            if proposed_sl:
                sl_dist = abs(entry - proposed_sl)
                sl_dist = min(sl_dist, atr * MAX_SL_ATR_MULT)
                sl = entry - sl_dist if "BULL" in direction else entry + sl_dist
            else:
                sl = entry - default_sl_dist if "BULL" in direction else entry + default_sl_dist

            tp = proposed_tp or (entry + atr*3 if "BULL" in direction else entry - atr*3)

            invalidated = check_opposing_invalidation(levels, direction, entry, atr)
            if invalidated:
                high_count -= 1
                if high_count < MIN_CONVICTION_FOR_ALERT:
                    return

            cash_risk = min(buffer * (risk_pct / 100), daily_limit or buffer)
            risk_dist = abs(entry - sl) + SPREAD_BUFFER_POINTS + SLIPPAGE_BUFFER_POINTS
            risk_per_lot = risk_dist * PIP_VALUE
            lots = cash_risk / risk_per_lot if risk_per_lot > 0 else 0.01
            lots = max(0.01, round(lots / 0.01) * 0.01)
            actual_risk = lots * risk_per_lot

            key = f"{style}_{direction}_{entry:.2f}_{sl:.2f}"
            if key == last["key"]:
                return

            event_warning = f"\n\n{warning_text}" if warning_text else ""
            msg = (
                f"**High Conviction Setup!** ({high_count}/3)\n"
                f"Style: {style}\nDirection: {direction}\n"
                f"Entry: ${entry:.2f}\nSL: ${sl:.2f}\nTP: ${tp:.2f} (R:R \~1:{rr:.1f})\n"
                f"**Proposed lots: {lots:.2f}** (risk \~${actual_risk:.0f})\n"
                f"Price: ${live_price:.2f} | RSI {rsi:.1f}{event_warning}"
            )
            priority = "high" if high_count == 3 else "normal"
            send_telegram(msg, priority)

            proposal_data = {
                "time": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                "style": style,
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "lots": lots,
                "risk": actual_risk,
                "event_warning": warning_text
            }
            save_last_alert(now_ts, key, proposal_data)

    except Exception as e:
        st.error(f"Check failed: {str(e)}")

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive engine | Safe auto-check while page open | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# Auto-check controls (always visible)
auto_enabled = st.checkbox("Enable auto-check while this page is open", value=False)  # Default off for safety
auto_interval_min = st.slider("Check every (minutes)", 10, 60, 15, step=5)

# Usage monitor
if st.button("🔍 Check Twelve Data Usage (costs 1 credit)"):
    usage = get_td_usage()
    st.json(usage)

# Settings screen
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = DEFAULT_BALANCE
    st.session_state.daily_limit = DEFAULT_DAILY_LIMIT
    st.session_state.floor = DEFAULT_FLOOR
    st.session_state.risk_pct = DEFAULT_RISK_PCT
    st.session_state.last_analysis = 0
    st.session_state.last_check_time = 0

if not st.session_state.analysis_done:
    st.header("Account Settings (RF Bronze 5K friendly)")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.balance = st.number_input("Balance ($)", min_value=0.0, value=st.session_state.balance, format="%.2f")
    with col2:
        st.session_state.daily_limit = st.number_input("Daily Limit ($)", min_value=0.0, value=st.session_state.daily_limit or 0.0, format="%.2f")

    st.session_state.floor = st.number_input("Floor ($)", value=st.session_state.floor, format="%.2f")
    st.session_state.risk_pct = st.slider("Risk % per trade", 5, 50, st.session_state.risk_pct, step=5)

    if st.button("🚀 Analyze & Suggest", type="primary"):
        if st.session_state.balance <= 0:
            st.error("Valid balance required")
        elif time.time() - st.session_state.last_analysis < 60:
            st.warning("Wait 60s")
        else:
            st.session_state.last_analysis = time.time()
            st.session_state.analysis_done = True
            st.rerun()

    if st.button("📡 Manual Alert Check Now (\~4 credits)"):
        with st.spinner("Running manual check..."):
            check_for_high_conviction_setup()
        st.success("Manual check done. See Telegram if setup found.")
else:
    cols = st.columns(5)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "Unlimited")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")

    last = load_last_alert()
    if last.get("proposal"):
        p = last["proposal"]
        with st.expander("**Last High-Conviction Alert**", expanded=True):
            st.markdown(f"**Time:** {p.get('time', 'unknown')}")
            st.markdown(f"**Style:** {p.get('style')} | **Direction:** {p.get('direction')}")
            col1, col2 = st.columns(2)
            col1.metric("Entry", f"${p.get('entry', 0):.2f}")
            col2.metric("SL / TP", f"${p.get('sl', 0):.2f} → ${p.get('tp', 0):.2f}")
            st.metric("Proposed Lots", f"{p.get('lots', 0.01):.2f}", f"Risk \~${p.get('risk', 0):.0f}")
            rr_value = p.get('rr', '?')
            st.caption(f"R:R \~1:{rr_value:.1f}" if isinstance(rr_value, (int, float)) else f"R:R \~1:{rr_value}")
            if p.get("event_warning"):
                st.warning(p["event_warning"])

    utc_now = datetime.now(timezone.utc)
    if utc_now.hour >= 21 and utc_now.hour < 23:
        st.warning("Late NY session — whipsaw risk ↑")

    # AI raw outputs (collapsible)
    with st.expander("Raw AI Outputs (debug)", expanded=False):
        cols = st.columns(3)
        cols[0].markdown("**Gemini**")
        cols[0].json(g_raw if 'g_raw' in locals() else "No data yet")
        cols[1].markdown("**Grok**")
        cols[1].json(k_raw if 'k_raw' in locals() else "No data yet")
        cols[2].markdown("**ChatGPT**")
        cols[2].json(c_raw if 'c_raw' in locals() else "No data yet")

# ─── SAFE AUTO-CHECK TIMER (runs on EVERY script execution) ───────────────────
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

# Manual alert button (always available)
if st.button("📡 Manual Alert Check Now (\~4 credits)"):
    with st.spinner("Running manual check..."):
        check_for_high_conviction_setup()
    st.success("Manual check done. See Telegram if setup found.")

# Reset button
if st.button("Reset to Settings"):
    st.session_state.analysis_done = False
    st.rerun()
