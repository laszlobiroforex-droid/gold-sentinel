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

ALERT_COOLDOWN_MIN = 30
MIN_CONVICTION_FOR_ALERT = 2     # 2 or 3 AIs saying ELITE/HIGH_CONV

# ─── DEFAULT ACCOUNT VALUES ──────────────────────────────────────────────────
DEFAULT_BALANCE = 5000.0
DEFAULT_DAILY_LIMIT = 250.0
DEFAULT_FLOOR = 4500.0
DEFAULT_RISK_PCT = 25
DEFAULT_MODE = "Standard (Swing – 15m + 1h alignment)"

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
    st.error(f"Critical: API setup failed → {e}\nCheck Streamlit secrets.")
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
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=8)
    except:
        pass

# ─── FRACTAL LEVELS ──────────────────────────────────────────────────────────
def get_fractal_levels(df, window=5, min_dist_factor=0.4):
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

# ─── FIXED DATA FETCH ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
@retry_api()
def fetch_15m_data():
    ts = td.time_series(
        symbol="XAU/USD",
        interval="15min",
        outputsize=500
    )
    ts = ts.with_rsi()                     # default period=14
    ts = ts.with_ema(time_period=200)
    ts = ts.with_ema(time_period=50)
    ts = ts.with_atr(time_period=14)
    return ts.as_pandas()

@st.cache_data(ttl=300)
@retry_api()
def fetch_htf_data(interval):
    out_size = 500 if interval == "5min" else 200
    ts = td.time_series(
        symbol="XAU/USD",
        interval=interval,
        outputsize=out_size
    )
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

@st.cache_data(ttl=60)
@retry_api()
def get_current_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

# ─── AI ADVICE ───────────────────────────────────────────────────────────────
@retry_api()
def get_ai_advice(market, setup, levels, buffer, mode):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[:6]]) if levels else "No clear levels"
    current_price = market['price']
    prompt = f"""
You are a high-conviction gold trading auditor for any account size.
Mode: {mode} ({'standard swing (15m + 1h)' if mode == 'Standard' else 'fast scalp (15m + 5m)'}).
Aggressive risk is user's choice — size is handled externally, do NOT suggest lots multiplier or position size changes.
Focus on math, pullback quality, structural confluence, risk/reward.
IMPORTANT: For buys, SL must be BELOW entry. For sells, SL must be ABOVE entry.

Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
NY session close \~22:00 UTC — factor in thinning liquidity and whipsaw risk after 21:30 UTC.
If any high-impact news likely within ±30 min, prefer to wait unless setup is exceptionally strong.

Current market price: ${current_price:.2f}
Buffer left: ${buffer:.2f}
Market: Price ${current_price:.2f}, RSI {market['rsi']:.1f}
Original setup: {setup['type']} at ${setup['entry']:.2f}, SL distance ${setup['sl_distance']:.2f}, ATR ${setup['atr']:.2f}, risk % {setup['risk_pct']:.0f}%
Fractals: {levels_str}

Be STRICTLY consistent:
- If original setup is low-edge, obsolete, missed, gamble, or chasing, your proposal MUST NOT re-use or slightly adjust the original entry price.
- Any proposal MUST respect current market price ${current_price:.2f} — never suggest entries significantly below current price in bullish mode or above in bearish mode unless clear reversal evidence exists.
- Direction must match detected bias unless verdict explicitly states "reversal".
- Only propose changes that meaningfully improve the setup (e.g. higher entry in continuation, different SL/TP, or skip).
- If no good alternative exists, clearly recommend skipping.

Respond ONLY in this exact structured format. Do not add extra text.

VERDICT: ELITE | HIGH_CONV | LOW_EDGE | GAMBLE | SKIP
REASON: [short explanation, 1-2 sentences]
PROPOSAL: [entry price] | [SL price] | [TP price] | [RR ratio e.g. 2.5] | [direction: BULLISH/BEARISH/NEUTRAL] | [reasoning, 1-2 sentences]
or
PROPOSAL: NONE
"""

    try:
        g_out = gemini_model.generate_content(prompt).text.strip()
    except:
        g_out = "VERDICT: SKIP\nREASON: Gemini Offline.\nPROPOSAL: NONE"

    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220
        )
        k_out = r.choices[0].message.content.strip()
    except:
        k_out = "VERDICT: SKIP\nREASON: Grok Error.\nPROPOSAL: NONE"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.65
        )
        c_out = response.choices[0].message.content.strip()
    except:
        c_out = "VERDICT: SKIP\nREASON: ChatGPT Error.\nPROPOSAL: NONE"

    return g_out, k_out, c_out

# ─── PARSE AI OUTPUT ─────────────────────────────────────────────────────────
def parse_ai_output(text):
    verdict = "UNKNOWN"
    reason = ""
    proposal = "NONE"
    entry = sl = tp = rr = direction = None

    v_match = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    if v_match:
        verdict = v_match.group(1).upper()

    r_match = re.search(r"REASON:\s*(.+?)(?=PROPOSAL:|$)", text, re.DOTALL | re.IGNORECASE)
    if r_match:
        reason = r_match.group(1).strip()

    p_match = re.search(r"PROPOSAL:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if p_match:
        proposal_raw = p_match.group(1).strip()
        if proposal_raw.upper() != "NONE":
            parts = [p.strip() for p in proposal_raw.split("|")]
            if len(parts) >= 5:
                try:
                    entry = float(parts[0])
                    sl = float(parts[1])
                    tp = float(parts[2])
                    rr = float(parts[3])
                    direction = parts[4].strip().upper()
                    proposal = "PROPOSAL"
                except:
                    pass

    return {
        "verdict": verdict,
        "reason": reason,
        "proposal": proposal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "direction": direction
    }

# ─── ALERT CHECK ─────────────────────────────────────────────────────────────
def check_for_high_conviction_setup():
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0
    if "last_alerted_key" not in st.session_state:
        st.session_state.last_alerted_key = None

    now = time.time()
    if now - st.session_state.last_alert_time < ALERT_COOLDOWN_MIN * 60:
        return

    try:
        live_price = get_current_price()
        ts_15m = fetch_15m_data()
        latest_15m = ts_15m.iloc[-1]
        rsi = latest_15m.get('rsi', 50.0)
        atr = latest_15m.get('atr', 10.0)

        market = {"price": live_price, "rsi": rsi}
        setup = {"type": "UNKNOWN", "entry": live_price, "sl_distance": atr*1.5, "atr": atr, "risk_pct": DEFAULT_RISK_PCT}
        levels = get_fractal_levels(ts_15m)
        buffer = DEFAULT_BALANCE - DEFAULT_FLOOR

        g_raw, k_raw, c_raw = get_ai_advice(market, setup, levels, buffer, DEFAULT_MODE)
        g_p = parse_ai_output(g_raw)
        k_p = parse_ai_output(k_raw)
        c_p = parse_ai_output(c_raw)

        high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

        if high_count >= MIN_CONVICTION_FOR_ALERT:
            direction = g_p.get("direction") or "UNKNOWN"
            entry = g_p.get("entry") or live_price
            sl = g_p.get("sl") or (entry - atr*1.5 if direction.startswith("BULL") else entry + atr*1.5)
            tp = g_p.get("tp") or (entry + atr*3 if direction.startswith("BULL") else entry - atr*3)
            rr = g_p.get("rr") or 2.0

            key = f"{direction}_{entry:.2f}_{sl:.2f}"
            if st.session_state.last_alerted_key == key:
                return

            msg = (
                f"**High Conviction Setup!** ({high_count}/3 AIs)\n"
                f"Direction: {direction}\n"
                f"Entry: ${entry:.2f}\n"
                f"SL: ${sl:.2f}\n"
                f"TP: ${tp:.2f}   (R:R \~1:{rr:.1f})\n"
                f"Current price: ${live_price:.2f} | RSI {rsi:.1f}"
            )
            priority = "high" if high_count == 3 else "normal"
            send_telegram(msg, priority)

            st.session_state.last_alert_time = now
            st.session_state.last_alerted_key = key

    except Exception:
        pass

# ─── BACKGROUND THREAD ───────────────────────────────────────────────────────
def run_background_checker():
    schedule.every(5).minutes.do(check_for_high_conviction_setup)
    while True:
        schedule.run_pending()
        time.sleep(1)

if "checker_started" not in st.session_state:
    st.session_state.checker_started = True
    t = threading.Thread(target=run_background_checker, daemon=True)
    t.start()

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# Session state defaults
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = DEFAULT_BALANCE
    st.session_state.daily_limit = DEFAULT_DAILY_LIMIT
    st.session_state.floor = DEFAULT_FLOOR
    st.session_state.risk_pct = DEFAULT_RISK_PCT
    st.session_state.mode = DEFAULT_MODE
    st.session_state.last_analysis = 0

# ─── INPUTS ──────────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    st.header("Account Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.balance = st.number_input(
            "Current Balance ($)", min_value=0.0, value=st.session_state.balance,
            format="%.2f"
        )
    with col2:
        st.session_state.daily_limit = st.number_input(
            "Daily Drawdown Limit ($)", min_value=0.0, value=st.session_state.daily_limit or 0.0,
            format="%.2f", help="Set 0 for no daily limit"
        )

    st.session_state.floor = st.number_input(
        "Survival Floor / Max DD ($)", value=st.session_state.floor, format="%.2f"
    )

    st.session_state.risk_pct = st.slider(
        "Risk % of Available Buffer", 5, 50, st.session_state.risk_pct, step=5
    )

    st.session_state.mode = st.radio(
        "Analysis Mode",
        ["Standard (Swing – 15m + 1h alignment)", "Scalp (Fast – 15m + 5m alignment)"],
        index=0 if st.session_state.mode.startswith("Standard") else 1
    )

    if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
        if st.session_state.balance <= 0:
            st.error("Enter valid balance > 0")
        elif time.time() - st.session_state.last_analysis < 60:
            st.warning("Wait 60s between analyses")
        else:
            st.session_state.last_analysis = time.time()
            st.session_state.analysis_done = True
            st.rerun()

else:
    st.info("Analysis locked — using your settings:")
    cols = st.columns(5)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "No limit")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")
    cols[4].metric("Mode", st.session_state.mode.split(" – ")[0])

    utc_now = datetime.now(timezone.utc)
    if utc_now.hour >= 21 and utc_now.hour < 23:
        st.warning("Late NY session — higher whipsaw risk after \~21:30 UTC")

    with st.spinner("Running triple AI audit first..."):
        try:
            live_price = get_current_price()
            ts_15m = fetch_15m_data()
            latest_15m = ts_15m.iloc[-1]
            rsi = latest_15m.get('rsi', 50.0)
            atr = latest_15m.get('atr', 10.0) or 10.0

            market = {"price": live_price, "rsi": rsi}
            levels = get_fractal_levels(ts_15m)
            buffer = st.session_state.balance - st.session_state.floor

            setup_placeholder = {
                "type": "PULLBACK",
                "entry": live_price,
                "sl_distance": atr * 1.5,
                "atr": atr,
                "risk_pct": st.session_state.risk_pct
            }

            g_raw, k_raw, c_raw = get_ai_advice(
                market, setup_placeholder, levels, buffer, st.session_state.mode
            )

            g_parsed = parse_ai_output(g_raw)
            k_parsed = parse_ai_output(k_raw)
            c_parsed = parse_ai_output(c_raw)

            st.divider()
            st.subheader("Triple AI Opinions (primary filter)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Gemini**")
                st.info(g_raw)
            with col2:
                st.markdown("**Grok**")
                st.info(k_raw)
            with col3:
                st.markdown("**ChatGPT**")
                st.info(c_raw)

            parsed_verdicts = [g_parsed["verdict"], k_parsed["verdict"], c_parsed["verdict"]]
            elite_high_count = sum(1 for v in parsed_verdicts if v in ["ELITE", "HIGH_CONV"])
            skip_count = sum(1 for v in parsed_verdicts if v == "SKIP")

            if elite_high_count == 3:
                st.success("3/3 High Conviction – Very strong signal")
            elif elite_high_count == 2:
                st.info("2/3 High Conviction – Reasonable edge")
            elif skip_count >= 2:
                st.warning(f"{skip_count}/3 Skip votes – Proceed with caution")
            else:
                st.markdown("Mixed AI views – review carefully")

            # ── Your original logic (secondary) ─────────────────────────────────
            st.divider()
            st.subheader("Your Structured Setup (secondary check)")
            try:
                interval = "1h" if st.session_state.mode.startswith("Standard") else "5min"
                ts_htf = fetch_htf_data(interval)

                latest_htf = ts_htf.iloc[-1]
                ema_cols_15m = sorted([c for c in ts_15m.columns if 'ema' in c.lower()])
                ema200_15m = latest_15m[ema_cols_15m[0]] if ema_cols_15m else live_price
                ema50_15m  = latest_15m[ema_cols_15m[1]] if len(ema_cols_15m) >= 2 else live_price

                ema_cols_htf = [c for c in ts_htf.columns if 'ema' in c.lower()]
                ema200_htf = latest_htf[sorted(ema_cols_htf)[0]] if ema_cols_htf else live_price

                aligned = (live_price > ema200_15m and live_price > ema200_htf) or \
                          (live_price < ema200_15m and live_price < ema200_htf)

                if not aligned:
                    st.warning("Trend misalignment between 15m and HTF EMA200 → no high-conviction continuation trade recommended")
                else:
                    bias = "BULLISH" if live_price > ema200_15m else "BEARISH"

                    resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
                    supports    = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

                    sl_dist = round(atr * 1.5, 2)
                    min_sl_distance = atr * 0.4

                    if bias == "BULLISH":
                        original_entry = ema50_15m if (live_price - ema50_15m) > (atr * 0.5) else live_price
                        valid_sup = [s for s in supports if s < original_entry]
                        candidate_sl = valid_sup[0] - (0.3 * atr) if valid_sup else original_entry - sl_dist
                        original_sl = min(candidate_sl, original_entry - min_sl_distance)
                        original_tp = resistances[0] if resistances else original_entry + (sl_dist * 2.5)
                        original_action = "BUY AT MARKET" if original_entry == live_price else "BUY LIMIT ORDER"
                    else:
                        original_entry = ema50_15m if (ema50_15m - live_price) > (atr * 0.5) else live_price
                        valid_res = [r for r in resistances if r > original_entry]
                        candidate_sl = valid_res[0] + (0.3 * atr) if valid_res else original_entry + sl_dist
                        original_sl = max(candidate_sl, original_entry + min_sl_distance)
                        original_tp = supports[0] if supports else original_entry - (sl_dist * 2.5)
                        original_action = "SELL AT MARKET" if original_entry == live_price else "SELL LIMIT ORDER"

                    cash_risk = min(buffer * (st.session_state.risk_pct / 100), st.session_state.daily_limit or buffer)

                    proposals = [p for p in [g_parsed, k_parsed, c_parsed] if p["proposal"] != "NONE" and p["entry"] is not None]
                    final_entry = original_entry
                    final_sl = original_sl
                    final_tp = original_tp
                    override_applied = False

                    if len(proposals) >= 2:
                        entries = [p["entry"] for p in proposals]
                        directions = [p["direction"] for p in proposals if p["direction"]]
                        if len(set(directions)) == 1 and max(entries) - min(entries) <= 10:
                            avg_entry = np.mean(entries)
                            avg_sl = np.mean([p["sl"] for p in proposals if p["sl"] is not None])
                            avg_tp = np.mean([p["tp"] for p in proposals if p["tp"] is not None])
                            if (bias == "BULLISH" and avg_sl < avg_entry) or (bias == "BEARISH" and avg_sl > avg_entry):
                                final_entry = avg_entry
                                final_sl = avg_sl
                                final_tp = avg_tp
                                override_applied = True
                                st.success("AI consensus override applied (averaged levels)")

                    risk_dist = abs(final_entry - final_sl)
                    reward_dist = abs(final_tp - final_entry)
                    actual_rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

                    risk_per_lot = (risk_dist + SPREAD_BUFFER_POINTS + SLIPPAGE_BUFFER_POINTS) * PIP_VALUE
                    lots = cash_risk / risk_per_lot if risk_per_lot > 0 else 0.01
                    lots = max(0.01, round(lots / 0.01) * 0.01)
                    actual_risk = lots * risk_per_lot

                    setup_valid = True
                    warnings = []

                    min_risk = max(atr * 100 * 0.01 * 2, buffer * 0.005, 10)
                    if cash_risk < min_risk:
                        warnings.append(f"Risk amount too small (${cash_risk:.2f} < ${min_risk:.2f})")
                        setup_valid = False
                    if not ((bias == "BULLISH" and final_sl < final_entry) or (bias == "BEARISH" and final_sl > final_entry)):
                        warnings.append("Invalid SL direction")
                        setup_valid = False
                    if actual_rr < 1.2:
                        warnings.append(f"R:R too low ({actual_rr:.2f}:1)")
                        setup_valid = False

                    if setup_valid:
                        action_text = original_action if not override_applied else f"AI-ADJUSTED {'BUY LIMIT' if bias == 'BULLISH' else 'SELL LIMIT'}"
                        st.markdown(f"### Recommended: {action_text}")
                        with st.container(border=True):
                            st.metric("Entry", f"${final_entry:.2f}")
                            col_sl, col_tp = st.columns(2)
                            col_sl.metric("Stop Loss", f"${final_sl:.2f}")
                            col_tp.metric("Take Profit", f"${final_tp:.2f}")
                            col_l, col_r = st.columns(2)
                            col_l.metric("Lots", f"{lots:.2f}")
                            col_r.metric("Risk \( ", f" \){actual_risk:.2f}")
                            st.metric("Actual R:R", f"1:{actual_rr:.2f}")
                        if override_applied:
                            st.caption("Levels averaged from AI proposals")
                    else:
                        st.warning("Your setup rejected:\n" + "\n".join(warnings))

                    with st.expander("Fractal Levels"):
                        st.write("Resistance above:", resistances[:4] or "None close")
                        st.write("Support below:", supports[:4] or "None close")

            except Exception as e:
                st.error(f"Original setup calculation failed: {e}\n→ Rely on AI verdicts above")

            if st.button("Reset & Re-analyze"):
                st.session_state.analysis_done = False
                st.rerun()

        except Exception as e:
            st.error(f"Core data / AI fetch failed: {str(e)}\nTry again in a few minutes.")
