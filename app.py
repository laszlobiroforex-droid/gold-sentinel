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

# ─── DATA FETCH (reduced sizes + both HTFs always) ───────────────────────────
@st.cache_data(ttl=300)
@retry_api()
def fetch_15m_data():
    ts = td.time_series(
        symbol="XAU/USD",
        interval="15min",
        outputsize=200
    )
    ts = ts.with_rsi()
    ts = ts.with_ema(time_period=200)
    ts = ts.with_ema(time_period=50)
    ts = ts.with_atr(time_period=14)
    return ts.as_pandas()

@st.cache_data(ttl=300)
@retry_api()
def fetch_htf_data(interval):
    out_size = 200 if interval == "5min" else 100
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

# ─── AI ADVICE (no mode, richer context) ─────────────────────────────────────
@retry_api()
def get_ai_advice(market, setup, levels, buffer, aligned_1h, aligned_5m):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[:6]]) if levels else "No clear levels"
    current_price = market['price']
    prompt = f"""
You are a high-conviction gold trading auditor.
Evaluate ANY high-edge pattern visible in current data: continuation pullbacks (scalp or swing), momentum breakouts, reversals at structure, range-bound mean-reversion, etc.
Prioritize setups with:
- RR ≥ 1.8
- Multiple confluences (fractals, EMAs, RSI, session/time)
- Clean risk (proper SL placement)
Be brutal — skip if low quality, chasing, or gamble.

Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} (NY close ~22:00 UTC, watch whipsaw after 21:30)
Market: ${current_price:.2f}, RSI {market['rsi']:.1f}
Trend context: 15m vs 1H EMA200 alignment: {'agree' if aligned_1h else 'disagree'}, 15m vs 5M EMA200: {'agree' if aligned_5m else 'disagree'}
Fractals: {levels_str}
Original levels: entry ~${setup['entry']:.2f}, ATR ${setup['atr']:.2f}, risk buffer ${buffer:.2f}

Respond in exact format:

VERDICT: ELITE | HIGH_CONV | LOW_EDGE | GAMBLE | SKIP
REASON: [1-2 sentences]
PROPOSAL: [entry price] | [SL price] | [TP price] | [RR ratio e.g. 2.5] | [STYLE: SCALP/SWING/BREAKOUT/REVERSAL/RANGE/NONE] | [direction: BULLISH/BEARISH/NEUTRAL] | [reasoning, 1-2 sentences]
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

# ─── PARSE AI OUTPUT (with STYLE) ────────────────────────────────────────────
def parse_ai_output(text):
    verdict = "UNKNOWN"
    reason = ""
    proposal = "NONE"
    entry = sl = tp = rr = direction = style = None

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
            if len(parts) >= 6:
                try:
                    entry = float(parts[0])
                    sl = float(parts[1])
                    tp = float(parts[2])
                    rr = float(parts[3])
                    style_match = re.search(r'STYLE:\s*(\w+)', parts[4], re.IGNORECASE)
                    style = style_match.group(1).upper() if style_match else "UNKNOWN"
                    direction = parts[5].strip().upper()
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
        "style": style,
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

        # Both HTFs for richer context
        ts_1h = fetch_htf_data("1h")
        ts_5m = fetch_htf_data("5min")

        ema200_1h = ts_1h.iloc[-1].get('ema_200', live_price) if not ts_1h.empty else live_price
        ema200_5m = ts_5m.iloc[-1].get('ema_200', live_price) if not ts_5m.empty else live_price

        aligned_1h = (live_price > ema200_1h) == (live_price > live_price)  # simplified agreement check
        aligned_5m = (live_price > ema200_5m) == (live_price > live_price)

        g_raw, k_raw, c_raw = get_ai_advice(market, setup, levels, buffer, aligned_1h, aligned_5m)
        g_p = parse_ai_output(g_raw)
        k_p = parse_ai_output(k_raw)
        c_p = parse_ai_output(c_raw)

        high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

        if high_count >= MIN_CONVICTION_FOR_ALERT:
            direction = g_p.get("direction") or "UNKNOWN"
            style = g_p.get("style") or "UNKNOWN"
            entry = g_p.get("entry") or live_price
            sl = g_p.get("sl") or (entry - atr*1.5 if direction.startswith("BULL") else entry + atr*1.5)
            tp = g_p.get("tp") or (entry + atr*3 if direction.startswith("BULL") else entry - atr*3)
            rr = g_p.get("rr") or 2.0

            key = f"{style}_{direction}_{entry:.2f}_{sl:.2f}"
            if st.session_state.last_alerted_key == key:
                return

            msg = (
                f"**High Conviction Setup!** ({high_count}/3 AIs)\n"
                f"Style: {style}\n"
                f"Direction: {direction}\n"
                f"Entry: ${entry:.2f}\n"
                f"SL: ${sl:.2f}\n"
                f"TP: ${tp:.2f}   (R:R ~1:{rr:.1f})\n"
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
    schedule.every(15).minutes.do(check_for_high_conviction_setup)
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
st.caption(f"Adaptive pullback engine | Background checks every 15 min (free tier credit optimization) | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# Session state defaults
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = DEFAULT_BALANCE
    st.session_state.daily_limit = DEFAULT_DAILY_LIMIT
    st.session_state.floor = DEFAULT_FLOOR
    st.session_state.risk_pct = DEFAULT_RISK_PCT
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

    utc_now = datetime.now(timezone.utc)
    if utc_now.hour >= 21 and utc_now.hour < 23:
        st.warning("Late NY session — higher whipsaw risk after ~21:30 UTC")

    with st.spinner("Fetching data & running triple AI audit (may take a moment)..."):
        try:
            live_price = get_current_price()
            ts_15m = fetch_15m_data()
            latest_15m = ts_15m.iloc[-1] if not ts_15m.empty else None
            rsi = latest_15m.get('rsi', 50.0) if latest_15m is not None else 50.0
            atr = latest_15m.get('atr', 10.0) if latest_15m is not None else 10.0

            # Both HTFs
            ts_1h = fetch_htf_data("1h")
            ts_5m = fetch_htf_data("5min")

            ema200_1h = ts_1h.iloc[-1].get('ema_200', live_price) if not ts_1h.empty else live_price
            ema200_5m = ts_5m.iloc[-1].get('ema_200', live_price) if not ts_5m.empty else live_price

            aligned_1h = (live_price > ema200_1h) == (live_price > (latest_15m.get('ema_200', live_price) if latest_15m is not None else live_price))
            aligned_5m = (live_price > ema200_5m) == (live_price > (latest_15m.get('ema_200', live_price) if latest_15m is not None else live_price))

            market = {"price": live_price, "rsi": rsi}
            levels = get_fractal_levels(ts_15m) if not ts_15m.empty else []
            buffer = st.session_state.balance - st.session_state.floor

            setup_placeholder = {
                "type": "UNKNOWN",
                "entry": live_price,
                "sl_distance": atr * 1.5,
                "atr": atr,
                "risk_pct": st.session_state.risk_pct
            }

            g_raw, k_raw, c_raw = get_ai_advice(
                market, setup_placeholder, levels, buffer, aligned_1h, aligned_5m
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
            st.subheader("Structured Setup (secondary check)")
            try:
                # Use 1h for continuation bias, but AIs already decided style
                ts_htf = ts_1h if not ts_1h.empty else ts_5m
                latest_htf = ts_htf.iloc[-1] if not ts_htf.empty else None
                ema200_htf = latest_htf.get('ema_200', live_price) if latest_htf is not None else live_price

                ema_cols_15m = sorted([c for c in ts_15m.columns if 'ema' in c.lower()])
                ema200_15m = latest_15m.get(ema_cols_15m[0], live_price) if ema_cols_15m and latest_15m is not None else live_price
                ema50_15m  = latest_15m.get(ema_cols_15m[1], live_price) if len(ema_cols_15m) >= 2 and latest_15m is not None else live_price

                aligned = (live_price > ema200_15m and live_price > ema200_htf) or \
                          (live_price < ema200_15m and live_price < ema200_htf)

                if not aligned:
                    st.warning("Trend misalignment detected → no high-conviction continuation recommended")
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
                        styles = [p["style"] for p in proposals if p["style"]]
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
                        st.warning("Setup rejected:\n" + "\n".join(warnings))

                    with st.expander("Fractal Levels"):
                        st.write("Resistance above:", resistances[:4] or "None close")
                        st.write("Support below:", supports[:4] or "None close")

            except Exception as e:
                st.error(f"Setup calculation failed: {e}\n→ Rely on AI verdicts above")

            if st.button("Reset & Re-analyze"):
                st.session_state.analysis_done = False
                st.rerun()

        except Exception as e:
            st.error(f"Core data / AI fetch failed: {str(e)}\nLikely Twelve Data credit limit or timeout – wait for reset or reduce load. Try again in a few minutes.")
