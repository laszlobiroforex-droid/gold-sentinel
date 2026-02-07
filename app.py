import streamlit as st
import json
from datetime import datetime, timezone, timedelta
import time
import numpy as np
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
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

def get_ema(series_or_row, period):
    candidates = [f"ema_{period}", f"ema ({period})", f"EMA_{period}", f"ema{period}", f"EMA{period}"]
    for key in candidates:
        if key in series_or_row:
            val = series_or_row[key]
            if pd.notna(val):
                return val
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
        st.warning(f"15m fetch failed: {str(e)}")
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
        st.warning(f"1h fetch failed: {str(e)}")
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

def calculate_strict_lot_size(entry, sl, max_risk_dollars):
    if not all(v is not None for v in [entry, sl, max_risk_dollars]):
        return 0.01, "Missing data — min lot"

    try:
        entry = float(entry)
        sl = float(sl)
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return 0.01, "Invalid SL — min lot"

        lot_size = max_risk_dollars / (price_diff * 100)
        lot_size_rounded = max(round(lot_size, 2), 0.01)

        actual_risk = lot_size_rounded * price_diff * 100
        note = f"Risk ${actual_risk:.2f}"
        if actual_risk > max_risk_dollars * 1.05:
            note += " (slightly over)"
        return lot_size_rounded, note
    except:
        return 0.01, "Calc error — min lot"

def run_check():
    current_time = datetime.now(timezone.utc)
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M UTC')

    # Soft session reminder
    hour_utc = current_time.hour
    if not (8 <= hour_utc <= 17):
        st.info("Outside major overlap — liquidity lower, watch for fakeouts.")

    with st.spinner("Fetching recent data..."):
        ts_15m = fetch_recent_15m()
        time.sleep(2)
        ts_1h  = fetch_recent_1h()
        if ts_15m is None or ts_15m.empty:
            st.error("No recent 15m data available")
            return
        price = ts_15m['close'].iloc[-1]

    df_1d = load_long_term_1d()
    df_4h = load_long_term_4h()

    long_term_summary = ""
    ht_trend = "neutral"
    if df_1d is not None and not df_1d.empty:
        ema200_1d = get_ema(df_1d.iloc[-1], 200)
        if ema200_1d is not None:
            lt_price = df_1d['close'].iloc[-1]
            ht_trend = "bullish" if lt_price > ema200_1d else "bearish"
            long_term_summary += f"\nDaily trend: {ht_trend} (price vs EMA200 1D)"
        else:
            long_term_summary += "\nDaily EMA200 not found"
    if df_4h is not None and not df_4h.empty:
        lt_price_4h = df_4h['close'].iloc[-1]
        long_term_summary += f"\n4H price: {lt_price_4h:.2f}"

    latest_15m = ts_15m.iloc[-1]
    rsi = latest_15m.get('rsi', 50.0)
    atr = latest_15m.get('atr', 10.0)
    ema200_15m = get_ema(latest_15m, 200) or price
    ema50_15m  = get_ema(latest_15m, 50) or price

    ema200_1h = get_ema(ts_1h.iloc[-1], 200) or price if ts_1h is not None and not ts_1h.empty else price

    # Fractal levels
    levels = []
    for i in range(5, min(40, len(ts_15m)-5)):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    dd_limit = st.session_state.get("dd_limit", 251.45)
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct", 25.0)
    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit else 50.0

    st.write(f"DD limit: ${dd_limit:.2f}")
    st.write(f"Risk % of DD: {risk_of_dd_pct}%")
    st.write(f"Max risk $: ${max_risk_dollars:.2f}")

    # Pre-filters
    filter_results = {}

    # 1. Higher TF trend alignment
    filter_results["ht_trend"] = ht_trend != "neutral"

    # 2. HTF structure proximity
    htf_near = False
    if df_4h is not None and 'close' in df_4h.columns:
        recent_4h = df_4h['close'].iloc[-1]
        if abs(price - recent_4h) <= atr * 1.5:
            htf_near = True
    filter_results["htf_near"] = htf_near

    # 3. Volatility expansion
    expansion = False
    if len(ts_15m) >= 21:
        recent_atr = ts_15m['atr'].tail(1).values[0]
        avg_atr = ts_15m['atr'].tail(20).mean()
        expansion = recent_atr > 1.3 * avg_atr
    filter_results["expansion"] = expansion

    # 4. LTF structure proximity
    ltf_near = any(abs(price - level_price) <= atr for _, level_price in levels[-8:])
    filter_results["ltf_near"] = ltf_near

    passed_filters = sum(filter_results.values())
    pre_filter_ok = passed_filters >= 3

    if not pre_filter_ok:
        st.warning(f"Pre-filters failed ({passed_filters}/4 passed) — no strong setup possible.")
        verdict = "NO_SETUP"
        lot_size = 0.01
        lot_note = "Pre-filters not met"
        st.markdown(f"**Verdict:** {verdict}")
        st.markdown(f"**Lot Size:** {lot_size:.2f} ({lot_note})")
        st.caption(f"Checked at {current_time_str} — manual run only")
        return

    # Narrow prompt
    prompt = f"""
Current UTC: {current_time_str}

Recent market data (15m):
Price: ${price:.2f}
RSI: {rsi:.1f}
ATR: {atr:.2f}
EMA50 / EMA200: {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 1h: {ema200_1h:.2f}

Long-term context: {long_term_summary.strip() or 'none'}

Recent S/R levels: {', '.join([f"{t}@{p}" for t,p in levels[-8:]]) or 'none'}

This is a pre-qualified {ht_trend} setup: price aligned with higher TFs, near structure, volatility expanding.

Explain why this is or is not a high-probability continuation trade.
Be extremely critical — if chop or weak, say so.

Output in this exact format:

Bias: bullish / bearish / neutral
Confidence: high / medium / low / no edge
Key levels: support/resistance zones
Suggested setup:
Entry zone: approximate range
Stop zone: approximate range
Target zone: approximate range
Reasoning: full explanation
"""

    with st.spinner("Consulting AIs..."):
        g_raw = "Bias: neutral\nConfidence: no edge\nReasoning: Gemini offline"
        try:
            g_raw = gemini_model.generate_content(prompt, generation_config={"temperature": 0.2}).text.strip()
        except:
            pass

        k_raw = "Bias: neutral\nConfidence: no edge\nReasoning: Grok offline"
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

        c_raw = "Bias: neutral\nConfidence: no edge\nReasoning: ChatGPT offline"
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

        for key, lst in [("Entry zone:", entry_zones), ("Stop zone:", stop_zones), ("Target zone:", target_zones)]:
            line = [l for l in text.split('\n') if l.strip().startswith(key)]
            if line:
                lst.append(line[0].split(":", 1)[1].strip())

    consensus_bias = max(set(biases), key=biases.count) if biases else "NEUTRAL"
    agreement_count = biases.count(consensus_bias)

    # Direction match with HTF trend
    if ht_trend != "neutral" and consensus_bias.lower() != ht_trend:
        verdict = "NO_CONVICTION (counter-trend)"
    else:
        if passed_filters >= 4 and agreement_count == 3 and sum(1 for c in confidences if c == "high") >= 2:
            verdict = "ELITE"
        elif passed_filters >= 3 and agreement_count >= 2 and sum(1 for c in confidences if c in ["high", "medium"]) >= 2:
            verdict = "HIGH_CONVICTION"
        else:
            verdict = "NO_CONVICTION"

    # Post-filter & educated guess
    reject_reason = ""
    entry_guess = sl_guess = tp_guess = None
    rr = 0.0
    if verdict in ["ELITE", "HIGH_CONVICTION"]:
        if len(entry_zones) < 2 or len(stop_zones) < 2:
            reject_reason = "Insufficient zone suggestions"
        else:
            try:
                entry_lows = [float(z.split('–')[0].strip()) for z in entry_zones if '–' in z]
                entry_highs = [float(z.split('–')[1].strip()) for z in entry_zones if '–' in z]
                stop_lows = [float(z.split('–')[0].strip()) for z in stop_zones if '–' in z]
                stop_highs = [float(z.split('–')[1].strip()) for z in stop_zones if '–' in z]
                target_mids = [np.mean([float(p) for p in z.split('–')]) for z in target_zones if '–' in z]

                entry_low = min(entry_lows)
                entry_high = max(entry_highs)
                worst_stop = min(stop_lows) if consensus_bias == "BULLISH" else max(stop_highs)
                median_target = np.median(target_mids) if target_mids else None

                # Educated guess
                entry_guess = entry_low if consensus_bias == "BULLISH" else entry_high
                sl_guess = worst_stop - 0.2 * atr if consensus_bias == "BULLISH" else worst_stop + 0.2 * atr
                tp_guess = median_target

                rr = abs(tp_guess - entry_guess) / abs(entry_guess - sl_guess) if tp_guess else 0

                if rr < 1.2:
                    reject_reason = f"RR low ({rr:.1f}) — modest reward"
                if abs(entry_guess - sl_guess) > 2 * atr:
                    reject_reason = "Stop too wide"
            except:
                reject_reason = "Zone calculation failed"

        if reject_reason and "low" not in reject_reason:
            verdict = "REJECTED"

    lot_size = 0.01
    lot_note = "Min lot — no strong signal"
    if verdict in ["ELITE", "HIGH_CONVICTION"] and not reject_reason and entry_guess and sl_guess:
        lot_size, lot_note = calculate_strict_lot_size(entry_guess, sl_guess, max_risk_dollars)

    st.divider()
    st.subheader("Signal Review")

    st.markdown(f"**Pre-filters passed:** {passed_filters}/4")
    st.markdown(f"**Consensus Bias:** {consensus_bias} ({agreement_count}/3)")
    st.markdown(f"**Verdict:** {verdict}")
    st.markdown(f"**Lot Size:** {lot_size:.2f} ({lot_note})")

    if entry_guess and sl_guess:
        st.markdown(f"**Educated Guess:**")
        st.markdown(f"Entry: \~{entry_guess:.1f}")
        st.markdown(f"SL: {sl_guess:.1f}")
        if tp_guess:
            st.markdown(f"TP: \~{tp_guess:.1f} (RR ≈ {rr:.1f})")

    if reject_reason:
        st.warning(f"Note: {reject_reason}")

    st.caption(f"Checked at {current_time_str} — manual run only")

    st.markdown("**Disclaimer**")
    st.markdown("Second opinion only. No guarantee of edge. Verify independently.")

    for resp in ai_responses:
        st.markdown(f"**{resp['name']} Explanation:**\n\n{resp['text']}")

    if verdict in ["ELITE", "HIGH_CONVICTION"] and not reject_reason and entry_guess and sl_guess:
        msg = f"**{verdict} {consensus_bias}** ({agreement_count}/3)\n"
        msg += f"Entry: \~{entry_guess:.1f}\n"
        msg += f"SL: {sl_guess:.1f}\n"
        if tp_guess:
            msg += f"TP: \~{tp_guess:.1f} (RR ≈ {rr:.1f})\n"
        msg += f"Lot: {lot_size:.2f} ({lot_note})\n{long_term_summary.strip()}"
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
    st.info("Download fresh context files for 1D and 4H.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Download 4H (60d)"):
            df = fetch_recent_1h(outputsize=360)
            if df is not None:
                st.download_button("Save 4H", df.to_csv(), "longterm_history_4H.csv")
    with col_b:
        if st.button("Download 1D (60d)"):
            df = fetch_recent_1h(outputsize=60)
            if df is not None:
                st.download_button("Save 1D", df.to_csv(), "longterm_history_1D.csv")

if st.button("Run Live Analysis", type="primary", use_container_width=True):
    run_check()
