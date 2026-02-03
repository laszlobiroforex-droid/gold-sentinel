import streamlit as st
import pandas as pd
import numpy as np
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
from datetime import datetime

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    grok_client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
except Exception as e:
    st.error(f"API setup failed: {e}\nCheck secrets: TWELVE_DATA_KEY, GEMINI_KEY, GROK_API_KEY")
    st.stop()

# ─── FRACTAL LEVELS ──────────────────────────────────────────────────────────
def get_fractal_levels(df, window=5):
    levels = []
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            levels.append(('RES', round(df['high'].iloc[i], 2)))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            levels.append(('SUP', round(df['low'].iloc[i], 2)))
    return levels

# ─── DUAL AUDITORS ───────────────────────────────────────────────────────────
def get_ai_advice(market, setup, levels, buffer, mode):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[-5:]]) if levels else "No clear levels"
    prompt = f"""
    You are a high-conviction gold trading auditor for any account size.
    Mode: {mode} ({'standard swing (15m + 1h)' if mode == 'Standard' else 'fast scalp (15m + 5m)'}).
    Aggressive risk is user's choice — do NOT suggest reducing % risk.
    Focus on math, pullback quality, structural confluence, risk/reward.
    IMPORTANT: For buys, SL must be BELOW entry. For sells, SL must be ABOVE entry.

    Buffer left: ${buffer:.2f}
    Market: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    Original setup: {setup['type']} at ${setup['entry']:.2f} risking ${setup['risk']:.2f}
    Fractals: {levels_str}

    First, give a blunt verdict on the original setup (elite or low-edge gamble? 2 sentences max).

    Then, if you see a meaningfully better or safer alternative (different entry, SL, TP, RR, lots, or even opposite bias), propose it clearly.
    Format your proposal like this:
    PROPOSAL: [brief description, e.g. "Move entry to $X, SL to $Y for 2.5:1 RR"]
    REASONING: [1-2 sentences why it's better]

    If no meaningful change is needed, just say "No better alternative".
    Keep total response under 5 sentences.
    """
    try:
        g_out = gemini_model.generate_content(prompt).text.strip()
    except:
        g_out = "Gemini Offline."

    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e:
        k_out = f"Grok Error: {e}"
    
    return g_out, k_out

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── SESSION STATE INIT ──────────────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = None
    st.session_state.daily_limit = None
    st.session_state.floor = 0.0
    st.session_state.risk_pct = 25
    st.session_state.mode = "Standard"

if "saved_setups" not in st.session_state:
    st.session_state.saved_setups = []

# ─── INPUTS ──────────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    st.header("Account Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.balance = st.number_input(
            "Current Balance ($)", min_value=0.0, value=st.session_state.balance,
            placeholder="Required", format="%.2f", key="balance_input"
        )
    with col2:
        st.session_state.daily_limit = st.number_input(
            "Daily Drawdown Limit ($)", min_value=0.0, value=st.session_state.daily_limit,
            placeholder="Optional (set to balance for no limit)", format="%.2f", key="limit_input"
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
        index=0 if st.session_state.mode == "Standard" else 1
    )

    if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
        if st.session_state.balance is None:
            st.error("❌ Enter current balance")
        else:
            st.session_state.analysis_done = True
            st.rerun()
else:
    # ─── REMINDER ────────────────────────────────────────────────────────────
    st.info("Analysis locked with your settings:")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "No limit")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")
    cols[3].metric("Mode", st.session_state.mode.split(" – ")[0])

    with st.spinner("Scanning structure..."):
        try:
            price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
            live_price = float(price_data["price"])

            # ─── DATA FETCH ──────────────────────────────────────────────────────
            ts_15m = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "15min",
                "outputsize": 100
            }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

            # Higher or lower timeframe
            if st.session_state.mode.startswith("Standard"):
                ts_htf = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "1h",
                    "outputsize": 50
                }).with_ema(**{"time_period": 200}).as_pandas()
                htf_label = "1H"
            else:
                ts_htf = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "5min",
                    "outputsize": 100
                }).with_ema(**{"time_period": 200}).as_pandas()
                htf_label = "5M"

            # ─── SAFE INDICATOR EXTRACTION ───────────────────────────────────────
            rsi = ts_15m['rsi'].iloc[0] if 'rsi' in ts_15m.columns else 50.0
            atr = ts_15m['atr'].iloc[0] if 'atr' in ts_15m.columns else 0.0

            ema_cols_15m = [c for c in ts_15m.columns if 'ema' in c.lower()]
            ema_cols_15m.sort()
            ema200_15m = ts_15m[ema_cols_15m[0]].iloc[0] if len(ema_cols_15m) >= 1 else live_price
            ema50_15m  = ts_15m[ema_cols_15m[1]].iloc[0] if len(ema_cols_15m) >= 2 else live_price

            ema_cols_htf = [c for c in ts_htf.columns if 'ema' in c.lower()]
            ema200_htf = ts_htf[ema_cols_htf[0]].iloc[0] if ema_cols_htf else live_price

            # ─── TREND ALIGNMENT ─────────────────────────────────────────────────
            aligned = (live_price > ema200_15m and live_price > ema200_htf) or \
                      (live_price < ema200_15m and live_price < ema200_htf)

            if not aligned:
                st.warning(f"Trend misalignment – {htf_label} EMA200 at ${ema200_htf:.2f}")
                st.markdown("**Short explanation:** The 15-minute and higher-timeframe trends are not aligned. This prevents trades against the larger trend.")
                st.markdown("**Suggested action:** Wait approximately **15 minutes** and press 'Analyze & Suggest' again.")
                st.stop()

            bias = "BULLISH" if live_price > ema200_15m else "BEARISH"

            # ─── FRACTAL LEVELS ──────────────────────────────────────────────────
            levels = get_fractal_levels(ts_15m)
            resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
            supports = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

            # ─── PULLBACK ENTRY + FORCED CORRECT SL DIRECTION ────────────────────
            sl_dist = round(atr * 1.5, 2)
            min_sl_distance = atr * 0.4
            min_rr = 1.2

            if bias == "BULLISH":
                entry = ema50_15m if (live_price - ema50_15m) > (atr * 0.5) else live_price
                
                # Valid support BELOW entry
                valid_sup = [s for s in supports if s < entry]
                candidate_sl = valid_sup[0] - (0.3 * atr) if valid_sup else entry - sl_dist
                
                sl = min(candidate_sl, entry - min_sl_distance)
                
                if sl >= entry:
                    st.warning("No valid stop-loss below entry – setup skipped")
                    st.stop()

                tp = resistances[0] if resistances else entry + (sl_dist * 2.5)

                risk_dist = entry - sl
                reward_dist = tp - entry
                actual_rr = reward_dist / risk_dist if risk_dist > 0 else 0
                
                if actual_rr < min_rr:
                    st.warning(f"Reward:risk too low ({actual_rr:.2f}:1) – setup skipped")
                    st.stop()

                action_header = "BUY AT MARKET" if entry == live_price else "BUY LIMIT ORDER"

            else:
                entry = ema50_15m if (ema50_15m - live_price) > (atr * 0.5) else live_price
                
                valid_res = [r for r in resistances if r > entry]
                candidate_sl = valid_res[0] + (0.3 * atr) if valid_res else entry + sl_dist
                
                sl = max(candidate_sl, entry + min_sl_distance)
                
                if sl <= entry:
                    st.warning("No valid stop-loss above entry – setup skipped")
                    st.stop()

                tp = supports[0] if supports else entry - (sl_dist * 2.5)

                risk_dist = sl - entry
                reward_dist = entry - tp
                actual_rr = reward_dist /
