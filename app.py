import streamlit as st
import pandas as pd
import numpy as np
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
from datetime import datetime, timedelta
import time

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
    st.error(f"API setup failed: {e}")
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
def get_ai_advice(market, setup, levels):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[-5:]]) if levels else "No clear levels"
    prompt = f"Prop Auditor: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}. Levels: {levels_str}. Setup: {setup}. Audit risk vs structure. 2 sentences."
    
    try: g_out = gemini_model.generate_content(prompt).text.strip()
    except: g_out = "Gemini Offline."
    
    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e: k_out = f"Grok Error: {e}"
    
    return g_out, k_out

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel 8.6 – Fractal Pullback")
st.caption(f"Phase 2 Protector | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── INPUT SECTION ───────────────────────────────────────────────────────────
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.user_balance = None
    st.session_state.user_daily_limit = None
    st.session_state.user_floor = 4500.0
    st.session_state.user_risk_pct = 25

if not st.session_state.analysis_done:
    st.header("Account Health (required)")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_balance = st.number_input(
            "Current Balance ($)", min_value=0.0, value=st.session_state.user_balance,
            placeholder="Required", format="%.2f", key="balance_input"
        )
    with col2:
        st.session_state.user_daily_limit = st.number_input(
            "Daily Drawdown Left ($)", min_value=0.0, value=st.session_state.user_daily_limit,
            placeholder="Required", format="%.2f", key="limit_input"
        )

    st.session_state.user_floor = st.number_input(
        "Max Overall Drawdown Floor ($)", value=st.session_state.user_floor, format="%.2f"
    )

    st.session_state.user_risk_pct = st.slider(
        "Risk % of Buffer", 10, 50, st.session_state.user_risk_pct, step=5
    )

    if st.button("🚀 Analyze Market Structure", type="primary", use_container_width=True):
        if st.session_state.user_balance is None or st.session_state.user_daily_limit is None:
            st.error("❌ Please fill balance and daily drawdown limit")
        else:
            st.session_state.analysis_done = True
            st.rerun()  # Hide inputs, show results
else:
    # Show entered values as reminder
    st.success("Account values locked for this analysis:")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${st.session_state.user_balance:.2f}")
    cols[1].metric("Daily DD Left", f"${st.session_state.user_daily_limit:.2f}")
    cols[2].metric("Floor", f"${st.session_state.user_floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.user_risk_pct}%")

    with st.spinner("Scanning structure..."):
        try:
            price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
            live_price = float(price_data["price"])

            ts_15m = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "15min",
                "outputsize": 100
            }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

            ts_1h = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "1h",
                "outputsize": 50
            }).with_ema(**{"time_period": 200}).as_pandas()

            # SAFE EMA DETECTION
            rsi = ts_15m['rsi'].iloc[0] if 'rsi' in ts_15m.columns else 50.0
            atr = ts_15m['atr'].iloc[0] if 'atr' in ts_15m.columns else 0.0

            ema_cols = [c for c in ts_15m.columns if 'ema' in c.lower()]
            ema_cols.sort()
            ema200_15 = ts_15m[ema_cols[0]].iloc[0] if len(ema_cols) >= 1 else live_price
            ema50_15  = ts_15m[ema_cols[1]].iloc[0] if len(ema_cols) >= 2 else live_price
            ema200_1h = ts_1h['ema_1'].iloc[0] if 'ema_1' in ts_1h.columns else live_price

            # TREND ALIGNMENT
            if (live_price > ema200_15 and live_price > ema200_1h):
                bias = "BULLISH"
            elif (live_price < ema200_15 and live_price < ema200_1h):
                bias = "BEARISH"
            else:
                st.warning(f"Trend misalignment – 1H EMA200 at ${ema200_1h:.2f}")
                st.markdown("**Description:** The 15-minute and 1-hour trends are not aligned. The higher timeframe (1H) is not supporting the shorter-term bias. This filter prevents fighting the bigger picture.")
                st.markdown("**Action:** Wait **15 minutes** (enough for 1 full 15-min candle to form) and press 'Analyze' again to check if structure has normalized.")
                st.stop()

            # FRACTAL LEVELS
            levels = get_fractal_levels(ts_15m)
            resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
            supports = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

            # PULLBACK ENTRY
            sl_dist = round(atr * 1.5, 2)
            if bias == "BULLISH":
                entry = ema50_15 if (live_price - ema50_15) > (atr * 0.5) else live_price
                sl = supports[0] - 1.0 if supports else entry - sl_dist
                tp = resistances[0] if resistances else entry + (sl_dist * 2.5)
                action_type = "BUY AT MARKET" if entry == live_price else "BUY LIMIT ORDER"
            else:
                entry = ema50_15 if (ema50_15 - live_price) > (atr * 0.5) else live_price
                sl = resistances[0] + 1.0 if resistances else entry + sl_dist
                tp = supports[0] if supports else entry - (sl_dist * 2.5)
                action_type = "SELL AT MARKET" if entry == live_price else "SELL LIMIT ORDER"

            # RISK & SAFETY
            buffer = balance - survival_floor
            cash_risk = min(buffer * (risk_pct / 100), daily_limit)

            if cash_risk < 20:
                st.warning("Calculated risk too small for minimum lot size – skipping")
                st.stop()

            sl_dist_actual = abs(entry - sl)
            lots = max(round(cash_risk / ((sl_dist_actual + 0.35) * 100), 2), 0.01)
            actual_risk = round(lots * (sl_dist_actual + 0.35) * 100, 2)

            # ─── SETUP BOX ───────────────────────────────────────────────────
            st.markdown(f"### {action_type}")
            with st.container(border=True):
                st.metric("Entry", f"${entry:.2f}")
                col_sl, col_tp = st.columns(2)
                col_sl.metric("Stop Loss", f"${sl:.2f}")
                col_tp.metric("Take Profit", f"${tp:.2f}")
                st.metric("Lots", f"{lots:.2f}")
                st.metric("Risk Amount", f"${actual_risk:.2f}")

            # Dual AI
            st.divider()
            st.subheader("AI Opinions")
            market = {"price": live_price, "rsi": rsi}
            setup = {"type": bias, "entry": entry, "risk": actual_risk}
            g_verdict, k_verdict = get_ai_advice(market, setup, levels)

            col_g, col_k = st.columns(2)
            with col_g:
                st.markdown("**Gemini (Cautious)**")
                st.info(g_verdict)
            with col_k:
                st.markdown("**Grok (Direct)**")
                st.info(k_verdict)

            # Consensus
            g_low, k_low = g_verdict.lower(), k_verdict.lower()
            if "elite" in g_low and "elite" in k_low:
                st.success("✅ Both AIs agree: Elite setup")
            elif "gamble" in g_low or "reckless" in g_low or "gamble" in k_low or "reckless" in k_low:
                st.warning("⚠️ At least one AI flags caution")
            else:
                st.info("Mixed or neutral opinions – review carefully")

            # Levels
            with st.expander("Detected Fractal Levels"):
                st.write("**Resistance above:**", resistances[:3] or "None nearby")
                st.write("**Support below:**", supports[:3] or "None nearby")

            # Save
            st.session_state.saved_setups.append({
                "time": datetime.utcnow().strftime("%H:%M UTC"),
                "bias": bias,
                "entry": round(entry, 2),
                "risk": actual_risk
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ─── HISTORY ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(10), use_container_width=True, hide_index=True)
else:
    st.info("No setups saved yet.")
