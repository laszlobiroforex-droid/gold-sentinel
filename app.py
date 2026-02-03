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
    prompt = f"""
    Trading auditor for any gold account (challenge or live cash).
    Aggressive risk is user's choice – do NOT suggest lowering %.
    Focus on math, pullback quality, structural confluence, risk/reward.

    Account buffer left: ${account['buffer']:.2f}
    Market: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    Setup: {setup['type']} at ${setup['entry']:.2f} risking ${setup['risk']:.2f}

    Blunt: Elite high-conviction entry or low-edge gamble? 3 sentences max.
    """
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
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── INPUTS (front and center) ───────────────────────────────────────────────
st.header("Account Settings")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")
with col2:
    daily_limit = st.number_input("Daily Drawdown Limit ($)", min_value=0.0, value=None, placeholder="Optional / set to balance for no limit", format="%.2f")

survival_floor = st.number_input("Survival Floor / Max DD Line ($)", value=0.0, format="%.2f")

risk_pct = st.slider("Risk % of Available Buffer", 5, 50, 25, step=5)

if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────
if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
    if balance is None:
        st.error("❌ Please enter current balance")
    else:
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
                    st.markdown("**Short explanation:** The 15-minute trend direction is not supported by the 1-hour timeframe. This filter prevents fighting the bigger trend and getting stopped out quickly.")
                    st.markdown("**Suggested action:** Wait approximately **15 minutes** (one full 15-min candle) and press 'Analyze & Suggest' again to see if the structure has normalized.")
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
                    action_header = "BUY AT MARKET" if entry == live_price else "BUY LIMIT ORDER"
                else:
                    entry = ema50_15 if (ema50_15 - live_price) > (atr * 0.5) else live_price
                    sl = resistances[0] + 1.0 if resistances else entry + sl_dist
                    tp = supports[0] if supports else entry - (sl_dist * 2.5)
                    action_header = "SELL AT MARKET" if entry == live_price else "SELL LIMIT ORDER"

                # RISK & SAFETY
                buffer = balance - survival_floor
                cash_risk = min(buffer * (risk_pct / 100), daily_limit or buffer)

                if cash_risk < 20:
                    st.warning("Calculated risk too small for minimum lot size – skipping")
                    st.stop()

                sl_dist_actual = abs(entry - sl)
                lots = max(round(cash_risk / ((sl_dist_actual + 0.35) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist_actual + 0.35) * 100, 2)

                # ─── ACTION BOX ──────────────────────────────────────────────────
                st.markdown(f"### {action_header}")
                with st.container(border=True):
                    st.metric("Entry", f"${entry:.2f}")
                    col_sl, col_tp = st.columns(2)
                    col_sl.metric("Stop Loss", f"${sl:.2f}")
                    col_tp.metric("Take Profit", f"${tp:.2f}")
                    st.metric("Lots", f"{lots:.2f}")
                    st.metric("Risk Amount", f"${actual_risk:.2f}")

                # Dual AI opinions
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
                    st.success("✅ Both AIs agree: High-conviction setup")
                elif "gamble" in g_low or "reckless" in g_low or "gamble" in k_low or "reckless" in k_low:
                    st.warning("⚠️ Caution: At least one AI flags risk")
                else:
                    st.info("Mixed or neutral – review both opinions")

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
    st.info("No setups saved yet in this session.")

# Reset button (to bring back inputs if needed)
if st.button("Reset Inputs & Start New Analysis"):
    for key in list(st.session_state.keys()):
        if key.startswith("user_") or key == "analysis_done":
            del st.session_state[key]
    st.rerun()
