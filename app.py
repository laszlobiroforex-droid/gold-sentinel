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
def get_ai_advice(market, setup, levels):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[-5:]]) if levels else "No clear levels detected"
    prompt = f"Prop Auditor: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}. Levels: {levels_str}. Setup: {setup}. Audit risk vs structure. 2 sentences."
    
    # Gemini
    try:
        g_out = gemini_model.generate_content(prompt).text.strip()
    except:
        g_out = "Gemini Offline."
    
    # Grok
    try:
        r = grok_client.chat.completions.create(
            model="grok-4",  # Change to "grok-beta" or your preferred if needed
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e:
        k_out = f"Grok Error: {e}"
    
    return g_out, k_out

# ─── APP SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel 8.5 – Fractal Pullback")
st.caption(f"Phase 2 Protector | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── MAIN INPUTS ─────────────────────────────────────────────────────────────
st.header("Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")

survival_floor = st.number_input("Max Overall Drawdown Floor ($)", value=4500.0, format="%.2f")

risk_pct = st.slider("Risk % of Buffer", 10, 50, 25, step=5)

if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────
if st.button("🚀 Analyze Market Structure", type="primary", use_container_width=True):
    if balance is None or daily_limit is None:
        st.error("❌ Please enter balance and daily drawdown limit")
    else:
        with st.spinner("Scanning fractal pivots and structure..."):
            try:
                # 1. FETCH DATA
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

                # 2. SAFE INDICATOR EXTRACTION (EMA workaround)
                rsi = ts_15m['rsi'].iloc[0] if 'rsi' in ts_15m.columns else 50.0
                atr = ts_15m['atr'].iloc[0] if 'atr' in ts_15m.columns else 0.0

                # Dynamic EMA detection
                ema_cols = [c for c in ts_15m.columns if 'ema' in c.lower()]
                ema_cols.sort()  # usually ema_1 = 200, ema_2 = 50
                ema200_15 = ts_15m[ema_cols[0]].iloc[0] if len(ema_cols) >= 1 else live_price
                ema50_15  = ts_15m[ema_cols[1]].iloc[0] if len(ema_cols) >= 2 else live_price

                ema200_1h = ts_1h['ema_1'].iloc[0] if 'ema_1' in ts_1h.columns else live_price

                if len(ema_cols) < 2:
                    st.warning("Warning: EMA columns not detected as expected. Using fallback values.")

                # 3. TREND ALIGNMENT GUARD
                if (live_price > ema200_15 and live_price > ema200_1h):
                    bias = "BULLISH"
                elif (live_price < ema200_15 and live_price < ema200_1h):
                    bias = "BEARISH"
                else:
                    st.warning(f"🚫 Trend misalignment – 1H EMA200 at ${ema200_1h:.2f}")
                    st.stop()

                # 4. FRACTAL LEVELS
                levels = get_fractal_levels(ts_15m)
                resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
                supports    = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

                # 5. PULLBACK ENTRY LOGIC
                sl_dist = round(atr * 1.5, 2)
                if bias == "BULLISH":
                    entry = ema50_15 if (live_price - ema50_15) > (atr * 0.5) else live_price
                    sl = supports[0] - 1.0 if supports else entry - sl_dist
                    tp = resistances[0] if resistances else entry + (sl_dist * 2.5)
                else:
                    entry = ema50_15 if (ema50_15 - live_price) > (atr * 0.5) else live_price
                    sl = resistances[0] + 1.0 if resistances else entry + sl_dist
                    tp = supports[0] if supports else entry - (sl_dist * 2.5)

                # 6. RISK MATH + SAFETY GATE
                buffer = balance - survival_floor
                cash_risk = min(buffer * (risk_pct / 100), daily_limit)

                if cash_risk < 20:
                    st.warning("Calculated risk too small for minimum lot size – skipping")
                    st.stop()

                sl_dist_actual = abs(entry - sl)
                lots = max(round(cash_risk / ((sl_dist_actual + 0.35) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist_actual + 0.35) * 100, 2)

                # 7. OUTPUT
                st.header(f"Bias: {bias}")
                if entry != live_price:
                    st.info(f"⏳ Set LIMIT {bias} order @ ${entry:.2f}")
                else:
                    st.success(f"🚀 Execute {bias} MARKET order @ ${entry:.2f}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lots", f"{lots:.2f}")
                c2.metric("SL", f"${sl:.2f}")
                c3.metric("TP", f"${tp:.2f}")
                c4.metric("Risk \( ", f" \){actual_risk:.2f}")

                # Dual AI
                st.divider()
                st.subheader("Dual AI Analysis")
                market = {"price": live_price, "rsi": rsi}
                setup = {"type": bias, "entry": entry, "risk": actual_risk}
                g_verdict, k_verdict = get_ai_advice(market, setup, levels)
                
                col_g, col_k = st.columns(2)
                col_g.info(f"**Gemini (Cautious)**\n{g_verdict}")
                col_k.info(f"**Grok (Direct)**\n{k_verdict}")

                # Levels expander
                with st.expander("Detected Fractal Levels"):
                    st.write("**Resistance above:**", resistances[:3] or "None nearby")
                    st.write("**Support below:**", supports[:3] or "None nearby")

                # Save to history
                st.session_state.saved_setups.append({
                    "time": datetime.utcnow().strftime("%H:%M UTC"),
                    "bias": bias,
                    "entry": round(entry, 2),
                    "risk": actual_risk
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─── HISTORY SECTION ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(10), use_container_width=True, hide_index=True)
else:
    st.info("No setups saved yet in this session.")
