import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai
from datetime import datetime

# ─── API CONFIGURATION (NOW SECURE) ──────────────────────────────────────────
# These now pull from the 'Secrets' tab in Streamlit, not your code!
try:
    TWELVE_DATA_KEY = st.secrets["TWELVE_DATA_KEY"]
    GEMINI_KEY      = st.secrets["GEMINI_KEY"]
except Exception:
    st.error("❌ Secrets not found! Add TWELVE_DATA_KEY and GEMINI_KEY to your Streamlit App Secrets.")
    st.stop()

td = TDClient(apikey=TWELVE_DATA_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ─── AI RISK AUDITOR ─────────────────────────────────────────────────────────
def get_ai_advice(market, account, setup):
    prompt = f"""
    You are a strict Gold risk auditor for a RebelsFunding Phase 2 account.
    Buffer: ${account['buffer']:.2f} | Market: ${market['price']:.2f}
    Proposed: {setup['type']} @ ${setup['entry']:.2f} with ${setup['risk']:.2f} risk.
    Analyze risk vs buffer. Be blunt. 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Brain Error: {str(e)}"

# ─── STREAMLIT SETUP ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 6.9.1")
st.caption("Secure Secrets Mode — Feb 2, 2026")

# ─── ACCOUNT INPUTS ───
st.header("Step 1: Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=0.0, value=None, placeholder="Required")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", min_value=0.0, value=None, placeholder="Required")
survival_floor = st.number_input("Max Overall Drawdown Floor ($)", value=4500.0)

# ─── RISK SETTINGS ───
st.header("Step 2: Risk Settings")
risk_pct = st.slider("Risk % of Buffer", 3, 30, 8, step=1)

if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── EXECUTION ───
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor:
        st.error("❌ Enter valid account data!")
    else:
        with st.spinner("Analyzing Market..."):
            try:
                # API Call (Correct Kwargs)
                price_resp = td.price(**{"symbol": "XAU/USD"}).as_json()
                live_price = float(price_resp["price"])

                ts_15m = td.time_series(**{"symbol": "XAU/USD", "interval": "15min", "outputsize": 200}) \
                    .with_rsi(**{}) \
                    .with_ema(**{"time_period": 200}) \
                    .with_ema(**{"time_period": 50}) \
                    .with_atr(**{"time_period": 14}) \
                    .as_pandas()

                # Dynamic Columns
                cols = ts_15m.columns.tolist()
                rsi = ts_15m[[c for c in cols if 'rsi' in c.lower()][0]].iloc[0]
                atr = ts_15m[[c for c in cols if 'atr' in c.lower()][0]].iloc[0]
                ema_cols = sorted([c for c in cols if 'ema' in c.lower()])
                ema200_15, ema50_15 = ts_15m[ema_cols[0]].iloc[0], ts_15m[ema_cols[1]].iloc[0]

                # 1H Filter
                ts_1h = td.time_series(**{"symbol": "XAU/USD", "interval": "1h", "outputsize": 100}).with_ema(**{"time_period": 200}).as_pandas()
                ema200_1h = ts_1h[[c for c in ts_1h.columns if 'ema' in c.lower()][0]].iloc[0]

                # Bias & Risk
                if (live_price > ema200_15 and ema50_15 > ema200_15 and live_price > ema200_1h): bias = "BULLISH"
                elif (live_price < ema200_15 and ema50_15 < ema200_15 and live_price < ema200_1h): bias = "BEARISH"
                else: 
                    st.warning(f"🚫 Trend Misalignment. 1H EMA: ${ema200_1h:.2f}")
                    st.stop()

                sl_dist, spread = round(atr * 1.5, 2), 0.35
                rr_ratio = 4.0 if (rsi < 25 or rsi > 75) else 2.5 if (rsi < 35 or rsi > 65) else 1.8
                buffer = balance - survival_floor
                cash_risk = min(buffer * (risk_pct / 100), daily_limit)

                if rr_ratio < 1.8 or cash_risk < 20:
                    st.warning(f"Low quality setup skipped (${cash_risk:.2f} risk).")
                    st.stop()

                lots = max(round(cash_risk / ((sl_dist + spread) * 100), 2), 0.01)
                st.subheader(f"Bias → {bias}")
                st.metric("Lots", f"{lots:.2f}")
                st.write(f"**Entry:** ${live_price:.2f} | **Risk:** ${lots * (sl_dist + spread) * 100:.2f}")
                
                # History & AI
                st.session_state.saved_setups.append({"time": datetime.utcnow().strftime("%H:%M"), "bias": bias, "entry": live_price})
                st.info(get_ai_advice({"price": live_price, "rsi": rsi}, {"buffer": buffer}, {"type": bias, "entry": live_price, "risk": cash_risk}))

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
