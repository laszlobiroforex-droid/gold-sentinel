import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai
from datetime import datetime

# ─── API CONFIGURATION ───────────────────────────────────────────────────────
TWELVE_DATA_KEY = "a7479c4fa2a24df483edd27fe4254de1"
GEMINI_KEY      = "AIzaSyAs5fIJJ9bFYiS9VxeIPrsiFW-6Gq06YbY"

td = TDClient(apikey=TWELVE_DATA_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ─── AI RISK AUDITOR ─────────────────────────────────────────────────────────
def get_ai_advice(market, account, setup):
    prompt = f"""
    You are a strict Gold risk auditor for a RebelsFunding Phase 2 account.
    Account Buffer above hard floor: ${account['buffer']:.2f}
    Market: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    Proposed: {setup['type']} at ${setup['entry']:.2f} with ${setup['risk']:.2f} risk.

    Audit the size relative to the buffer and prop rules. Be blunt.
    Elite execution or Gambling? 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Brain Error: {str(e)}"

# ─── STREAMLIT SETUP ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 6.8 (Fixed)")
st.caption("RebelsFunding Phase 2 Protector — Feb 2, 2026")

# ─── ACCOUNT INPUTS ──────────────────────────────────────────────────────────
st.header("Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=4500.0, value=4616.28, step=0.01)
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", min_value=0.0, value=None, placeholder="Required")

survival_floor = st.number_input("Max Overall Drawdown Floor ($)", value=4500.0)

# ─── RISK SETTINGS ───────────────────────────────────────────────────────────
st.header("Risk Settings")
risk_pct = st.slider("Risk % of Available Buffer", 3, 30, 8, step=1)

if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── MAIN BUTTON LOGIC ───────────────────────────────────────────────────────
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor:
        st.error("❌ Invalid balance. Protect the $4500 floor!")
    else:
        with st.spinner("Fetching market data..."):
            try:
                # ─── FIX: USE EXPLICIT KWARGS FOR ALL TD CALLS ───
                price_resp = td.price(**{"symbol": "XAU/USD"}).as_json()
                live_price = float(price_resp["price"])

                # 15-min Data with named technical parameters
                ts_15m = td.time_series(**{"symbol": "XAU/USD", "interval": "15min", "outputsize": 200}) \
                    .with_rsi(**{}) \
                    .with_ema(**{"time_period": 200}) \
                    .with_ema(**{"time_period": 50}) \
                    .with_atr(**{"time_period": 14}) \
                    .as_pandas()

                rsi       = ts_15m["rsi"].iloc[0]
                atr       = ts_15m["atr"].iloc[0]
                ema200_15 = ts_15m["ema_1"].iloc[0]
                ema50_15  = ts_15m["ema_2"].iloc[0]

                # 1-hour Filter
                ts_1h = td.time_series(**{"symbol": "XAU/USD", "interval": "1h", "outputsize": 100}) \
                    .with_ema(**{"time_period": 200}) \
                    .as_pandas()
                ema200_1h = ts_1h["ema_1"].iloc[0]

                # ─── TREND ALIGNMENT ───
                if (live_price > ema200_15 and ema50_15 > ema200_15 and live_price > ema200_1h):
                    bias = "BULLISH"
                elif (live_price < ema200_15 and ema50_15 < ema200_15 and live_price < ema200_1h):
                    bias = "BEARISH"
                else:
                    st.warning(f"🚫 No setup — trend misalignment. 1H EMA 200: ${ema200_1h:.2f}")
                    st.stop()

                # ─── RISK ENGINE ───
                sl_dist     = round(atr * 1.5, 2)
                spread_cost = 0.35
                rr_ratio    = 4.0 if (rsi < 25 or rsi > 75) else 2.5 if (rsi < 35 or rsi > 65) else 1.8

                buffer     = balance - survival_floor
                cash_risk  = min(buffer * (risk_pct / 100), daily_limit)

                if rr_ratio < 1.8 or cash_risk < 20:
                    st.warning(f"Setup skipped — low quality (${cash_risk:.2f} risk).")
                    st.stop()

                tp_dist     = round(sl_dist * rr_ratio, 2)
                lots        = max(round(cash_risk / ((sl_dist + spread_cost) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist + spread_cost) * 100, 2)

                entry = live_price
                sl = round(entry - sl_dist, 2) if bias == "BULLISH" else round(entry + sl_dist, 2)
                tp = round(entry + tp_dist, 2) if bias == "BULLISH" else round(entry - tp_dist, 2)

                # DISPLAY
                st.subheader(f"Bias → {bias}")
                if bias == "BULLISH": st.success(f"📈 BUY @ ${entry:.2f}")
                else: st.warning(f"📉 SELL @ ${entry:.2f}")

                cols = st.columns(4)
                cols[0].metric("Lots", f"{lots:.2f}")
                cols[1].metric("R:R", f"1:{rr_ratio}")
                cols[2].metric("Risk $", f"${actual_risk:.2f}")
                cols[3].metric("Buffer", f"${buffer:.2f}")

                st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")
                st.line_chart(ts_15m[['close', 'ema_1', 'ema_2']].tail(60).rename(columns={'close':'Price', 'ema_1':'EMA200', 'ema_2':'EMA50'}))

                # SAVE HISTORY
                setup_record = {"time": datetime.utcnow().strftime("%H:%M:%S"), "bias": bias, "entry": entry, "risk": actual_risk, "rr": rr_ratio}
                st.session_state.saved_setups.append(setup_record)

                # AI AUDIT
                st.divider()
                st.subheader("🧠 Gemini Risk Auditor")
                st.info(get_ai_advice({"price": live_price, "rsi": rsi}, {"buffer": buffer}, {"type": bias, "entry": entry, "risk": actual_risk}))

            except Exception as e:
                st.error(f"❌ Market Data Error: {str(e)}")

# ─── HISTORY ───
if st.session_state.saved_setups:
    st.divider()
    st.subheader("Session History")
    st.dataframe(pd.DataFrame(st.session_state.saved_setups).tail(5), use_container_width=True)
