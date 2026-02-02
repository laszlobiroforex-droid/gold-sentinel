import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai
from datetime import datetime

# ─── SECURE API SETUP ────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"❌ API setup failed: {str(e)}\nCheck Streamlit Secrets (TWELVE_DATA_KEY & GEMINI_KEY)")
    st.stop()

# ─── AI AUDITOR ──────────────────────────────────────────────────────────────
def get_ai_advice(market, account, setup):
    prompt = f"""
    SYSTEM: You are a high-stakes risk auditor for a $116 buffer Gold challenge.
    USER STRATEGY: The user is intentionally using aggressive 25-30% risk to pass Phase 2. 
    DO NOT tell them to lower the risk. Audit the MATH and the ENTRY.

    ACCOUNT: ${account['buffer']:.2f} buffer left.
    MARKET: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    PROPOSED: {setup['type']} at ${setup['entry']:.2f} with ${setup['risk']:.2f} risk.

    Is the entry point elite or gambling based on the trend? 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "Brain offline. Stick to the math."

# ─── APP CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 7.1")
st.caption(f"RebelsFunding Phase 2 Protector | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── INPUTS ──────────────────────────────────────────────────────────────────
st.header("Step 1: Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")

survival_floor = st.number_input("Max Overall Drawdown Floor ($)", value=4500.0, format="%.2f")

st.header("Step 2: Risk Settings")
risk_pct = st.slider("Risk % of Buffer", 3, 50, 25, step=5)

# Session history
if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor:
        st.error("❌ Enter valid account values first!")
    else:
        with st.spinner("Fetching market data..."):
            try:
                # ─── PRICE (explicit kwargs) ─────────────────────────────────────
                price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
                live_price = float(price_data["price"])

                # ─── TIME SERIES (explicit kwargs + time_period) ─────────────────
                ts = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "15min",
                    "outputsize": 100
                }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

                # ─── SAFE COLUMN EXTRACTION ──────────────────────────────────────
                rsi_col  = next(c for c in ts.columns if 'rsi'  in c.lower())
                atr_col  = next(c for c in ts.columns if 'atr'  in c.lower())
                ema_cols = sorted([c for c in ts.columns if 'ema' in c.lower()])  # ema_1 usually 200, ema_2 = 50
                ema200_col, ema50_col = ema_cols[0], ema_cols[1]

                rsi   = ts[rsi_col].iloc[0]
                atr   = ts[atr_col].iloc[0]
                ema200 = ts[ema200_col].iloc[0]
                ema50  = ts[ema50_col].iloc[0]  # extra for future use

                # ─── BIAS & CALCS ────────────────────────────────────────────────
                bias = "BULLISH" if live_price > ema200 else "BEARISH"

                sl_dist     = round(atr * 1.5, 2)
                spread      = 0.35
                rr_ratio    = 4.0 if (rsi < 25 or rsi > 75) else 2.5 if (rsi < 35 or rsi > 65) else 1.8

                buffer      = balance - survival_floor
                cash_risk   = min(buffer * (risk_pct / 100), daily_limit)

                # Quality gate (adjust <20 if too strict for your style)
                if cash_risk < 20 or rr_ratio < 1.8:
                    st.warning(f"Risk too low (${cash_risk:.2f}) or RR weak. Skipping to protect buffer.")
                    st.stop()

                lots        = max(round(cash_risk / ((sl_dist + spread) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist + spread) * 100, 2)

                entry = live_price
                sl = round(entry - sl_dist, 2) if bias == "BULLISH" else round(entry + sl_dist, 2)
                tp = round(entry + (sl_dist * rr_ratio), 2) if bias == "BULLISH" else round(entry - (sl_dist * rr_ratio), 2)

                # ─── DISPLAY ─────────────────────────────────────────────────────
                st.subheader(f"Sentinel Bias: {bias}")
                if bias == "BULLISH":
                    st.success(f"📈 BUY @ ${entry:.2f}")
                else:
                    st.warning(f"📉 SELL @ ${entry:.2f}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lots", f"{lots:.2f}")
                c2.metric("R:R", f"1:{rr_ratio}")
                c3.metric("Total Risk", f"${actual_risk:.2f}")
                c4.metric("Buffer", f"${buffer:.2f}")

                st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")

                # Chart
                chart_df = ts[['close', ema200_col, ema50_col]].tail(50).rename(
                    columns={'close': 'Price', ema200_col: 'EMA 200', ema50_col: 'EMA 50'}
                )
                st.line_chart(chart_df)

                # ─── SAVE & AI ───────────────────────────────────────────────────
                setup_record = {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "bias": bias,
                    "entry": round(entry, 2),
                    "risk": actual_risk,
                    "rr": rr_ratio
                }
                st.session_state.saved_setups.append(setup_record)

                # Optional alert sound (uncomment if wanted)
                # st.components.v1.html("""<audio autoplay><source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav"></audio>""", height=0)

                st.divider()
                st.subheader("🧠 Gemini Risk Auditor")
                setup_info = {"type": bias, "entry": entry, "risk": actual_risk}
                st.info(get_ai_advice({"price": live_price, "rsi": rsi}, {"buffer": buffer}, setup_info))

            except Exception as e:
                st.error(f"❌ Market Error: {str(e)}\n(Upgrade twelvedata package or check rate limits?)")

# ─── HISTORY ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups (session)")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(8), use_container_width=True, hide_index=True)
else:
    st.info("No setups saved yet.")
