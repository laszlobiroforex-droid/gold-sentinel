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
    Buffer above hard floor: ${account['buffer']:.2f}
    Market: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    Proposed: {setup['type']} at ${setup['entry']:.2f} with ${setup['risk']:.2f} risk.

    Audit the size relative to the buffer and prop rules. Be very blunt.
    Elite execution or Gambling? 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Brain Error: {str(e)}"

# ─── STREAMLIT SETUP ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 6.8")
st.caption("RebelsFunding Phase 2 Protector — Not financial advice — Use for idea generation only")

# ─── ACCOUNT INPUTS ──────────────────────────────────────────────────────────
st.header("Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", 
                             min_value=4500.0, value=4616.28, step=0.01, 
                             format="%.2f")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", 
                                 min_value=0.0, value=None, 
                                 placeholder="Required", format="%.2f")

survival_floor = st.number_input("Max Overall Drawdown Floor ($)", 
                                value=4500.0, format="%.2f")

# ─── RISK SETTINGS ───────────────────────────────────────────────────────────
st.header("Risk Settings")
risk_pct = st.slider("Risk % of Available Buffer", 3, 30, 8, step=1)

# Session state for history
if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# ─── MAIN BUTTON LOGIC ───────────────────────────────────────────────────────
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor + 10:
        st.error("❌ Invalid or critically low balance. Protect the $4500 floor!")
    else:
        with st.spinner("Fetching market data (15m + 1h) ..."):
            try:
                now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.caption(f"Data timestamp: {now_utc} | XAU/USD")

                # ─── REAL-TIME PRICE ─────────────────────────────────────────────
                price_resp = td.price(symbol="XAU/USD").as_json()
                live_price = float(price_resp["price"])

                # ─── 15-MIN TIMEFRAME ────────────────────────────────────────────
                ts_15m = td.time_series(
                    symbol="XAU/USD", interval="15min", outputsize=200
                ).with_rsi().with_ema(200).with_ema(50).with_atr(14).as_pandas()

                rsi       = ts_15m["rsi"].iloc[0]
                atr       = ts_15m["atr"].iloc[0]
                ema200_15 = ts_15m["ema_1"].iloc[0]
                ema50_15  = ts_15m["ema_2"].iloc[0]

                # ─── 1-HOUR TIMEFRAME FILTER ─────────────────────────────────────
                ts_1h = td.time_series(
                    symbol="XAU/USD", interval="1h", outputsize=100
                ).with_ema(200).as_pandas()
                ema200_1h = ts_1h["ema_1"].iloc[0]

                # ─── TREND ALIGNMENT CHECK ───────────────────────────────────────
                if (live_price > ema200_15 and 
                    ema50_15  > ema200_15 and 
                    live_price > ema200_1h):
                    bias = "BULLISH"
                elif (live_price < ema200_15 and 
                      ema50_15  < ema200_15 and 
                      live_price < ema200_1h):
                    bias = "BEARISH"
                else:
                    st.warning(
                        f"🚫 No setup — trend misalignment\n"
                        f"1H EMA 200 is at ${ema200_1h:.2f}"
                    )
                    st.stop()

                # ─── RISK & POSITION CALC ────────────────────────────────────────
                sl_dist     = round(atr * 1.5, 2)
                spread_cost = 0.35
                rr_ratio    = 4.0 if (rsi < 25 or rsi > 75) else \
                              2.5 if (rsi < 35 or rsi > 65) else 1.8

                buffer     = balance - survival_floor
                cash_risk  = min(buffer * (risk_pct / 100), daily_limit)

                # QUALITY GATE ─ very important with small buffer
                if rr_ratio < 1.8 or cash_risk < 20:
                    st.warning(
                        f"Setup skipped — quality too low or risk too small "
                        f"(${cash_risk:.2f}). Protect the account."
                    )
                    st.stop()

                tp_dist     = round(sl_dist * rr_ratio, 2)
                lots        = max(round(cash_risk / ((sl_dist + spread_cost) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist + spread_cost) * 100, 2)

                entry = live_price
                if bias == "BULLISH":
                    sl = round(entry - sl_dist, 2)
                    tp = round(entry + tp_dist, 2)
                    st.success(f"📈 BUY  @  ${entry:.2f}")
                else:
                    sl = round(entry + sl_dist, 2)
                    tp = round(entry - tp_dist, 2)
                    st.warning(f"📉 SELL  @  ${entry:.2f}")

                # ─── DISPLAY METRICS ─────────────────────────────────────────────
                cols = st.columns(4)
                cols[0].metric("Lots", f"{lots:.2f}")
                cols[1].metric("R:R", f"1 : {rr_ratio}")
                cols[2].metric("Risk \( ", f" \){actual_risk:.2f}")
                cols[3].metric("Buffer left", f"${buffer:.2f}")

                st.write(f"**SL** → ${sl:.2f}    **TP** → ${tp:.2f}")
                st.caption(f"SL distance used: {sl_dist} pts  •  spread approx ${spread_cost}")

                # Chart
                chart_df = ts_15m[['close', 'ema_1', 'ema_2']].tail(60).copy()
                chart_df.columns = ['Price', 'EMA 200 (15m)', 'EMA 50 (15m)']
                st.line_chart(chart_df)

                st.caption(f"1H EMA 200 filter: ${ema200_1h:.2f}")

                # ─── SAVE & ALERT ────────────────────────────────────────────────
                setup_record = {
                    "time": now_utc,
                    "bias": bias,
                    "entry": round(entry, 2),
                    "sl": sl,
                    "tp": tp,
                    "lots": lots,
                    "rr": rr_ratio,
                    "risk": actual_risk
                }
                st.session_state.saved_setups.append(setup_record)

                st.success("Setup saved to session history!")

                # Short alert sound
                st.components.v1.html("""
                    <audio autoplay volume="0.4">
                      <source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav">
                    </audio>
                """, height=0)

                # ─── GEMINI AUDIT ────────────────────────────────────────────────
                st.divider()
                st.subheader("🧠 Gemini Risk Auditor")
                setup_info = {"type": bias, "entry": entry, "risk": actual_risk}
                st.info(get_ai_advice(
                    {"price": live_price, "rsi": rsi},
                    {"buffer": buffer},
                    setup_info
                ))

            except Exception as e:
                st.error(f"Market data error: {str(e)}\n(Check API keys, rate limits, or internet)")

# ─── HISTORY SECTION ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Session History (most recent first)")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(
        df.sort_values("time", ascending=False).head(10)[
            ["time", "bias", "entry", "risk", "rr", "lots"]
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No setups saved in this session yet.")
