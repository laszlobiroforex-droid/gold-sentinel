import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai
from datetime import datetime

# ─── SECURE API INIT ─────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"❌ API setup failed: {str(e)}\nVerify secrets.")
    st.stop()

# ─── AI AUDITOR ──────────────────────────────────────────────────────────────
def get_ai_advice(market, account, setup, extra_context):
    prompt = f"""
    SYSTEM: You are a high-stakes risk auditor for RebelsFunding Copper 5K Phase 2 challenge.
    USER CONSTRAINTS & STRATEGY:
    - Min lot 0.01 on XAU/USD → $1 P/L per $1 move.
    - Natural risk floor ~$15–$60 due to min lot + SL distances.
    - User deliberately runs aggressive 25–40% risk of buffer to pass Phase 2 fast.
    - DO NOT suggest lowering % risk — accept the style.
    - Judge only math, entry quality (pullback confluence, trend support), prop DD safety.
    - Prefer pullback entries over blind trend chases.

    ACCOUNT: ${account['buffer']:.2f} buffer left.
    MARKET: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}, ATR {extra_context['atr']:.2f}, EMA200 {extra_context['ema200']:.2f}, EMA50 {extra_context['ema50']:.2f}
    SETUP: {setup['type']} @ ${setup['entry']:.2f} risking ${setup['risk']:.2f} (SL ~{extra_context['sl_dist']:.2f} pts).

    Blunt verdict: Elite pullback entry with confluence or gambling? 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return "Auditor offline."

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 7.9")
st.caption(f"Phase 2 Protector | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Inputs
st.header("Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", min_value=0.0, value=None, placeholder="Required", format="%.2f")

survival_floor = st.number_input("Max Drawdown Floor ($)", value=4500.0, format="%.2f")

st.header("Risk Settings")
risk_pct = st.slider("Risk % of Buffer", 10, 50, 25, step=5)

if 'saved_setups' not in st.session_state:
    st.session_state.saved_setups = []

# Main button
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor:
        st.error("❌ Enter valid values!")
    else:
        with st.spinner("Analyzing..."):
            try:
                price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
                live_price = float(price_data["price"])

                ts = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "15min",
                    "outputsize": 100
                }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

                rsi_col = next((c for c in ts.columns if 'rsi' in c.lower()), None)
                atr_col = next((c for c in ts.columns if 'atr' in c.lower()), None)
                ema_cols = sorted([c for c in ts.columns if 'ema' in c.lower()])
                ema200_col = ema_cols[0] if ema_cols else None
                ema50_col = ema_cols[1] if len(ema_cols) > 1 else None

                rsi = ts[rsi_col].iloc[0] if rsi_col else 50.0
                atr = ts[atr_col].iloc[0] if atr_col else 0.0
                ema200 = ts[ema200_col].iloc[0] if ema200_col else live_price
                ema50 = ts[ema50_col].iloc[0] if ema50_col else live_price

                # ─── PULLBACK-FOCUSED BIAS ──────────────────────────────────────
                sl_dist = round(atr * 1.5, 2)
                pullback_zone = sl_dist * 1.0  # price near EMA50 within ~1 ATR

                if live_price > ema200:  # overall uptrend
                    if abs(live_price - ema50) <= pullback_zone and rsi > 35:  # pullback to EMA50, not oversold
                        bias = "BULLISH PULLBACK"
                    else:
                        bias = "NO SETUP – wait for pullback in uptrend"
                else:  # overall downtrend
                    if abs(live_price - ema50) <= pullback_zone and rsi < 65:  # rally to EMA50, not overbought
                        bias = "BEARISH PULLBACK"
                    else:
                        bias = "NO SETUP – wait for rally in downtrend"

                if "NO SETUP" in bias:
                    st.warning(bias)
                    st.stop()

                rr_ratio = 4.0 if (rsi < 25 or rsi > 75) else 2.5 if (rsi < 35 or rsi > 65) else 1.3  # lowered min RR

                buffer = balance - survival_floor
                cash_risk = min(buffer * (risk_pct / 100), daily_limit)

                if cash_risk < 20:
                    st.warning(f"Risk ${cash_risk:.2f} too small for min lot — skipping.")
                    st.stop()

                lots = max(round(cash_risk / ((sl_dist + 0.35) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist + 0.35) * 100, 2)

                entry = live_price
                sl = round(entry - sl_dist, 2) if "BULLISH" in bias else round(entry + sl_dist, 2)
                tp = round(entry + (sl_dist * rr_ratio), 2) if "BULLISH" in bias else round(entry - (sl_dist * rr_ratio), 2)

                # Display
                st.subheader(f"Bias: {bias}")
                if "BULLISH" in bias:
                    st.success(f"📈 BUY @ ${entry:.2f}")
                else:
                    st.warning(f"📉 SELL @ ${entry:.2f}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lots", f"{lots:.2f}")
                c2.metric("R:R", f"1:{rr_ratio}")
                c3.metric("Risk \( ", f" \){actual_risk:.2f}")
                c4.metric("Buffer $", f"{buffer:.2f}")

                st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")

                # Save
                setup_record = {
                    "time": datetime.utcnow().strftime("%H:%M UTC"),
                    "bias": bias,
                    "entry": round(entry, 2),
                    "risk": actual_risk,
                    "rr": rr_ratio
                }
                st.session_state.saved_setups.append(setup_record)

                # Auditor
                st.divider()
                st.subheader("🧠 Gemini Auditor")
                market_data = {"price": live_price, "rsi": rsi}
                account_data = {"buffer": buffer}
                setup_data = {"type": bias, "entry": entry, "risk": actual_risk}
                extra = {"atr": atr, "ema200": ema200, "ema50": ema50, "sl_dist": sl_dist}
                st.info(get_ai_advice(market_data, account_data, setup_data, extra))

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─── HISTORY ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(10), use_container_width=True, hide_index=True)
else:
    st.info("No setups yet.")
