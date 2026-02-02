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
    st.error(f"❌ API setup failed: {str(e)}\nVerify secrets: TWELVE_DATA_KEY and GEMINI_KEY")
    st.stop()

# ─── IMPROVED AI AUDITOR ─────────────────────────────────────────────────────
def get_ai_advice(market, account, setup, extra_context):
    prompt = f"""
    SYSTEM: You are a high-stakes risk auditor for RebelsFunding Copper 5K Phase 2 challenge.
    USER CONSTRAINTS & STRATEGY:
    - Minimum lot size is 0.01 on XAU/USD → 0.01 lot = $1 P/L per $1 price move.
    - This creates a natural minimum risk floor of roughly $15–$60 depending on SL distance (cannot go meaningfully below ~$20 risk without violating min lot).
    - The user is DELIBERATELY using aggressive 25–40% risk-per-trade of remaining buffer to push through Phase 2 quickly.
    - DO NOT advise lowering the risk percentage — accept it as the chosen style.
    - Focus ONLY on: math consistency, entry quality (trend confluence from provided indicators), prop-rule safety (daily/overall DD not breached), and whether the entry looks elite or gambling given the data.

    ACCOUNT: ${account['buffer']:.2f} buffer left above $4500 floor.
    MARKET CONTEXT: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}, ATR(14) {extra_context['atr']:.2f}, EMA200 {extra_context['ema200']:.2f}, EMA50 {extra_context['ema50']:.2f}
    PROPOSED SETUP: {setup['type']} at ${setup['entry']:.2f} with ${setup['risk']:.2f} risk (SL distance implied ~{extra_context['sl_dist']:.2f} points).

    Be blunt: Is this an elite, math-consistent entry with some confluence, or pure gambling? 3 sentences max.
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return "Auditor offline — verify math manually."

# ─── APP CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel Adaptive 7.6")
st.caption(f"Phase 2 Protector | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── INPUTS ──────────────────────────────────────────────────────────────────
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

# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────
if st.button("🚀 Get a Setup!", type="primary", use_container_width=True):
    if balance is None or daily_limit is None or balance <= survival_floor:
        st.error("❌ Enter valid account values!")
    else:
        with st.spinner("Fetching XAU/USD data..."):
            try:
                # Price fetch
                price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
                live_price = float(price_data["price"])

                # Time series (still needed for RSI, ATR, EMAs in logic & auditor)
                ts = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "15min",
                    "outputsize": 100
                }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

                # Column extraction
                rsi_col  = next((c for c in ts.columns if 'rsi'  in c.lower()), None)
                atr_col  = next((c for c in ts.columns if 'atr'  in c.lower()), None)
                ema_cols = [c for c in ts.columns if 'ema' in c.lower()]
                ema_cols.sort()
                ema200_col = ema_cols[0] if len(ema_cols) > 0 else None
                ema50_col  = ema_cols[1] if len(ema_cols) > 1 else None

                rsi    = ts[rsi_col].iloc[0] if rsi_col else 50.0
                atr    = ts[atr_col].iloc[0] if atr_col else 0.0
                ema200 = ts[ema200_col].iloc[0] if ema200_col else live_price
                ema50  = ts[ema50_col].iloc[0] if ema50_col else live_price

                bias = "BULLISH" if live_price > ema200 else "BEARISH"

                sl_dist = round(atr * 1.5, 2)
                spread  = 0.35
                rr_ratio = 4.0 if (rsi < 25 or rsi > 75) else 2.5 if (rsi < 35 or rsi > 65) else 1.8

                buffer     = balance - survival_floor
                cash_risk  = min(buffer * (risk_pct / 100), daily_limit)

                if cash_risk < 20 or rr_ratio < 1.8:
                    st.warning(f"Risk ${cash_risk:.2f} too small for min lot or RR weak — skipping.")
                    st.stop()

                lots        = max(round(cash_risk / ((sl_dist + spread) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist + spread) * 100, 2)

                entry = live_price
                sl = round(entry - sl_dist, 2) if bias == "BULLISH" else round(entry + sl_dist, 2)
                tp = round(entry + (sl_dist * rr_ratio), 2) if bias == "BULLISH" else round(entry - (sl_dist * rr_ratio), 2)

                # ─── OUTPUT ──────────────────────────────────────────────────────
                st.subheader(f"Bias: {bias}")
                if bias == "BULLISH":
                    st.success(f"📈 BUY @ ${entry:.2f}")
                else:
                    st.warning(f"📉 SELL @ ${entry:.2f}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lots", f"{lots:.2f}")
                c2.metric("R:R", f"1:{rr_ratio}")
                c3.metric("Risk \( ", f" \){actual_risk:.2f}")
                c4.metric("Buffer $", f"{buffer:.2f}")

                st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")

                # ─── TRADINGVIEW CHART EMBED (OANDA:XAUUSD, 15 min, compact) ─────
                st.subheader("XAU/USD 15 min (OANDA via TradingView)")
                st.components.v1.html("""
                <div class="tradingview-widget-container">
                  <div id="tradingview_widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget(
                  {
                    "autosize": true,
                    "symbol": "OANDA:XAUUSD",
                    "interval": "15",
                    "timezone": "Etc/UTC",
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "allow_symbol_change": false,
                    "hide_side_toolbar": true,
                    "studies": [],
                    "show_popup_button": false,
                    "popup_width": "1000",
                    "popup_height": "650",
                    "container_id": "tradingview_widget",
                    "height": 450,
                    "width": "100%",
                    "range": "1D"  // shows recent ~1 day, but you see current action
                  }
                  );
                  </script>
                </div>
                """, height=480)

                st.caption("Live 15-min chart from TradingView (OANDA:XAUUSD) – recent price action only")

                # Save to history
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
                st.error(f"Market error: {str(e)}\nTry refreshing or check rate limits")

# ─── HISTORY ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(10), use_container_width=True, hide_index=True)
else:
    st.info("No setups yet.")
