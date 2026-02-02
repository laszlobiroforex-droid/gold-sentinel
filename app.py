import streamlit as st
import pandas as pd
from twelvedata import TDClient

# 1. INITIALIZATION
TWELVE_DATA_API_KEY = "a7479c4fa2a24df483edd27fe4254de1"
td = TDClient(apikey=TWELVE_DATA_API_KEY)

st.set_page_config(page_title="Gold Sentinel Adaptive", page_icon="🥇")
st.title("🥇 Gold Sentinel Adaptive")

# 2. INPUTS - MANUAL ENTRY (No Defaults)
st.header("Step 1: Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", value=None, placeholder="Enter Balance...")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", value=None, placeholder="Enter Limit...")

survival_floor = st.number_input("Max Overall Drawdown Line ($)", value=4500.0)

st.header("Step 2: Risk Strategy")
risk_pct_of_buffer = st.slider("Risk % of Remaining Buffer", 5, 50, 30)

st.divider()

# 3. TRIGGER: FETCH FRESH DATA
if st.button('🚀 Get a Setup!'):
    # Validation Check: Ensure user entered their data
    if balance is None or daily_limit is None:
        st.error("❌ MATE! Enter your Balance and Daily Limit first!")
    else:
        with st.spinner('Analyzing Volatility & Trend...'):
            available_buffer = balance - survival_floor
            
            # Fetching fresh data with indicators
            ts = td.time_series(
                symbol="XAU/USD", 
                interval="15min", 
                outputsize=200
            ).with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14).as_pandas()
            
            live_price = ts['close'].iloc[0]
            rsi = ts['rsi'].iloc[0]
            ema_200 = ts['ema1'].iloc[0]
            ema_50 = ts['ema2'].iloc[0]
            atr = ts['atr'].iloc[0]
            
            # 4. VOLATILITY-BASED STOP LOSS
            sl_distance = round(atr * 1.5, 2)
            
            # 5. DYNAMIC RR LOGIC
            if rsi < 25 or rsi > 75:
                rr_ratio = 4.0
                setup_quality = "ELITE (High RR Reversal)"
            elif rsi < 35 or rsi > 65:
                rr_ratio = 2.25
                setup_quality = "STRONG (Trend Continuation)"
            else:
                rr_ratio = 1.0
                setup_quality = "SCALP (Low Confidence)"
                
            tp_distance = round(sl_distance * rr_ratio, 2)

            # 6. POSITION SIZING
            cash_risk = min((available_buffer * (risk_pct_of_buffer / 100)), daily_limit)
            calculated_lots = cash_risk / (sl_distance * 100)
            final_lots = max(round(calculated_lots, 2), 0.01)

            # 7. DIRECTIONAL BIAS
            trend_bias = "BULLISH" if live_price > ema_200 else "BEARISH"

            # 8. OUTPUT
            st.subheader(f"Setup Type: {setup_quality}")
            st.write(f"**Price:** ${live_price:.2f} | **Trend:** {trend_bias}")
            
            st.divider()
            
            if trend_bias == "BULLISH":
                entry_price = live_price
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
                st.success(f"📈 SUGGESTION: BUY @ ${entry_price:.2f}")
            else:
                entry_price = live_price
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
                st.warning(f"📉 SUGGESTION: SELL @ ${entry_price:.2f}")

            # FINAL METRICS
            col_lots, col_rr, col_risk = st.columns(3)
            col_lots.metric("Lots", f"{final_lots}")
            col_rr.metric("RR Ratio", f"1:{rr_ratio}")
            col_risk.metric("Total Risk", f"${final_lots * sl_distance * 100:.2f}")
            
            st.write(f"**Stop Loss:** ${sl_price:.2f}")
            st.write(f"**Take Profit:** ${tp_price:.2f}")
