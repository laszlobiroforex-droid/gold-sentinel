import streamlit as st
import pandas as pd
from twelvedata import TDClient

# 1. INITIALIZATION
TWELVE_DATA_API_KEY = "a7479c4fa2a24df483edd27fe4254de1"
td = TDClient(apikey=TWELVE_DATA_API_KEY)

st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇")
st.title("🥇 Gold Sentinel Pro")

# 2. DYNAMIC INPUTS
with st.sidebar:
    st.header("Account Parameters")
    balance = st.number_input("Current Balance ($)", value=4616.28)
    daily_limit = st.number_input("Daily Drawdown Left ($)", value=25.49)
    survival_floor = st.number_input("Max Overall Drawdown Line ($)", value=4500.0)
    
    st.header("Risk Strategy")
    st.caption("How much of your REMAINING buffer to risk?")
    risk_pct_of_buffer = st.slider("Risk % of Buffer", 5, 50, 20)

# 3. TRIGGER: FETCH FRESH DATA ONLY ON CLICK
if st.button('🚀 Get a Setup!'):
    with st.spinner('Analyzing Trend & Liquidity...'):
        # Fresh API Call - Prevents stale coffee-break prices
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200).with_rsi().with_ema(time_period=200).with_ema(time_period=50).as_pandas()
        
        # Data Extraction
        live_price = ts['close'].iloc[0]
        rsi = ts['rsi'].iloc[0]
        ema_200 = ts['ema1'].iloc[0] # ema1 is 200 per order of definition
        ema_50 = ts['ema2'].iloc[0]  # ema2 is 50
        
        # 4. PROFESSIONAL RISK ENGINE
        # Calculate available 'life' in the account
        available_buffer = balance - survival_floor
        # Total $ amount we are allowed to lose on this specific trade
        cash_risk = min((available_buffer * (risk_pct_of_buffer / 100)), daily_limit)
        
        # 5. ELITE TREND LOGIC
        # Trend is defined by EMA 200. Pullback is defined by EMA 50.
        trend_bias = "BULLISH" if live_price > ema_200 else "BEARISH"
        
        # Suggested SL distance based on current Gold volatility (approx $20 move)
        sl_distance = 20.0 
        tp_distance = 45.0 # targeting 1:2.25 Risk/Reward

        # 6. CALCULATE LOT SIZE (The Professional Way)
        # Gold Formula: 1 Lot = 100oz. $1 move = $100 profit/loss.
        # Lot Size = Cash Risk / (SL Distance * 100)
        calculated_lots = cash_risk / (sl_distance * 100)
        final_lots = max(round(calculated_lots, 2), 0.01) # Minimum 0.01

        # 7. OUTPUT DISPLAY
        st.subheader(f"Market Status: {trend_bias}")
        st.write(f"**Price:** ${live_price:.2f} | **RSI:** {rsi:.1f}")
        
        st.divider()
        
        # Trade Suggestions based on Pullback
        if trend_bias == "BULLISH":
            if live_price <= (ema_50 * 1.002): # Price near or below EMA 50
                st.success("✅ ELITE BUY SETUP: Trend Pullback Detected")
                entry_price = live_price
                st.write(f"**Order:** BUY LIMIT @ ${entry_price:.2f}")
            else:
                st.info("🚦 BULLISH: Waiting for pullback to EMA 50 support.")
                entry_price = ema_50
                st.write(f"**Suggested Order:** BUY LIMIT @ ${entry_price:.2f}")
            
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance

        else: # BEARISH
            if live_price >= (ema_50 * 0.998):
                st.warning("✅ ELITE SELL SETUP: Bearish Retest Detected")
                entry_price = live_price
                st.write(f"**Order:** SELL LIMIT @ ${entry_price:.2f}")
            else:
                st.info("🚦 BEARISH: Waiting for retest of EMA 50 resistance.")
                entry_price = ema_50
                st.write(f"**Suggested Order:** SELL LIMIT @ ${entry_price:.2f}")
                
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # FINAL RISK METRICS
        st.divider()
        col_lots, col_risk = st.columns(2)
        col_lots.metric("Recommended Lots", f"{final_lots}")
        col_risk.metric("Total Risk ($)", f"${final_lots * sl_distance * 100:.2f}")
        
        st.write(f"**Stop Loss:** ${sl_price:.2f}")
        st.write(f"**Take Profit:** ${tp_price:.2f}")
        st.caption(f"Based on a ${sl_distance} stop distance and {risk_pct_of_buffer}% risk of your ${available_buffer:.2f} buffer.")
