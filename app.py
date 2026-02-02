import streamlit as st
from twelvedata import TDClient

# 1. SETUP YOUR KEYS
TWELVE_DATA_API_KEY = "a7479c4fa2a24df483edd27fe4254de1" # Get yours at twelvedata.com
td = TDClient(apikey=TWELVE_DATA_API_KEY)

# 2. THE INTERFACE
st.title("Gold Sentinel")
st.write("Survival Line: $4,500.00")

col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance", value=4616.28)
with col2:
    daily_limit = st.number_input("Daily Drawdown Left", value=25.49)

# 3. THE MAGIC BUTTON
if st.button('Get SETUP!'):
    with st.spinner('Scanning the Waterfall...'):
        # A. Fetch Live Price & RSI
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=14).with_rsi().as_pandas()
        price = ts['close'].iloc[0]
        rsi = ts['rsi'].iloc[0]
        
        # B. Risk Logic
        buffer_to_death = balance - 4500
        max_loss = min(daily_limit, buffer_to_death, 50.0) # Never risk more than $25
        
        # C. Strategy Decision
        st.subheader(f"Live Price: ${price:.2f} | RSI: {rsi:.1f}")
        
        if rsi < 30 and price < 4450:
            st.success("🔥 HIGH PROBABILITY BUY: Ghost Floor Near")
            st.write(f"**Entry:** Market @ ${price}")
            st.write(f"**SL:** ${price - 25:.2f} (Hard $25 Risk)")
            st.write(f"**TP:** $4,420.00")
        elif rsi > 70:
            st.warning("⚠️ OVERBOUGHT: Wait for rejection at $4,894 (Friday Close)")
        else:
            st.info("🚦 STATUS: Road is messy. Stay Parked.")
            
        st.caption(f"Note: 0.01 Lot will risk ${max_loss:.2f} if SL is hit.")
