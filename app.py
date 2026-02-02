import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai

# 1. API CONFIGURATION
# MAKE SURE BOTH ARE INSIDE "QUOTES"
TWELVE_DATA_KEY = "a7479c4fa2a24df483edd27fe4254de1"
GEMINI_KEY = "AIzaSyAs5fIJJ9bFYiS9VxeIPrsiFW-6Gq06YbY"

# Initialize Clients
td = TDClient(apikey=TWELVE_DATA_KEY)
genai.configure(api_key=GEMINI_KEY)

# Using the 2026 Stable Flash Model
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. THE AI BRAIN FUNCTION
def get_ai_advice(market_data, account_info):
    prompt = f"""
    You are an elite Gold trader. 
    Context: {market_data}
    Account: {account_info}
    Provide a 3-sentence risk assessment. Be blunt.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # This will show you the ACTUAL error if it fails
        return f"AI Error: {str(e)}"

# 3. INTERFACE SETUP
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇")
st.title("🥇 Gold Sentinel Adaptive")

# 4. STEP 1: ACCOUNT HEALTH
st.header("Step 1: Account Health")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Current Balance ($)", value=None, placeholder="Type Balance...")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", value=None, placeholder="Type Limit...")

survival_floor = st.number_input("Max Overall Drawdown Line ($)", value=4500.0)

st.header("Step 2: Risk Strategy")
risk_pct = st.slider("Risk % of Remaining Buffer", 5, 50, 30)

st.divider()

# 5. STEP 3: THE EXECUTION
if st.button('🚀 Get a Setup!'):
    if balance is None or daily_limit is None:
        st.error("❌ MATE! Enter your Balance and Daily Limit first!")
    else:
        with st.spinner('Sentinel & AI are analyzing the tape...'):
            ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200).with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14).as_pandas()
            
            live_price = ts['close'].iloc[0]
            rsi = ts['rsi'].iloc[0]
            ema_200 = ts['ema1'].iloc[0]
            ema_50 = ts['ema2'].iloc[0]
            atr = ts['atr'].iloc[0]
            
            # MATH
            sl_dist = round(atr * 1.5, 2)
            rr_ratio = 4.0 if (rsi < 25 or rsi > 75) else 2.25 if (rsi < 35 or rsi > 65) else 1.0
            tp_dist = round(sl_dist * rr_ratio, 2)
            buffer = balance - survival_floor
            cash_risk = min((buffer * (risk_pct / 100)), daily_limit)
            lots = max(round(cash_risk / (sl_dist * 100), 2), 0.01)
            bias = "BULLISH" if live_price > ema_200 else "BEARISH"

            # 6. AI CONSULTATION
            market_summary = f"Price: {live_price}, RSI: {rsi}, Trend: {bias}"
            account_summary = f"Buffer: ${buffer:.2f}, Risk: ${cash_risk:.2f}"
            ai_advice = get_ai_advice(market_summary, account_summary)

            # 7. OUTPUT
            st.subheader(f"Setup: {bias}")
            if bias == "BULLISH":
                st.success(f"📈 BUY @ ${live_price:.2f}")
                sl, tp = live_price - sl_dist, live_price + tp_dist
            else:
                st.warning(f"📉 SELL @ ${live_price:.2f}")
                sl, tp = live_price + sl_dist, live_price - tp_dist

            c1, c2, c3 = st.columns(3)
            c1.metric("Lots", f"{lots}")
            c2.metric("RR", f"1:{rr_ratio}")
            c3.metric("Risk $", f"${lots * sl_dist * 100:.2f}")
            st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")

            st.divider()
            st.subheader("🧠 Gemini's Strategic Take")
            st.info(ai_advice)
