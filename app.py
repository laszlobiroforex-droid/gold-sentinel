import streamlit as st
import pandas as pd
from twelvedata import TDClient
import google.generativeai as genai

# 1. API CONFIGURATION
TWELVE_DATA_KEY = "a7479c4fa2a24df483edd27fe4254de1"
GEMINI_KEY = "AIzaSyAs5fIJJ9bFYiS9VxeIPrsiFW-6Gq06YbY"

# Initialize Clients
td = TDClient(apikey=TWELVE_DATA_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# 2. THE AI BRAIN FUNCTION
def get_ai_advice(market_data, account_info, setup_details):
    prompt = f"""
    SYSTEM: You are a risk-management auditor. Use ONLY these numbers:
    ACCOUNT: {account_info}
    MARKET: {market_data}
    PROPOSED SETUP: {setup_details}

    TASK: 
    1. Validate the {setup_details['type']} at {setup_details['entry']}.
    2. Audit the ${setup_details['risk']} risk against the available buffer.
    3. Be blunt: Is this 'Elite' or 'Gambling'? (3 sentences max).
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Brain Error: {str(e)}"

# 3. INTERFACE
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇")
st.title("🥇 Gold Sentinel Adaptive")

# STEP 1: ACCOUNT HEALTH (Fully Manual)
st.header("Step 1: Account Health")
col1, col2 = st.columns(2)
with col1:
    # No default value here; you must enter it
    balance = st.number_input("Current Balance ($)", value=None, placeholder="Type Balance...")
with col2:
    daily_limit = st.number_input("Daily Drawdown Left ($)", value=None, placeholder="Type Limit...")

survival_floor = st.number_input("Max Overall Drawdown Line ($)", value=4500.0)

st.header("Step 2: Risk Strategy")
risk_pct = st.slider("Risk % of Available Buffer", 5, 50, 30)

st.divider()

# 4. EXECUTION
if st.button('🚀 Get a Setup!'):
    if balance is None or daily_limit is None:
        st.error("❌ Enter your data first! The Sentinel cannot calculate risk without your balance.")
    else:
        with st.spinner('Sentinel & AI Analyzing...'):
            ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200).with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14).as_pandas()
            
            live_price = ts['close'].iloc[0]
            rsi = ts['rsi'].iloc[0]
            ema_200 = ts['ema1'].iloc[0]
            atr = ts['atr'].iloc[0]
            
            # DYNAMIC MATH ENGINE
            sl_dist = round(atr * 1.5, 2)
            rr_ratio = 4.0 if (rsi < 25 or rsi > 75) else 2.25 if (rsi < 35 or rsi > 65) else 1.0
            tp_dist = round(sl_dist * rr_ratio, 2)
            
            # Calculate the ACTUAL buffer dynamically
            available_buffer = balance - survival_floor
            cash_risk = min((available_buffer * (risk_pct / 100)), daily_limit)
            
            lots = max(round(cash_risk / (sl_dist * 100), 2), 0.01)
            bias = "BULLISH" if live_price > ema_200 else "BEARISH"

            # PREPARE DATA FOR AI
            entry = live_price
            sl = entry - sl_dist if bias == "BULLISH" else entry + sl_dist
            tp = entry + tp_dist if bias == "BULLISH" else entry - tp_dist

            setup_details = {
                "type": f"{bias} Order",
                "entry": round(entry, 2),
                "risk": round(lots * sl_dist * 100, 2),
                "rr": f"1:{rr_ratio}"
            }

            # AI CONSULT
            market_summary = f"Price: {live_price}, RSI: {rsi}, Bias: {bias}"
            account_summary = f"Total Buffer: ${available_buffer:.2f}, Risking: ${cash_risk:.2f}"
            ai_advice = get_ai_advice(market_summary, account_summary, setup_details)

            # OUTPUT
            st.subheader(f"Sentinel Bias: {bias}")
            if bias == "BULLISH":
                st.success(f"📈 BUY @ ${entry:.2f}")
            else:
                st.warning(f"📉 SELL @ ${entry:.2f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Lots", f"{lots}")
            c2.metric("RR", f"1:{rr_ratio}")
            c3.metric("Total Risk", f"${setup_details['risk']}")
            st.write(f"**SL:** ${sl:.2f} | **TP:** ${tp:.2f}")

            st.divider()
            st.subheader("🧠 Gemini's Strategic Take")
            st.info(ai_advice)
