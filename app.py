import streamlit as st
import pandas as pd
import numpy as np
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
from pywebpush import webpush, WebPushException
import json
from datetime import datetime

# ─── API & PUSH CONFIG ───
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    grok_client = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
except Exception as e:
    st.error(f"API Setup Error: {e}")
    st.stop()

# ─── PUSH NOTIFICATION SENDER ───
def send_push_notification(title, body):
    sub_info = st.session_state.get('push_sub')
    if not sub_info:
        return
    try:
        webpush(
            subscription_info=sub_info,
            data=json.dumps({"title": title, "body": body}),
            vapid_private_key=st.secrets["VAPID_PRIVATE_KEY"],
            vapid_claims={"sub": "mailto:admin@gold-sentinel.com"}
        )
    except WebPushException as ex:
        print(f"Push failed: {ex}")

# ─── FRACTAL LEVELS ───
def get_fractal_levels(df, window=5):
    levels = []
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            levels.append(('RES', round(df['high'].iloc[i], 2)))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            levels.append(('SUP', round(df['low'].iloc[i], 2)))
    return levels

# ─── UI SETUP ───
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction")

# ─── JAVASCRIPT: NOTIFICATION ENABLER ───
# This registers the service worker and grabs your push 'address'
st.components.v1.html(f"""
<script>
    const vapidKey = "{st.secrets['VAPID_PUBLIC_KEY']}";
    
    function urlBase64ToUint8Array(base64String) {{
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = window.atob(base64);
        return Uint8Array.from([...rawData].map((char) => char.charCodeAt(0)));
    }}

    async function subscribe() {{
        const reg = await navigator.serviceWorker.register('/sw.js');
        const sub = await reg.pushManager.subscribe({{
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array(vapidKey)
        }});
        // Send subscription to Streamlit session
        window.parent.postMessage({{type: 'push_subscription', data: sub}}, "*");
    }}

    if ('serviceWorker' in navigator) {{
        subscribe().catch(err => console.error(err));
    }}
</script>
""", height=0)

# ─── INPUTS ───
st.header("Account Settings")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Balance ($)", value=None, placeholder="Required")
with col2:
    daily_limit = st.number_input("Daily Limit ($)", value=None, placeholder="Required")
survival_floor = st.number_input("Survival Floor ($)", value=4500.0)
risk_pct = st.slider("Risk % of Buffer", 5, 50, 25)

# ─── MAIN LOGIC ───
if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
    if balance is None:
        st.error("Enter balance")
    else:
        with st.spinner("Scanning Structure..."):
            try:
                # 1. FETCH DATA
                price_data = td.price(symbol="XAU/USD").as_json()
                live_price = float(price_data["price"])
                ts_15m = td.time_series(symbol="XAU/USD", interval="15min", outputsize=100).with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14).as_pandas()
                ts_1h = td.time_series(symbol="XAU/USD", interval="1h", outputsize=50).with_ema(time_period=200).as_pandas()

                # 2. INDICATORS
                rsi, atr = ts_15m['rsi'].iloc[0], ts_15m['atr'].iloc[0]
                ema200_15, ema50_15 = ts_15m['ema_1'].iloc[0], ts_15m['ema_2'].iloc[0]
                ema200_1h = ts_1h['ema_1'].iloc[0]

                # 3. TREND & LEVELS
                if (live_price > ema200_15 and live_price > ema200_1h): bias = "BULLISH"
                elif (live_price < ema200_15 and live_price < ema200_1h): bias = "BEARISH"
                else:
                    st.warning("Trend Misalignment")
                    st.stop()

                levels = get_fractal_levels(ts_15m)
                res = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
                sup = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

                # 4. ENTRY TYPE
                sl_dist = round(atr * 1.5, 2)
                if bias == "BULLISH":
                    entry = live_price if (live_price - ema50_15) <= (atr * 0.4) else ema50_15
                else:
                    entry = live_price if (ema50_15 - live_price) <= (atr * 0.4) else ema50_15

                # 5. EXECUTION & PUSH
                if entry == live_price:
                    action = f"MARKET {bias}"
                    send_push_notification("🥇 GOLD ALERT", f"Elite {bias} Market Entry at ${entry:.2f}")
                else:
                    action = f"LIMIT {bias}"

                st.success(f"### {action} @ ${entry:.2f}")
                
            except Exception as e:
                st.error(f"Error: {e}")
