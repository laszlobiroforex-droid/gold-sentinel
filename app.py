import streamlit as st
import pandas as pd
import numpy as np
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
from datetime import datetime

st.markdown("""
<script src="https://unpkg.com/eruda"></script>
<script>
  eruda.init({
    defaults: {
      container: document.body,
      tool: ['console', 'elements', 'network'],
      theme: 'Monokai Pro'
    }
  });
  eruda.show();  // force show immediately
</script>
""", unsafe_allow_html=True)

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    grok_client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
except Exception as e:
    st.error(f"API setup failed: {e}\nCheck secrets: TWELVE_DATA_KEY, GEMINI_KEY, GROK_API_KEY")
    st.stop()

# ─── FRACTAL LEVELS ──────────────────────────────────────────────────────────
def get_fractal_levels(df, window=5):
    levels = []
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            levels.append(('RES', round(df['high'].iloc[i], 2)))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            levels.append(('SUP', round(df['low'].iloc[i], 2)))
    return levels

# ─── DUAL AUDITORS ───────────────────────────────────────────────────────────
def get_ai_advice(market, setup, levels, buffer):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[-5:]]) if levels else "No clear levels"
    prompt = f"""
    High-conviction gold trading auditor for any account size.
    Aggressive risk is user's choice — do NOT suggest reducing %.
    Focus on math, pullback quality, structural confluence, risk/reward.

    Buffer left: ${buffer:.2f}
    Market: Price ${market['price']:.2f}, RSI {market['rsi']:.1f}
    Setup: {setup['type']} at ${setup['entry']:.2f} risking ${setup['risk']:.2f}
    Fractals: {levels_str}

    Blunt verdict: Elite high-conviction entry or low-edge gamble? 3 sentences max.
    """
    try: g_out = gemini_model.generate_content(prompt).text.strip()
    except: g_out = "Gemini Offline."

    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e: k_out = f"Grok Error: {e}"
    
    return g_out, k_out

# ─── PWA + PUSH SUPPORT ──────────────────────────────────────────────────────
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#000000">
<link rel="icon" href="https://img.icons8.com/fluency/192/000000/gold-bar.png">
""", unsafe_allow_html=True)

st.markdown(f"""
<script>
  const vapidPublicKey = "{st.secrets.get('VAPID_PUBLIC_KEY', '')}";

  // Register service worker
  if ('serviceWorker' in navigator) {{
    window.addEventListener('load', () => {{
      navigator.serviceWorker.register('/sw.js')
        .then(reg => console.log('Service Worker registered'))
        .catch(err => console.log('SW failed:', err));
    }});
  }}

  // Request push permission & subscribe
  async function subscribeToPush() {{
    if (!vapidPublicKey) {{
      console.log('VAPID key missing');
      return;
    }}
    if (!('PushManager' in window)) {{
      console.log('Push not supported');
      return;
    }}
    const permission = await Notification.requestPermission();
    if (permission !== 'granted') {{
      console.log('Notifications denied');
      return;
    }}

    const reg = await navigator.serviceWorker.ready;
    const sub = await reg.pushManager.subscribe({{
      userVisibleOnly: true,
      applicationServerKey: vapidPublicKey
    }});

    console.log('Push subscribed');
  }}

  if (Notification.permission === 'default') {{
    subscribeToPush();
  }}

  // Accept setup → pause notifications
  window.acceptSetup = function() {{
    localStorage.setItem('setup_accepted', Date.now());
    clearInterval(window.checkInterval);
    alert('Setup accepted — notifications paused for 24 hours');
  }};

  // 15-min auto-check
  window.checkInterval = setInterval(async () => {{
    if (localStorage.getItem('setup_accepted')) {{
      const accepted = parseInt(localStorage.getItem('setup_accepted'));
      if (Date.now() - accepted < 24 * 60 * 60 * 1000) return;
      localStorage.removeItem('setup_accepted');
    }}

    try {{
      const res = await fetch('/check_latest_setup');
      const data = await res.json();

      if (data.setup && !data.accepted) {{
        const reg = await navigator.serviceWorker.ready;
        reg.showNotification('High Conviction Setup Available', {{
          body: `\( {{data.setup.bias}} | Entry ~ \){{data.setup.entry}} | Risk $${{data.setup.risk}}`,
          icon: 'https://img.icons8.com/fluency/192/000000/gold-bar.png',
          tag: 'sentinel-setup-' + data.setup.timestamp,
          renotify: true
        }});
      }}
    }} catch (err) {{
      console.log('Check failed:', err);
    }}
  }}, 15 * 60 * 1000);
</script>
""", unsafe_allow_html=True)

# ─── APP ─────────────────────────────────────────────────────────────────────
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Initialize session state
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = None
    st.session_state.daily_limit = None
    st.session_state.floor = 0.0
    st.session_state.risk_pct = 25

if "saved_setups" not in st.session_state:
    st.session_state.saved_setups = []

# ─── INPUTS ──────────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    st.header("Account Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.balance = st.number_input(
            "Current Balance ($)", min_value=0.0, value=st.session_state.balance,
            placeholder="Required", format="%.2f", key="balance_input"
        )
    with col2:
        st.session_state.daily_limit = st.number_input(
            "Daily Drawdown Limit ($)", min_value=0.0, value=st.session_state.daily_limit,
            placeholder="Optional (set to balance for no limit)", format="%.2f", key="limit_input"
        )

    st.session_state.floor = st.number_input(
        "Survival Floor / Max DD ($)", value=st.session_state.floor, format="%.2f"
    )

    st.session_state.risk_pct = st.slider(
        "Risk % of Available Buffer", 5, 50, st.session_state.risk_pct, step=5
    )

    if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
        if st.session_state.balance is None:
            st.error("❌ Enter current balance")
        else:
            st.session_state.analysis_done = True
            st.rerun()
else:
    # Reminder
    st.info("Analysis locked:")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "No limit")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")

    with st.spinner("Scanning structure..."):
        try:
            price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
            live_price = float(price_data["price"])

            ts_15m = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "15min",
                "outputsize": 100
            }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

            ts_1h = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "1h",
                "outputsize": 50
            }).with_ema(**{"time_period": 200}).as_pandas()

            # SAFE EMA DETECTION
            rsi = ts_15m['rsi'].iloc[0] if 'rsi' in ts_15m.columns else 50.0
            atr = ts_15m['atr'].iloc[0] if 'atr' in ts_15m.columns else 0.0

            ema_cols = [c for c in ts_15m.columns if 'ema' in c.lower()]
            ema_cols.sort()
            ema200_15 = ts_15m[ema_cols[0]].iloc[0] if len(ema_cols) >= 1 else live_price
            ema50_15  = ts_15m[ema_cols[1]].iloc[0] if len(ema_cols) >= 2 else live_price
            ema200_1h = ts_1h['ema_1'].iloc[0] if 'ema_1' in ts_1h.columns else live_price

            # TREND ALIGNMENT
            if (live_price > ema200_15 and live_price > ema200_1h):
                bias = "BULLISH"
            elif (live_price < ema200_15 and live_price < ema200_1h):
                bias = "BEARISH"
            else:
                st.warning(f"Trend misalignment – 1H EMA200 at ${ema200_1h:.2f}")
                st.markdown("**Short explanation:** The 15-minute and 1-hour trends are not aligned. This prevents trades against the larger trend, which often leads to quick stops.")
                st.markdown("**Suggested action:** Wait approximately **15 minutes** (one full 15-min candle) and press 'Analyze & Suggest' again to check if structure has normalized.")
                st.stop()

            # FRACTAL LEVELS
            levels = get_fractal_levels(ts_15m)
            resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
            supports = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

            # PULLBACK ENTRY
            sl_dist = round(atr * 1.5, 2)
            if bias == "BULLISH":
                entry = ema50_15 if (live_price - ema50_15) > (atr * 0.5) else live_price
                sl = supports[0] - (0.3 * atr) if supports else entry - sl_dist
                tp = resistances[0] if resistances else entry + (sl_dist * 2.5)
                action_header = "BUY AT MARKET" if entry == live_price else "BUY LIMIT ORDER"
            else:
                entry = ema50_15 if (ema50_15 - live_price) > (atr * 0.5) else live_price
                sl = resistances[0] + (0.3 * atr) if resistances else entry + sl_dist
                tp = supports[0] if supports else entry - (sl_dist * 2.5)
                action_header = "SELL AT MARKET" if entry == live_price else "SELL LIMIT ORDER"

            # RISK & SAFETY
            buffer = st.session_state.balance - st.session_state.floor
            cash_risk = min(buffer * (st.session_state.risk_pct / 100), st.session_state.daily_limit or buffer)

            if cash_risk < 20:
                st.warning("Calculated risk too small for minimum lot size – skipping")
                st.stop()

            sl_dist_actual = abs(entry - sl)
            lots = max(round(cash_risk / ((sl_dist_actual + 0.35) * 100), 2), 0.01)
            actual_risk = round(lots * (sl_dist_actual + 0.35) * 100, 2)

            # ─── AI OPINIONS FIRST ───────────────────────────────────────────────
            st.divider()
            st.subheader("AI Opinions")
            market = {"price": live_price, "rsi": rsi}
            setup = {"type": bias, "entry": entry, "risk": actual_risk}
            g_verdict, k_verdict = get_ai_advice(market, setup, levels, buffer)

            col_g, col_k = st.columns(2)
            with col_g:
                st.markdown("**Gemini (Cautious)**")
                st.info(g_verdict)
            with col_k:
                st.markdown("**Grok (Direct)**")
                st.info(k_verdict)

            st.caption("AI opinions are probabilistic assessments, not trading signals.")

            # ─── ACTION BOX AFTER AI ─────────────────────────────────────────────
            st.divider()
            st.markdown(f"### {action_header}")
            with st.container(border=True):
                st.metric("Entry", f"${entry:.2f}")
                col_sl, col_tp = st.columns(2)
                col_sl.metric("Stop Loss", f"${sl:.2f}")
                col_tp.metric("Take Profit", f"${tp:.2f}")
                col_lots, col_risk = st.columns(2)
                col_lots.metric("Lots", f"{lots:.2f}")
                col_risk.metric("Risk Amount", f"${actual_risk:.2f}")

            # Levels
            with st.expander("Detected Fractal Levels"):
                st.write("**Resistance above:**", resistances[:3] or "None nearby")
                st.write("**Support below:**", supports[:3] or "None nearby")

            # Accept button
            if st.button("✅ Accept This Setup & Pause Notifications"):
                st.markdown("<script>window.acceptSetup();</script>", unsafe_allow_html=True)
                st.success("Accepted! Notifications paused for 24 hours (or until reset).")

            # Save
            st.session_state.saved_setups.append({
                "time": datetime.utcnow().strftime("%H:%M UTC"),
                "bias": bias,
                "entry": round(entry, 2),
                "risk": actual_risk
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ─── HISTORY ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Setups")
if st.session_state.saved_setups:
    df = pd.DataFrame(st.session_state.saved_setups)
    st.dataframe(df.sort_values("time", ascending=False).head(10), use_container_width=True, hide_index=True)
else:
    st.info("No setups saved yet.")

# Reset button
if st.button("Reset & Enter New Account Settings"):
    st.session_state.analysis_done = False
    st.rerun()

    st.markdown("""
<button onclick="console.log('Polling active: ' + (window.checkInterval ? 'Yes' : 'No'))">
  Debug: Check polling status
</button>
""", unsafe_allow_html=True)
