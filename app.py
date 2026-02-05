import streamlit as st
import json
from datetime import datetime, timezone
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
CHECK_INTERVAL_MIN = 15

# ─── API INIT ──────────────────────────────────────────────────────────────
td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
genai.configure(api_key=st.secrets["GEMINI_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

grok_client   = OpenAI(api_key=st.secrets["GROK_API_KEY"],  base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── TELEGRAM ──────────────────────────────────────────────────────────────
def send_telegram(message, priority="normal"):
    token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.warning("Telegram not configured")
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except:
        st.error("Telegram send failed")

# ─── FRESH DATA FETCH (no caching) ─────────────────────────────────────────
def get_live_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

def fetch_15m():
    ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=120)
    ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
    return ts.as_pandas()

def fetch_1h():
    ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=60)
    ts = ts.with_ema(time_period=200)
    return ts.as_pandas()

# ─── PARSE AI OUTPUT ───────────────────────────────────────────────────────
def parse_ai_output(text):
    if not text or not isinstance(text, str):
        return {"verdict": "UNKNOWN", "reason": "No output"}

    start = text.find('{')
    end   = text.rfind('}') + 1
    if start == -1 or end == 0:
        return {"verdict": "PARSE_ERROR", "reason": "No JSON found"}

    json_str = text[start:end].strip()
    json_str = json_str.replace('```json', '').replace('```', '').strip()

    try:
        data = json.loads(json_str)
        return {
            "verdict": data.get("verdict", "UNKNOWN").upper(),
            "reason": data.get("reason", "No reason"),
            "entry": data.get("entry"),
            "sl": data.get("sl"),
            "tp": data.get("tp"),
            "rr": data.get("rr"),
            "style": data.get("style", "NONE").upper(),
            "direction": data.get("direction", "NEUTRAL").upper(),
            "reasoning": data.get("reasoning", "")
        }
    except:
        return {
            "verdict": "PARSE_ERROR",
            "reason": f"Invalid JSON. Raw: {text[:150]}..."
        }

# ─── MAIN CHECK ─────────────────────────────────────────────────────────────
def run_check():
    with st.spinner("Fetching fresh market data..."):
        price = get_live_price()
        ts_15m = fetch_15m()
        ts_1h  = fetch_1h()

    if ts_15m.empty:
        st.error("No 15m data received")
        return

    latest_15m = ts_15m.iloc[-1]
    rsi = latest_15m.get('rsi', 50.0)
    atr = latest_15m.get('atr', 10.0)
    ema200_15m = latest_15m.get('ema_200', price)
    ema50_15m  = latest_15m.get('ema_50',  price)

    ema200_1h = ts_1h.iloc[-1].get('ema_200', price) if not ts_1h.empty else price

    # Optional recent fractals
    levels = []
    for i in range(5, len(ts_15m)-5):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    balance   = st.session_state.get("balance")
    floor     = st.session_state.get("floor")
    risk_pct  = st.session_state.get("risk_pct")
    buffer    = (balance - floor) if balance is not None and floor is not None else None

    prompt = f"""
You are a high-conviction gold trading auditor.
Use ONLY the provided data — do NOT hallucinate prices, levels, or facts.

Current UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
NY session close \~22:00 UTC — watch whipsaw after 21:30 UTC.

Market data:
Price: ${price:.2f}
RSI (15m): {rsi:.1f}
ATR (15m): {atr:.2f}
EMA50 / EMA200 (15m): {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 (1h): {ema200_1h:.2f}

Recent fractals: {', '.join([f"{t}@{p}" for t,p in levels[-6:]]) or 'None'}

Account info: balance {balance if balance else 'unknown'}, risk % {risk_pct if risk_pct else 'unknown'}, buffer {buffer if buffer else 'unknown'}

Analyze current Gold setup. Be brutal — if low edge, say skip.
Propose entry / SL / TP / RR / style / direction only if high conviction exists.

Respond **ONLY** with valid JSON. No fences, no markdown, no extra text:

{{
  "verdict": "ELITE" | "HIGH_CONV" | "LOW_EDGE" | "GAMBLE" | "SKIP",
  "reason": "short explanation",
  "entry": number or null,
  "sl": number or null,
  "tp": number or null,
  "rr": number or null,
  "style": "SCALP" | "SWING" | "BREAKOUT" | "REVERSAL" | "RANGE" | "NONE",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "1-2 sentences"
}}
"""

    with st.spinner("Consulting AIs..."):
        try:
            g_raw = gemini_model.generate_content(prompt).text.strip()
        except:
            g_raw = '{"verdict":"SKIP","reason":"Gemini offline"}'

        try:
            r = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4
            )
            k_raw = r.choices[0].message.content.strip()
        except:
            k_raw = '{"verdict":"SKIP","reason":"Grok error"}'

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )
            c_raw = resp.choices[0].message.content.strip()
        except:
            c_raw = '{"verdict":"SKIP","reason":"ChatGPT error"}'

    g_p = parse_ai_output(g_raw)
    k_p = parse_ai_output(k_raw)
    c_p = parse_ai_output(c_raw)

    high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

    # ─── CLEAN DISPLAY ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("AI Verdicts")

    def format_verdict(p, ai_name):
        if p["verdict"] == "PARSE_ERROR":
            return f"**{ai_name}** — Parse error: {p['reason']}"

        verdict = p["verdict"]
        reason = p["reason"]
        direction = p["direction"]
        style = p["style"]
        entry = f"${p['entry']:.2f}" if p["entry"] is not None else "—"
        sl = f"${p['sl']:.2f}" if p["sl"] is not None else "—"
        tp = f"${p['tp']:.2f}" if p["tp"] is not None else "—"
        rr = f"1:{p['rr']:.1f}" if p["rr"] is not None else "—"
        reasoning = p["reasoning"]

        colors = {
            "ELITE": "#2ecc71",
            "HIGH_CONV": "#3498db",
            "LOW_EDGE": "#e67e22",
            "GAMBLE": "#e74c3c",
            "SKIP": "#95a5a6",
            "PARSE_ERROR": "#7f8c8d",
            "UNKNOWN": "#7f8c8d"
        }
        color = colors.get(verdict, "#7f8c8d")

        return f"""
        <div style="
            background: {color}11;
            border-left: 5px solid {color};
            padding: 16px 20px;
            margin: 16px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 1.3em; font-weight: bold; color: {color}; margin-bottom: 8px;">
                {ai_name} — {verdict}
            </div>
            <div style="margin-bottom: 12px; color: #555;">
                {reason}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0;">
                <div><strong>Direction:</strong> {direction}</div>
                <div><strong>Style:</strong> {style}</div>
                <div><strong>Entry:</strong> {entry}</div>
                <div><strong>SL:</strong> {sl}</div>
                <div><strong>TP:</strong> {tp}</div>
                <div><strong>RR:</strong> {rr}</div>
            </div>
            <div style="font-style: italic; color: #444; margin-top: 12px;">
                {reasoning}
            </div>
        </div>
        """

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(format_verdict(g_p, "Gemini"), unsafe_allow_html=True)

    with col2:
        st.markdown(format_verdict(k_p, "Grok"), unsafe_allow_html=True)

    with col3:
        st.markdown(format_verdict(c_p, "ChatGPT"), unsafe_allow_html=True)

    # Consensus & Telegram
    if high_count >= 2:
        entries = [p["entry"] for p in [g_p, k_p, c_p] if p["entry"] is not None]
        consensus_note = ""
        if len(entries) >= 2:
            spread = max(entries) - min(entries)
            if spread <= atr * 0.8:
                median = np.median(entries)
                consensus_note = f"\nConsensus entry (within \~0.8 ATR): ${median:.2f}"
            else:
                consensus_note = "\nEntries too spread — review manually"

        best = max([g_p, k_p, c_p], key=lambda p: 2 if p["verdict"] == "ELITE" else 1 if p["verdict"] == "HIGH_CONV" else 0)

        msg = (
            f"**Consensus High Conviction ({high_count}/3)**\n"
            f"Direction: {best['direction']}\n"
            f"Style: {best['style']}\n"
            f"Entry: ${best['entry']:.2f}\n"
            f"SL: ${best['sl']:.2f}\n"
            f"TP: ${best['tp']:.2f} (RR \~1:{best['rr']})\n"
            f"Reason: {best['reason']}{consensus_note}\n\n"
            f"Full verdicts above."
        )
        send_telegram(msg, priority="high" if high_count == 3 else "normal")
        st.success("High conviction consensus — Telegram sent!")
    else:
        st.info("No strong consensus — no Telegram sent.")

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-CHECK TIMER
# ─────────────────────────────────────────────────────────────────────────────
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

auto_enabled = st.checkbox("Run automatically every 15 minutes (keep tab open & active)", value=False)

if auto_enabled:
    now = time.time()
    time_since_last = now - st.session_state.last_check_time

    if time_since_last >= CHECK_INTERVAL_MIN * 60:
        st.session_state.last_check_time = now
        st.info(f"Auto-check running... (last was {int(time_since_last/60)} min ago)")
        run_check()
        st.rerun()
    else:
        remaining = CHECK_INTERVAL_MIN * 60 - int(time_since_last)
        st.caption(f"Next auto-check in {remaining // 60} min {remaining % 60} sec (tab must stay open & active)")
else:
    st.info("Auto-check off — press button manually or enable above.")

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Three AIs decide everything | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# Optional account inputs
with st.expander("Account Info (all optional)", expanded=False):
    balance_input = st.number_input("Balance ($)", min_value=0.0, step=0.01, format="%.2f", value=None)
    floor_input   = st.number_input("Floor ($)",   min_value=0.0, step=0.01, format="%.2f", value=None)
    risk_input    = st.number_input("Risk %",      min_value=0.0, step=0.1, format="%.1f", value=None)

    if balance_input is not None:
        st.session_state.balance = balance_input
    if floor_input is not None:
        st.session_state.floor = floor_input
    if risk_input is not None:
        st.session_state.risk_pct = risk_input

if st.button("📡 Run Analysis Now (\~8 credits)", type="primary", use_container_width=True):
    run_check()
