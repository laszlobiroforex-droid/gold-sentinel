import streamlit as st
import json
from datetime import datetime, timezone
import time
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
import requests
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
CHECK_INTERVAL_MIN = 30

# ─── API INIT ──────────────────────────────────────────────────────────────
td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
genai.configure(api_key=st.secrets["GEMINI_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

grok_client   = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── TELEGRAM ──────────────────────────────────────────────────────────────
def send_telegram(message, priority="normal"):
    token   = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in secrets)")
        return

    emoji = "🟢 ELITE" if priority == "high" else "🔵 Conviction"
    text = f"{emoji} Gold Setup Alert\n\n{message}\n\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# ─── DATA FETCH ────────────────────────────────────────────────────────────
def get_live_price():
    try:
        return float(td.price(symbol="XAU/USD").as_json()["price"])
    except:
        return None

def fetch_15m():
    try:
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=120)
        ts = ts.with_rsi().with_ema(time_period=200).with_ema(time_period=50).with_atr(time_period=14)
        return ts.as_pandas()
    except:
        return None

def fetch_1h():
    try:
        ts = td.time_series(symbol="XAU/USD", interval="1h", outputsize=60)
        ts = ts.with_ema(time_period=200)
        return ts.as_pandas()
    except:
        return None

# ─── PARSE AI OUTPUT ───────────────────────────────────────────────────────
def parse_ai_output(text):
    if not text or not isinstance(text, str):
        return {"verdict": "UNKNOWN", "reason": "No output"}

    start = text.find('{')
    end   = text.rfind('}') + 1
    if start == -1 or end == 0:
        return {"verdict": "PARSE_ERROR", "reason": "No JSON found"}

    json_str = text[start:end].strip().replace('```json', '').replace('```', '').strip()

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
    except Exception as e:
        return {
            "verdict": "PARSE_ERROR",
            "reason": f"Invalid JSON: {str(e)}. Raw: {text[:150]}..."
        }

# ─── LOT SIZE CALCULATION ──────────────────────────────────────────────────
def calculate_lot_size(entry, sl, balance, dd_limit, risk_of_dd_pct):
    if not all(v is not None for v in [entry, sl, balance, dd_limit, risk_of_dd_pct]):
        return None, "Missing data for lot calc (need entry, SL, balance, DD limit and risk %)"

    try:
        entry = float(entry)
        sl = float(sl)
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return None, "Invalid SL distance (entry == SL)"

        max_risk_dollars = (dd_limit * risk_of_dd_pct) / 100.0

        # XAU/USD: 0.01 lot = $1 per $1 price move
        # Risk $ = lots × price_diff × 100
        lot_size = max_risk_dollars / (price_diff * 100)
        lot_size_rounded = round(lot_size, 2)

        note = ""
        if lot_size_rounded < 0.01:
            lot_size_rounded = 0.01
            note = "(minimum 0.01 lot applied – actual risk lower)"

        return lot_size_rounded, f"${max_risk_dollars:.2f} risk → {lot_size_rounded:.2f} lots {note}"
    except Exception as e:
        return None, f"Lot calc error: {str(e)}"

# ─── MAIN ANALYSIS ─────────────────────────────────────────────────────────
def run_check():
    with st.spinner("Fetching market data..."):
        price = get_live_price()
        ts_15m = fetch_15m()
        ts_1h  = fetch_1h()

    if price is None:
        st.error("Failed to fetch live price")
        return
    if ts_15m is None or ts_15m.empty:
        st.error("No 15m data received")
        return

    latest_15m = ts_15m.iloc[-1]
    rsi = latest_15m.get('rsi', 50.0)
    atr = latest_15m.get('atr', 10.0)
    ema200_15m = latest_15m.get('ema_200', price)
    ema50_15m  = latest_15m.get('ema_50', price)

    ema200_1h = ts_1h.iloc[-1].get('ema_200', price) if ts_1h is not None and not ts_1h.empty else price

    levels = []
    for i in range(5, len(ts_15m)-5):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    balance        = st.session_state.get("balance")
    dd_limit       = st.session_state.get("dd_limit")
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct")

    prompt = f"""
You are an experienced gold (XAU/USD) trader. Analyze ONLY the provided real-time data.
Do NOT hallucinate prices, news, levels, or external information.

Current UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

Market data:
Current price: ${price:.2f}
RSI (15m): {rsi:.1f}
ATR (15m): {atr:.2f}
EMA50 / EMA200 (15m): {ema50_15m:.2f} / {ema200_15m:.2f}
EMA200 (1h): {ema200_1h:.2f}

Recent support/resistance fractals: {', '.join([f"{t}@{p}" for t,p in levels[-8:]]) or 'None'}

Account context: Balance \~\( {balance if balance else 'unknown'}, Daily DD limit \~ \){dd_limit if dd_limit else 'unknown'}

Identify the best realistic trading setup available right now — even if the edge is only moderate.
Always propose concrete entry, SL, TP, RR unless the market has truly no structure or direction.
Be objective and favor probability.

Verdict levels (use these exactly):
- ELITE
- HIGH_CONV
- MODERATE
- LOW_EDGE

Respond **ONLY** with valid JSON. No extra text, no fences:

{{
  "verdict": "ELITE" | "HIGH_CONV" | "MODERATE" | "LOW_EDGE",
  "reason": "one sentence explaining edge strength",
  "entry": number or null,
  "sl": number or null,
  "tp": number or null,
  "rr": number or null,
  "style": "SCALP" | "SWING" | "BREAKOUT" | "REVERSAL" | "RANGE" | "NONE",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "1-2 sentences explaining the setup"
}}
"""

    with st.spinner("Consulting AIs..."):
        g_raw = k_raw = c_raw = '{"verdict":"ERROR","reason":"AI offline"}'

        try:
            g_raw = gemini_model.generate_content(prompt).text.strip()
        except Exception as e:
            st.warning(f"Gemini failed: {str(e)}")

        try:
            r = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4
            )
            k_raw = r.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Grok failed: {str(e)}")

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )
            c_raw = resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"ChatGPT failed: {str(e)}")

    g_p = parse_ai_output(g_raw)
    k_p = parse_ai_output(k_raw)
    c_p = parse_ai_output(c_raw)

    high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

    # Select best AI for consensus lot sizing
    verdict_scores = {"ELITE": 4, "HIGH_CONV": 3, "MODERATE": 2, "LOW_EDGE": 1}
    best_p = max([g_p, k_p, c_p], key=lambda p: verdict_scores.get(p["verdict"], 0))

    lot_size, lot_note = calculate_lot_size(
        best_p.get("entry"), best_p.get("sl"),
        balance, dd_limit, risk_of_dd_pct
    )

    # Show why lot size failed (if applicable)
    if lot_size is None and "Missing data" in lot_note:
        st.info(f"Lot size not calculated: {lot_note}\n\nPlease fill in and save the risk settings below.")

    # ─── DISPLAY VERDICTS ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("AI Verdicts")

    def format_verdict(p, ai_name, lot=None, note=""):
        if p["verdict"] == "PARSE_ERROR":
            return f"**{ai_name}** — Parse error: {p['reason']}"

        verdict = p["verdict"]
        colors = {
            "ELITE": "#2ecc71", "HIGH_CONV": "#3498db",
            "MODERATE": "#f1c40f", "LOW_EDGE": "#e67e22",
            "PARSE_ERROR": "#7f8c8d", "UNKNOWN": "#7f8c8d"
        }
        color = colors.get(verdict, "#7f8c8d")

        entry = f"${p['entry']:.2f}" if p["entry"] is not None else "—"
        sl    = f"${p['sl']:.2f}"    if p["sl"]    is not None else "—"
        tp    = f"${p['tp']:.2f}"    if p["tp"]    is not None else "—"
        rr    = f"1:{p['rr']:.1f}"   if p["rr"]    is not None else "—"

        html = f"""
        <div style="background:{color}11; border-left:5px solid {color}; padding:16px 20px; margin:16px 0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="font-size:1.3em; font-weight:bold; color:{color}; margin-bottom:8px;">
                {ai_name} — {verdict}
            </div>
            <div style="margin-bottom:12px; color:#555;">{p['reason']}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0;">
                <div><strong>Direction:</strong> {p['direction']}</div>
                <div><strong>Style:</strong> {p['style']}</div>
                <div><strong>Entry:</strong> {entry}</div>
                <div><strong>SL:</strong> {sl}</div>
                <div><strong>TP:</strong> {tp}</div>
                <div><strong>RR:</strong> {rr}</div>
            </div>
            <div style="margin:12px 0; font-weight:bold; color:#2c3e50;">Lot size: {lot if lot is not None else '—'}</div>
            <div style="color:#555; font-size:0.9em;">{note}</div>
            <div style="font-style:italic; color:#444; margin-top:12px;">{p['reasoning']}</div>
        </div>
        """
        return html

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(format_verdict(g_p, "Gemini", lot_size, lot_note), unsafe_allow_html=True)
    with col2:
        st.markdown(format_verdict(k_p, "Grok", lot_size, lot_note), unsafe_allow_html=True)
    with col3:
        st.markdown(format_verdict(c_p, "ChatGPT", lot_size, lot_note), unsafe_allow_html=True)

    # ─── CONSENSUS & TELEGRAM ──────────────────────────────────────────────────
    if high_count >= 2:
        entries = [p["entry"] for p in [g_p, k_p, c_p] if p["entry"] is not None]
        consensus_note = ""
        if len(entries) >= 2:
            spread = max(entries) - min(entries)
            if spread <= atr * 0.8:
                median = np.median(entries)
                consensus_note = f"\nConsensus entry ≈ ${median:.2f}"
            else:
                consensus_note = "\nEntries spread — review manually"

        msg = (
            f"**Consensus High Conviction ({high_count}/3)**\n"
            f"Direction: {best_p['direction']}\n"
            f"Style: {best_p['style']}\n"
            f"Entry: ${best_p['entry']:.2f}\n"
            f"SL: ${best_p['sl']:.2f}\n"
            f"TP: ${best_p['tp']:.2f} (RR \~1:{best_p['rr']})\n"
            f"Reason: {best_p['reason']}{consensus_note}\n"
        )
        if lot_size is not None:
            msg += f"\nSuggested lot: {lot_size:.2f} lots ({lot_note})"

        priority = "high" if high_count == 3 else "normal"
        send_telegram(msg, priority=priority)
        st.success("High conviction consensus — Telegram sent!")
    else:
        st.info("No strong consensus (need 2+ ELITE/HIGH_CONV) — no Telegram sent.")

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Gemini • Grok • ChatGPT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ─── RISK SETTINGS (persistent defaults) ────────────────────────────────────
with st.expander("Prop Challenge / Risk Settings (required for lot sizing)", expanded=True):
    default_balance = st.session_state.get("balance", 0)
    default_dd = st.session_state.get("dd_limit", 0)
    default_risk_pct = st.session_state.get("risk_of_dd_pct", 25.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        balance_input = st.number_input("Current Balance ($)", min_value=0.0, step=0.01, value=default_balance, format="%.2f")
    with col2:
        dd_input = st.number_input("Daily Drawdown Limit ($)", min_value=0.0, step=1.0, value=default_dd, format="%.2f")
    with col3:
        risk_input = st.number_input("Accepted risk % of Daily DD", min_value=1.0, max_value=100.0, value=default_risk_pct, step=1.0, format="%.0f")

    if st.button("Save & Apply Settings", type="primary", use_container_width=True):
        st.session_state.balance = balance_input
        st.session_state.dd_limit = dd_input
        st.session_state.risk_of_dd_pct = risk_input
        st.success("Settings saved! Lot sizing will now work on next run.")
        st.rerun()  # Refresh UI to show updated defaults

# ─── AUTO / MANUAL CONTROLS ────────────────────────────────────────────────
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

auto_enabled = st.checkbox("Auto-run every 15 minutes (keep tab open)", value=False)

if auto_enabled:
    now = time.time()
    time_since = now - st.session_state.last_check_time
    if time_since >= CHECK_INTERVAL_MIN * 60:
        st.session_state.last_check_time = now
        run_check()
    else:
        remaining = int(CHECK_INTERVAL_MIN * 60 - time_since)
        st.caption(f"Next auto-run in {remaining // 60} min {remaining % 60} sec")
else:
    st.info("Auto mode off — use the button below")

if st.button("📡 Run Analysis Now", type="primary", use_container_width=True):
    run_check()
