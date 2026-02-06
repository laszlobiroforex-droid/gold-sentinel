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
        st.warning("Telegram not configured (missing token or chat ID)")
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
            "entry_type": data.get("entry_type"),
            "entry_price": data.get("entry_price"),
            "sl": data.get("sl"),
            "tp": data.get("tp"),
            "rr": data.get("rr"),
            "estimated_win_prob": data.get("estimated_win_prob"),
            "suggested_lot": data.get("suggested_lot"),
            "risk_dollars": data.get("risk_dollars"),
            "risk_pct_of_dd": data.get("risk_pct_of_dd"),
            "exceeds_preferred_risk": data.get("exceeds_preferred_risk", False),
            "style": data.get("style", "NONE").upper(),
            "direction": data.get("direction", "NEUTRAL").upper(),
            "reasoning": data.get("reasoning", "")
        }
    except Exception as e:
        return {
            "verdict": "PARSE_ERROR",
            "reason": f"Invalid JSON: {str(e)}. Raw: {text[:150]}..."
        }

# ─── FALLBACK LOT CALC ─────────────────────────────────────────────────────
def calculate_lot_size(entry, sl, balance, dd_limit, risk_of_dd_pct):
    if not all(v is not None for v in [entry, sl, balance, dd_limit, risk_of_dd_pct]):
        return None, "Missing data for lot calc"

    try:
        entry = float(entry)
        sl = float(sl)
        price_diff = abs(entry - sl)
        if price_diff <= 0:
            return None, "Invalid SL distance"

        max_risk_dollars = (dd_limit * risk_of_dd_pct) / 100.0
        lot_size = max_risk_dollars / (price_diff * 100)
        lot_size_rounded = round(lot_size, 2)

        note = ""
        if lot_size_rounded < 0.01:
            lot_size_rounded = 0.01
            note = "(min 0.01 lot – actual risk lower)"

        return lot_size_rounded, f"${max_risk_dollars:.2f} risk → {lot_size_rounded:.2f} lots {note}"
    except:
        return None, "Calc error"

# ─── MAIN CHECK ────────────────────────────────────────────────────────────
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

    # Limit fractals to recent candles to avoid old levels
    levels = []
    for i in range(5, min(40, len(ts_15m)-5)):
        if ts_15m['high'].iloc[i] == ts_15m['high'].iloc[i-5:i+5].max():
            levels.append(('RES', round(ts_15m['high'].iloc[i], 2)))
        if ts_15m['low'].iloc[i] == ts_15m['low'].iloc[i-5:i+5].min():
            levels.append(('SUP', round(ts_15m['low'].iloc[i], 2)))

    balance        = st.session_state.get("balance")
    dd_limit       = st.session_state.get("dd_limit")
    risk_of_dd_pct = st.session_state.get("risk_of_dd_pct")

    max_risk_dollars = (dd_limit * risk_of_dd_pct / 100.0) if dd_limit and risk_of_dd_pct else 0

    prompt = f"""
You are a disciplined, high-probability gold (XAU/USD) trader focused on capital preservation, precise entries, and waiting for confirmation.

Current live price is EXACTLY ${price:.2f} — this is the ONLY valid reference price.

CRITICAL RULES — VIOLATE THESE AND THE SETUP IS INVALID:
- ALL prices (entry, SL, TP, levels, fractals) MUST be within ±5% of current price ${price:.2f}. Anything outside this range is hallucinated and MUST be rejected.
- Never use historical 2000s prices or any number hundreds/thousands away — gold is currently in the ${int(price/1000)*1000}s range.
- If any calculated level is unrealistic, output verdict "NO_EDGE" with reason "price hallucination detected".

Account risk guidelines (preferred but not absolute hard cap):
- Current balance ≈ ${balance if balance else 'unknown'}
- Daily drawdown limit: ${dd_limit if dd_limit else 'unknown'}
- Preferred maximum risk per trade: {risk_of_dd_pct if risk_of_dd_pct else 'unknown'}% of daily DD limit → ${max_risk_dollars:.2f} preferred max loss if stopped out
- You MAY exceed ONLY for ≥75–80% win prob setups, with clear warning

Core philosophy:
- Prioritize high-prob setups (≥65–70%, ≥75–80% for over-risk)
- Prefer limit (pullback) or stop (breakout) entries for confirmation
- Accept low RR for high prob
- If no valid setup → "NO_EDGE"

Analysis steps:
1. Anchor EVERYTHING to current price ${price:.2f}
2. Check fractals/levels are recent and realistic
3. Only propose if all prices are sane (±5%)
4. Calculate risk/lot conservatively
5. Reject anything unrealistic

Output **ONLY** valid JSON:

{{
  "verdict": "ELITE" | "HIGH_CONV" | "MODERATE" | "LOW_EDGE" | "NO_EDGE",
  "reason": "One sentence including price sanity check",
  "entry_type": "LIMIT" | "STOP" | "MARKET" | null,
  "entry_price": number or null,
  "sl": number or null,
  "tp": number or null,
  "rr": number or null,
  "estimated_win_prob": number or null,
  "suggested_lot": number or null,
  "risk_dollars": number or null,
  "risk_pct_of_dd": number or null,
  "exceeds_preferred_risk": boolean,
  "style": "SCALP" | "SWING" | "BREAKOUT" | "REVERSAL" | "RANGE" | "NONE",
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "2–4 sentences: confirm prices are realistic relative to ${price:.2f}, structure, probability, risk"
}}

Use "NO_EDGE" if any price looks hallucinated or unrealistic.
"""

    with st.spinner("Consulting AIs..."):
        g_raw = k_raw = c_raw = '{"verdict":"ERROR","reason":"AI offline"}'

        try:
            g_raw = gemini_model.generate_content(prompt, generation_config={"temperature": 0.2}).text.strip()
        except Exception as e:
            st.warning(f"Gemini failed: {str(e)}")

        try:
            r = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4
            )
            k_raw = r.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Grok failed: {str(e)}")

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5
            )
            c_raw = resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"ChatGPT failed: {str(e)}")

    g_p = parse_ai_output(g_raw)
    k_p = parse_ai_output(k_raw)
    c_p = parse_ai_output(c_raw)

    high_count = sum(1 for p in [g_p, k_p, c_p] if p["verdict"] in ["ELITE", "HIGH_CONV"])

    best_p = max([g_p, k_p, c_p], key=lambda p: {"ELITE":4, "HIGH_CONV":3, "MODERATE":2, "LOW_EDGE":1, "NO_EDGE":-1}.get(p["verdict"], 0))

    lot_size = best_p.get("suggested_lot")
    lot_note = None
    if lot_size is not None:
        lot_note = f"AI suggested: {lot_size:.2f} lots (${best_p.get('risk_dollars', '—')})"
        if best_p.get("exceeds_preferred_risk"):
            lot_note += " — EXCEEDS preferred risk!"
    else:
        lot_size, lot_note = calculate_lot_size(
            best_p.get("entry_price") or best_p.get("entry"),
            best_p.get("sl"),
            balance, dd_limit, risk_of_dd_pct
        )

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
            "NO_EDGE": "#95a5a6", "PARSE_ERROR": "#7f8c8d"
        }
        color = colors.get(verdict, "#7f8c8d")

        # Safe entry display
        entry_type = p.get('entry_type', '—')
        entry_price_val = p.get('entry_price')
        if isinstance(entry_price_val, (int, float)):
            entry_price_str = f"${entry_price_val:.2f}"
        else:
            entry_price_str = "—"
        entry_str = f"{entry_type} @ {entry_price_str}"

        # Safe SL, TP, RR
        sl_val = p.get('sl')
        sl_str = f"${sl_val:.2f}" if isinstance(sl_val, (int, float)) else "—"

        tp_val = p.get('tp')
        tp_str = f"${tp_val:.2f}" if isinstance(tp_val, (int, float)) else "—"

        rr_val = p.get('rr')
        rr_str = f"1:{rr_val:.1f}" if isinstance(rr_val, (int, float)) else "—"

        # Prob
        prob_val = p.get('estimated_win_prob')
        prob_str = f"{prob_val}%" if isinstance(prob_val, (int, float)) else "—"

        # Safe risk
        risk_dollars_val = p.get('risk_dollars')
        risk_pct_val     = p.get('risk_pct_of_dd')
        exceed_flag      = p.get('exceeds_preferred_risk', False)

        if isinstance(risk_dollars_val, (int, float)):
            dollars_part = f"${risk_dollars_val:.2f}"
        else:
            dollars_part = "—"

        if isinstance(risk_pct_val, (int, float)):
            pct_part = f"{risk_pct_val:.1f}%"
        else:
            pct_part = "—"

        risk_str = f"{dollars_part} ({pct_part} of DD)"
        exceed_warning = '<span style="color:#e74c3c; font-weight:bold;">EXCEEDS preferred risk!</span>' if exceed_flag else ""

        # Entry distance sanity check
        if isinstance(entry_price_val, (int, float)) and price and abs(entry_price_val - price) / price > 0.10:
            entry_str += ' <span style="color:#e74c3c;">(unrealistic distance!)</span>'

        lot_str = f"{lot:.2f}" if lot is not None else "—"

        html = f"""
        <div style="background:{color}11; border-left:5px solid {color}; padding:16px 20px; margin:16px 0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="font-size:1.3em; font-weight:bold; color:{color}; margin-bottom:8px;">
                {ai_name} — {verdict}
            </div>
            <div style="margin-bottom:12px; color:#555;">{p['reason']}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0;">
                <div><strong>Direction:</strong> {p['direction']}</div>
                <div><strong>Style:</strong> {p['style']}</div>
                <div><strong>Entry:</strong> {entry_str}</div>
                <div><strong>SL:</strong> {sl_str}</div>
                <div><strong>TP:</strong> {tp_str}</div>
                <div><strong>RR:</strong> {rr_str}</div>
                <div><strong>Est. Win Prob:</strong> {prob_str}</div>
                <div><strong>Risk:</strong> {risk_str} {exceed_warning}</div>
                <div><strong>Lot size:</strong> {lot_str}</div>
            </div>
            <div style="color:#555; font-size:0.9em; margin:8px 0;">{note or ''}</div>
            <div style="font-style:italic; color:#444; margin-top:12px;">{p['reasoning']}</div>
        </div>
        """
        return html

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(format_verdict(g_p, "Gemini",   lot_size, lot_note), unsafe_allow_html=True)
    with col2: st.markdown(format_verdict(k_p, "Grok",     lot_size, lot_note), unsafe_allow_html=True)
    with col3: st.markdown(format_verdict(c_p, "ChatGPT",  lot_size, lot_note), unsafe_allow_html=True)

    # ─── CONSENSUS & TELEGRAM ──────────────────────────────────────────────────
    if high_count >= 2:
        entries = [p.get("entry_price") or p.get("entry") for p in [g_p, k_p, c_p] if p.get("entry_price") or p.get("entry")]
        consensus_note = ""
        if len(entries) >= 2:
            spread = max(entries) - min(entries)
            if spread <= atr * 0.8:
                median = np.median(entries)
                consensus_note = f"\nConsensus entry ≈ ${median:.2f}"
            else:
                consensus_note = "\nEntries spread — review manually"

        exceed_flag = any(p.get("exceeds_preferred_risk") for p in [g_p, k_p, c_p])
        exceed_text = " — **EXCEEDS preferred risk on at least one AI**" if exceed_flag else ""

        msg = (
            f"**Consensus High Conviction ({high_count}/3)**\n"
            f"Direction: {best_p['direction']}\n"
            f"Style: {best_p['style']}\n"
            f"Entry: {best_p.get('entry_type', '—')} @ ${best_p.get('entry_price', best_p.get('entry', '—')):.2f}\n"
            f"SL: ${best_p.get('sl', '—'):.2f}\n"
            f"TP: ${best_p.get('tp', '—'):.2f} (RR \~1:{best_p.get('rr', '—')})\n"
            f"Est. Win Prob: {best_p.get('estimated_win_prob', '—')}%\n"
            f"Reason: {best_p['reason']}{consensus_note}{exceed_text}\n"
        )
        if lot_size:
            msg += f"\nSuggested lot: {lot_size:.2f} ({lot_note})"

        priority = "high" if high_count == 3 else "normal"
        send_telegram(msg, priority=priority)
        st.success("High conviction consensus — Telegram sent!")
    else:
        st.info("No strong consensus (need 2+ ELITE/HIGH_CONV) — no Telegram sent.")

# ─── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – AI-Driven Gold Setups")
st.caption(f"Gemini • Grok • ChatGPT | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.expander("Prop Challenge / Risk Settings (required for lot sizing)", expanded=True):
    default_balance = st.session_state.get("balance", 5029.00)
    default_dd = st.session_state.get("dd_limit", 251.45)
    default_risk_pct = st.session_state.get("risk_of_dd_pct", 20.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        balance_input = st.number_input("Current Balance ($)", min_value=0.0, step=0.01, value=default_balance, format="%.2f")
    with col2:
        dd_input = st.number_input("Daily Drawdown Limit ($)", min_value=0.0, step=1.0, value=default_dd, format="%.2f")
    with col3:
        risk_input = st.number_input("Preferred risk % of Daily DD", min_value=1.0, max_value=100.0, value=default_risk_pct, step=1.0, format="%.0f")

    if st.button("Save & Apply Settings", type="primary", use_container_width=True):
        st.session_state.balance = balance_input
        st.session_state.dd_limit = dd_input
        st.session_state.risk_of_dd_pct = risk_input
        st.success("Settings saved!")
        st.rerun()

if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

auto_enabled = st.checkbox(f"Auto-run every {CHECK_INTERVAL_MIN} minutes (keep tab open)", value=False)

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
