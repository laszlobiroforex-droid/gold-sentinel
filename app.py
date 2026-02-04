import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timezone
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SPREAD_BUFFER_POINTS = 0.30
SLIPPAGE_BUFFER_POINTS = 0.20
COMMISSION_PER_LOT_RT = 1.00     # USD round-turn per lot
PIP_VALUE = 100                  # $ per 1.00 move per standard lot

# ─── RETRY HELPER ────────────────────────────────────────────────────────────
def retry_api(max_attempts=3, backoff=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(backoff * (attempt + 1))
            return None
        return wrapper
    return decorator

# ─── API INIT ────────────────────────────────────────────────────────────────
try:
    td = TDClient(apikey=st.secrets["TWELVE_DATA_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    grok_client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

except Exception as e:
    st.error(f"API setup failed: {e}\nCheck secrets.")
    st.stop()

# ─── FRACTAL LEVELS with filter ──────────────────────────────────────────────
def get_fractal_levels(df, window=5, min_dist_factor=0.4):
    levels = []
    atr_approx = (df['high'] - df['low']).rolling(14).mean().iloc[-1]  # rough ATR proxy if needed
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            price = round(df['high'].iloc[i], 2)
            if abs(price - df['close'].iloc[-1]) > min_dist_factor * atr_approx:
                levels.append(('RES', price))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            price = round(df['low'].iloc[i], 2)
            if abs(price - df['close'].iloc[-1]) > min_dist_factor * atr_approx:
                levels.append(('SUP', price))
    return sorted(levels, key=lambda x: x[1], reverse=True)  # sort descending price

# ─── CACHED DATA FETCH ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
@retry_api()
def fetch_15m_data():
    return td.time_series(**{
        "symbol": "XAU/USD",
        "interval": "15min",
        "outputsize": 500                      # ~5 trading days
    }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

@st.cache_data(ttl=300)
@retry_api()
def fetch_htf_data(interval):
    out_size = 500 if interval == "5min" else 200   # 200 for 1h ≈ 8-10 days
    return td.time_series(**{
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": out_size
    }).with_ema(**{"time_period": 200}).as_pandas()

@st.cache_data(ttl=60)
@retry_api()
def get_current_price():
    return float(td.price(symbol="XAU/USD").as_json()["price"])

# ─── STRUCTURED AI AUDIT ─────────────────────────────────────────────────────
@retry_api()
def get_ai_advice(market, setup, levels, buffer, mode):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[:6]]) if levels else "No clear levels"
    current_price = market['price']
    prompt = f"""
You are a high-conviction gold trading auditor for any account size.
Mode: {mode} ({'standard swing (15m + 1h)' if mode == 'Standard' else 'fast scalp (15m + 5m)'}).
Aggressive risk is user's choice — size is handled externally, do NOT suggest lots multiplier or position size changes.
Focus on math, pullback quality, structural confluence, risk/reward.
IMPORTANT: For buys, SL must be BELOW entry. For sells, SL must be ABOVE entry.

Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
NY session close ~22:00 UTC — factor in thinning liquidity and whipsaw risk after 21:30 UTC.
If any high-impact news likely within ±30 min, prefer to wait unless setup is exceptionally strong.

Current market price: ${current_price:.2f}
Buffer left: ${buffer:.2f}
Market: Price ${current_price:.2f}, RSI {market['rsi']:.1f}
Original setup: {setup['type']} at ${setup['entry']:.2f}, SL distance ${setup['sl_distance']:.2f}, ATR ${setup['atr']:.2f}, risk % {setup['risk_pct']:.0f}%
Fractals: {levels_str}

Be STRICTLY consistent:
- If original setup is low-edge, obsolete, missed, gamble, or chasing, your proposal MUST NOT re-use or slightly adjust the original entry price.
- Any proposal MUST respect current market price ${current_price:.2f} — never suggest entries significantly below current price in bullish mode or above in bearish mode unless clear reversal evidence exists.
- Direction must match detected bias unless verdict explicitly states "reversal".
- Only propose changes that meaningfully improve the setup (e.g. higher entry in continuation, different SL/TP, or skip).
- If no good alternative exists, clearly recommend skipping.

Respond ONLY in this exact structured format. Do not add extra text.

VERDICT: ELITE | HIGH_CONV | LOW_EDGE | GAMBLE | SKIP
REASON: [short explanation, 1-2 sentences]
PROPOSAL: [entry price] | [SL price] | [TP price] | [RR ratio e.g. 2.5] | [direction: BULLISH/BEARISH/NEUTRAL] | [reasoning, 1-2 sentences]
or
PROPOSAL: NONE
"""

    try:
        g_out = gemini_model.generate_content(prompt).text.strip()
    except:
        g_out = "VERDICT: SKIP\nREASON: Gemini Offline.\nPROPOSAL: NONE"

    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e:
        k_out = f"VERDICT: SKIP\nREASON: Grok Error: {e}\nPROPOSAL: NONE"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.65
        )
        c_out = response.choices[0].message.content.strip()
    except Exception as e:
        c_out = f"VERDICT: SKIP\nREASON: ChatGPT Error: {e}\nPROPOSAL: NONE"

    return g_out, k_out, c_out

# ─── PARSE ───────────────────────────────────────────────────────────────────
def parse_ai_output(text):
    verdict = "UNKNOWN"
    reason = ""
    proposal = "NONE"
    entry = sl = tp = rr = None
    direction = None

    v_match = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    if v_match:
        verdict = v_match.group(1).upper()

    r_match = re.search(r"REASON:\s*(.+?)(?=PROPOSAL:|$)", text, re.DOTALL | re.IGNORECASE)
    if r_match:
        reason = r_match.group(1).strip()

    p_match = re.search(r"PROPOSAL:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if p_match:
        proposal_raw = p_match.group(1).strip()
        if proposal_raw.upper() != "NONE":
            parts = [p.strip() for p in proposal_raw.split("|")]
            if len(parts) >= 5:
                try:
                    entry = float(parts[0])
                    sl = float(parts[1])
                    tp = float(parts[2])
                    rr = float(parts[3])
                    direction = parts[4].strip().upper()
                    proposal = "PROPOSAL"
                except:
                    pass

    return {
        "verdict": verdict,
        "reason": reason,
        "proposal": proposal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "direction": direction
    }

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ─── SESSION STATE ───────────────────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = None
    st.session_state.daily_limit = None
    st.session_state.floor = 0.0
    st.session_state.risk_pct = 25
    st.session_state.mode = "Standard"
    st.session_state.last_analysis = 0

if "saved_setups" not in st.session_state:
    st.session_state.saved_setups = []

# ─── INPUTS ──────────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    st.header("Account Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.balance = st.number_input(
            "Current Balance ($)", min_value=0.0, value=st.session_state.balance or 10000.0,
            placeholder="Required", format="%.2f", key="balance_input"
        )
    with col2:
        st.session_state.daily_limit = st.number_input(
            "Daily Drawdown Limit ($)", min_value=0.0, value=st.session_state.daily_limit,
            placeholder="Optional (set to balance for no limit)", format="%.2f"
        )

    st.session_state.floor = st.number_input(
        "Survival Floor / Max DD ($)", value=st.session_state.floor, format="%.2f"
    )

    st.session_state.risk_pct = st.slider(
        "Risk % of Available Buffer", 5, 50, st.session_state.risk_pct, step=5
    )

    st.session_state.mode = st.radio(
        "Analysis Mode",
        ["Standard (Swing – 15m + 1h alignment)", "Scalp (Fast – 15m + 5m alignment)"],
        index=0 if st.session_state.mode == "Standard" else 1
    )

    if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
        if st.session_state.balance is None or st.session_state.balance <= 0:
            st.error("❌ Enter valid current balance")
        elif time.time() - st.session_state.last_analysis < 60:
            st.warning("⏳ Wait 60s between analyses (rate limits)")
        else:
            st.session_state.last_analysis = time.time()
            st.session_state.analysis_done = True
            st.rerun()
else:
    st.info("Analysis locked with your settings:")
    cols = st.columns(5)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "No limit")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")
    cols[4].metric("Mode", st.session_state.mode.split(" – ")[0])

    utc_now = datetime.now(timezone.utc)
    is_late_session = utc_now.hour >= 21 and utc_now.hour < 23  # rough NY fade

    if is_late_session:
        st.warning("⚠️ Late NY session (after ~21:30 UTC) — thinning liquidity, higher whipsaw risk. Consider waiting for London open unless setup is very strong.")

    with st.spinner("Scanning structure..."):
        try:
            live_price = get_current_price()

            ts_15m = fetch_15m_data()
            interval = "1h" if st.session_state.mode.startswith("Standard") else "5min"
            ts_htf = fetch_htf_data(interval)
            htf_label = "1H" if interval == "1h" else "5M"

            latest_15m = ts_15m.iloc[-1]
            latest_htf = ts_htf.iloc[-1]

            rsi = latest_15m.get('rsi', 50.0)
            atr = latest_15m.get('atr', 0.0) or 10.0  # fallback

            ema_cols_15m = sorted([c for c in ts_15m.columns if 'ema' in c.lower()])
            ema200_15m = latest_15m[ema_cols_15m[0]] if ema_cols_15m else live_price
            ema50_15m  = latest_15m[ema_cols_15m[1]] if len(ema_cols_15m) >= 2 else live_price

            ema_cols_htf = [c for c in ts_htf.columns if 'ema' in c.lower()]
            ema200_htf = latest_htf[sorted(ema_cols_htf)[0]] if ema_cols_htf else live_price

            # Trend alignment
            aligned = (live_price > ema200_15m and live_price > ema200_htf) or \
                      (live_price < ema200_15m and live_price < ema200_htf)

            if not aligned:
                st.warning(f"Trend misalignment – {htf_label} EMA200 at ${ema200_htf:.2f}")
                st.markdown("**Explanation:** 15m and HTF trends conflict → avoid counter-trend trades.")
                st.markdown("**Action:** Wait ~15–30 min and re-analyze.")
                st.stop()

            bias = "BULLISH" if live_price > ema200_15m else "BEARISH"

            # Filtered fractals
            levels = get_fractal_levels(ts_15m, min_dist_factor=0.4)
            resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
            supports    = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

            # Original setup (fallback)
            sl_dist = round(atr * 1.5, 2)
            min_sl_distance = atr * 0.4
            min_rr = 1.2

            if bias == "BULLISH":
                original_entry = ema50_15m if (live_price - ema50_15m) > (atr * 0.5) else live_price
                valid_sup = [s for s in supports if s < original_entry]
                candidate_sl = valid_sup[0] - (0.3 * atr) if valid_sup else original_entry - sl_dist
                original_sl = min(candidate_sl, original_entry - min_sl_distance)
                original_tp = resistances[0] if resistances else original_entry + (sl_dist * 2.5)
                original_action = "BUY AT MARKET" if original_entry == live_price else "BUY LIMIT ORDER"
            else:
                original_entry = ema50_15m if (ema50_15m - live_price) > (atr * 0.5) else live_price
                valid_res = [r for r in resistances if r > original_entry]
                candidate_sl = valid_res[0] + (0.3 * atr) if valid_res else original_entry + sl_dist
                original_sl = max(candidate_sl, original_entry + min_sl_distance)
                original_tp = supports[0] if supports else original_entry - (sl_dist * 2.5)
                original_action = "SELL AT MARKET" if original_entry == live_price else "SELL LIMIT ORDER"

            # Risk early
            buffer = st.session_state.balance - st.session_state.floor
            cash_risk = min(buffer * (st.session_state.risk_pct / 100), st.session_state.daily_limit or buffer)

            min_risk_vol = atr * 100 * 0.01 * 2
            min_risk_pct = buffer * 0.005
            min_risk_hard = 10
            min_risk_overall = max(min_risk_vol, min_risk_pct, min_risk_hard)

            # Triple AI
            st.divider()
            st.subheader("Triple AI Opinions")
            market = {"price": live_price, "rsi": rsi}
            setup = {
                "type": bias,
                "entry": original_entry,
                "sl_distance": abs(original_entry - original_sl),
                "atr": atr,
                "risk_pct": st.session_state.risk_pct,
                "risk": cash_risk
            }
            g_verdict, k_verdict, c_verdict = get_ai_advice(market, setup, levels, buffer, st.session_state.mode)

            g_parsed = parse_ai_output(g_verdict)
            k_parsed = parse_ai_output(k_verdict)
            c_parsed = parse_ai_output(c_verdict)

            col1, col2, col3 = st.columns(3)
            with col1: st.markdown("**Gemini**"); st.info(g_verdict)
            with col2: st.markdown("**Grok**"); st.info(k_verdict)
            with col3: st.markdown("**ChatGPT**"); st.info(c_verdict)

            # Consensus
            parsed_verdicts = [g_parsed["verdict"], k_parsed["verdict"], c_parsed["verdict"]]
            elite_count = sum(1 for v in parsed_verdicts if v in ["ELITE", "HIGH_CONV"])
            skip_count = sum(1 for v in parsed_verdicts if v == "SKIP")

            if elite_count == 3:
                st.success("3/3 High Conviction – Strongest signal")
            elif elite_count == 2:
                st.info("2/3 High Conviction – Reasonable confidence")
            elif skip_count == 3:
                st.error("3/3 Skip – No high-conviction trade")
            elif skip_count == 2:
                st.warning("2/3 Skip – Caution advised")
            else:
                st.markdown("<div style='padding:10px; background:#555; color:white; border-radius:5px;'>Mixed consensus – Review or skip</div>", unsafe_allow_html=True)

            # Proposal override + validation
            proposals = [p for p in [g_parsed, k_parsed, c_parsed] if p["proposal"] != "NONE" and p["entry"] is not None]
            final_entry = original_entry
            final_sl = original_sl
            final_tp = original_tp
            final_action = original_action
            override_applied = False

            if len(proposals) >= 2:
                entries = [p["entry"] for p in proposals]
                directions = [p["direction"] for p in proposals if p["direction"]]
                if len(set(directions)) == 1:
                    entry_spread = max(entries) - min(entries)
                    if entry_spread <= 10:
                        avg_entry = np.mean(entries)
                        avg_sl = np.mean([p["sl"] for p in proposals if p["sl"] is not None])
                        avg_tp = np.mean([p["tp"] for p in proposals if p["tp"] is not None])
                        # Validate direction after avg
                        valid = (bias == "BULLISH" and avg_sl < avg_entry) or (bias == "BEARISH" and avg_sl > avg_entry)
                        if valid:
                            final_entry = avg_entry
                            final_sl = avg_sl
                            final_tp = avg_tp
                            final_action = "BUY LIMIT ORDER" if directions[0] == "BULLISH" else "SELL LIMIT ORDER"
                            override_applied = True
                            st.success(f"AI Consensus Override ({len(proposals)} proposals) – Averaged levels applied")
                        else:
                            st.warning("AI override created invalid SL direction — using original setup")

            # Final calc & display
            risk_dist = abs(final_entry - final_sl)
            reward_dist = abs(final_tp - final_entry)
            actual_rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

            risk_per_lot = (risk_dist + SPREAD_BUFFER_POINTS + SLIPPAGE_BUFFER_POINTS) * PIP_VALUE
            lots = cash_risk / risk_per_lot
            lots = max(0.01, round(lots / 0.01) * 0.01)  # nearest 0.01
            actual_risk = lots * risk_per_lot

            setup_valid = True
            warning_msgs = []

            if cash_risk < min_risk_overall:
                warning_msgs.append(f"Risk too small (${cash_risk:.2f} < dynamic min ${min_risk_overall:.2f})")
                setup_valid = False
            if not ((bias == "BULLISH" and final_sl < final_entry) or (bias == "BEARISH" and final_sl > final_entry)):
                warning_msgs.append("Invalid risk direction")
                setup_valid = False
            if actual_rr < min_rr:
                warning_msgs.append(f"R:R too low ({actual_rr:.2f}:1)")
                setup_valid = False

            if setup_valid:
                st.divider()
                st.markdown(f"### {final_action}")
                with st.container(border=True):
                    st.metric("Entry", f"${final_entry:.2f}")
                    col_sl, col_tp = st.columns(2)
                    col_sl.metric("Stop Loss", f"${final_sl:.2f}")
                    col_tp.metric("Take Profit", f"${final_tp:.2f}")
                    col_lots, col_risk = st.columns(2)
                    col_lots.metric("Lots", f"{lots:.2f}")
                    col_risk.metric("Risk Amount", f"${actual_risk:.2f}")
                    st.metric("Actual R:R", f"1:{actual_rr:.2f}")

                if override_applied:
                    st.caption("R:R shown reflects AI-averaged levels")

            else:
                st.warning("Setup rejected:\n" + "\n".join(warning_msgs))

            with st.expander("Detected Fractal Levels (filtered)"):
                st.write("**Resistance above:**", resistances[:4] or "None nearby")
                st.write("**Support below:**", supports[:4] or "None nearby")

            if st.button("✅ Accept This Setup"):
                st.success("Setup accepted! (Add notification logic in future versions)")

        except Exception as e:
            st.error(f"Analysis error: {e}")
            if st.button("Retry Analysis"):
                st.rerun()
