import streamlit as st
import pandas as pd
import numpy as np
import re
from twelvedata import TDClient
import google.generativeai as genai
from openai import OpenAI
from datetime import datetime

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
    st.error(f"API setup failed: {e}\nCheck secrets: TWELVE_DATA_KEY, GEMINI_KEY, GROK_API_KEY, OPENAI_API_KEY")
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

# ─── TRIPLE AUDITORS ─────────────────────────────────────────────────────────
def get_ai_advice(market, setup, levels, buffer, mode):
    levels_str = ", ".join([f"{l[0]}@{l[1]}" for l in levels[-5:]]) if levels else "No clear levels"
    current_price = market['price']
    prompt = f"""
You are a high-conviction gold trading auditor for any account size.
Mode: {mode} ({'standard swing (15m + 1h)' if mode == 'Standard' else 'fast scalp (15m + 5m)'}).
Aggressive risk is user's choice — do NOT suggest reducing % risk.
Focus on math, pullback quality, structural confluence, risk/reward.
IMPORTANT: For buys, SL must be BELOW entry. For sells, SL must be ABOVE entry.

Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
NY session close ~22:00 UTC — factor in thinning liquidity and whipsaw risk after 21:30 UTC.
If any high-impact news likely within ±30 min, prefer to wait unless setup is exceptionally strong.

Current market price: ${current_price:.2f}
Buffer left: ${buffer:.2f}
Market: Price ${current_price:.2f}, RSI {market['rsi']:.1f}
Original setup: {setup['type']} at ${setup['entry']:.2f} risking ${setup['risk']:.2f}
Fractals: {levels_str}

Be STRICTLY consistent:
- If you judge the original setup as low-edge, obsolete, missed, gamble, or chasing, your proposal MUST NOT simply re-use or slightly adjust the original entry price — that would contradict your verdict.
- Any proposal MUST respect current market price ${current_price:.2f} — never suggest entries significantly below current price in bullish mode or above in bearish mode unless clear reversal evidence exists.
- Only propose changes that meaningfully improve the setup (e.g. higher entry in trend continuation, opposite bias, different SL/TP, or skip entirely).
- If no good alternative exists, clearly recommend skipping.

First, give a blunt verdict on the original setup (elite or low-edge gamble? 2 sentences max).

Then, if you see a meaningfully better or safer alternative, propose it clearly.
Format your proposal like this:
PROPOSAL: [brief description, e.g. "Move entry to $X, SL to $Y for 2.5:1 RR"]
REASONING: [1-2 sentences why it's better]

If no meaningful change is needed or no good trade exists, just say "No better alternative — skip or wait".
Keep total response under 5 sentences.
"""

    # Gemini
    try:
        g_out = gemini_model.generate_content(prompt).text.strip()
    except:
        g_out = "Gemini Offline."

    # Grok
    try:
        r = grok_client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220
        )
        k_out = r.choices[0].message.content.strip()
    except Exception as e:
        k_out = f"Grok Error: {e}"

    # ChatGPT (gpt-4o-mini)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.65
        )
        c_out = response.choices[0].message.content.strip()
    except Exception as e:
        c_out = f"ChatGPT Error: {e}"

    return g_out, k_out, c_out

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gold Sentinel Pro", page_icon="🥇", layout="wide")
st.title("🥇 Gold Sentinel – High Conviction Gold Entries")
st.caption(f"Adaptive pullback engine | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─── SESSION STATE INIT ──────────────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.balance = None
    st.session_state.daily_limit = None
    st.session_state.floor = 0.0
    st.session_state.risk_pct = 25
    st.session_state.mode = "Standard"

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

    st.session_state.mode = st.radio(
        "Analysis Mode",
        ["Standard (Swing – 15m + 1h alignment)", "Scalp (Fast – 15m + 5m alignment)"],
        index=0 if st.session_state.mode == "Standard" else 1
    )

    if st.button("🚀 Analyze & Suggest", type="primary", use_container_width=True):
        if st.session_state.balance is None:
            st.error("❌ Enter current balance")
        else:
            st.session_state.analysis_done = True
            st.rerun()
else:
    # ─── REMINDER ────────────────────────────────────────────────────────────
    st.info("Analysis locked with your settings:")
    cols = st.columns(5)
    cols[0].metric("Balance", f"${st.session_state.balance:.2f}")
    cols[1].metric("Daily Limit", f"${st.session_state.daily_limit:.2f}" if st.session_state.daily_limit else "No limit")
    cols[2].metric("Floor", f"${st.session_state.floor:.2f}")
    cols[3].metric("Risk %", f"{st.session_state.risk_pct}%")
    cols[4].metric("Mode", st.session_state.mode.split(" – ")[0])

    with st.spinner("Scanning structure..."):
        try:
            price_data = td.price(**{"symbol": "XAU/USD"}).as_json()
            live_price = float(price_data["price"])

            # ─── DATA FETCH ──────────────────────────────────────────────────────
            ts_15m = td.time_series(**{
                "symbol": "XAU/USD",
                "interval": "15min",
                "outputsize": 100
            }).with_rsi(**{}).with_ema(**{"time_period": 200}).with_ema(**{"time_period": 50}).with_atr(**{"time_period": 14}).as_pandas()

            if st.session_state.mode.startswith("Standard"):
                ts_htf = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "1h",
                    "outputsize": 50
                }).with_ema(**{"time_period": 200}).as_pandas()
                htf_label = "1H"
            else:
                ts_htf = td.time_series(**{
                    "symbol": "XAU/USD",
                    "interval": "5min",
                    "outputsize": 100
                }).with_ema(**{"time_period": 200}).as_pandas()
                htf_label = "5M"

            # ─── SAFE INDICATOR EXTRACTION ───────────────────────────────────────
            rsi = ts_15m['rsi'].iloc[0] if 'rsi' in ts_15m.columns else 50.0
            atr = ts_15m['atr'].iloc[0] if 'atr' in ts_15m.columns else 0.0

            ema_cols_15m = [c for c in ts_15m.columns if 'ema' in c.lower()]
            ema_cols_15m.sort()
            ema200_15m = ts_15m[ema_cols_15m[0]].iloc[0] if len(ema_cols_15m) >= 1 else live_price
            ema50_15m  = ts_15m[ema_cols_15m[1]].iloc[0] if len(ema_cols_15m) >= 2 else live_price

            ema_cols_htf = [c for c in ts_htf.columns if 'ema' in c.lower()]
            ema200_htf = ts_htf[ema_cols_htf[0]].iloc[0] if ema_cols_htf else live_price

            # ─── TREND ALIGNMENT ─────────────────────────────────────────────────
            aligned = (live_price > ema200_15m and live_price > ema200_htf) or \
                      (live_price < ema200_15m and live_price < ema200_htf)

            if not aligned:
                st.warning(f"Trend misalignment – {htf_label} EMA200 at ${ema200_htf:.2f}")
                st.markdown("**Short explanation:** The 15-minute and higher-timeframe trends are not aligned. This prevents trades against the larger trend.")
                st.markdown("**Suggested action:** Wait approximately **15 minutes** and press 'Analyze & Suggest' again.")
                st.stop()

            bias = "BULLISH" if live_price > ema200_15m else "BEARISH"

            # ─── FRACTAL LEVELS ──────────────────────────────────────────────────
            levels = get_fractal_levels(ts_15m)
            resistances = sorted([l[1] for l in levels if l[0] == 'RES' and l[1] > live_price])
            supports = sorted([l[1] for l in levels if l[0] == 'SUP' and l[1] < live_price], reverse=True)

            # ─── CALCULATE ORIGINAL SETUP ────────────────────────────────────────
            sl_dist = round(atr * 1.5, 2)
            min_sl_distance = atr * 0.4
            min_rr = 1.2

            if bias == "BULLISH":
                entry = ema50_15m if (live_price - ema50_15m) > (atr * 0.5) else live_price
                
                valid_sup = [s for s in supports if s < entry]
                candidate_sl = valid_sup[0] - (0.3 * atr) if valid_sup else entry - sl_dist
                
                sl = min(candidate_sl, entry - min_sl_distance)
                
                tp = resistances[0] if resistances else entry + (sl_dist * 2.5)

            else:
                entry = ema50_15m if (ema50_15m - live_price) > (atr * 0.5) else live_price
                
                valid_res = [r for r in resistances if r > entry]
                candidate_sl = valid_res[0] + (0.3 * atr) if valid_res else entry + sl_dist
                
                sl = max(candidate_sl, entry + min_sl_distance)
                
                tp = supports[0] if supports else entry - (sl_dist * 2.5)

            # ─── RISK CALC EARLY ─────────────────────────────────────────────────
            buffer = st.session_state.balance - st.session_state.floor
            cash_risk = min(buffer * (st.session_state.risk_pct / 100), st.session_state.daily_limit or buffer)

            # Dynamic min risk
            min_risk_vol = atr * 100 * 0.01 * 2
            min_risk_pct = buffer * 0.005
            min_risk_hard = 10
            min_risk_overall = max(min_risk_vol, min_risk_pct, min_risk_hard)

            # Temporary lots/risk for AI
            sl_dist_actual_temp = abs(entry - sl)
            lots_temp = max(round(cash_risk / ((sl_dist_actual_temp + 0.35) * 100), 2), 0.01) if sl_dist_actual_temp > 0 else 0.01
            actual_risk_temp = round(lots_temp * (sl_dist_actual_temp + 0.35) * 100, 2)

            # ─── TRIPLE AI OPINIONS ──────────────────────────────────────────────
            st.divider()
            st.subheader("Triple AI Opinions")
            market = {"price": live_price, "rsi": rsi}
            setup = {"type": bias, "entry": entry, "risk": actual_risk_temp}
            g_verdict, k_verdict, c_verdict = get_ai_advice(market, setup, levels, buffer, st.session_state.mode)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Gemini (Cautious)**")
                st.info(g_verdict)
            with col2:
                st.markdown("**Grok (Direct)**")
                st.info(k_verdict)
            with col3:
                st.markdown("**ChatGPT (Balanced)**")
                st.info(c_verdict)

            st.caption("AI opinions are probabilistic assessments, not trading signals.")

            # ─── CONSENSUS SUMMARY ───────────────────────────────────────────────
            verdicts = [g_verdict.lower(), k_verdict.lower(), c_verdict.lower()]
            proposals = [v for v in [g_verdict, k_verdict, c_verdict] if "PROPOSAL:" in v]
            elite_count = sum(1 for v in verdicts if "elite" in v)
            gamble_count = sum(1 for v in verdicts if "gamble" in v or "low-edge" in v)

            # Main consensus badge
            if elite_count == 3:
                st.success("3/3 High Conviction – Strongest signal")
            elif elite_count == 2:
                st.info("2/3 High Conviction – Reasonable confidence")
            elif gamble_count >= 2:
                st.warning(f"{gamble_count}/3 Gamble/Low-edge – Caution advised")
            else:
                st.info("Mixed opinions – Review all three carefully")

            # ─── PROPOSALS BOX (only if 2+ similar proposals) ────────────────────
            if len(proposals) >= 2:
                # Simple extraction of proposed entry/SL/TP (regex, crude but good enough)
                entries = []
                sls = []
                tps = []
                for p in proposals:
                    entry_match = re.search(r"entry to \$?([\d\.]+)", p, re.IGNORECASE)
                    sl_match = re.search(r"SL to \$?([\d\.]+)", p, re.IGNORECASE)
                    tp_match = re.search(r"target(?:ing)? .*\$?([\d\.]+)", p, re.IGNORECASE)
                    
                    if entry_match: entries.append(float(entry_match.group(1)))
                    if sl_match: sls.append(float(sl_match.group(1)))
                    if tp_match: tps.append(float(tp_match.group(1)))

                if len(entries) >= 2:
                    avg_entry = np.mean(entries)
                    entry_spread = max(entries) - min(entries)
                    color = "green" if entry_spread <= 10 else "orange"
                    st.markdown(f"<div style='padding:10px; background-color:{color}; color:white; border-radius:5px;'>"
                                f"Proposal consensus ({len(entries)}/{len(proposals)}): Buy/Sell near ${avg_entry:.2f} "
                                f"(spread ${entry_spread:.2f})</div>", unsafe_allow_html=True)

                if len(sls) >= 2:
                    avg_sl = np.mean(sls)
                    st.markdown(f"<div style='padding:8px; background-color:#444; color:white;'>Avg proposed SL: ${avg_sl:.2f}</div>", unsafe_allow_html=True)

                if len(tps) >= 2:
                    avg_tp = np.mean(tps)
                    st.markdown(f"<div style='padding:8px; background-color:#444; color:white;'>Avg proposed TP: ${avg_tp:.2f}</div>", unsafe_allow_html=True)

            # ─── APPLY HARD FILTERS AFTER AI ─────────────────────────────────────
            valid_direction = (bias == "BULLISH" and sl < entry) or (bias == "BEARISH" and sl > entry)
            risk_dist = abs(entry - sl)
            reward_dist = abs(tp - entry)
            actual_rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

            setup_valid = True
            warning_msgs = []

            if cash_risk < min_risk_overall:
                warning_msgs.append(f"Risk too small (\( {cash_risk:.2f}) vs dynamic min ( \){min_risk_overall:.2f})")
                setup_valid = False
            if not valid_direction:
                warning_msgs.append("Invalid risk direction (SL on wrong side of entry)")
                setup_valid = False
            if actual_rr < min_rr:
                warning_msgs.append(f"Reward:risk too low ({actual_rr:.2f}:1)")
                setup_valid = False

            if setup_valid:
                sl_dist_actual = risk_dist
                lots = max(round(cash_risk / ((sl_dist_actual + 0.35) * 100), 2), 0.01)
                actual_risk = round(lots * (sl_dist_actual + 0.35) * 100, 2)

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
                    col_rr = st.columns(1)[0]
                    col_rr.metric("Actual R:R", f"1:{actual_rr:.2f}")

            else:
                st.warning("Setup rejected by filters:\n" + "\n".join(warning_msgs) + "\nSee AI opinions for alternatives or skip")

            # Levels (always show)
            with st.expander("Detected Fractal Levels"):
                st.write("**Resistance above:**", resistances[:3] or "None nearby")
                st.write("**Support below:**", supports[:3] or "None nearby")

            # Accept button
            if st.button("✅ Accept This Setup"):
                st.success("Setup accepted! (Notification pause logic can be added in PWA version)")

            # Save to history
            st.session_state.saved_setups.append({
                "time": datetime.utcnow().strftime("%H:%M UTC"),
                "mode": st.session_state.mode,
                "bias": bias,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "lots": lots_temp if 'lots_temp' in locals() else 0.01,
                "risk": cash_risk,
                "rr": actual_rr,
                "status": "Rejected by filters" if not setup_valid else "Valid"
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
