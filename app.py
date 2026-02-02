if st.button('🚀 Get a Setup!'):
    with st.spinner('Calculating Trend & Risk...'):
        # 1. FETCH DATA (Need 200 periods for the long-term trend)
        ts = td.time_series(symbol="XAU/USD", interval="15min", outputsize=200).with_rsi().as_pandas()
        
        # 2. INDICATORS
        ts['EMA_200'] = ts['close'].ewm(span=200).mean()
        ts['EMA_50'] = ts['close'].ewm(span=50).mean()
        live_price = ts['close'].iloc[0]
        rsi = ts['rsi'].iloc[0]
        ema_200 = ts['EMA_200'].iloc[0]
        ema_50 = ts['EMA_50'].iloc[0]

        # 3. DYNAMIC RISK (Based on your balance/daily limit)
        buffer_to_death = balance - survival_floor
        max_allowed_risk = min(calculated_risk, daily_limit, buffer_to_death)
        
        # 4. TREND CONTINUATION LOGIC
        trend = "BULLISH" if live_price > ema_200 else "BEARISH"
        
        st.subheader(f"Price: ${live_price:.2f} | Trend: {trend}")
        
        # SETUP VARIABLES
        entry_type = "MARKET"
        sl_dist = 20.0 # Standard $20 move SL
        tp_dist = 40.0 # 1:2 Risk/Reward

        if trend == "BULLISH" and live_price <= ema_50:
            st.success("📈 BULLISH PULLBACK: Trend is Up, Price is at Support.")
            entry_type = "BUY LIMIT"
            target_price = live_price + 2.0 # Enter slightly above current
        elif trend == "BEARISH" and live_price >= ema_50:
            st.warning("📉 BEARISH PULLBACK: Trend is Down, Price is at Resistance.")
            entry_type = "SELL LIMIT"
            target_price = live_price - 2.0
        else:
            st.info("🚦 NO CLEAR PULLBACK: Price is too far from Moving Averages.")
            target_price = live_price

        # 5. LOT SIZING (1 Lot = $100 per $1 move in Gold)
        lots = max_allowed_risk / (sl_dist * 100)
        final_lots = round(lots, 2)

        # 6. FINAL OUTPUT
        st.divider()
        if final_lots >= 0.01:
            st.metric("Suggested Lot Size", f"{final_lots}")
            st.write(f"**Order Type:** {entry_type} @ ${target_price:.2f}")
            st.write(f"**Stop Loss:** ${target_price - sl_dist if trend == 'BULLISH' else target_price + sl_dist:.2f}")
            st.write(f"**Take Profit:** ${target_price + tp_dist if trend == 'BULLISH' else target_price - tp_dist:.2f}")
            st.write(f"**Total Risk:** ${final_lots * sl_dist * 100:.2f}")
        else:
            st.error("Account Buffer too low for this setup.")
