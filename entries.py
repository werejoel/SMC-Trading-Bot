def identify_advanced_entries(df, order_blocks, fvgs, liquidity_zones, window=20):
    """
    Identify high-probability entries where multiple SMC concepts align:
    - Order block test
    - Fair Value Gap presence
    - Break of Market Structure (BMS)
    - Liquidity swipe
    """
    entries = []
    current_price = df['close'].iloc[-1]
    
    # Get unmitigated components
    unmitigated_obs = order_blocks[order_blocks['mitigated'] == False]
    unmitigated_fvgs = fvgs[fvgs['filled'] == False]
    
    # Identify BMS points
    structure_breaks = identify_market_structure_breaks(df)
    recent_breaks = structure_breaks[structure_breaks['timestamp'] >= df['timestamp'].iloc[-window]]
    
    for i in range(len(df)-window, len(df)-1):
        current_candle = df.iloc[i]
        next_candle = df.iloc[i+1]
        
        # === BULLISH ENTRY CONDITIONS ===
        bullish_conditions = []
        
        # 1. Check for bullish order block test
        has_bullish_ob = False
        for _, ob in unmitigated_obs[unmitigated_obs['type'] == 'bullish'].iterrows():
            if current_candle['low'] <= ob['high'] and current_candle['close'] > ob['low']:
                has_bullish_ob = True
                ob_level = ob['high']
                bullish_conditions.append("Bullish OB test")
                break
        
        # 2. Check for bullish FVG above
        has_bullish_fvg = False
        for _, fvg in unmitigated_fvgs[unmitigated_fvgs['type'] == 'bullish'].iterrows():
            if current_price < fvg['bottom'] and abs(current_price - fvg['bottom'])/current_price < 0.02:
                has_bullish_fvg = True
                fvg_level = fvg['bottom']
                bullish_conditions.append("Bullish FVG above")
                break
        
        # 3. Check for recent bullish BMS
        has_bullish_bms = False
        for _, bms in recent_breaks[recent_breaks['type'].str.contains('Bullish')].iterrows():
            if bms['timestamp'] <= current_candle['timestamp'] and bms['price'] < current_price:
                has_bullish_bms = True
                bms_level = bms['price']
                bullish_conditions.append("Bullish BMS")
                break
        
        # 4. Check for liquidity swipe below
        has_liquidity_swipe = False
        for _, liq in liquidity_zones[liquidity_zones['type'] == 'liquidity_low'].iterrows():
            if current_candle['low'] <= liq['level'] and next_candle['close'] > liq['level']:
                has_liquidity_swipe = True
                liquidity_level = liq['level']
                bullish_conditions.append("Liquidity swipe")
                break
        
        # === BEARISH ENTRY CONDITIONS ===
        bearish_conditions = []
        
        # 1. Check for bearish order block test
        has_bearish_ob = False
        for _, ob in unmitigated_obs[unmitigated_obs['type'] == 'bearish'].iterrows():
            if current_candle['high'] >= ob['low'] and current_candle['close'] < ob['high']:
                has_bearish_ob = True
                ob_level = ob['low']
                bearish_conditions.append("Bearish OB test")
                break
        
        # 2. Check for bearish FVG below
        has_bearish_fvg = False
        for _, fvg in unmitigated_fvgs[unmitigated_fvgs['type'] == 'bearish'].iterrows():
            if current_price > fvg['top'] and abs(current_price - fvg['top'])/current_price < 0.02:
                has_bearish_fvg = True
                fvg_level = fvg['top']
                bearish_conditions.append("Bearish FVG below")
                break
        
        # 3. Check for recent bearish BMS
        has_bearish_bms = False
        for _, bms in recent_breaks[recent_breaks['type'].str.contains('Bearish')].iterrows():
            if bms['timestamp'] <= current_candle['timestamp'] and bms['price'] > current_price:
                has_bearish_bms = True
                bms_level = bms['price']
                bearish_conditions.append("Bearish BMS")
                break
        
        # 4. Check for liquidity swipe above
        has_bearish_liquidity = False
        for _, liq in liquidity_zones[liquidity_zones['type'] == 'liquidity_high'].iterrows():
            if current_candle['high'] >= liq['level'] and next_candle['close'] < liq['level']:
                has_bearish_liquidity = True
                liquidity_level = liq['level']
                bearish_conditions.append("Liquidity swipe")
                break
        
        # === GENERATE ENTRY SIGNALS ===
        # Bullish Entry (minimum 3 conditions)
        if len(bullish_conditions) >= 3:
            stop_loss = min(current_candle['low'], liquidity_level if has_liquidity_swipe else float('inf')) * 0.998
            take_profit = current_price + (current_price - stop_loss) * 2
            
            entries.append({
                'timestamp': current_candle['timestamp'],
                'type': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 'high' if len(bullish_conditions) >= 4 else 'medium',
                'conditions': ', '.join(bullish_conditions),
                'risk_reward': round((take_profit - current_price) / (current_price - stop_loss), 2)
            })
        
        # Bearish Entry (minimum 3 conditions)
        if len(bearish_conditions) >= 3:
            stop_loss = max(current_candle['high'], liquidity_level if has_bearish_liquidity else float('-inf')) * 1.002
            take_profit = current_price - (stop_loss - current_price) * 2
            
            entries.append({
                'timestamp': current_candle['timestamp'],
                'type': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 'high' if len(bearish_conditions) >= 4 else 'medium',
                'conditions': ', '.join(bearish_conditions),
                'risk_reward': round((current_price - take_profit) / (stop_loss - current_price), 2)
            })
    
    return pd.DataFrame(entries)