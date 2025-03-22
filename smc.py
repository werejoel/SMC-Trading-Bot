import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import requests
import ta
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')
from config import MT5_CONFIG, API_KEYS

# Replace API Configuration section
ALPHA_VANTAGE_API_KEY = API_KEYS['alphavantage']
OANDA_API_KEY = API_KEYS['oanda']
MT5_TERMINAL_PATH = MT5_CONFIG['path']

# Initialize exchange (using ccxt)
def initialize_exchange(exchange_id='binance', api_key=None, secret=None):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
    })
    return exchange

exchange = initialize_exchange()

def initialize_forex_connection():
    """Initialize connection for forex data"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(
            path=MT5_CONFIG['path'],
            login=MT5_CONFIG['login'],
            password=MT5_CONFIG['password'],
            server=MT5_CONFIG['server'],
            timeout=MT5_CONFIG['timeout']
        ):
            print("MT5 initialization failed")
            return None
        return mt5
    except ImportError:
        print("MetaTrader5 package not found. Using alternative data source.")
        return None

def fetch_forex_data(symbol, timeframe, limit=200):
    """
    Fetch forex/commodities data from multiple sources
    Supported symbols: XAUUSD, GBPJPY, USDJPY, USOIL
    """
    # Convert timeframe to MT5 format
    tf_map = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '4h': '240',
        '1d': '1440'
    }
    
    # Try MetaTrader5 first
    mt5 = initialize_forex_connection()
    if mt5:
        try:
            # Convert symbol format
            mt5_symbol = symbol.replace('XAU', 'GOLD').replace('/', '')
            
            # Get timeframe constant
            tf_constant = getattr(mt5.TIMEFRAME_M1 if tf_map[timeframe] == '1' else
                                mt5.TIMEFRAME_M5 if tf_map[timeframe] == '5' else
                                mt5.TIMEFRAME_M15 if tf_map[timeframe] == '15' else
                                mt5.TIMEFRAME_M30 if tf_map[timeframe] == '30' else
                                mt5.TIMEFRAME_H1 if tf_map[timeframe] == '60' else
                                mt5.TIMEFRAME_H4 if tf_map[timeframe] == '240' else
                                mt5.TIMEFRAME_D1)

            # Fetch data
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_constant, 0, limit)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'tick_volume': 'volume'
                })
                return df

        except Exception as e:
            print(f"MT5 data fetch error: {e}")

    # Fallback to Alpha Vantage for forex data
    if 'USD' in symbol or 'JPY' in symbol:
        try:
            base = symbol[:3]
            quote = symbol[3:]
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={quote}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series FX (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient='index')
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close'
                })
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df['volume'] = 0
                df = df.reset_index()
                df = df.rename(columns={'index': 'timestamp'})
                df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
                return df.tail(limit)

        except Exception as e:
            print(f"Alpha Vantage data fetch error: {e}")

    # Last resort: Try CCXT for available symbols
    try:
        # Modify symbol format for CCXT
        ccxt_symbol = symbol.replace('XAU', 'GOLD')
        if 'OIL' in symbol:
            ccxt_symbol = 'USOIL/USD'
            
        return fetch_data(ccxt_symbol, timeframe, limit, source='ccxt')
    except Exception as e:
        print(f"CCXT data fetch error: {e}")
        return None

# Fetch market data with retry mechanism
def fetch_data(symbol, timeframe, limit=200, source='ccxt'):
    if source == 'alphavantage':
        try:
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={symbol[:3]}&to_symbol={symbol[4:]}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series FX (Daily)" not in data:
                print(f"Error fetching data from Alpha Vantage: {data}")
                return None
            
            df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient='index')
            df = df.rename(columns={
                '1. open': 'open', 
                '2. high': 'high', 
                '3. low': 'low', 
                '4. close': 'close' 
            })
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df['volume'] = 0  # AlphaVantage FX data doesn't include volume
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})
            df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
            
            return df.tail(limit)
        except Exception as e:
            print(f"Error fetching AlphaVantage data: {e}")
            return None



    # Default to CCXT
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
            return df
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2)  # Wait before retry
    
    print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return None



# ===== TECHNICAL INDICATORS =====
def add_technical_indicators(df):
    """Add various technical indicators to the dataframe"""
    # Volume indicators
    df['volume_ema'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    
    # Trend indicators
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_diff'] = ta.trend.macd_diff(df['close'])
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    
    # Volatility indicators
    df['bollinger_high'] = ta.volatility.bollinger_hband(df['close'])
    df['bollinger_low'] = ta.volatility.bollinger_lband(df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    return df


# ===== LIQUIDITY DETECTION =====
def identify_liquidity_zones(df, window=10, threshold_pct=0.005):
    """
    Identify liquidity zones where price might hunt for stops
    - window: period to look for swing highs/lows
    - threshold_pct: minimum distance between zones (as % of price)
    """
    df = df.copy()
    
    # Calculate threshold based on average price
    avg_price = df['close'].mean()
    threshold = avg_price * threshold_pct
    
    liquidity_zones = []
    for i in range(window, len(df) - window):
        # Check if this is a local high/low compared to surrounding candles
        is_swing_high = all(df['high'][i] >= df['high'][j] for j in range(i-window, i+window+1) if j != i)
        is_swing_low = all(df['low'][i] <= df['low'][j] for j in range(i-window, i+window+1) if j != i)
        
        # Verify if it's significant (avoid noise)
        if is_swing_high:
            # Check if we already have a nearby liquidity zone
            is_unique = all(abs(zone['level'] - df['high'][i]) > threshold for zone in liquidity_zones if zone['type'] == 'liquidity_high')
            if is_unique:
                liquidity_zones.append({
                    'type': 'liquidity_high', 
                    'level': df['high'][i], 
                    'timestamp': df['timestamp'][i],
                    'strength': sum(1 for j in range(i-window*2, i+window*2) if j >= 0 and j < len(df) and abs(df['high'][j] - df['high'][i]) < threshold)
                })
        
        if is_swing_low:
            is_unique = all(abs(zone['level'] - df['low'][i]) > threshold for zone in liquidity_zones if zone['type'] == 'liquidity_low')
            if is_unique:
                liquidity_zones.append({
                    'type': 'liquidity_low', 
                    'level': df['low'][i], 
                    'timestamp': df['timestamp'][i],
                    'strength': sum(1 for j in range(i-window*2, i+window*2) if j >= 0 and j < len(df) and abs(df['low'][j] - df['low'][i]) < threshold)
                })
    
    # Filter out weak liquidity zones (only keep those that have been tested multiple times)
    strong_liquidity_zones = [zone for zone in liquidity_zones if zone['strength'] >= 3]
    
    return pd.DataFrame(strong_liquidity_zones)



# ===== SUPPLY AND DEMAND ZONES =====
def identify_supply_demand_zones(df, window=20, rejection_threshold=0.02):
    """
    Identify supply and demand zones based on price rejection and volume.
    Refined criteria for demand zones to improve accuracy.
    """
    zones = []
    
    for i in range(window, len(df) - window):
        # Supply zone criteria: 
        # - A strong bearish candle (close much lower than open)
        # - Followed by price moving down
        # - Higher than average volume
        
        if (df['open'][i] - df['close'][i]) > (df['high'][i] - df['low'][i]) * 0.5:  # Strong bearish candle
            # Check if price moved down after this candle
            avg_price_after = df['close'][i+1:i+6].mean()
            if avg_price_after < df['close'][i]:
                # Check if volume was significant
                if df['volume'][i] > df['volume'][i-10:i+10].mean() * 1.5:
                    zones.append({
                        'type': 'supply',
                        'top': df['high'][i],
                        'bottom': df['open'][i],
                        'timestamp': df['timestamp'][i],
                        'strength': df['volume'][i] / df['volume'][i-10:i+10].mean()
                    })
        
        # Refined demand zone criteria:
        # - A strong bullish candle (close much higher than open)
        # - Followed by price moving up
        # - Higher than average volume
        # - Candle body must be at least 50% of the total range
        
        if (df['close'][i] - df['open'][i]) > (df['high'][i] - df['low'][i]) * 0.5:  # Strong bullish candle
            # Check if price moved up after this candle
            avg_price_after = df['close'][i+1:i+6].mean()
            if avg_price_after > df['close'][i]:
                # Check if volume was significant
                if df['volume'][i] > df['volume'][i-10:i+10].mean() * 1.5:
                    zones.append({
                        'type': 'demand',
                        'top': df['close'][i],
                        'bottom': df['open'][i],
                        'timestamp': df['timestamp'][i],
                        'strength': df['volume'][i] / df['volume'][i-10:i+10].mean()
                    })
    
    return pd.DataFrame(zones)



# ===== ORDER BLOCKS =====
def identify_order_blocks(df, lookback=3):
    """
    Identify order blocks (mitigated and unmitigated)
    - Order blocks are areas where price started a strong move
    """
    order_blocks = []
    
    for i in range(lookback, len(df) - lookback):
        # Bullish order block: A bearish candle followed by strong bullish moves
        if df['close'][i] < df['open'][i]:  # Bearish candle
            # Check if next candles show strong bullish momentum
            next_candles_bullish = all(df['close'][j] > df['open'][j] for j in range(i+1, min(i+lookback+1, len(df))))
            next_candles_higher = df['high'][i+lookback] > df['high'][i] + (df['high'][i] - df['low'][i])
            
            if next_candles_bullish and next_candles_higher:
                # Check if this order block is still unmitigated (price hasn't returned to it)
                mitigated = any(df['low'][j] <= df['low'][i] for j in range(i+lookback+1, len(df)))
                
                order_blocks.append({
                    'type': 'bullish',
                    'high': df['high'][i],
                    'low': df['low'][i],
                    'timestamp': df['timestamp'][i],
                    'mitigated': mitigated
                })
        
        # Bearish order block: A bullish candle followed by strong bearish moves
        if df['close'][i] > df['open'][i]:  # Bullish candle
            # Check if next candles show strong bearish momentum
            next_candles_bearish = all(df['close'][j] < df['open'][j] for j in range(i+1, min(i+lookback+1, len(df))))
            next_candles_lower = df['low'][i+lookback] < df['low'][i] - (df['high'][i] - df['low'][i])
            
            if next_candles_bearish and next_candles_lower:
                # Check if this order block is still unmitigated (price hasn't returned to it)
                mitigated = any(df['high'][j] >= df['high'][i] for j in range(i+lookback+1, len(df)))
                
                order_blocks.append({
                    'type': 'bearish',
                    'high': df['high'][i],
                    'low': df['low'][i],
                    'timestamp': df['timestamp'][i],
                    'mitigated': mitigated
                })
    
    return pd.DataFrame(order_blocks)



# ===== CANDLESTICK PATTERNS =====
def identify_candlestick_patterns(df):
    """
    Identify common candlestick patterns
    """
    patterns = []
    
    for i in range(5, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        prev2_row = df.iloc[i-2]
        
        candle_size = current_row['high'] - current_row['low']
        body_size = abs(current_row['close'] - current_row['open'])
        prev_body_size = abs(prev_row['close'] - prev_row['open'])
        
        # Doji: Very small body compared to range
        if body_size <= candle_size * 0.1:
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'doji',
                'significance': 'neutral'
            })
        
        # Hammer: Small body at the top, long lower wick, little/no upper wick
        if (current_row['close'] > current_row['open'] and  # Bullish
            body_size <= candle_size * 0.3 and
            current_row['high'] - max(current_row['open'], current_row['close']) <= body_size * 0.3 and
            min(current_row['open'], current_row['close']) - current_row['low'] >= candle_size * 0.6):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'hammer',
                'significance': 'bullish'
            })
        
        # Hanging Man: Same as hammer but in an uptrend
        if (i > 5 and
            all(df.iloc[j]['close'] > df.iloc[j-1]['close'] for j in range(i-4, i)) and
            body_size <= candle_size * 0.3 and
            current_row['high'] - max(current_row['open'], current_row['close']) <= body_size * 0.3 and
            min(current_row['open'], current_row['close']) - current_row['low'] >= candle_size * 0.6):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'hanging_man',
                'significance': 'bearish'
            })
        
        # Engulfing patterns
        if (current_row['open'] < prev_row['close'] and
            current_row['close'] > prev_row['open'] and
            body_size > prev_body_size):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'bullish_engulfing',
                'significance': 'bullish'
            })
        
        if (current_row['open'] > prev_row['close'] and
            current_row['close'] < prev_row['open'] and
            body_size > prev_body_size):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'bearish_engulfing',
                'significance': 'bearish'
            })
        
        # Evening Star (bearish reversal)
        if (i > 5 and
            prev2_row['close'] > prev2_row['open'] and  # First candle bullish
            abs(prev_row['close'] - prev_row['open']) < prev_body_size * 0.3 and  # Second candle small
            current_row['close'] < current_row['open'] and  # Third candle bearish
            current_row['close'] < (prev2_row['open'] + prev2_row['close']) / 2):  # Close below midpoint of first candle
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'evening_star',
                'significance': 'bearish'
            })
        
        # Morning Star (bullish reversal)
        if (i > 5 and
            prev2_row['close'] < prev2_row['open'] and  # First candle bearish
            abs(prev_row['close'] - prev_row['open']) < prev_body_size * 0.3 and  # Second candle small
            current_row['close'] > current_row['open'] and  # Third candle bullish
            current_row['close'] > (prev2_row['open'] + prev2_row['close']) / 2):  # Close above midpoint of first candle
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'morning_star',
                'significance': 'bullish'
            })
        
        # Three White Soldiers (strong bullish trend)
        if (i > 5 and
            all(df.iloc[j]['close'] > df.iloc[j]['open'] for j in range(i-2, i+1)) and
            all(df.iloc[j]['close'] > df.iloc[j-1]['close'] for j in range(i-1, i+1)) and
            all(df.iloc[j]['open'] > df.iloc[j-1]['open'] for j in range(i-1, i+1))):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'three_white_soldiers',
                'significance': 'bullish'
            })
        
        # Three Black Crows (strong bearish trend)
        if (i > 5 and
            all(df.iloc[j]['close'] < df.iloc[j]['open'] for j in range(i-2, i+1)) and
            all(df.iloc[j]['close'] < df.iloc[j-1]['close'] for j in range(i-1, i+1)) and
            all(df.iloc[j]['open'] < df.iloc[j-1]['open'] for j in range(i-1, i+1))):
            patterns.append({
                'timestamp': current_row['timestamp'],
                'pattern': 'three_black_crows',
                'significance': 'bearish'
            })
    
    return pd.DataFrame(patterns)



# ===== SMART ENTRY POINTS =====
def calculate_smart_entry_points(df, liquidity_zones, supply_demand_zones, candlestick_patterns, fair_value_gaps, order_blocks):
    """
    Calculate smart entry points based on multiple factors.
    Entry points are valid only when:
    - Unmitigated FVG is present
    - Order block is present
    - Liquidity is present to tap into the point
    - Institutional entry alignment is satisfied
    """
    entries = []
    last_row = df.iloc[-1]
    current_price = last_row['close']
    
    # Filter unmitigated FVGs
    unmitigated_fvgs = fair_value_gaps[fair_value_gaps['filled'] == False]
    
    # Filter liquidity zones to only include those that are required to be taken out
    required_liquidity = liquidity_zones[
        (liquidity_zones['type'] == 'liquidity_high') & (liquidity_zones['level'] > current_price) |
        (liquidity_zones['type'] == 'liquidity_low') & (liquidity_zones['level'] < current_price)
    ]
    
    # Filter unmitigated order blocks
    unmitigated_blocks = order_blocks[order_blocks['mitigated'] == False]
    
    # Factor 1: Entry near strong demand zones during uptrends
    if last_row['ema20'] > last_row['ema50']:  # Uptrend condition
        for _, zone in supply_demand_zones[supply_demand_zones['type'] == 'demand'].iterrows():
            if zone['top'] < current_price < zone['top'] * 1.02:  # Price just above demand zone
                # Check if required liquidity is taken out
                if not required_liquidity.empty and all(current_price > liquidity['level'] for _, liquidity in required_liquidity.iterrows()):
                    # Check if unmitigated FVG is present
                    if not unmitigated_fvgs.empty and any(zone['bottom'] <= fvg['bottom'] <= zone['top'] for _, fvg in unmitigated_fvgs.iterrows()):
                        # Check if unmitigated order block is present
                        if not unmitigated_blocks.empty and any(zone['bottom'] <= block['low'] <= zone['top'] for _, block in unmitigated_blocks.iterrows()):
                            entries.append({
                                'type': 'long',
                                'entry_price': current_price,
                                'stop_loss': zone['bottom'] * 0.99,  # Stop just below demand zone
                                'take_profit': current_price + (current_price - zone['bottom']) * 2,  # 1:2 risk-reward
                                'confidence': 'high',
                                'reason': 'Price at demand zone with unmitigated FVG, order block, and liquidity alignment'
                            })
    
    # Factor 2: Entry near strong supply zones during downtrends
    if last_row['ema20'] < last_row['ema50']:  # Downtrend condition
        for _, zone in supply_demand_zones[supply_demand_zones['type'] == 'supply'].iterrows():
            if zone['bottom'] > current_price > zone['bottom'] * 0.98:  # Price just below supply zone
                # Check if required liquidity is taken out
                if not required_liquidity.empty and all(current_price < liquidity['level'] for _, liquidity in required_liquidity.iterrows()):
                    # Check if unmitigated FVG is present
                    if not unmitigated_fvgs.empty and any(zone['bottom'] <= fvg['top'] <= zone['top'] for _, fvg in unmitigated_fvgs.iterrows()):
                        # Check if unmitigated order block is present
                        if not unmitigated_blocks.empty and any(zone['bottom'] <= block['high'] <= zone['top'] for _, block in unmitigated_blocks.iterrows()):
                            entries.append({
                                'type': 'short',
                                'entry_price': current_price,
                                'stop_loss': zone['top'] * 1.01,  # Stop just above supply zone
                                'take_profit': current_price - (zone['top'] - current_price) * 2,  # 1:2 risk-reward
                                'confidence': 'high',
                                'reason': 'Price at supply zone with unmitigated FVG, order block, and liquidity alignment'
                            })
    
    return pd.DataFrame(entries)



# ===== PLOT AND DATA VISUALIZATION =====
def plot_chart(df, liquidity_zones, supply_demand_zones, order_blocks, 
               candlestick_patterns, smart_entries, fair_value_gaps, symbol):
    """
    Plot a clean chart with only the most significant zones and levels.
    Highlight only the required liquidity zones, unmitigated FVGs, and order blocks.
    """
    # Identify market structure breaks and FVGs
    structure_breaks = identify_market_structure_breaks(df)
    fair_value_gaps = identify_fair_value_gaps(df)
    
    # Set the style and figure size for better spacing
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 11))
    
    # Create main price subplot with adjusted margins
    ax_main = fig.add_subplot(111)
    plt.subplots_adjust(right=0.85, left=0.12, top=0.95, bottom=0.15)
    
    # Professional color scheme
    colors = {
        'bg': '#f0f3fa',           # Light blue-gray background
        'grid': '#e6e9f0',         # Subtle grid lines
        'bar_up': '#089981',       # Forest green for bullish
        'bar_down': '#f23645',     # Crimson red for bearish
        'ma1': '#2962ff',          # Royal blue for EMA 20
        'ma2': '#9c27b0',         # Purple for EMA 50
        'ma3': '#ff6d00',         # Orange for EMA 200
        'text': '#131722',         # Dark gray text
        'liquidity': '#1E88E5',    # Main blue for liquidity zones
        'supply': '#FF5252',       # Red for supply zones
        'demand': '#00C853',       # Green for demand zones
        'fvg_bull': '#4CAF50',     # Green for bullish FVG
        'fvg_bear': '#F44336',     # Red for bearish FVG
        'label_bg': '#E3F2FD',     # Very light blue for label backgrounds
    }
    
    # Set background color
    fig.patch.set_facecolor(colors['bg'])
    ax_main.set_facecolor(colors['bg'])
    
    # Calculate price range for improved label spacing
    price_range = df['high'].max() - df['low'].min()
    label_spacing = price_range * 0.04  # Increased to 4% for even better spacing
    
    # Plot candlesticks
    bar_width = 0.8 * (df['timestamp_mpl'].iloc[1] - df['timestamp_mpl'].iloc[0])
    
    for i in range(len(df)):
        is_bullish = df['close'].iloc[i] >= df['open'].iloc[i]
        color = colors['bar_up'] if is_bullish else colors['bar_down']
        
        # Plot bar body
        bottom = min(df['open'].iloc[i], df['close'].iloc[i])
        height = abs(df['close'].iloc[i] - df['open'].iloc[i])
        ax_main.bar(df['timestamp_mpl'].iloc[i], height, bottom=bottom,
                   width=bar_width, color=color, alpha=0.9)
        
        # Plot wicks
        ax_main.plot([df['timestamp_mpl'].iloc[i], df['timestamp_mpl'].iloc[i]], 
                    [df['low'].iloc[i], df['high'].iloc[i]], 
                    color=color, linewidth=0.5, alpha=0.5)
    
    # Plot EMAs
    ax_main.plot(df['timestamp_mpl'], df['ema20'], color=colors['ma1'], 
                 linewidth=1.5, label='EMA 20', alpha=0.9)
    ax_main.plot(df['timestamp_mpl'], df['ema50'], color=colors['ma2'], 
                 linewidth=1.5, label='EMA 50', alpha=0.9)
    ax_main.plot(df['timestamp_mpl'], df['ema200'], color=colors['ma3'], 
                 linewidth=1.5, label='EMA 200', alpha=0.9)
    
    # Plot entry points, stop-loss, and take-profit levels with prices
    if not smart_entries.empty:
        last_timestamp = df['timestamp_mpl'].iloc[-1]
        for _, entry in smart_entries.iterrows():
            # Entry point line and price
            ax_main.axhline(y=entry['entry_price'], color='yellow', linestyle='--', alpha=0.8)
            ax_main.text(last_timestamp, entry['entry_price'], 
                        f" Entry @ {entry['entry_price']:.4f}", 
                        color='yellow', fontweight='bold', va='bottom', ha='left',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='yellow'))

            # Stop-loss line and price
            ax_main.axhline(y=entry['stop_loss'], color='red', linestyle='--', alpha=0.8)
            ax_main.text(last_timestamp, entry['stop_loss'], 
                        f" SL @ {entry['stop_loss']:.4f}", 
                        color='red', fontweight='bold', va='bottom', ha='left',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='red'))

            # Take-profit line and price
            ax_main.axhline(y=entry['take_profit'], color='green', linestyle='--', alpha=0.8)
            ax_main.text(last_timestamp, entry['take_profit'], 
                        f" TP @ {entry['take_profit']:.4f}", 
                        color='green', fontweight='bold', va='bottom', ha='left',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='green'))

    # Initialize label position trackers
    right_labels = []
    left_labels = []
    
    # Filter valid supply and demand zones
    valid_zones = supply_demand_zones[supply_demand_zones['strength'] >= 1.5]  # Adjust strength threshold as needed

    # Plot supply and demand zones
    for _, zone in valid_zones.iterrows():
        if zone['type'] == 'supply':
            color = colors['supply']
            label = 'Supply Zone'
        else:
            color = colors['demand']
            label = 'Demand Zone'
        
        # Plot zone as a filled rectangle
        ax_main.fill_between(
            [df['timestamp_mpl'].iloc[0], df['timestamp_mpl'].iloc[-1]],
            zone['bottom'], zone['top'],
            color=color, alpha=0.2, label=label if label not in ax_main.get_legend_handles_labels()[1] else ""
        )

    # Remove other unnecessary labels and lines
    # Comment out or remove plotting for liquidity zones, order blocks, and other markers

    # Filter required liquidity zones
    required_liquidity = liquidity_zones[
        (liquidity_zones['type'] == 'liquidity_high') & (liquidity_zones['level'] > df['close'].iloc[-1]) |
        (liquidity_zones['type'] == 'liquidity_low') & (liquidity_zones['level'] < df['close'].iloc[-1])
    ]

    # Plot required liquidity zones with $ symbol on the left
    for _, zone in required_liquidity.iterrows():
        label = '$ Required Liquidity'  # Add $ symbol
        color = colors['liquidity']
        
        # Plot zone lines
        ax_main.plot([df['timestamp_mpl'].iloc[0], df['timestamp_mpl'].iloc[-1]], 
                     [zone['level'], zone['level']], '--', color=color, alpha=0.8)
        
        # Add label on the left side
        ax_main.annotate(f"$ {zone['level']:.4f}",
                         xy=(df['timestamp_mpl'].iloc[0], zone['level']),
                         xytext=(-10, 0),
                         textcoords='offset points',
                         fontsize=10,
                         color=color,
                         bbox=dict(facecolor=colors['label_bg'],
                                   edgecolor=color,
                                   alpha=0.9,
                                   boxstyle='round,pad=0.5'),
                         ha='right',
                         va='center')

    # Plot unmitigated FVGs
    for _, fvg in fair_value_gaps[fair_value_gaps['filled'] == False].iterrows():
        color = colors['fvg_bull'] if fvg['type'] == 'bullish' else colors['fvg_bear']
        
        # Plot FVG zone with gradient fill
        ax_main.fill_between([mdates.date2num(fvg['timestamp']), df['timestamp_mpl'].iloc[-1]], 
                             [fvg['bottom'], fvg['bottom']], 
                             [fvg['top'], fvg['top']], 
                             color=color, alpha=0.1)
        
        # Add label with strength
        label = f"FVG ({fvg['strength']:.1f}%)"
        ax_main.annotate(label,
                         xy=(mdates.date2num(fvg['timestamp']), (fvg['top'] + fvg['bottom']) / 2),
                         xytext=(10, 0),
                         textcoords='offset points',
                         fontsize=10,
                         color=color,
                         bbox=dict(facecolor=colors['label_bg'],
                                   edgecolor=color,
                                   alpha=0.9,
                                   boxstyle='round,pad=0.5'))
        
        # Add an arrow pointing to the FVG
        ax_main.annotate('', 
                         xy=(mdates.date2num(fvg['timestamp']), fvg['bottom']),
                         xytext=(mdates.date2num(fvg['timestamp']), fvg['top']),
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.8))

    # Plot unmitigated order blocks
    for _, block in order_blocks[order_blocks['mitigated'] == False].iterrows():
        color = colors['bar_up'] if block['type'] == 'bullish' else colors['bar_down']
        
        # Add label
        y_pos = (block['high'] + block['low']) / 2
        label = f"{'Bullish' if block['type'] == 'bullish' else 'Bearish'} Order Block"
        ax_main.annotate(label, 
                         xy=(df['timestamp_mpl'].iloc[0], y_pos),
                         xytext=(-50, 0),
                         textcoords='offset points',
                         fontsize=10,
                         color=color,
                         bbox=dict(facecolor=colors['label_bg'],
                                   edgecolor=color,
                                   alpha=0.9,
                                   boxstyle='round,pad=0.5'),
                         arrowprops=dict(arrowstyle='->',
                                         color=color,
                                         alpha=0.8,
                                         connectionstyle='arc3,rad=-0.2'),
                         va='center',
                         ha='right')

    # Mark swing highs and lows
    window = 10
    swing_labels = []  # Track label positions for spacing

    for i in range(window, len(df) - window):
        # Swing high
        if all(df['high'].iloc[i] > df['high'].iloc[j] for j in range(i-window, i+window+1) if j != i):
            y_pos = df['high'].iloc[i]
            # Adjust label position if too close to existing labels
            while any(abs(y - y_pos) < label_spacing for y in swing_labels):
                y_pos += label_spacing
            swing_labels.append(y_pos)
            
            ax_main.annotate('Swing High', 
                             xy=(df['timestamp_mpl'].iloc[i], df['high'].iloc[i]),
                             xytext=(-30, 20),
                             textcoords='offset points',
                             fontsize=10,
                             color=colors['liquidity'],
                             bbox=dict(facecolor=colors['label_bg'],
                                       edgecolor=colors['liquidity'],
                                       alpha=0.9,
                                       boxstyle='round,pad=0.5'),
                             arrowprops=dict(arrowstyle='->',
                                             color=colors['liquidity'],
                                             alpha=0.8,
                                             connectionstyle='arc3,rad=-0.2'),
                             va='center',
                             ha='right')
        
        # Swing low
        if all(df['low'].iloc[i] < df['low'].iloc[j] for j in range(i-window, i+window+1) if j != i):
            y_pos = df['low'].iloc[i]
            # Adjust label position if too close to existing labels
            while any(abs(y - y_pos) < label_spacing for y in swing_labels):
                y_pos -= label_spacing
            swing_labels.append(y_pos)
            
            ax_main.annotate('Swing Low', 
                             xy=(df['timestamp_mpl'].iloc[i], df['low'].iloc[i]),
                             xytext=(-30, -20),
                             textcoords='offset points',
                             fontsize=10,
                             color=colors['liquidity'],
                             bbox=dict(facecolor=colors['label_bg'],
                                       edgecolor=colors['liquidity'],
                                       alpha=0.9,
                                       boxstyle='round,pad=0.5'),
                             arrowprops=dict(arrowstyle='->',
                                             color=colors['liquidity'],
                                             alpha=0.8,
                                             connectionstyle='arc3,rad=0.2'),
                             va='center',
                             ha='right')

    # Customize grid and spines
    ax_main.grid(True, color=colors['grid'], alpha=0.3, linestyle='-', linewidth=1)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_color(colors['grid'])
    ax_main.spines['left'].set_color(colors['grid'])
    ax_main.tick_params(colors=colors['text'])
    
    # Set title and labels
    plt.suptitle(f"{symbol} Analysis (1H Timeframe)", y=0.95, color=colors['text'], fontsize=14)
    ax_main.set_title(f"Last Update: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}", 
                      color=colors['text'], fontsize=12, pad=10)
    
    # Format x-axis
    ax_main.xaxis_date()
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, color=colors['text'])
    
    # Add legend
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                  facecolor=colors['label_bg'], edgecolor=colors['grid'])
    
    # Save with high quality
    plt.savefig(f"{symbol.replace('/', '_')}_analysis.png", 
                dpi=300, bbox_inches='tight',
                facecolor=colors['bg'])
    
    plt.show()
    return fig

# ===== ANALYSIS AND SIGNAL GENERATION =====
def analyze_market(symbol, chart_timeframe='4h', entry_timeframe='15m'):
    """
    Perform comprehensive market analysis for a symbol
    chart_timeframe: timeframe for the main chart display (4h)
    entry_timeframe: timeframe for calculating entry points (15m)
    """
    print(f"\n--- Analyzing {symbol} ---")
    
    # Fetch data for multiple timeframes
    df_chart = fetch_data(symbol, chart_timeframe, limit=200)  # 4h data for chart
    df_entry = fetch_data(symbol, entry_timeframe, limit=200)  # 15m data for entries
    
    if df_chart is None or df_entry is None:
        print(f"Could not fetch data for {symbol}")
        return None
    
    # Add technical indicators to both timeframes
    df_chart = add_technical_indicators(df_chart)
    df_entry = add_technical_indicators(df_entry)
    
    # Determine trend from 4h timeframe
    trend = "bullish" if df_chart['ema50'].iloc[-1] > df_chart['ema200'].iloc[-1] else "bearish"
    print(f"Chart timeframe trend: {trend}")
    
    # Use 15m timeframe for detailed analysis
    liquidity_zones = identify_liquidity_zones(df_entry)
    supply_demand_zones = identify_supply_demand_zones(df_entry)
    order_blocks = identify_order_blocks(df_entry)
    candlestick_patterns = identify_candlestick_patterns(df_entry)
    fair_value_gaps = identify_fair_value_gaps(df_entry)
    
    # Calculate smart entry points using 15m data
    smart_entries = calculate_smart_entry_points(df_entry, liquidity_zones, supply_demand_zones, candlestick_patterns, fair_value_gaps, order_blocks)
    
    # Generate trading signals using 15m data but considering 4h trend
    signals = generate_trading_signals(df_entry, liquidity_zones, supply_demand_zones, 
                                    order_blocks, candlestick_patterns, smart_entries, trend)
    
    # Log any potential trade setups
    if not signals.empty:
        log_trades(symbol, signals)
        print(f"Found {len(signals)} trading signals for {symbol}")
    else:
        print(f"No trading signals found for {symbol}")
    
    # Plot analysis chart using 4h timeframe
    print("\nDebug Info before plotting:")
    print(f"Data points in chart timeframe: {len(df_chart)}")
    print(f"Data points in entry timeframe: {len(df_entry)}")
    print(f"Liquidity zones found: {len(liquidity_zones)}")
    print(f"Supply/Demand zones found: {len(supply_demand_zones)}")
    print(f"Order blocks found: {len(order_blocks)}")
    print(f"Candlestick patterns found: {len(candlestick_patterns)}")
    print(f"Smart entries found: {len(smart_entries)}")
    
    # Plot using 4h data but mark entry points from 15m analysis
    plot_chart(df_chart, liquidity_zones, supply_demand_zones, order_blocks, 
                  candlestick_patterns, smart_entries, fair_value_gaps, symbol)
    
    return {
        'symbol': symbol,
        'trend': trend,
        'liquidity_zones': liquidity_zones,
        'supply_demand_zones': supply_demand_zones,
        'order_blocks': order_blocks,
        'candlestick_patterns': candlestick_patterns,
        'signals': signals
    }

def generate_trading_signals(df, liquidity_zones, supply_demand_zones, 
                           order_blocks, candlestick_patterns, smart_entries, higher_trend):
    """
    Generate actionable trading signals based on all analysis
    """
    signals = []
    current_price = df['close'].iloc[-1]
    
    # If we have smart entries, prioritize those
    if not smart_entries.empty:
        for _, entry in smart_entries.iterrows():
            # Check if entry aligns with higher timeframe trend
            if (higher_trend == "bullish" and entry['type'] == "long") or \
               (higher_trend == "bearish" and entry['type'] == "short"):
                confidence = "high"  # Trend alignment increases confidence
            else:
                confidence = entry['confidence']  # Keep original confidence
                
            signals.append({
                'timestamp': df['timestamp'].iloc[-1],
                'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                'signal_type': entry['type'],
                'entry_price': entry['entry_price'],
                'stop_loss': entry['stop_loss'],
                'take_profit': entry['take_profit'],
                'confidence': confidence,
                'reason': entry['reason'],
                'risk_reward_ratio': abs(entry['take_profit'] - entry['entry_price']) / abs(entry['entry_price'] - entry['stop_loss'])
            })
        
        return pd.DataFrame(signals)
    


    # If no smart entries, check for other setups
    # 1. Check for liquidity sweeps (price just broke above/below significant levels)
    recent_liquidity = liquidity_zones[liquidity_zones['timestamp'] > df['timestamp'].iloc[-20]]
    
    for _, zone in recent_liquidity.iterrows():
        if zone['type'] == 'liquidity_high' and df['high'].iloc[-1] > zone['level'] and df['high'].iloc[-2] < zone['level']:
            # Price just swept a high, potential short opportunity if in a downtrend
            if higher_trend == "bearish":
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'short',
                    'entry_price': current_price,
                    'stop_loss': zone['level'] * 1.01,
                    'take_profit': current_price - (zone['level'] * 1.01 - current_price) * 2,
                    'confidence': 'medium',
                    'reason': 'Liquidity sweep at swing high in bearish trend',
                    'risk_reward_ratio': 2.0
                })
        
        if zone['type'] == 'liquidity_low' and df['low'].iloc[-1] < zone['level'] and df['low'].iloc[-2] > zone['level']:
            # Price just swept a low, potential long opportunity if in an uptrend
            if higher_trend == "bullish":
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'long',
                    'entry_price': current_price,
                    'stop_loss': zone['level'] * 0.99,
                    'take_profit': current_price + (current_price - zone['level'] * 0.99) * 2,
                    'confidence': 'medium',
                    'reason': 'Liquidity sweep at swing low in bullish trend',
                    'risk_reward_ratio': 2.0
                })

    # 2. Check for unmitigated order blocks near current price
    unmitigated_blocks = order_blocks[order_blocks['mitigated'] == False]
    
    for _, block in unmitigated_blocks.iterrows():
        # Price approaching a bullish order block from above
        if block['type'] == 'bullish' and \
           current_price <= block['high'] * 1.02 and current_price >= block['low'] * 0.98:
            if higher_trend == "bullish":  # Align with higher timeframe
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'long',
                    'entry_price': current_price,
                    'stop_loss': block['low'] * 0.98,
                    'take_profit': current_price + (current_price - block['low'] * 0.98) * 2,
                    'confidence': 'high',
                    'reason': 'Price at unmitigated bullish order block in bullish trend',
                    'risk_reward_ratio': 2.0
                })
        
        # Price approaching a bearish order block from below
        if block['type'] == 'bearish' and \
           current_price >= block['low'] * 0.98 and current_price <= block['high'] * 1.02:
            if higher_trend == "bearish":  # Align with higher timeframe
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'short',
                    'entry_price': current_price,
                    'stop_loss': block['high'] * 1.02,
                    'take_profit': current_price - (block['high'] * 1.02 - current_price) * 2,
                    'confidence': 'high',
                    'reason': 'Price at unmitigated bearish order block in bearish trend',
                    'risk_reward_ratio': 2.0
                })
    
    # 3. Check for recent candlestick patterns
    recent_patterns = candlestick_patterns[candlestick_patterns['timestamp'] > df['timestamp'].iloc[-3]]
    
    if not recent_patterns.empty:
        latest_pattern = recent_patterns.iloc[-1]
        
        if latest_pattern['significance'] == 'bullish' and higher_trend == 'bullish':
            # Look for nearby support (demand zone or liquidity low)
            nearby_support = False
            support_level = 0
            
            for _, zone in supply_demand_zones[supply_demand_zones['type'] == 'demand'].iterrows():
                if current_price >= zone['bottom'] and current_price <= zone['top'] * 1.05:
                    nearby_support = True
                    support_level = zone['bottom']
                    break
            
            if not nearby_support:
                for _, zone in liquidity_zones[liquidity_zones['type'] == 'liquidity_low'].iterrows():
                    if current_price >= zone['level'] * 0.95 and current_price <= zone['level'] * 1.05:
                        nearby_support = True
                        support_level = zone['level']
                        break
            
            if nearby_support:
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'long',
                    'entry_price': current_price,
                    'stop_loss': support_level * 0.98,
                    'take_profit': current_price + (current_price - support_level * 0.98) * 2,
                    'confidence': 'medium',
                    'reason': f'Bullish {latest_pattern["pattern"]} pattern at support in bullish trend',
                    'risk_reward_ratio': 2.0
                })
        
        if latest_pattern['significance'] == 'bearish' and higher_trend == 'bearish':
            # Look for nearby resistance (supply zone or liquidity high)
            nearby_resistance = False
            resistance_level = 0
            
            for _, zone in supply_demand_zones[supply_demand_zones['type'] == 'supply'].iterrows():
                if current_price <= zone['top'] and current_price >= zone['bottom'] * 0.95:
                    nearby_resistance = True
                    resistance_level = zone['top']
                    break
            
            if not nearby_resistance:
                for _, zone in liquidity_zones[liquidity_zones['type'] == 'liquidity_high'].iterrows():
                    if current_price <= zone['level'] * 1.05 and current_price >= zone['level'] * 0.95:
                        nearby_resistance = True
                        resistance_level = zone['level']
                        break
            
            if nearby_resistance:
                signals.append({
                    'timestamp': df['timestamp'].iloc[-1],
                    'symbol': df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    'signal_type': 'short',
                    'entry_price': current_price,
                    'stop_loss': resistance_level * 1.02,
                    'take_profit': current_price - (resistance_level * 1.02 - current_price) * 2,
                    'confidence': 'medium',
                    'reason': f'Bearish {latest_pattern["pattern"]} pattern at resistance in bearish trend',
                    'risk_reward_ratio': 2.0
                })
    
    return pd.DataFrame(signals)

def log_trades(symbol, signals):
    """
    Log potential trade setups to CSV file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for _, signal in signals.iterrows():
        trade_data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal_type': signal['signal_type'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'confidence': signal['confidence'],
            'reason': signal['reason'],
            'risk_reward_ratio': signal['risk_reward_ratio']
        }
        
        # Create DataFrame and append to CSV
        trade_df = pd.DataFrame([trade_data])
        
        try:
            # Check if file exists
            try:
                existing_df = pd.read_csv('trade_log.csv')
                combined_df = pd.concat([existing_df, trade_df])
                combined_df.to_csv('trade_log.csv', index=False)
            except:
                # File doesn't exist, create new one
                trade_df.to_csv('trade_log.csv', index=False)
                
            print(f"Trade logged: {signal['signal_type']} {symbol} at {signal['entry_price']}")
        except Exception as e:
            print(f"Error logging trade: {e}")


# ===== MULTI-TIMEFRAME ANALYSIS =====
def analyze_multi_timeframe(symbol, timeframes=['1d', '4h', '1h']):
    """
    Perform analysis across multiple timeframes to find high-probability setups
    """
    results = {}
    
    # Analyze each timeframe
    for tf in timeframes:
        df = fetch_data(symbol, tf, limit=200)
        if df is None:
            continue
            
        df = add_technical_indicators(df)
        
        # Store results for this timeframe
        results[tf] = {
            'trend': 'bullish' if df['ema20'].iloc[-1] > df['ema50'].iloc[-1] else 'bearish',
            'rsi': df['rsi'].iloc[-1],
            'atr': df['atr'].iloc[-1],
            'liquidity_zones': identify_liquidity_zones(df),
            'supply_demand_zones': identify_supply_demand_zones(df),
            'order_blocks': identify_order_blocks(df),
            'candlestick_patterns': identify_candlestick_patterns(df)
        }
    
    # Look for confluence across timeframes
    confluence_signals = []
    
    # Check if trend direction is aligned across timeframes
    trend_aligned = all(results[tf]['trend'] == results[timeframes[0]]['trend'] for tf in timeframes if tf in results)
    
    # If we have alignment, look for specific setups
    if trend_aligned and len(results) >= 2:
        trend = results[timeframes[0]]['trend']
        current_price = fetch_data(symbol, '1h', limit=1)['close'].iloc[-1]
        
        if trend == 'bullish':
            # Look for bullish setups with confluence
            
            # Check for demand zones near current price across timeframes
            demand_zone_confluence = False
            for tf in timeframes:
                if tf not in results:
                    continue
                    
                demand_zones = results[tf]['supply_demand_zones']
                demand_zones = demand_zones[demand_zones['type'] == 'demand']
                
                for _, zone in demand_zones.iterrows():
                    if current_price >= zone['bottom'] * 0.98 and current_price <= zone['top'] * 1.05:
                        demand_zone_confluence = True
                        break
            
            # Check for unmitigated bullish order blocks near price
            bullish_ob_confluence = False
            for tf in timeframes:
                if tf not in results:
                    continue
                    
                order_blocks = results[tf]['order_blocks']
                bullish_obs = order_blocks[(order_blocks['type'] == 'bullish') & (order_blocks['mitigated'] == False)]
                
                for _, ob in bullish_obs.iterrows():
                    if current_price >= ob['low'] * 0.98 and current_price <= ob['high'] * 1.05:
                        bullish_ob_confluence = True
                        break
            
            # If we have confluence, generate a high-probability signal
            if demand_zone_confluence and bullish_ob_confluence:
                # Find nearest support level for stop loss
                support_levels = []
                for tf in timeframes:
                    if tf not in results:
                        continue
                        
                    for _, zone in results[tf]['supply_demand_zones'][results[tf]['supply_demand_zones']['type'] == 'demand'].iterrows():
                        support_levels.append(zone['bottom'])
                    
                    for _, zone in results[tf]['liquidity_zones'][results[tf]['liquidity_zones']['type'] == 'liquidity_low'].iterrows():
                        support_levels.append(zone['level'])
                
                if support_levels:
                    # Find closest support below current price
                    valid_supports = [s for s in support_levels if s < current_price]
                    if valid_supports:
                        stop_loss = max(valid_supports) * 0.98
                        take_profit = current_price + (current_price - stop_loss) * 3  # Higher R:R for high-probability setups
                        
                        confluence_signals.append({
                            'symbol': symbol,
                            'signal_type': 'long',
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': 'very_high',
                            'reason': 'Multi-timeframe bullish confluence at demand zone and order block',
                            'risk_reward_ratio': round((take_profit - current_price) / (current_price - stop_loss), 2)
                        })
        
        elif trend == 'bearish':
            # Look for bearish setups with confluence
            
            # Check for supply zones near current price across timeframes
            supply_zone_confluence = False
            for tf in timeframes:
                if tf not in results:
                    continue
                    
                supply_zones = results[tf]['supply_demand_zones']
                supply_zones = supply_zones[supply_zones['type'] == 'supply']
                
                for _, zone in supply_zones.iterrows():
                    if current_price <= zone['top'] * 1.02 and current_price >= zone['bottom'] * 0.95:
                        supply_zone_confluence = True
                        break
            
            # Check for unmitigated bearish order blocks near price
            bearish_ob_confluence = False
            for tf in timeframes:
                if tf not in results:
                    continue
                    
                order_blocks = results[tf]['order_blocks']
                bearish_obs = order_blocks[(order_blocks['type'] == 'bearish') & (order_blocks['mitigated'] == False)]
                
                for _, ob in bearish_obs.iterrows():
                    if current_price <= ob['high'] * 1.02 and current_price >= ob['low'] * 0.95:
                        bearish_ob_confluence = True
                        break
            
            # If we have confluence, generate a high-probability signal
            if supply_zone_confluence and bearish_ob_confluence:
                # Find nearest resistance level for stop loss
                resistance_levels = []
                for tf in timeframes:
                    if tf not in results:
                        continue
                        
                    for _, zone in results[tf]['supply_demand_zones'][results[tf]['supply_demand_zones']['type'] == 'supply'].iterrows():
                        resistance_levels.append(zone['top'])
                    
                    for _, zone in results[tf]['liquidity_zones'][results[tf]['liquidity_zones']['type'] == 'liquidity_high'].iterrows():
                        resistance_levels.append(zone['level'])
                
                if resistance_levels:
                    # Find closest resistance above current price
                    valid_resistances = [r for r in resistance_levels if r > current_price]
                    if valid_resistances:
                        stop_loss = min(valid_resistances) * 1.02
                        take_profit = current_price - (stop_loss - current_price) * 3  # Higher R:R for high-probability setups
                        
                        confluence_signals.append({
                            'symbol': symbol,
                            'signal_type': 'short',
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': 'very_high',
                            'reason': 'Multi-timeframe bearish confluence at supply zone and order block',
                            'risk_reward_ratio': round((current_price - take_profit) / (stop_loss - current_price), 2)
                        })
    
    return pd.DataFrame(confluence_signals) if confluence_signals else pd.DataFrame()

def identify_market_structure_breaks(df, window=10):
    """
    Identify valid breaks of market structure and change of character
    Returns DataFrame with BMS and CHoCH points
    """
    structure_breaks = []
    
    for i in range(window, len(df) - window):
        # Look for significant swing points
        prev_highs = df['high'].iloc[i-window:i].max()
        prev_lows = df['low'].iloc[i-window:i].min()
        next_highs = df['high'].iloc[i+1:i+window].max()
        next_lows = df['low'].iloc[i+1:i+window].min()
        
        # Bullish BMS: Break above significant structure after downtrend
        if (df['low'].iloc[i-window:i].is_monotonic_decreasing and  # Downtrend
            df['high'].iloc[i] > prev_highs and  # Breaks above previous high
            df['close'].iloc[i] > df['open'].iloc[i] and  # Strong bullish close
            next_lows > df['low'].iloc[i]):  # Maintained break
            
            structure_breaks.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': df['high'].iloc[i],
                'type': 'Bullish BMS',
                'valid': True
            })
        
        # Bearish BMS: Break below significant structure after uptrend
        if (df['high'].iloc[i-window:i].is_monotonic_increasing and  # Uptrend
            df['low'].iloc[i] < prev_lows and  # Breaks below previous low
            df['close'].iloc[i] < df['open'].iloc[i] and  # Strong bearish close
            next_highs < df['high'].iloc[i]):  # Maintained break
            
            structure_breaks.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': df['low'].iloc[i],
                'type': 'Bearish BMS',
                'valid': True
            })
        
        # Bullish CHoCH: Higher high after series of lower highs
        if (i > window*2 and
            all(df['high'].iloc[j] > df['high'].iloc[j-1] for j in range(i-window+1, i)) and
            df['high'].iloc[i] > df['high'].iloc[i-window:i].max() and
            next_lows > df['low'].iloc[i-window:i].min()):
            
            structure_breaks.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': df['high'].iloc[i],
                'type': 'Bullish CHoCH',
                'valid': True
            })
        
        # Bearish CHoCH: Lower low after series of higher lows
        if (i > window*2 and
            all(df['low'].iloc[j] < df['low'].iloc[j-1] for j in range(i-window+1, i)) and
            df['low'].iloc[i] < df['low'].iloc[i-window:i].min() and
            next_highs < df['high'].iloc[i-window:i].max()):
            
            structure_breaks.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': df['low'].iloc[i],
                'type': 'Bearish CHoCH',
                'valid': True
            })
    
    return pd.DataFrame(structure_breaks)

def identify_fair_value_gaps(df, lookback=30):
    """
    Identify valid unmitigated Fair Value Gaps (FVG) near recent market structure
    - Only shows FVGs that haven't been filled
    - Focuses on FVGs near potential entry points
    - Validates FVGs against market structure
    """
    fvgs = []
    current_price = df['close'].iloc[-1]
    
    for i in range(len(df)-lookback, len(df)-2):  # Only look at recent price action
        # Bullish FVG
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            # Check if FVG is still valid (not filled)
            is_filled = any(df['low'].iloc[i+3:] <= df['high'].iloc[i])
            
            # Check if FVG is near potential entry (within 2% of current price)
            price_range = abs(df['high'].iloc[i] - current_price) / current_price
            is_relevant = price_range <= 0.02
            
            if not is_filled and is_relevant:
                fvgs.append({
                    'type': 'bullish',
                    'top': df['low'].iloc[i+2],
                    'bottom': df['high'].iloc[i],
                    'timestamp': df['timestamp'].iloc[i+1],
                    'filled': False,
                    'strength': (df['low'].iloc[i+2] - df['high'].iloc[i]) / df['high'].iloc[i] * 100  # Gap size as percentage
                })
        
        # Bearish FVG
        if df['high'].iloc[i+2] < df['low'].iloc[i]:
            # Check if FVG is still valid (not filled)
            is_filled = any(df['high'].iloc[i+3:] >= df['low'].iloc[i])
            
            # Check if FVG is near potential entry (within 2% of current price)
            price_range = abs(df['low'].iloc[i] - current_price) / current_price
            is_relevant = price_range <= 0.02
            
            if not is_filled and is_relevant:
                fvgs.append({
                    'type': 'bearish',
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i+2],
                    'timestamp': df['timestamp'].iloc[i+1],
                    'filled': False,
                    'strength': (df['low'].iloc[i] - df['high'].iloc[i+2]) / df['low'].iloc[i] * 100  # Gap size as percentage
                })
    
    # Sort FVGs by strength and take only the strongest ones
    fvgs_df = pd.DataFrame(fvgs)
    if not fvgs_df.empty:
        fvgs_df = fvgs_df.nlargest(3, 'strength')
    return fvgs_df

# ===== MAIN FUNCTION =====

def main():
    """
    Main function
    """
    print("=" * 50)
    print("Advanced Smart Money Trading Bot By LitoProgrammer")
    print("=" * 50)
    
    # Define symbols to analyze (now including forex and commodities)
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'ADA/USDT']
    forex_symbols = ['XAUUSD', 'GBPJPY', 'USDJPY', 'USOIL']
    
    # Define timeframes
    chart_timeframe = '1h'    # Main chart display timeframe (changed to 1h)
    entry_timeframe = '15m'   # Entry analysis timeframe
    
    all_signals = []
    
    # Run analysis with specified timeframes
    for symbol in crypto_symbols:
        try:
            result = analyze_market(symbol, chart_timeframe=chart_timeframe, entry_timeframe=entry_timeframe)
            if result is not None and 'signals' in result and not result['signals'].empty:
                all_signals.append(result['signals'])
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Run multi-timeframe analysis for confluence signals
    for symbol in crypto_symbols:
        try:
            mtf_signals = analyze_multi_timeframe(symbol, timeframes=['1d', '4h', '1h', '15m'])
            if not mtf_signals.empty:
                all_signals.append(mtf_signals)
                print(f"Found multi-timeframe confluence signal for {symbol}")
        except Exception as e:
            print(f"Error in multi-timeframe analysis for {symbol}: {e}")
    
    # Analyze forex pairs
    for symbol in forex_symbols:
        try:
            # Use forex-specific data fetching
            df_chart = fetch_forex_data(symbol, chart_timeframe)
            df_entry = fetch_forex_data(symbol, entry_timeframe)
            
            if df_chart is not None and df_entry is not None:
                result = analyze_market(symbol, chart_timeframe=chart_timeframe, entry_timeframe=entry_timeframe)
                if result is not None and 'signals' in result and not result['signals'].empty:
                    all_signals.append(result['signals'])
                    
                # Run multi-timeframe analysis
                mtf_signals = analyze_multi_timeframe(symbol, timeframes=['1d', '4h', '1h', '15m'])
                if not mtf_signals.empty:
                    all_signals.append(mtf_signals)
                    print(f"Found multi-timeframe confluence signal for {symbol}")
                    
        except Exception as e:
            print(f"Error analyzing forex pair {symbol}: {e}")

    # Combine all signals
    if all_signals:
        combined_signals = pd.concat(all_signals)
        print("\n=== HIGH PROBABILITY TRADE SETUPS ===")
        
        # Sort by confidence
        confidence_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        combined_signals['confidence_value'] = combined_signals['confidence'].map(confidence_order)
        combined_signals = combined_signals.sort_values('confidence_value')
        
        # Display top signals
        for _, signal in combined_signals.iterrows():
            print(f"\nSymbol: {signal['symbol']}")
            print(f"Signal: {signal['signal_type'].upper()}")
            print(f"Entry: {signal['entry_price']:.8f}")
            print(f"Stop Loss: {signal['stop_loss']:.8f}")
            print(f"Take Profit: {signal['take_profit']:.8f}")
            print(f"Confidence: {signal['confidence'].upper()}")
            print(f"Reason: {signal['reason']}")
            print(f"Risk/Reward: 1:{signal['risk_reward_ratio']}")
            print("-" * 30)

if __name__ == "__main__":
    main()