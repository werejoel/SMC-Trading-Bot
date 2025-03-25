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
FINNHUB_API_KEY = API_KEYS['finnhub']


# Initialize exchange (using ccxt)
def initialize_exchange(exchange_id='binance', api_key=None, secret=None):
    """
    Enhanced exchange initialization with better error handling and rate limiting
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
                'validateMarkets': True,  # Add market validation
                'fetchOrderBookLimit': 100
            },
            'timeout': 30000,
            'rateLimit': 1000  # Conservative rate limit
        })

        # Load markets with retry mechanism
        for attempt in range(3):
            try:
                exchange.load_markets()
                break
            except Exception as e:
                print(f"Market loading attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return None
                time.sleep(2)

        # Test connection with multiple pairs
        test_pairs = ['BTC/USDT', 'ETH/USDT']
        for pair in test_pairs:
            try:
                exchange.fetch_ticker(pair)
            except Exception as e:
                print(f"Error testing {pair}: {e}")
                continue
                
        return exchange
    except Exception as e:
        print(f"Error initializing exchange: {e}")
        return None

# Initialize with better error handling
exchange = initialize_exchange()
if exchange is None:
    print("Warning: Exchange initialization failed. Using backup data source.")
   

   # Initialize MT5 connection
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
    Enhanced forex/commodities data fetching with multiple data sources
    Now includes Finnhub as primary source for forex
    """
    try:
        # Map timeframe to Finnhub format
        tf_map = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        
        # Format symbol for Finnhub
        if 'XAU' in symbol:
            finnhub_symbol = 'OANDA:XAU_USD'
        elif 'USD' in symbol:
            finnhub_symbol = f'OANDA:{symbol[:3]}_{symbol[3:]}'
        else:
            finnhub_symbol = f'OANDA:{symbol.replace("/", "_")}'
            
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (limit * tf_map[timeframe] * 60)  # Convert to seconds
        
        # Make API request
        url = f'https://finnhub.io/api/v1/forex/candle'
        params = {
            'symbol': finnhub_symbol,
            'resolution': tf_map[timeframe],
            'from': start_time,
            'to': end_time,
            'token': FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('s') == 'ok' and len(data.get('t', [])) > 0:
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
            return df
            
        else:
            print(f"Finnhub data fetch failed for {symbol}, trying MT5...")
            
    except Exception as e:
        print(f"Finnhub error: {e}")

    # Try MetaTrader5 first (primary source for forex)
    mt5 = initialize_forex_connection()
    if mt5:
        try:
            # Format symbol for MT5
            mt5_symbol = symbol.replace('/', '')
            if 'XAU' in symbol:
                mt5_symbol = 'XAUUSD'
            elif 'OIL' in symbol:
                mt5_symbol = 'USOIL'
            
            # Ensure symbol exists in MT5
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info is None:
                print(f"Symbol {mt5_symbol} not found in MT5")
                raise LookupError(f"Symbol {mt5_symbol} not found in MT5")
            
            # Fetch data with retry mechanism
            for attempt in range(3):
                try:
                    rates = mt5.copy_rates_from_pos(mt5_symbol, tf_map[timeframe], 0, limit)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)
                        df = df.rename(columns={
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'tick_volume': 'volume',
                            'spread': 'spread',
                            'real_volume': 'real_volume'
                        })
                        return df
                except Exception as e:
                    print(f"MT5 attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            print("All MT5 attempts failed, trying alternative source")
            raise Exception("MT5 data fetch failed")
            
        except Exception as e:
            print(f"MT5 error: {e}")

    # Try Alpha Vantage as backup for forex pairs
    if any(pair in symbol for pair in ['USD', 'EUR', 'GBP', 'JPY', 'XAU']):
        try:
            # Format symbol for Alpha Vantage
            base = symbol[:3]
            quote = symbol[3:]
            
            # Adjust timeframe for Alpha Vantage limitations
            if timeframe not in ['1d', '1h']:
                print(f"Adjusting timeframe from {timeframe} to 1h for Alpha Vantage")
                timeframe = '1h'
            
            # Select appropriate API endpoint
            if timeframe == '1d':
                endpoint = 'FX_DAILY'
            else:
                endpoint = 'FX_INTRADAY'
            
            url = f"https://www.alphavantage.co/query?function={endpoint}&from_symbol={base}&to_symbol={quote}&interval={timeframe}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                raise Exception(f"Alpha Vantage error: {data['Error Message']}")
            
            # Extract time series data
            time_series_key = f"Time Series FX ({timeframe})" if timeframe != '1d' else "Time Series FX (Daily)"
            
            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
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
        # Map forex symbols to available CCXT markets
        ccxt_symbols = {
            'XAUUSD': 'GOLD/USD',
            'USOIL': 'USOIL/USD',
            'BTCUSD': 'BTC/USD'
        }
        
        ccxt_symbol = ccxt_symbols.get(symbol.replace('/', ''), symbol)
        return fetch_data(ccxt_symbol, timeframe, limit, source='ccxt')
        
    except Exception as e:
        print(f"All data sources failed for {symbol}: {e}")
        return None

# Fetch market data with retry mechanism
def fetch_data(symbol, timeframe, limit=200, source='ccxt'):
    """
    Enhanced data fetching with better error handling and data validation
    """
    if source == 'ccxt':
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Validate symbol format
                if '/' not in symbol:
                    symbol = f"{symbol[:3]}/{symbol[3:]}"

                # Ensure timeframe is supported
                supported_timeframes = exchange.timeframes
                if timeframe not in supported_timeframes:
                    print(f"Timeframe {timeframe} not supported. Using 1h instead.")
                    timeframe = '1h'

                # Fetch OHLCV with validation
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv or len(ohlcv) < limit/2:
                    raise Exception("Insufficient data points")

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Add data validation
                df = df[df['high'] >= df['low']]  # Remove invalid candles
                df = df[df['volume'] > 0]  # Remove zero volume candles
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['timestamp_mpl'] = df['timestamp'].apply(mdates.date2num)

                # Check for gaps in data
                time_diff = df['timestamp'].diff()
                expected_diff = pd.Timedelta(timeframe.replace('m', 'min').replace('h', 'hour'))
                if time_diff.max() > expected_diff * 2:
                    print(f"Warning: Data gaps detected in {symbol}")

                return df
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(2)  # Wait before retry
        
        print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        return None

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
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    
    # Volatility indicators
    df['bollinger_high'] = ta.volatility.bollinger_hband(df['close'])
    df['bollinger_low'] = ta.volatility.bollinger_lband(df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    return df

def add_crypto_indicators(df):
    """
    Add crypto-specific technical indicators
    """
    df = add_technical_indicators(df)  # Add base indicators first
    
    # Add crypto-specific indicators
    df['mf_index'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
    
    # Add volatility bands for crypto
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2.5)  # Wider bands for crypto
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2.5)
    
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



# ===== SUPPLY AND DEMAND ZONES Logic =====
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
        
        # Demand zone criteria:
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
def identify_order_blocks(df, lookback=3, min_volume_multiplier=2.0):
    """
    Identify institutional order blocks with clearer rules:
    1. Strong volume spike compared to average
    2. Clear price rejection in opposite direction
    3. Clean break in intended direction
    4. Block hasn't been mitigated (price hasn't returned)
    """
    order_blocks = []
    avg_volume = df['volume'].rolling(20).mean()
    atr = df['atr'].fillna(df['atr'].mean())  # Handle NaN values
    
    for i in range(lookback, len(df) - lookback):
        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
        
        # Check for significant volume
        volume_spike = df['volume'].iloc[i] > avg_volume.iloc[i] * min_volume_multiplier
        
        if not volume_spike:
            continue
            
        # Bullish Order Block (Bearish candle followed by strong bullish move)
        if df['close'].iloc[i] < df['open'].iloc[i]:  # Bearish candle
            # Measure momentum after the candle
            next_candles_momentum = sum(1 for j in range(i+1, min(i+lookback+1, len(df)))
                                     if df['close'].iloc[j] > df['open'].iloc[j])
            
            price_movement = (df['high'].iloc[i+1:i+lookback+1].max() - df['low'].iloc[i]) / atr.iloc[i]
            
            if next_candles_momentum >= 2 and price_movement > 1.5:  # Strong bullish follow-through
                # Check if block hasn't been mitigated
                mitigated = any(df['low'].iloc[j] <= df['low'].iloc[i] 
                              for j in range(i+lookback+1, len(df)))
                
                # Calculate block strength based on volume and follow-through
                strength = (df['volume'].iloc[i] / avg_volume.iloc[i]) * price_movement
                
                order_blocks.append({
                    'type': 'bullish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'entry': df['close'].iloc[i],  # Add entry price
                    'timestamp': df['timestamp'].iloc[i],
                    'mitigated': mitigated,
                    'strength': strength,
                    'volume_multiple': df['volume'].iloc[i] / avg_volume.iloc[i]
                })
        
        # Bearish Order Block (Bullish candle followed by strong bearish move)
        if df['close'].iloc[i] > df['open'].iloc[i]:  # Bullish candle
            next_candles_momentum = sum(1 for j in range(i+1, min(i+lookback+1, len(df)))
                                     if df['close'].iloc[j] < df['open'].iloc[j])
            
            price_movement = (df['high'].iloc[i] - df['low'].iloc[i+1:i+lookback+1].min()) / atr.iloc[i]
            
            if next_candles_momentum >= 2 and price_movement > 1.5:  # Strong bearish follow-through
                # Check if block hasn't been mitigated
                mitigated = any(df['high'].iloc[j] >= df['high'].iloc[i] 
                              for j in range(i+lookback+1, len(df)))
                
                # Calculate block strength
                strength = (df['volume'].iloc[i] / avg_volume.iloc[i]) * price_movement
                
                order_blocks.append({
                    'type': 'bearish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'entry': df['close'].iloc[i],  # Add entry price
                    'timestamp': df['timestamp'].iloc[i],
                    'mitigated': mitigated,
                    'strength': strength,
                    'volume_multiple': df['volume'].iloc[i] / avg_volume.iloc[i]
                })
    
    # Convert to DataFrame and sort by strength
    df_blocks = pd.DataFrame(order_blocks)
    if not df_blocks.empty:
        df_blocks = df_blocks.sort_values('strength', ascending=False).reset_index(drop=True)
        # Keep only top 3 strongest blocks of each type
        df_blocks = pd.concat([
            df_blocks[df_blocks['type'] == 'bullish'].head(3),
            df_blocks[df_blocks['type'] == 'bearish'].head(3)
        ]).reset_index(drop=True)
    
    return df_blocks



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
    
    # Remove EMA conditions and use other criteria for trend determination
    trend = "bullish" if len(identify_market_structure_breaks(df)) > 0 and identify_market_structure_breaks(df).iloc[-1]['type'] == 'Bullish BMS' else "bearish"
    
    # Factor 1: Entry near strong demand zones during uptrends
    if trend == "bullish":
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
    if trend == "bearish":
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
        'bar_up': '#2596be',       # Forest green for bullish
        'bar_down': '#010406',     # Crimson red for bearish
        'ma1': '#2962ff',          # Royal blue for EMA 20
        'ma2': '#9c27b0',          # Purple for EMA 50
        'ma3': '#ff6d00',          # Orange for EMA 200
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
    
    
    
    # Plot entry points, stop-loss, and take-profit levels with prices
    if not smart_entries.empty:
        last_timestamp = df['timestamp_mpl'].iloc[-1]
        timestamp_width = last_timestamp - df['timestamp_mpl'].iloc[-2]
        
        for _, entry in smart_entries.iterrows():
            # Entry point line and price
            ax_main.axhline(y=entry['entry_price'], color='yellow', linestyle='--', linewidth=2, alpha=0.8)
            ax_main.plot([last_timestamp - timestamp_width, last_timestamp + timestamp_width], 
                        [entry['entry_price'], entry['entry_price']], 
                        color='yellow', linewidth=2, alpha=1)
            ax_main.text(last_timestamp + timestamp_width*0.5, entry['entry_price'], 
                        f" Entry @ {entry['entry_price']:.4f}", 
                        color='yellow', fontweight='bold', va='center', ha='left',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='yellow', pad=3))

            # Stop-loss line and price
            ax_main.axhline(y=entry['stop_loss'], color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax_main.plot([last_timestamp - timestamp_width, last_timestamp + timestamp_width], 
                        [entry['stop_loss'], entry['stop_loss']], 
                        color='red', linewidth=2, alpha=1)
            ax_main.text(last_timestamp + timestamp_width*0.5, entry['stop_loss'], 
                        f" SL @ {entry['stop_loss']:.4f}", 
                        color='red', fontweight='bold', va='center', ha='left',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='red', pad=3))

            # Take-profit line and price
            ax_main.axhline(y=entry['take_profit'], color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax_main.plot([last_timestamp - timestamp_width, last_timestamp + timestamp_width], 
                        [entry['take_profit'], entry['take_profit']], 
                        color='green', linewidth=2, alpha=1)
            ax_main.text(last_timestamp + timestamp_width*0.5, entry['take_profit'], 
                        f" TP @ {entry['take_profit']:.4f}", 
                        color='green', fontweight='bold', va='center', ha='left',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='green', pad=3))

            # Add connecting lines with arrows
            ax_main.annotate('', xy=(last_timestamp, entry['stop_loss']), 
                           xytext=(last_timestamp, entry['entry_price']),
                           arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5, alpha=0.6))
            ax_main.annotate('', xy=(last_timestamp, entry['entry_price']), 
                           xytext=(last_timestamp, entry['take_profit']),
                           arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5, alpha=0.6))

            # Add Risk:Reward ratio and trade direction
            risk = abs(entry['entry_price'] - entry['stop_loss'])
            reward = abs(entry['take_profit'] - entry['entry_price'])
            rr_ratio = reward / risk
            mid_price = entry['entry_price'] + (entry['take_profit'] - entry['entry_price']) / 2
            
            # Display R:R ratio
            ax_main.text(last_timestamp + timestamp_width*0.5, mid_price,
                        f"R:R = 1:{rr_ratio:.1f}",
                        color='white', fontweight='bold', ha='left', va='center',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='yellow', pad=3))

            # Display trade direction
            direction_color = 'yellow' if entry['type'] == 'long' else 'red'
            ax_main.text(last_timestamp + timestamp_width*0.5, 
                        entry['entry_price'] - (price_range * 0.02),
                        f"{entry['type'].upper()} TRADE",
                        color=direction_color, fontweight='bold', ha='left', va='top',
                        bbox=dict(facecolor='black', alpha=0.8, 
                                edgecolor=direction_color, pad=3))

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
        alpha = 0.3 if block['mitigated'] else 0.4
        
        # Plot order block zone
        ax_main.fill_between(
            [mdates.date2num(block['timestamp']), df['timestamp_mpl'].iloc[-1]],
            [block['low'], block['low']],
            [block['high'], block['high']],
            color=color, alpha=alpha
        )
        
        # Add detailed label
        label = f"{'Bull' if block['type'] == 'bullish' else 'Bear'} OB\nVol: {block['volume_multiple']:.1f}x\nStr: {block['strength']:.1f}"
        ax_main.annotate(
            label,
            xy=(mdates.date2num(block['timestamp']), block['entry']),
            xytext=(-50, 0),
            textcoords='offset points',
            fontsize=9,
            color=color,
            bbox=dict(
                facecolor=colors['label_bg'],
                edgecolor=color,
                alpha=0.9,
                boxstyle='round,pad=0.5'
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                alpha=0.8,
                connectionstyle='arc3,rad=-0.2'
            )
        )

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
    Enhanced market analysis with better crypto-specific handling
    """
    print(f"\n--- Analyzing {symbol} ---")
    
    # Validate timeframes for crypto
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    if chart_timeframe not in valid_timeframes:
        chart_timeframe = '1h'
    if entry_timeframe not in valid_timeframes:
        entry_timeframe = '15m'
    
    # Fetch data with volume validation
    df_chart = fetch_data(symbol, chart_timeframe, limit=200)
    df_entry = fetch_data(symbol, entry_timeframe, limit=200)
    
    if df_chart is None or df_entry is None:
        print(f"Could not fetch valid data for {symbol}")
        return None
    
    # Validate volume data
    if df_chart['volume'].mean() == 0 or df_entry['volume'].mean() == 0:
        print(f"Invalid volume data for {symbol}")
        return None

    # Add crypto-specific market analysis
    df_chart = add_crypto_indicators(df_chart)
    df_entry = add_crypto_indicators(df_entry)
    
    # Add technical indicators to both timeframes
    df_chart = add_technical_indicators(df_chart)
    df_entry = add_technical_indicators(df_entry)
    
    # Determine trend using market structure instead of EMAs
    trend = "bullish" if len(identify_market_structure_breaks(df_chart)) > 0 and identify_market_structure_breaks(df_chart).iloc[-1]['type'] == 'Bullish BMS' else "bearish"
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
    
    # Convert signals to smart_entries format for plotting
    if not signals.empty:
        smart_entries = pd.DataFrame([{
            'type': signal['signal_type'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit']
        } for _, signal in signals.iterrows()])
    
    # Plot using 4h data but mark entry points from signals
    plot_chart(df_chart, liquidity_zones, supply_demand_zones, order_blocks, 
              candlestick_patterns, smart_entries, fair_value_gaps, symbol)
    
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
            'trend': 'bullish' if len(identify_market_structure_breaks(df)) > 0 and identify_market_structure_breaks(df).iloc[-1]['type'] == 'Bullish BMS' else 'bearish',
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
    print("=" * 120)
    print("=======================  Welcome To Our Trading Bot =======================================================")    
    print("   =======================  This Bot Analyze Crypto Market And Give You The Best Signals =====================")
    print("   =======================  This Bot Will Also Analyze Forex And Commodities =================================")
    print("   =======================  Only Enter The Trade After You Have Checked The Signals Thoroughly ================")
    print("   =======================  This Robot was Developed BY Lito Programmer's Team ================================")
    print("   =======================  Trade With Easy Need Help Contact Us +256-705-672-545/789-251-487 =================")
    print("   =======================  Email: joellitoprogrammer2@gmail.com ===============================================")
    print("=" * 120)
    
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
        print("\n==================================== HIGH PROBABILITY TRADE SETUPS FOUND ======================================")
        
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