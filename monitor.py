import streamlit as st
import pandas as pd
import time
from datetime import datetime
import threading
from queue import Queue
from smc import analyze_market, analyze_multi_timeframe, fetch_data, fetch_forex_data

class SignalMonitor:
    def __init__(self):
        self.signal_queue = Queue()
        self.running = False
        self.last_signals = {}
        
    def start_monitoring(self, symbols, timeframes):
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(symbols, timeframes),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.running = False
        
    def _monitor_loop(self, symbols, timeframes):
        while self.running:
            for symbol in symbols:
                try:
                    # Check if enough time has passed since last signal
                    current_time = datetime.now()
                    if symbol in self.last_signals:
                        time_diff = (current_time - self.last_signals[symbol]).total_seconds()
                        if time_diff < 300:  # Wait 5 minutes between checks
                            continue
                    
                    # Get multi-timeframe confluence signals
                    mtf_signals = analyze_multi_timeframe(symbol, timeframes)
                    
                    if not mtf_signals.empty:
                        for _, signal in mtf_signals.iterrows():
                            signal_data = {
                                'timestamp': current_time,
                                'symbol': symbol,
                                'type': signal['signal_type'],
                                'entry': signal['entry_price'],
                                'stop_loss': signal['stop_loss'],
                                'target': signal['take_profit'],
                                'confidence': signal['confidence'],
                                'reason': signal['reason'],
                                'timeframe': 'Multi-TF'
                            }
                            self.signal_queue.put(signal_data)
                            self.last_signals[symbol] = current_time
                            
                    # Get single timeframe signals
                    result = analyze_market(symbol, chart_timeframe='1h', entry_timeframe='15m')
                    if result and 'signals' in result and not result['signals'].empty:
                        for _, signal in result['signals'].iterrows():
                            signal_data = {
                                'timestamp': current_time,
                                'symbol': symbol,
                                'type': signal['signal_type'],
                                'entry': signal['entry_price'],
                                'stop_loss': signal['stop_loss'],
                                'target': signal['take_profit'],
                                'confidence': signal['confidence'],
                                'reason': signal['reason'],
                                'timeframe': '1H/15M'
                            }
                            self.signal_queue.put(signal_data)
                            self.last_signals[symbol] = current_time
                            
                except Exception as e:
                    print(f"Error monitoring {symbol}: {e}")
                    
            time.sleep(60)  # Check every minute
