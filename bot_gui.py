import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
from datetime import datetime
import os
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import ccxt
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from smc import analyze_market, analyze_multi_timeframe, fetch_data, fetch_forex_data
from styles import ModernTheme, ModernWidget

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SMC Trading Bot")
        self.root.geometry("1200x800")
        self.root.configure(bg=ModernTheme.DARK)
        
        # Apply modern theme
        ModernTheme.apply_theme(root)

        # Create main containers
        self.create_header()
        self.create_symbol_frame()
        self.create_chart_frame()
        self.create_signals_frame()
        self.create_status_bar()
        
        # Add market statistics frame
        self.create_market_stats_frame()

        # Initialize variables
        self.analyzing = False
        self.current_symbol = None
        
        # Add message queue for thread communication
        self.queue = queue.Queue()
        self.root.after(100, self.check_queue)
        
        # Initialize exchange with better error handling
        self.exchange = self.initialize_exchange()
        
        # Start market data updates
        self.update_market_data()
        self.tradingview_url = None

    def initialize_exchange(self):
        """Initialize exchange with proper configuration and error handling"""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Use spot market instead of futures
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                },
                'timeout': 30000,
            })
            
            # Test connection with basic market load
            exchange.load_markets()
            return exchange
            
        except Exception as e:
            self.update_status(f"Error initializing Binance: {str(e)}")
            # Try backup exchange
            try:
                backup_exchange = ccxt.kucoin({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                backup_exchange.load_markets()
                self.update_status("Using KuCoin as backup exchange")
                return backup_exchange
            except:
                self.update_status("Failed to initialize any exchange. Using limited functionality.")
                return None

    def create_header(self):
        header = tk.Frame(self.root, bg=ModernTheme.SECONDARY, height=60)
        header.pack(fill=tk.X, padx=5, pady=5)
        header.pack_propagate(False)

        title = tk.Label(header, 
                        text="SMC Trading Bot", 
                        font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZES["header"], 'bold'),
                        bg=ModernTheme.SECONDARY,
                        fg=ModernTheme.LIGHT)
        title.pack(side=tk.LEFT, padx=20)

        # Add time label
        self.time_label = tk.Label(header,
                                 font=('Helvetica', 12),
                                 bg=ModernTheme.SECONDARY,
                                 fg=ModernTheme.LIGHT)
        self.time_label.pack(side=tk.RIGHT, padx=20)
        self.update_time()

    def create_symbol_frame(self):
        frame = ttk.LabelFrame(self.root, text="Trading Pairs", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Create symbol selection with modern styling
        symbols_frame = ttk.Frame(frame)
        symbols_frame.pack(fill=tk.X)

        # Crypto pairs - limited to BTC/USDT and XRP/USDT only
        self.crypto_var = tk.StringVar()
        crypto_pairs = ['BTC/USDT', 'XRP/USDT']
        ttk.Label(symbols_frame, text="Crypto:").pack(side=tk.LEFT, padx=5)
        crypto_combo = ttk.Combobox(symbols_frame, textvariable=self.crypto_var, values=crypto_pairs)
        crypto_combo.pack(side=tk.LEFT, padx=5)
        crypto_combo.set(crypto_pairs[0])

        # Forex pairs
        self.forex_var = tk.StringVar()
        forex_pairs = ['XAUUSD', 'GBPJPY', 'USDJPY', 'USOIL']
        ttk.Label(symbols_frame, text="Forex:").pack(side=tk.LEFT, padx=5)
        forex_combo = ttk.Combobox(symbols_frame, textvariable=self.forex_var, values=forex_pairs)
        forex_combo.pack(side=tk.LEFT, padx=5)
        forex_combo.set(forex_pairs[0])

        # Timeframe selection
        self.timeframe_var = tk.StringVar()
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        ttk.Label(symbols_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        timeframe_combo = ttk.Combobox(symbols_frame, textvariable=self.timeframe_var, values=timeframes)
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        timeframe_combo.set('1h')

        # Style the comboboxes
        combo_style = ttk.Style()
        combo_style.configure('Custom.TCombobox', 
                            background=ModernTheme.PRIMARY,
                            fieldbackground=ModernTheme.LIGHT,
                            selectbackground=ModernTheme.PRIMARY)

        # Replace the analyze button with modern styled button
        analyze_btn = ModernWidget.create_button(
            symbols_frame, 
            "Analyze",
            self.start_analysis,
            style="primary"
        )
        analyze_btn.pack(side=tk.LEFT, padx=20)

    def create_chart_frame(self):
        self.chart_frame = ttk.LabelFrame(self.root, text="Chart Analysis", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create frame for chart controls
        controls_frame = ttk.Frame(self.chart_frame)
        controls_frame.pack(fill=tk.X, pady=5)

        # Add button to open TradingView
        self.open_chart_btn = ModernWidget.create_button(
            controls_frame,
            "Open Chart in TradingView",
            self.open_tradingview,
            style="info"
        )
        self.open_chart_btn.pack(side=tk.LEFT, padx=5)

        # Create placeholder text
        placeholder = ttk.Label(
            self.chart_frame,
            text="Click 'Open Chart in TradingView' to view the chart",
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZES["large"]),
            justify=tk.CENTER
        )
        placeholder.pack(expand=True, pady=20)

    def open_tradingview(self):
        """Open TradingView chart in default browser"""
        symbol = self.crypto_var.get() if self.crypto_var.get() != '' else self.forex_var.get()
        timeframe = self.timeframe_var.get()
        
        # Convert timeframe to TradingView format
        tv_timeframe = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }.get(timeframe, '60')
        
        # Create TradingView URL
        symbol_formatted = symbol.replace('/', '')
        self.tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_formatted}&interval={tv_timeframe}"
        
        # Open in browser
        webbrowser.open(self.tradingview_url)

    def create_signals_frame(self):
        signals_frame = ttk.LabelFrame(self.root, text="Trading Signals", padding=10)
        signals_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create scrolled text widget for signals
        self.signals_text = scrolledtext.ScrolledText(signals_frame, height=10)
        self.signals_text.pack(fill=tk.X)

    def create_status_bar(self):
        self.status_bar = tk.Label(self.root,
                                 text="Ready",
                                 bd=1,
                                 relief=tk.SUNKEN,
                                 anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5)

    def create_market_stats_frame(self):
        stats_frame = ttk.LabelFrame(self.root, text="Market Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create modern styled frames
        left_frame = ttk.Frame(stats_frame, style='Custom.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        right_frame = ttk.Frame(stats_frame, style='Custom.TFrame')
        right_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        # Update widget styles with modern theme
        self.stats_widgets = {}
        self.indicator_widgets = {}
        
        # Price Stats (Left Frame)
        row = 0
        for label in ["Price", "24h Change", "24h Volume", "Market Cap"]:
            ttk.Label(left_frame, text=f"{label}:", font=('Helvetica', 10, 'bold')).grid(
                row=row, column=0, padx=5, pady=2, sticky='w'
            )
            self.stats_widgets[label] = tk.Label(
                left_frame, 
                text="Loading...",
                font=('Helvetica', 10),
                bg=ModernTheme.DARK,
                fg=ModernTheme.LIGHT
            )
            self.stats_widgets[label].grid(row=row, column=1, padx=5, pady=2, sticky='w')
            row += 1

        # Technical Indicators (Right Frame)
        row = 0
        for indicator in ["RSI", "SMA20", "BB Bands", "Trend"]:
            ttk.Label(right_frame, text=f"{indicator}:", font=('Helvetica', 10, 'bold')).grid(
                row=row, column=0, padx=5, pady=2, sticky='w'
            )
            self.indicator_widgets[indicator] = tk.Label(
                right_frame,
                text="Loading...",
                font=('Helvetica', 10),
                bg=ModernTheme.DARK,
                fg=ModernTheme.LIGHT
            )
            self.indicator_widgets[indicator].grid(row=row, column=1, padx=5, pady=2, sticky='w')
            row += 1

    def update_time(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def update_status(self, message):
        self.status_bar.config(text=message)

    def add_signal(self, message):
        self.signals_text.insert(tk.END, message + "\n")
        self.signals_text.see(tk.END)

    def start_analysis(self):
        if self.analyzing:
            return

        self.analyzing = True
        self.update_status("Analyzing...")
        self.signals_text.delete(1.0, tk.END)

        # Get selected symbol
        symbol = self.crypto_var.get() if self.crypto_var.get() != '' else self.forex_var.get()
        timeframe = self.timeframe_var.get()

        # Start analysis in a separate thread
        thread = threading.Thread(target=self.run_analysis, args=(symbol, timeframe))
        thread.daemon = True
        thread.start()

    def check_queue(self):
        """Check for messages from the analysis thread"""
        try:
            while True:
                msg = self.queue.get_nowait()
                
                if msg['type'] == 'status':
                    self.update_status(msg['data'])
                elif msg['type'] == 'signal':
                    self.add_signal(msg['data'])
                elif msg['type'] == 'figure':
                    # Update chart in main thread
                    self.update_chart(msg['data'])
                
                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.check_queue)

    def update_chart(self, fig_data):
        """Update chart in the main thread"""
        try:
            # Clear existing figure
            self.fig.clear()
            
            # Create new subplot and copy data from analysis figure
            ax = self.fig.add_subplot(111)
            for artist in fig_data.get_axes()[0].get_children():
                artist_copy = artist.copy()
                ax.add_artist(artist_copy)
            
            # Update canvas
            self.canvas.draw()
        except Exception as e:
            self.update_status(f"Error updating chart: {str(e)}")

    def run_analysis(self, symbol, timeframe):
        try:
            # Validate symbol
            allowed_symbols = ['BTC/USDT', 'XRP/USDT']
            if symbol not in allowed_symbols:
                self.queue.put({
                    'type': 'status',
                    'data': f"Error: Only {', '.join(allowed_symbols)} analysis is supported"
                })
                return

            # Fetch data and run analysis
            self.queue.put({
                'type': 'status',
                'data': f"Analyzing {symbol} on {timeframe} timeframe..."
            })

            # Run market analysis
            result = analyze_market(symbol, chart_timeframe=timeframe)
            if result and 'signals' in result and not result['signals'].empty:
                self.queue.put({
                    'type': 'signal',
                    'data': f"\n=== {symbol} Analysis ===\n{result['signals']}\n"
                })
            
            # Run multi-timeframe analysis with key timeframes
            mtf_signals = analyze_multi_timeframe(symbol, timeframes=['1d', '4h', '1h', '15m'])
            if not mtf_signals.empty:
                self.queue.put({
                    'type': 'signal',
                    'data': f"\n=== Multi-Timeframe Analysis for {symbol} ===\n{mtf_signals}\n"
                })

            self.queue.put({
                'type': 'status',
                'data': f"Analysis completed for {symbol}"
            })
            
        except Exception as e:
            self.queue.put({
                'type': 'status',
                'data': f"Error during analysis: {str(e)}"
            })
        finally:
            self.analyzing = False

    def display_signals(self, signals, prefix="Signal: "):
        for _, signal in signals.iterrows():
            message = f"\n{prefix}{signal['symbol']}\n"
            message += f"Type: {signal['signal_type'].upper()}\n"
            message += f"Entry: {signal['entry_price']:.8f}\n"
            message += f"Stop Loss: {signal['stop_loss']:.8f}\n"
            message += f"Take Profit: {signal['take_profit']:.8f}\n"
            message += f"Confidence: {signal['confidence'].upper()}\n"
            message += f"Reason: {signal['reason']}\n"
            message += "-" * 50
            self.add_signal(message)

    def cleanup(self):
        """Cleanup function to handle program exit"""
        try:
            # Clear the queue
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
            
            # Close matplotlib figure
            plt.close(self.fig)
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.root.destroy()

    def update_market_data(self):
        """Update market statistics periodically"""
        try:
            if not self.exchange:
                self.update_status("No exchange connection available")
                return

            symbol = self.crypto_var.get() if self.crypto_var.get() != '' else self.forex_var.get()
            
            # Add error handling for ticker fetch
            try:
                ticker = self.exchange.fetch_ticker(symbol)
            except Exception as e:
                self.update_status(f"Error fetching ticker: {str(e)}")
                # Reset stats to N/A
                for widget in self.stats_widgets.values():
                    widget.configure(text="N/A", bg=ModernTheme.DARK, fg=ModernTheme.LIGHT)
                for widget in self.indicator_widgets.values():
                    widget.configure(text="N/A", bg=ModernTheme.DARK, fg=ModernTheme.LIGHT)
                return

            # Rest of the update_market_data function remains the same
            # Update price stats with explicit widget updates
            self.stats_widgets["Price"].configure(
                text=f"${ticker['last']:,.2f}",
                bg=ModernTheme.DARK
            )
            
            change_color = '#2ecc71' if ticker['percentage'] > 0 else '#e74c3c'
            self.stats_widgets["24h Change"].configure(
                text=f"{ticker['percentage']:+.2f}%",
                fg=change_color,
                bg=ModernTheme.DARK
            )
            
            self.stats_widgets["24h Volume"].configure(
                text=f"${ticker['quoteVolume']:,.0f}",
                bg=ModernTheme.DARK
            )
            
            # Market cap (if available)
            if 'info' in ticker and 'marketCap' in ticker['info']:
                market_cap = f"${ticker['info']['marketCap']:,.0f}"
            else:
                market_cap = "N/A"
            self.stats_widgets["Market Cap"].configure(text=market_cap, bg=ModernTheme.DARK)
            
            # Fetch OHLCV data for indicators
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            rsi = RSIIndicator(df['close']).rsi()
            sma20 = SMAIndicator(df['close'], window=20).sma_indicator()
            bb = BollingerBands(df['close'])
            
            # Update indicator widgets with explicit configure calls
            rsi_value = rsi.iloc[-1]
            rsi_color = '#2ecc71' if rsi_value > 50 else '#e74c3c'
            self.indicator_widgets["RSI"].configure(
                text=f"{rsi_value:.1f}",
                fg=rsi_color,
                bg=ModernTheme.DARK
            )
            
            sma_color = '#2ecc71' if df['close'].iloc[-1] > sma20.iloc[-1] else '#e74c3c'
            self.indicator_widgets["SMA20"].configure(
                text=f"${sma20.iloc[-1]:.2f}",
                fg=sma_color,
                bg=ModernTheme.DARK
            )
            
            bb_position = (df['close'].iloc[-1] - bb.bollinger_lband()[-1]) / (bb.bollinger_hband()[-1] - bb.bollinger_lband()[-1])
            bb_color = '#2ecc71' if bb_position > 0.5 else '#e74c3c'
            self.indicator_widgets["BB Bands"].configure(
                text=f"{bb_position:.1%}",
                fg=bb_color,
                bg=ModernTheme.DARK
            )
            
            # Determine trend with more conditions
            trend = "Bullish" if (df['close'].iloc[-1] > sma20.iloc[-1] and 
                                rsi_value > 50 and 
                                bb_position > 0.5) else "Bearish"
            trend_color = '#2ecc71' if trend == "Bullish" else '#e74c3c'
            self.indicator_widgets["Trend"].configure(
                text=trend,
                fg=trend_color,
                bg=ModernTheme.DARK
            )

            # Force widget updates
            self.root.update_idletasks()

        except Exception as e:
            self.update_status(f"Error updating market data: {str(e)}")
        finally:
            # Schedule next update with longer interval if there were errors
            if not self.exchange:
                self.root.after(30000, self.update_market_data)  # 30 seconds if no exchange
            else:
                self.root.after(5000, self.update_market_data)   # 5 seconds normally

def main():
    root = tk.Tk()
    app = TradingBotGUI(root)
    
    # Add cleanup on window close
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Error in main loop: {e}")
        app.cleanup()
        root.destroy()

if __name__ == "__main__":
    main()
