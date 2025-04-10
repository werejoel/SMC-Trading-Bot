import streamlit as st
import smc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Welcome To Our Bot Developer Litoprogrammer", layout="wide")

def main():
    st.title("Welcome To Our Dashboard")
    st.sidebar.header("Settings")
    
    # Timeframe selection
    chart_timeframe = st.sidebar.selectbox(
        "Chart Timeframe",
        ["1h", "4h", "1d"],
        index=0
    )
    
    entry_timeframe = st.sidebar.selectbox(
        "Entry Timeframe",
        ["5m", "15m", "30m"],
        index=1
    )
    
    # Symbol selection
    symbol_type = st.sidebar.radio("Market Type", ["Crypto", "Forex"])
    
    if symbol_type == "Crypto":
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT"]
    else:
        symbols = ["XAUUSD", "GBPJPY", "USDJPY"]
    
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Real-time Chart", "Market Analysis", "Asset Monitoring"])

    with tab1:
        st.subheader("TradingView Chart")
        # Format symbol for TradingView
        if symbol_type == "Crypto":
            symbol_tv = f"BINANCE:{selected_symbol.replace('/', '').upper()}"
        else:
            # Handle forex symbols properly
            if selected_symbol == "XAUUSD":
                symbol_tv = "OANDA:XAUUSD"
            else:
                symbol_tv = f"OANDA:{selected_symbol}"
        
        # Convert timeframe to TradingView format
        tv_timeframe = chart_timeframe.replace('h', '').replace('d', 'D')
            
        tradingview_widget = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
            <div id="tradingview_chart"></div>
        </div>
        <style>
            .tradingview-widget-container {{
                height: 600px;
                width: 100%;
                background-color: #131722;
            }}
            #tradingview_chart {{
                height: 100%;
                width: 100%;
            }}
        </style>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            new TradingView.widget({{
                "autosize": true,
                "symbol": "{symbol_tv}",
                "interval": "{tv_timeframe}",
                "timezone": "exchange",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_chart",
                "studies": ["Volume@tv-basicstudies"]
            }});
        }});
        </script>
        <!-- TradingView Widget END -->
        """
        
        st.components.v1.html(tradingview_widget, height=650)
        
        # Add error handling message
        st.caption("If the chart doesn't load, please try selecting a different symbol or timeframe.")

    with tab2:
        # Add additional analysis metrics display
        if st.button("Click Button To Analyze Market"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                try:
                    # Get analyses
                    market_analysis = smc.analyze_market(selected_symbol, 
                        chart_timeframe=chart_timeframe,
                        entry_timeframe=entry_timeframe)
                    
                    mtf_signals = smc.analyze_multi_timeframe(selected_symbol, 
                        timeframes=['1d', '4h', '1h', '15m'])
                    
                    # Create three columns for better organization
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.subheader(" Current Market Chart & Analysis From the bot")
                        if market_analysis and 'error' not in market_analysis:
                            # Display chart
                            if 'fig' in market_analysis:
                                st.pyplot(market_analysis['fig'])
                            
                            # Display market structure
                            st.write("Market Structure/Bias:")
                            st.write(f"- Trend: {market_analysis.get('trend', 'N/A')}")
                            
                            # Safely access data quality information
                            if 'data_quality' in market_analysis:
                                st.write(f"- Data Points Analyzed: {market_analysis['data_quality'].get('chart_points', 'N/A')}")
                            else:
                                st.write("- Data Points Analyzed: N/A")
                    
                    with col2:
                        st.subheader("Technical Levels")
                        if market_analysis and 'analysis' in market_analysis:
                            analysis = market_analysis['analysis']
                            
                            # Display liquidity zones
                            if not analysis['liquidity_zones'].empty:
                                st.write("Key Liquidity Zones:")
                                st.dataframe(analysis['liquidity_zones'][['type', 'level', 'strength']])
                            
                            # Display order blocks
                            if not analysis['order_blocks'].empty:
                                st.write("Active Order Blocks:")
                                st.dataframe(analysis['order_blocks'][['type', 'high', 'low', 'strength']])
                    
                    with col3:
                        st.subheader("Trading Signals")
                        # Debug information
                        #st.write("Debug Info:")
                        #st.write(f"MTF Signals Type: {type(mtf_signals)}")
                        #st.write(f"Market Analysis Keys: {market_analysis.keys() if market_analysis else 'None'}")
                        
                        # Display MTF signals with proper checks
                        if isinstance(mtf_signals, pd.DataFrame) and not mtf_signals.empty:
                            st.success("High-Probability Setups")
                            for _, signal in mtf_signals.iterrows():
                                display_signal_card(signal)
                        else:
                            st.info("No multi-timeframe signals found")
                        
                        # Display regular signals with proper checks
                        if (market_analysis and 
                            'signals' in market_analysis and 
                            isinstance(market_analysis['signals'], pd.DataFrame) and 
                            not market_analysis['signals'].empty):
                            st.info("Additional Opportunities")
                            for _, signal in market_analysis['signals'].iterrows():
                                display_signal_card(signal)
                        else:
                            st.info("No additional signals found")
                
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
                    st.exception(e)

    with tab3:
        st.subheader("Real-time Asset Monitoring")
        # Create columns for monitoring metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        try:
            # Get real-time market data from smc module
            market_data = smc.get_market_data(selected_symbol)
            
            with m_col1:
                st.metric("Current Price", 
                         f"${market_data['price']:.2f}", 
                         f"{market_data['price_change_24h']:.2f}%")
            
            with m_col2:
                st.metric("24h Volume", 
                         f"${market_data['volume_24h']:,.0f}", 
                         f"{market_data['volume_change_24h']:.2f}%")
            
            with m_col3:
                st.metric("24h High", 
                         f"${market_data['high_24h']:.2f}")
            
            with m_col4:
                st.metric("24h Low", 
                         f"${market_data['low_24h']:.2f}")

            # Add price chart
            price_data = pd.DataFrame(market_data['price_history'])
            fig = go.Figure(data=[go.Candlestick(x=price_data['timestamp'],
                                               open=price_data['open'],
                                               high=price_data['high'],
                                               low=price_data['low'],
                                               close=price_data['close'])])
            fig.update_layout(title=f"{selected_symbol} Price Movement",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching real-time data: {str(e)}")

    # Display disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("""
        **Disclaimer!**: This bot provides trading signals To Use it please contact Developer litoprogrammer256@gmail.com.
        Always conduct your own analysis  to confirm alignment with the bot Thank you for your support(0705672545/789251487).
    """)

def display_signal_card(signal):
    """Helper function to display signal information in a card format"""
    try:
        with st.container():
            # Format numeric values with proper type checking
            def format_value(value, precision=4):
                if isinstance(value, (int, float)):
                    return f"{value:.{precision}f}"
                return str(value)
            
            st.markdown(f"""
            **{signal.get('signal_type', 'UNKNOWN').upper()} SIGNAL**
            - Entry: {format_value(signal.get('entry_price', 'N/A'))}
            - Stop Loss: {format_value(signal.get('stop_loss', 'N/A'))}
            - Take Profit: {format_value(signal.get('take_profit', 'N/A'))}
            - R:R Ratio: 1:{format_value(signal.get('risk_reward_ratio', 'N/A'), 2)}
            - Confidence: {signal.get('confidence', 'N/A').upper()}
            """)
            st.markdown("---")
    except Exception as e:
        st.error(f"Error displaying signal: {str(e)}")
        st.write("Raw signal data:", signal)

if __name__ == "__main__":
    main()
