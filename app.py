import streamlit as st
import smc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Welcome To Our Bot Dashboard", layout="wide")

def main():
    st.title("Trading Bot Dashboard")
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
        symbols = ["XAUUSD", "GBPJPY", "USDJPY", "USOIL"]
    
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)
    
    # Add additional analysis metrics display
    if st.sidebar.button("Analyze Market"):
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
                    st.subheader("Market Chart & Analysis")
                    if market_analysis and 'error' not in market_analysis:
                        # Display chart
                        if 'fig' in market_analysis:
                            st.pyplot(market_analysis['fig'])
                        
                        # Display market structure
                        st.write("Market Structure:")
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
                    st.write("Debug Info:")
                    st.write(f"MTF Signals Type: {type(mtf_signals)}")
                    st.write(f"Market Analysis Keys: {market_analysis.keys() if market_analysis else 'None'}")
                    
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
    
    # Display disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("""
        **Disclaimer**: This bot provides trading signals for educational purposes only.
        Always conduct your own analysis and trade at your own risk.
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
