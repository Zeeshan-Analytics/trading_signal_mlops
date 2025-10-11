"""
Streamlit dashboard for trading signal predictions.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config

# Page config
st.set_page_config(
    page_title="Trading Signal Generator",
    page_icon="📈",
    layout="wide"
)

# Load config
config = load_config()

# API URL (update this when deploying)
API_URL = "http://localhost:8000"

# Title
st.title("📈 Trading Signal Generator")
st.markdown("Generate trading signals using machine learning")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This dashboard uses a neural network to predict trading signals "
    "based on technical indicators."
)

st.sidebar.header("Signal Types")
st.sidebar.success("🟢 **Strong Buy** - High confidence upward movement")
st.sidebar.success("⬆️ **Buy** - Moderate upward movement")
st.sidebar.warning("➡️ **Hold** - Neutral, no clear direction")
st.sidebar.error("⬇️ **Sell** - Moderate downward movement")
st.sidebar.error("🔴 **Strong Sell** - High confidence downward movement")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "ℹ️ Model Info", "🔍 Batch Prediction"])

# Tab 1: Single Prediction
with tab1:
    st.header("Make a Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Moving Averages")
        sma_10 = st.number_input("SMA 10", value=150.0, format="%.2f")
        sma_20 = st.number_input("SMA 20", value=148.0, format="%.2f")
        sma_50 = st.number_input("SMA 50", value=145.0, format="%.2f")
        ema_12 = st.number_input("EMA 12", value=151.0, format="%.2f")
        ema_26 = st.number_input("EMA 26", value=149.0, format="%.2f")
    
    with col2:
        st.subheader("Momentum Indicators")
        rsi_14 = st.number_input("RSI 14", value=65.0, min_value=0.0, max_value=100.0, format="%.2f")
        macd = st.number_input("MACD", value=1.5, format="%.4f")
        macd_signal = st.number_input("MACD Signal", value=1.2, format="%.4f")
        macd_hist = st.number_input("MACD Histogram", value=0.3, format="%.4f")
    
    with col3:
        st.subheader("Other Indicators")
        bb_upper = st.number_input("Bollinger Upper", value=155.0, format="%.2f")
        bb_middle = st.number_input("Bollinger Middle", value=150.0, format="%.2f")
        bb_lower = st.number_input("Bollinger Lower", value=145.0, format="%.2f")
        volume_sma_20 = st.number_input("Volume SMA 20", value=1000000.0, format="%.2f")
        price_change = st.number_input("Price Change", value=0.02, format="%.4f")
    
    if st.button("🎯 Get Prediction", type="primary"):
        # Prepare data
        features = {
            "SMA_10": sma_10,
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "EMA_12": ema_12,
            "EMA_26": ema_26,
            "RSI_14": rsi_14,
            "MACD": macd,
            "MACD_signal": macd_signal,
            "MACD_hist": macd_hist,
            "BB_upper": bb_upper,
            "BB_middle": bb_middle,
            "BB_lower": bb_lower,
            "volume_sma_20": volume_sma_20,
            "price_change": price_change
        }
        
        # Make API request
        try:
            with st.spinner("Making prediction..."):
                response = requests.post(f"{API_URL}/predict", json=features)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display result
                    st.success("Prediction complete!")
                    
                    # Signal with emoji
                    signal_emoji = {
                        "strong_buy": "🟢",
                        "buy": "⬆️",
                        "hold": "➡️",
                        "sell": "⬇️",
                        "strong_sell": "🔴"
                    }
                    
                    signal = result['signal']
                    confidence = result['confidence']
                    
                    st.markdown(f"## {signal_emoji.get(signal, '❓')} Signal: **{signal.upper().replace('_', ' ')}**")
                    st.markdown(f"### Confidence: **{confidence:.1%}**")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Probabilities chart
                    st.subheader("Probability Distribution")
                    
                    probs = result['probabilities']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(probs.keys()),
                            y=list(probs.values()),
                            marker_color=['red' if 'sell' in k else 'yellow' if k == 'hold' else 'green' for k in probs.keys()]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Signal Probabilities",
                        xaxis_title="Signal",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Make sure the API is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Tab 2: Model Info
with tab2:
    st.header("Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        
        if response.status_code == 200:
            info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Name", info['model_name'])
                st.metric("Number of Features", info['num_features'])
            
            with col2:
                st.metric("Signal Classes", len(info['signal_classes']))
            
            st.subheader("Features Used")
            st.dataframe(pd.DataFrame({"Feature": info['features']}), use_container_width=True)
            
            st.subheader("Signal Classes")
            classes_df = pd.DataFrame([
                {"ID": k, "Signal": v} 
                for k, v in info['signal_classes'].items()
            ])
            st.dataframe(classes_df, use_container_width=True)
        
        else:
            st.error("Could not fetch model info")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tab 3: Batch Prediction
with tab3:
    st.header("Batch Prediction")
    st.info("Upload a CSV file with features to get predictions for multiple samples")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Run Batch Prediction"):
            # Convert DataFrame to list of dicts
            features_list = df.to_dict('records')
            
            try:
                with st.spinner("Making predictions..."):
                    response = requests.post(f"{API_URL}/batch-predict", json=features_list)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Add results to DataFrame
                        df['predicted_signal'] = [r['signal'] for r in results]
                        df['confidence'] = [r['confidence'] for r in results]
                        
                        st.success(f"Predictions completed for {len(results)} samples!")
                        
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI and Streamlit")