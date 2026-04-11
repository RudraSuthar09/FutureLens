import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

# --- Configuration ---
st.set_page_config(page_title="FutureLens", layout="wide")
API_URL = "http://localhost:8000"

st.title("FutureLens")
st.markdown("*FutureLens doesn't just predict the future — it tells you how to change it.*")

# --- Session State ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = "demo"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# --- Helpers ---
def load_data(file_bytes=None, filename="data.csv"):
    """
    Calls the backend /upload endpoint and stores the result in session state.
    """
    with st.spinner("Analyzing data and generating forecast..."):
        try:
            if file_bytes:
                files = {"file": (filename, file_bytes, "text/csv")}
            else:
                st.error("No file provided.")
                return

            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                st.session_state.data = response.json()
                st.session_state.session_id = st.session_state.data.get("session_id", "demo")
                st.success("Analysis complete!")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}. Ensure the FastAPI backend is running on {API_URL}.")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Run Analysis"):
            load_data(uploaded_file.getvalue(), uploaded_file.name)
            
    st.markdown("---")
    if st.button("🕹️ Demo Mode", use_container_width=True):
        st.session_state.demo_mode = True
        import os, sys
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_data.csv")
        # Generate sample data if it doesn't exist
        if not os.path.exists(sample_path):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import generate_data
            generate_data.create_sample_data()
        try:
            with open(sample_path, "rb") as f:
                load_data(f.read(), "sample_data.csv")
            # Prefill chat message for demo
            st.session_state.chat_history.append({"role": "user", "content": "What happened in week 67?"})
            # Mock Gemini reply
            st.session_state.chat_history.append({
                "role": "model",
                "content": "In Demo Mode, I detected an anomaly at week 67. The primary driver is a drop in ad_spend which contributed heavily to the sales decline."
            })
        except FileNotFoundError:
            st.error("Sample data not found. Please run generate_data.py first.")

# --- Main App ---
if st.session_state.data:
    data = st.session_state.data

    # Extract shared variables once so all tabs can use them
    historical_dates = data.get("historical_dates", [])
    historical = data.get("historical", [])
    future_dates = data.get("dates", [])
    forecast = data.get("forecast", [])
    lower = data.get("lower", [])
    upper = data.get("upper", [])
    anomalies = data.get("anomalies", [])
    truth_score = data.get("truth_score", 0.0)

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Root Cause", "Scenario", "Chat"])
    
    # === TAB 1: Forecast ===
    with tab1:
        st.subheader("Predictive Forecast Analysis")

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(x=historical_dates, y=historical, line=dict(color='blue'), mode='lines', name='Historical'))
        
        # Forecast Band (Lower to Upper)
        fig.add_trace(go.Scatter(
            name='Lower Bound',
            x=future_dates,
            y=lower,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            name='Forecast Band',
            x=future_dates,
            y=upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=True
        ))
        
        # Forecast likely line
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, line=dict(color='orange'), mode='lines', name='Likely Forecast'))
        
        # Anomalies
        if anomalies:
            anom_dates = [a['date'] for a in anomalies]
            anom_values = [a['actual'] for a in anomalies]
            fig.add_trace(go.Scatter(
                x=anom_dates, y=anom_values, mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Anomalies'
            ))

        st.plotly_chart(fig, use_container_width=True)
        
        # Truth Meter
        st.subheader("Truth Meter")
        
        # Color coding: > 20% better is green, otherwise red/orange
        color = "red"
        if truth_score > 10: color = "green"
        elif truth_score > 0: color = "orange"
        
        # Normalize cap to 100 for st.progress
        st.progress(min(int(truth_score), 100))
        st.markdown(f"**Model is {truth_score:.1f}% better than baseline**")

    # === TAB 2: Root Cause ===
    with tab2:
        st.subheader("Root Cause Analysis (SHAP)")
        shap_res = data.get("shap_results", [])
        if shap_res:
            features = [s["feature"] for s in shap_res]
            importances = [s["importance"] for s in shap_res]
            
            fig_shap = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='indigo'
            ))
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.info(data.get("rca_explanation", "No explanation provided."))
            
        else:
            st.write("No SHAP results available for this dataset.")

    # === TAB 3: Scenario ===
    with tab3:
        st.subheader("Iterative Scenario Simulation")
        change_pct = st.slider("Change input by X%", min_value=-50, max_value=50, value=0)
        
        if st.button("Run Scenario"):
            st.write("Comparing Baseline Forecast against Scenario Adjustment...")
            scenario_forecast = [val * (1 + (change_pct / 100.0)) for val in forecast]
            
            fig_scenario = go.Figure()
            fig_scenario.add_trace(go.Scatter(x=future_dates, y=forecast, line=dict(dash='dash', color='orange'), name='Baseline'))
            fig_scenario.add_trace(go.Scatter(x=future_dates, y=scenario_forecast, line=dict(color='green'), name=f'Scenario ({change_pct}%)'))
            st.plotly_chart(fig_scenario, use_container_width=True)

    # === TAB 4: Chat ===
    with tab4:
        st.subheader("Conversational Insights")
        
        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_input = st.chat_input("Ask about the forecast, anomalies, or actions...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.chat_message("model"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(f"{API_URL}/chat", json={
                            "message": user_input,
                            "session_context": st.session_state.chat_history,
                            "session_id": st.session_state.session_id
                        })
                        if resp.status_code == 200:
                            bot_reply = resp.json().get("response", "No response.")
                        else:
                            bot_reply = f"API Error {resp.status_code}: {resp.text}"
                    except Exception as e:
                        bot_reply = f"Error communicating with backend: {e}"
                        
                st.write(bot_reply)
                st.session_state.chat_history.append({"role": "model", "content": bot_reply})
else:
    st.info("Please upload a CSV or click the Demo Mode button in the sidebar to begin.")
