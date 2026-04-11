import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(page_title="FutureLens AI - Powered by NatWest", layout="wide", page_icon="🟣")
API_URL = "http://localhost:8000"

st.markdown("""
<style>
    /* ------------------------------------- */
    /* ENHANCED NATWEST THEME                */
    /* ------------------------------------- */
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #5A287D !important;
        border-right: none;
    }
    [data-testid="stSidebar"] p:not(.stButton p), 
    [data-testid="stSidebar"] span:not(.stButton span), 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
    }
    
    /* SIDEBAR BUTTONS FIX (Demo Button Text Visibility) */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 10px;
        transition: all 0.3s ease;
        padding: 5px 15px;
    }
    [data-testid="stSidebar"] .stButton>button p,
    [data-testid="stSidebar"] .stButton>button div {
        color: #5A287D !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #F8F4FA !important;
        border: 2px solid #00A859 !important; /* Green Accent on hover */
    }
    [data-testid="stSidebar"] .stButton>button:hover p,
    [data-testid="stSidebar"] .stButton>button:hover div {
        color: #00A859 !important; /* Green Accent text */
    }
    
    /* FILE UPLOADER */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px dashed rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 4px 15px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button div {
        color: #5A287D !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background-color: #5A287D !important;
        border: 2px solid #FFFFFF !important;
        transform: scale(1.06) !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover div {
        color: #FFFFFF !important;
    }
    
    /* MAIN AREA BUTTONS */
    .stMain .stButton>button {
        color: #FFFFFF !important;
        background-color: #5A287D !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(90, 40, 125, 0.2);
    }
    .stMain .stButton>button:hover {
        background-color: #A20067 !important;
        box-shadow: 0 6px 12px rgba(162, 0, 103, 0.3);
    }
    .stMain h1, .stMain h2, .stMain h3 {
        color: #5A287D !important;
    }
    
    /* HEADER (Larger, more premium) */
    .header-container {
        display: flex;
        align-items: center;
        padding-bottom: 20px;
        border-bottom: 3px solid #5A287D;
        margin-bottom: 20px;
    }
    .header-title {
        color: #5A287D !important;
        font-size: 4.2rem !important;
        font-weight: 900 !important;
        letter-spacing: -2px !important;
        margin-left: 15px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        line-height: 1.1 !important;
    }
    .sub-header {
        color: #555 !important;
        font-size: 1.3rem !important;
        margin-bottom: 35px !important;
        font-weight: 500 !important;
    }
    
    /* CUSTOM SPINNER / LOADER STYLING */
    .stSpinner > div > div {
        border-top-color: #A20067 !important;
        border-right-color: #5A287D !important;
        border-bottom-color: #A20067 !important;
        border-left-color: transparent !important;
    }
    .stSpinner p {
        color: #5A287D !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }

    /* TABS (Navbar / Tabs Interface) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F4FA;
        border-radius: 30px;
        padding: 5px 25px;
        border: 1px solid #EBE2F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #5A287D !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border: 1px solid #A20067;
        background-color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5A287D !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(90, 40, 125, 0.3) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    
    /* CHAT INTERFACE (Purple and White) */
    [data-testid="stChatMessage"] {
        background-color: #F8F4FA;
        border-radius: 15px;
        padding: 15px 20px;
        border-left: 6px solid #5A287D;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    /* Style Chat Avatars */
    [data-testid="stChatMessageAvatarUser"] { background-color: #00A859 !important; color: white !important; } /* Using nice green accent here */
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #5A287D !important; color: white !important;}
    
    /* Chat Input */
    [data-testid="stChatInput"] {
        border: 2px solid #5A287D !important;
        border-radius: 15px !important;
    }
    img[src*="natwest_logo.jpg"] {
        width: 290px !important;  /* change this value to resize the logo */
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 12], vertical_alignment="center")
with col1:
    st.image("assets/natwest_logo.jpg", width=160)
with col2:
    st.markdown('<div class="header-title">FutureLens AI</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Powered by NatWest — FutureLens doesn\'t just predict the future, it tells you how to change it.</div>', unsafe_allow_html=True)

# --- Session State ---
if "data" not in st.session_state:
    st.session_state.data = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "demo"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# --- Helpers ---
def load_data(file_bytes=None, filename="data.csv"):
    """Calls the backend /upload endpoint and stores the result in session state."""
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
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        if st.button("Run Analysis"):
            load_data(uploaded_file.getvalue(), uploaded_file.name)

    st.markdown("---")
    if st.button("🕹️ Demo Mode", width='stretch'):
        st.session_state.demo_mode = True
        import os, sys
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_data.csv")
        if not os.path.exists(sample_path):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import generate_data
            generate_data.create_sample_data()
        try:
            with open(sample_path, "rb") as f:
                load_data(f.read(), "sample_data.csv")
            st.session_state.chat_history.append({"role": "user", "content": "What happened in week 67?"})
            st.session_state.chat_history.append(
                {
                    "role": "model",
                    "content": "In Demo Mode, I detected an anomaly at week 67. The primary driver is a drop in ad_spend which contributed heavily to the sales decline.",
                }
            )
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
    truth_meter = data.get("truth_meter", {})
    detected_freq = data.get("detected_freq", "W")
    forecast_horizon = data.get("forecast_horizon", len(future_dates))
    confidence_level = data.get("confidence_level", 90)
    data_quality = data.get("data_quality", {})
    dq_warning = data_quality.get("warning", "")
    detected_cols = data.get("detected_columns", {})
    target_col_name = detected_cols.get("target", "target")
    date_col_name = detected_cols.get("date", "date")
    feature_cols = detected_cols.get("features", [])
    dataset_profile = data.get("dataset_profile", {})
    group_forecasts = data.get("group_forecasts", None)

    # Show detected-columns info box right after upload
    st.info(
        f"📊 Forecasting **'{target_col_name}'** using **'{date_col_name}'** as the date. "
        f"Detected frequency: **{detected_freq}**. "
        f"{len(feature_cols)} additional numeric feature column(s) found."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Root Cause", "Scenario", "Chat"])

    # === TAB 1: Forecast ===
    with tab1:
        st.subheader("Predictive Forecast Analysis")

        # Convert dates for plotting
        historical_dates_dt = pd.to_datetime(historical_dates, errors="coerce") if historical_dates else []
        future_dates_dt = pd.to_datetime(future_dates, errors="coerce") if future_dates else []

        # --- Validate dates before plotting ---
        if future_dates_dt is not None and len(future_dates_dt) > 0:
            min_year = pd.to_datetime(future_dates_dt).min().year
            if min_year < 1980:
                st.error(
                    "⚠️ Date parsing error detected — dates appear to be from 1970. "
                    "Please check your CSV date column format."
                )
                st.stop()

        fig = go.Figure()

        # Historical line
        fig.add_trace(
            go.Scatter(
                x=historical_dates_dt if len(historical_dates_dt) else historical_dates,
                y=historical,
                line=dict(color="royalblue", width=2),
                mode="lines",
                name="Historical",
            )
        )

        # Confidence band (future only)
        if lower and upper and future_dates:
            fig.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=future_dates_dt if len(future_dates_dt) else future_dates,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name=f"{confidence_level}% Confidence Band",
                    x=future_dates_dt if len(future_dates_dt) else future_dates,
                    y=upper,
                    mode="lines",
                    marker=dict(color="#A20067"),
                    line=dict(width=0),
                    fillcolor="rgba(162,0,103,0.15)",
                    fill="tonexty",
                    showlegend=True,
                )
            )

        # Forecast central line (future only)
        if forecast and future_dates:
            fig.add_trace(
                go.Scatter(
                    x=future_dates_dt if len(future_dates_dt) else future_dates,
                    y=forecast,
                    line=dict(color="orange", width=2, dash="dash"),
                    mode="lines",
                    name="Forecast",
                )
            )

        # Anomaly markers
        if anomalies:
            anom_dates_raw = [a.get("date") for a in anomalies]
            anom_values = [a.get("actual") for a in anomalies]
            anom_dates_dt = pd.to_datetime(anom_dates_raw, errors="coerce")

            fig.add_trace(
                go.Scatter(
                    x=anom_dates_dt,
                    y=anom_values,
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="x"),
                    name="Anomalies",
                )
            )

        # Vertical dashed line at forecast start (safe for datetime/string x)
        if historical_dates_dt is not None and len(historical_dates_dt) > 0 and pd.notna(historical_dates_dt[-1]):
            vline_x = historical_dates_dt[-1]
            if isinstance(vline_x, pd.Timestamp):
                vline_x = vline_x.to_pydatetime()

            fig.add_shape(
                type="line",
                x0=vline_x,
                x1=vline_x,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="gray", dash="dash"),
            )
            fig.add_annotation(
                x=vline_x,
                y=1,
                xref="x",
                yref="paper",
                text="Forecast starts here",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                yshift=10,
                font=dict(color="gray"),
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=target_col_name,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Forecast summary box ---
        if forecast and historical:
            last_hist = float(historical[-1])
            last_fc = float(forecast[-1])
            change_pct = (last_fc - last_hist) / abs(last_hist) * 100 if last_hist != 0 else 0.0
            lower_pct = ((float(lower[-1]) - last_hist) / abs(last_hist) * 100) if lower and last_hist != 0 else 0.0
            upper_pct = ((float(upper[-1]) - last_hist) / abs(last_hist) * 100) if upper and last_hist != 0 else 0.0
            direction_word = "up" if change_pct >= 0 else "down"
            summary_lines = [
                f"📈 Next **{forecast_horizon}** {detected_freq} period(s): central estimate **{direction_word} {abs(change_pct):.1f}%** from current level.",
                f"Lower bound: **{lower_pct:+.1f}%** · Upper bound: **{upper_pct:+.1f}%**.",
            ]
            if dq_warning:
                summary_lines.append(f"⚠️ {dq_warning}")
            st.info("\n\n".join(summary_lines))

        # Group forecast ranking (if available)
        if group_forecasts:
            st.subheader("Group Growth Outlook")
            st.caption("Forecasted expected change by top groups (lightweight group forecast).")
            st.dataframe(group_forecasts, use_container_width=True)

        # Dataset overview
        if dataset_profile:
            with st.expander("Dataset overview (detected columns)"):
                st.write("Columns:", dataset_profile.get("columns", []))
                st.write("Categorical columns (top values):")
                st.json(dataset_profile.get("categorical_summary", {}))
                st.write("Numeric columns summary:")
                st.json(dataset_profile.get("numeric_summary", {}))

        # Truth Meter
        st.subheader("Truth Meter")
        msg = truth_meter.get("message", f"Truth score: {truth_score:.1f}% vs baseline")
        color = truth_meter.get("color", "red")

        # For progress bar: map score to 0..100 (keep display message separate)
        score_for_bar = int(min(max(float(truth_meter.get("score", truth_score)) + 50.0, 0.0), 100.0))
        st.progress(score_for_bar)
        st.markdown(f"**:{color}[{msg}]**")

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
                marker_color='#5A287D'
            ))
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)

            st.info(data.get("rca_explanation", "No explanation provided."))
        else:
            st.write("No SHAP results available for this dataset.")

    # === TAB 3: Scenario ===
    with tab3:
        st.subheader("Iterative Scenario Simulation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Growth Scenario")
            growth_pct = st.slider(
                "Adjust growth rate", min_value=-50, max_value=50, value=10, key="growth_slider"
            )
            run_growth = st.button("Run Growth Scenario", key="btn_growth")

        with col2:
            st.markdown("#### 🔄 Trend Scenario")
            trend_type = st.radio(
                "Apply trend",
                options=["flat", "recent_trend", "remove_outliers"],
                format_func=lambda x: {
                    "flat": "Apply flat trend",
                    "recent_trend": "Keep recent trend",
                    "remove_outliers": "Remove outliers",
                }[x],
                key="trend_radio",
            )
            run_trend = st.button("Run Trend Scenario", key="btn_trend")

        def _call_simulate(change_percent: float, scenario_type: str):
            try:
                resp = requests.post(
                    f"{API_URL}/simulate",
                    json={
                        "session_id": st.session_state.session_id,
                        "change_percent": change_percent,
                        "scenario_type": scenario_type,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
                else:
                    st.error(f"Simulation error {resp.status_code}: {resp.text}")
                    return None
            except Exception as e:
                st.error(f"Request failed: {e}")
                return None

        def _render_scenario(sim_data: dict, label: str):
            if not sim_data:
                return
            sc_dates = sim_data.get("dates", future_dates)
            baseline_vals = sim_data.get("baseline", forecast)
            scenario_vals = sim_data.get("scenario", [])
            summary_text = sim_data.get("summary", "")

            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sc_dates, errors="coerce"),
                    y=baseline_vals,
                    line=dict(color="#5A287D", dash="dash"),
                    name="Baseline",
                )
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sc_dates, errors="coerce"),
                    y=scenario_vals,
                    line=dict(color="#A20067", width=3),
                    name=label,
                )
            )
            fig_sc.update_layout(xaxis_title="Date", yaxis_title=target_col_name, hovermode="x unified")
            st.plotly_chart(fig_sc, use_container_width=True)

            if summary_text:
                st.info(f"📊 {summary_text}")

        if run_growth:
            with st.spinner("Running growth scenario..."):
                result = _call_simulate(growth_pct, "growth")
            _render_scenario(result, f"Growth ({growth_pct:+d}%)")

        if run_trend:
            with st.spinner(f"Running {trend_type} scenario..."):
                result = _call_simulate(0.0, trend_type)
            _render_scenario(result, trend_type.replace("_", " ").title())

    # === TAB 4: Chat ===
    with tab4:
        st.subheader("Conversational Insights")

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
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "message": user_input,
                                "session_context": st.session_state.chat_history,
                                "session_id": st.session_state.session_id,
                            },
                        )
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