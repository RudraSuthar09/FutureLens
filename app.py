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
        border-top-color: #39FF14 !important;
        border-right-color: #00A859 !important;
        border-bottom-color: #39FF14 !important;
        border-left-color: transparent !important;
    }
    .stSpinner p {
        color: #00A859 !important; /* NatWest Green */
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        text-shadow: none !important;
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
        border: 1px solid #00A859;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 168, 89, 0.2);
        transform: translateY(-2px);
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

    /* CUSTOM INFO & SUCCESS BOXES */
    .sidebar-success {
        background-color: #00A859 !important;
        color: #FFFFFF !important;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid #FFFFFF;
    }
    .custom-info-box {
        background-color: #E6F4EA !important; /* Soft green */
        border-left: 5px solid #00A859 !important; /* NatWest green accent */
        padding: 15px !important;
        border-radius: 8px !important;
        color: #004D27 !important; /* Dark green text */
        margin-bottom: 20px !important;
        font-weight: bold !important;
        font-size: 1.05rem !important;
    }

    /* TRUTH METER / PROGRESS BAR */
    [data-testid="stProgressBar"] > div > div {
        background-color: #39FF14 !important; /* Neon Green */
    }
    
    /* TABLE STYLING */
    [data-testid="stTable"] {
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stTable"] table {
        border: 2px solid #00A859 !important;
    }
    [data-testid="stTable"] th {
        background-color: #00A859 !important;
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
    }
    }

    /* GRAPH CONTAINER (UI Card) */
    [data-testid="stPlotlyChart"], .stPlotlyChart {
        border: 2px solid #EBE2F0 !important;
        border-radius: 10px !important;
        padding: 5px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05) !important;
    }

    /* CHAT SPINNER (Subtle) */
    [data-testid="stChatMessage"] .stSpinner > div > div {
        border-top-color: #5A287D !important;
        border-right-color: #EBE2F0 !important;
        border-bottom-color: #5A287D !important;
        border-left-color: transparent !important;
    }
    [data-testid="stChatMessage"] .stSpinner p {
        color: #5A287D !important; 
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        text-shadow: none !important;
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
if "forecast_update" not in st.session_state:
    st.session_state.forecast_update = None
if "forecast_notification" not in st.session_state:
    st.session_state.forecast_notification = None
 


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
                st.markdown('<div class="sidebar-success">✅ Analysis complete!</div>', unsafe_allow_html=True)
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
            st.session_state.chat_history = [
                {"role": "user",
                 "content": "What will sales look like "
                            "next few weeks?"},
                {"role": "assistant",
                 "content": "Next 4 weeks: central estimate "
                            "+6.2% growth. Lower: -2.1%. "
                            "Upper: +12.4%. Seasonal spike "
                            "expected in Week 3. "
                            "Top driver: ad spend has a "
                            "positive impact."},
                {"role": "user",
                 "content": "Are there any unusual changes?"},
                {"role": "assistant",
                 "content": "1 unusual point detected. "
                            "Sales dropped 28% on Week 67, "
                            "exceeding the forecast band. "
                            "Likely driver: reduced ad spend. "
                            "Suggested action: increase "
                            "ad spend by 8%."}
            ]
        except FileNotFoundError:
            st.error("Sample data not found. Please run generate_data.py first.")

msg_count = len(st.session_state.get("chat_history", []))
user_msgs = msg_count // 2  # user + assistant pairs
st.sidebar.markdown("---")
st.sidebar.caption(
    f"💬 {user_msgs} questions asked this session."
)

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
    st.markdown(
        f"""
        <div class="custom-info-box">
            📊 Forecasting <strong>'{target_col_name}'</strong> using <strong>'{date_col_name}'</strong> as the date.<br/>
            Detected frequency: <strong>{detected_freq}</strong>.<br/>
            {len(feature_cols)} additional numeric feature column(s) found.
        </div>
        """,
        unsafe_allow_html=True
    )
    forecast_query = st.text_input(
    label        = "🔍 Forecast query",
    placeholder  = "e.g. 'next 3 weeks', 'West region forecast', 'show me next month'",
    key          = "forecast_query_input",
    label_visibility = "collapsed",
    )
 
    col_fq1, col_fq2 = st.columns([5, 1])
    with col_fq1:
        st.caption("Type a forecast question above — the chart below updates automatically.")
    with col_fq2:
        run_fq = st.button("Update chart", key="btn_forecast_query")
    
    if run_fq and forecast_query.strip() and st.session_state.data:
        try:
            resp = requests.post(
                f"{API_URL}/forecast_query",
                json={
                    "message":    forecast_query.strip(),
                    "session_id": st.session_state.session_id,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                fq_data = resp.json()
                fu = fq_data.get("forecast_update")
                if fu:
                    st.session_state.forecast_update       = fu
                    st.session_state.forecast_notification = fu.get("label", "Forecast updated")
                else:
                    st.session_state.forecast_notification = "No forecast update — try a more specific query."
            else:
                st.session_state.forecast_notification = "Could not parse query."
        except Exception as e:
            st.session_state.forecast_notification = f"Error: {e}"
        st.rerun()
    
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Root Cause", "Scenario", "Chat"])

    # === TAB 1: Forecast ===
    with tab1:
        fu = st.session_state.get("forecast_update")
        notif = st.session_state.get("forecast_notification")
        if notif:
            st.success(f"📊 Forecast chart updated — {notif}")
            # Clear after showing once
            st.session_state.forecast_notification = None
    
        st.subheader("Predictive Forecast Analysis")
        # ── Determine which data to plot ─────────────────────────────────────────
        # Defaults: use whatever was returned at upload time
        plot_forecast = forecast       # list from data
        plot_lower    = lower
        plot_upper    = upper
        plot_dates    = future_dates
        chart_subtitle = None
    
        if fu:
            fu_type = fu.get("type")
    
            if fu_type == "horizon":
                # Slice the forecast to the requested number of periods
                n_periods = int(fu.get("periods", len(forecast)))
                plot_forecast = forecast[:n_periods]
                plot_lower    = lower[:n_periods]
                plot_upper    = upper[:n_periods]
                plot_dates    = future_dates[:n_periods]
                chart_subtitle = f"Showing: {fu.get('label', '')}"
    
            elif fu_type == "group":
                # Find matching group in group_forecasts and highlight it
                kw = fu.get("keyword", "").lower()
                chart_subtitle = f"Highlighted group: {fu.get('label', kw.title())}"
                # The main chart stays the same; we add a callout below for the group
    
            elif fu_type == "default":
                chart_subtitle = fu.get("label", "")
    
        if chart_subtitle:
            st.caption(f"🔎 {chart_subtitle}")
    
        # ── Plot (uses plot_forecast / plot_lower / plot_upper / plot_dates) ─────
        # [keep your existing fig = go.Figure() block here, but replace:
        #    forecast  → plot_forecast
        #    lower     → plot_lower
        #    upper     → plot_upper
        #    future_dates → plot_dates
        #  in all the fig.add_trace() calls]
    
        # ── Group highlight callout (shown below chart when type=group) ───────────
        if fu and fu.get("type") == "group" and group_forecasts:
            kw = fu.get("keyword", "").lower()
            matched = [
                g for g in group_forecasts
                if kw in str(g.get("group", "")).lower()
            ]
            if matched:
                g = matched[0]
                change = g.get("expected_change_percent", 0)
                direction = "up" if change >= 0 else "down"
                color = "green" if change >= 0 else "red"
                st.markdown(
                    f"""
                    <div style="
                        background-color: #E6F4EA;
                        border-left: 5px solid #00A859;
                        padding: 12px 16px;
                        border-radius: 8px;
                        margin-top: 12px;
                    ">
                    <strong>📍 {g.get('group_col','Group')}: {g.get('group')}</strong><br/>
                    Current: <strong>{g.get('last_hist', 0):,.0f}</strong> &nbsp;→&nbsp;
                    Forecast: <strong>{g.get('last_forecast', 0):,.0f}</strong>
                    &nbsp;(<span style="color:{'green' if change>=0 else 'red'}">
                    {change:+.1f}%</span> expected change)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No group matching '{kw}' found in this dataset.")
    
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showline=True, linewidth=1, linecolor='black',
                showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                tickfont=dict(size=14, color='black', family="Arial, sans-serif")
            ),
            yaxis=dict(
                showline=True, linewidth=1, linecolor='black',
                showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                tickfont=dict(size=14, color='black', family="Arial, sans-serif")
            )
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
                f"📈 Next <strong>{forecast_horizon}</strong> {detected_freq} period(s): central estimate <strong>{direction_word} {abs(change_pct):.1f}%</strong> from current level.",
                f"Lower bound: <strong>{lower_pct:+.1f}%</strong> · Upper bound: <strong>{upper_pct:+.1f}%</strong>.",
            ]
            if dq_warning:
                summary_lines.append(f"⚠️ {dq_warning}")
            summary_html = "<br/><br/>".join(summary_lines)
            st.markdown(f'<div class="custom-info-box">{summary_html}</div>', unsafe_allow_html=True)

        # Group forecast ranking (if available)
        if group_forecasts:
            st.subheader("Group Growth Outlook")
            st.caption("Forecasted expected change by top groups (lightweight group forecast).")
            st.table(group_forecasts)

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
            fig_shap.update_layout(
                yaxis=dict(
                    categoryorder='total ascending',
                    showline=True, linewidth=1, linecolor='black',
                    title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=14, color='black', family="Arial, sans-serif")
                ),
                xaxis=dict(
                    showline=True, linewidth=1, linecolor='black',
                    showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                    title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=14, color='black', family="Arial, sans-serif")
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown(f'<div class="custom-info-box">{data.get("rca_explanation", "No explanation provided.")}</div>', unsafe_allow_html=True)
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
            fig_sc.update_layout(
                xaxis_title="Date",
                yaxis_title=target_col_name,
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showline=True, linewidth=1, linecolor='black',
                    showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                    title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=14, color='black', family="Arial, sans-serif")
                ),
                yaxis=dict(
                    showline=True, linewidth=1, linecolor='black',
                    showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
                    title_font=dict(size=16, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=14, color='black', family="Arial, sans-serif")
                )
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            if summary_text:
                st.markdown(f'<div class="custom-info-box">📊 {summary_text}</div>', unsafe_allow_html=True)

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

        # Auto-send pending questions (from suggested question buttons)
        if "pending_question" in st.session_state:
            auto_q = st.session_state.pop("pending_question")
            st.session_state.chat_history.append({
                "role": "user",
                "content": auto_q
            })
            auto_payload = {
                "message": auto_q,
                "session_id": st.session_state.get("session_id", "demo")
            }
            try:
                auto_resp = requests.post(
                    f"{API_URL}/chat",
                    json=auto_payload
                ).json()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": auto_resp.get("response", "")
                })
            except Exception:
                pass
            st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about the forecast, anomalies, or actions...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    rate_limited = False
                    try:
                        payload = {
                            "message": user_input,
                            "session_id": st.session_state.get(
                                "session_id", "demo")
                        }
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json=payload,
                        )
                        if resp.status_code == 200:
                            response_data = resp.json()
                            bot_reply = response_data.get("response", "No response.")
                            rate_limited = response_data.get("rate_limited", False)
                        else:
                            bot_reply = f"API Error {resp.status_code}: {resp.text}"
                            response_data = {}
                    except Exception as e:
                        bot_reply = f"Error communicating with backend: {e}"
                        response_data = {}

                st.markdown(bot_reply)
                chart_ctx = response_data.get("chart_context")
                if chart_ctx:
                    ctx_type = chart_ctx.get("type")

                    if ctx_type == "highlight_forecast":
                        periods = chart_ctx.get("periods", 4)
                        label = chart_ctx.get("label", "Forecast")
                        st.markdown(f"**📈 {label} — live forecast view:**")

                        # Show mini forecast chart inline in chat
                        fc_slice = forecast[:periods]
                        lo_slice = lower[:periods]
                        hi_slice = upper[:periods]
                        dt_slice = future_dates[:periods]

                        if fc_slice and dt_slice:
                            fig_mini = go.Figure()
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=fc_slice,
                                name=label,
                                line=dict(color="#5A287D", width=2),
                                mode="lines+markers"
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=hi_slice, name="Upper", line=dict(width=0), showlegend=False
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=lo_slice, name="Lower",
                                fill="tonexty",
                                fillcolor="rgba(90,40,125,0.15)",
                                line=dict(width=0), showlegend=False
                            ))
                            fig_mini.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=10, b=10),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                yaxis_title=target_col_name,
                                showlegend=False,
                            )
                            st.plotly_chart(fig_mini, use_container_width=True)

                    elif ctx_type == "highlight_group":
                        keyword = chart_ctx.get("keyword", "")
                        if group_forecasts:
                            matched = [g for g in group_forecasts
                                    if keyword.lower() in str(g.get("group","")).lower()]
                            if matched:
                                st.markdown(f"**📊 Group breakdown — '{keyword}':**")
                                st.table(matched)

                if rate_limited:
                    st.markdown(
                        '<p style="color:#e53e3e; font-size:0.82rem; margin-top:4px;">'
                        '⚠️ AI response limit reached. If answer seems incomplete, wait ~60 seconds and ask again.'
                        '</p>',
                        unsafe_allow_html=True
                    )

                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

                # Display suggested questions as buttons
                suggested = response_data.get("suggested_questions", [])
                if suggested:
                    st.markdown("**💡 Try asking:**")
                    cols = st.columns(len(suggested))
                    for i, q in enumerate(suggested):
                        with cols[i]:
                            if st.button(q,
                                         key=f"sq_{i}_{q[:15]}",
                                         use_container_width=True):
                                st.session_state.pending_question = q
                                st.rerun()
else:
    st.markdown('<div class="custom-info-box" style="margin-top: 50px;">Please upload a CSV or click the Demo Mode button in the sidebar to begin.</div>', unsafe_allow_html=True)