import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import os
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(page_title="FutureLens AI", layout="wide", page_icon="🟣")



load_dotenv()  # Load from .env file

API_URL = os.getenv("API_URL", "http://localhost:8000")
print(f"Using API_URL: {API_URL}")  # Debug line

# --- Color Palette Constants ---
THEME_COLORS = {
    "bg_main": "#F7F8FA",
    "bg_card": "#FFFFFF",
    "border_card": "1px solid #EBEBEB",
    "shadow_card": "0px 4px 10px rgba(0, 0, 0, 0.05)",
    "bg_navy_dark": "#1A1F2B",
    "bg_navy_light": "#232A38",
    "accent_primary": "#E8923C",
    "accent_hover": "#F2A65A",
    "text_primary": "#1A1F2B",
    "text_on_dark": "#E8E8E8",
    "text_secondary": "#9CA3AF"
}


st.markdown("""
<style>
    :root {
        --bg-main: #F7F8FA;
        --bg-card: #FFFFFF;
        --border-card: 1px solid #EBEBEB;
        --shadow-card: 0px 4px 10px rgba(0, 0, 0, 0.05);
        --bg-navy-dark: #1A1F2B;
        --bg-navy-light: #232A38;
        --accent-primary: #E8923C;
        --accent-hover: #F2A65A;
        --text-primary: #1A1F2B;
        --text-on-dark: #E8E8E8;
        --text-secondary: #9CA3AF;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: var(--bg-navy-dark) !important;
        border-right: none;
    }
    /* Pull sidebar content up */
    [data-testid="stSidebar"] div[class*="stVerticalBlock"] {
        padding-top: 15px !important;
    }
    .sidebar-logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: -30px !important;
        margin-bottom: 25px !important;
        padding: 5px 0px;
    }
    .sidebar-logo-icon {
        background-color: #8C5300; /* Rich golden-bronze background to match mockup */
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .sidebar-logo-text-block {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sidebar-logo-title {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.3px !important;
    }
    .sidebar-logo-subtitle {
        color: #E8923C !important; /* Golden Accent */
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        margin-top: 2px !important;
        text-transform: uppercase;
        opacity: 0.95;
    }
    [data-testid="stSidebar"] p:not(.stButton p),
    [data-testid="stSidebar"] span:not(.stButton span),
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        color: var(--text-on-dark) !important;
    }
    [data-testid="stSidebarNav"] ul li a {
        color: var(--text-on-dark) !important;
        background-color: transparent !important;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebarNav"] ul li a:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: var(--accent-primary) !important;
    }
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] {
        background-color: var(--accent-primary) !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: var(--bg-navy-light) !important;
        border: 2px solid var(--bg-navy-light) !important;
        border-radius: 10px;
        transition: all 0.3s ease;
        padding: 5px 15px;
    }
    [data-testid="stSidebar"] .stButton>button p,
    [data-testid="stSidebar"] .stButton>button div {
        color: var(--text-on-dark) !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: var(--accent-primary) !important;
        border: 2px solid var(--accent-primary) !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover p,
    [data-testid="stSidebar"] .stButton>button:hover div {
        color: var(--bg-navy-dark) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: var(--bg-navy-light) !important;
        border: 2px solid var(--bg-navy-light) !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 4px 15px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button div {
        color: var(--text-on-dark) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background-color: var(--accent-primary) !important;
        border: 2px solid var(--accent-primary) !important;
        transform: scale(1.06) !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover div {
        color: var(--bg-navy-dark) !important;
    }
    .stMain .stButton>button {
        color: #FFFFFF !important;
        background-color: var(--accent-primary) !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(232, 146, 60, 0.2);
    }
    .stMain .stButton>button:hover {
        background-color: var(--accent-hover) !important;
        box-shadow: 0 6px 12px rgba(242, 166, 90, 0.3);
    }
    .stMain h1, .stMain h2, .stMain h3 {
        color: var(--text-primary) !important;
    }
    .hero-banner {
        background-color: var(--bg-navy-dark);
        padding: 40px;
        border-radius: 16px;
        margin-bottom: 30px;
        color: var(--text-on-dark);
        box-shadow: var(--shadow-card);
    }
    .hero-badge {
        background-color: rgba(232, 146, 60, 0.15);
        color: var(--accent-primary);
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .hero-title {
        color: #FFFFFF !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        margin-bottom: 15px !important;
    }
    .hero-subtitle {
        color: var(--text-secondary) !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        max-width: 800px;
        margin-bottom: 0px !important;
    }
    .hero-btn-primary {
        background-color: var(--accent-primary) !important;
        color: #FFFFFF !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        display: inline-block;
        border: 1px solid var(--accent-primary);
    }
    .hero-btn-primary:hover {
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover);
        transform: translateY(-2px);
        color: #FFFFFF !important;
    }
    .hero-btn-secondary {
        background-color: transparent !important;
        color: #FFFFFF !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        display: inline-block;
        border: 1px solid #FFFFFF;
    }
    .hero-btn-secondary:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-2px);
        color: #FFFFFF !important;
    }
    .stSpinner > div > div {
        border-top-color: var(--accent-primary) !important;
        border-right-color: var(--accent-hover) !important;
        border-bottom-color: var(--accent-primary) !important;
        border-left-color: transparent !important;
    }
    .stSpinner p {
        color: var(--accent-primary) !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-main);
        border-radius: 30px;
        padding: 5px 25px;
        border: 1px solid #EBE2F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"] p { color: var(--text-primary) !important; font-weight: 600; }
    .stTabs [data-baseweb="tab"]:hover {
        border: 1px solid var(--accent-primary);
        background-color: var(--bg-card);
        box-shadow: 0 4px 6px rgba(232, 146, 60, 0.15);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-primary) !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(232, 146, 60, 0.3) !important;
    }
    .stTabs [aria-selected="true"] p { color: #FFFFFF !important; font-weight: bold !important; }
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main) !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    [data-testid="stChatMessage"] {
        background-color: var(--bg-card) !important;
        border: var(--border-card) !important;
        border-radius: 15px;
        padding: 15px 20px;
        border-left: 6px solid var(--accent-primary) !important;
        box-shadow: var(--shadow-card);
        margin-bottom: 15px;
    }
    [data-testid="stChatMessageAvatarUser"]      { background-color: var(--accent-primary) !important; color: white !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: var(--bg-navy-dark) !important; color: white !important; }
    [data-testid="stChatInput"] {
        border: 2px solid var(--accent-primary) !important;
        border-radius: 15px !important;
    }
    .sidebar-success {
        background-color: var(--accent-primary) !important;
        color: #FFFFFF !important;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid #FFFFFF;
    }
    .custom-info-box {
        background-color: var(--bg-navy-dark) !important;
        border-left: 5px solid var(--accent-primary) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        color: var(--text-on-dark) !important;
        margin-bottom: 20px !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
        box-shadow: var(--shadow-card);
    }
    .custom-info-box strong {
        color: var(--accent-primary) !important;
    }
    [data-testid="stProgressBar"] > div > div { background-color: var(--accent-primary) !important; }
    [data-testid="stTable"] {
        background-color: var(--bg-card) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-card) !important;
        border: var(--border-card) !important;
        margin-bottom: 25px !important;
    }
    [data-testid="stTable"] table {
        background-color: var(--bg-card) !important;
        border-collapse: collapse !important;
        width: 100% !important;
        border: none !important;
    }
    [data-testid="stTable"] th {
        background-color: var(--bg-navy-dark) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 12px 16px !important;
        border: none !important;
        text-align: left !important;
    }
    [data-testid="stTable"] td {
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #EBEBEB !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTable"] tr:last-child td {
        border-bottom: none !important;
    }
    [data-testid="stTable"] td:last-child {
        color: var(--accent-primary) !important;
        font-weight: 700 !important;
    }

    /* Status Badges Styling */
    .status-badge, .badge, .status {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-align: center !important;
    }
    .status-badge.validated, .badge-validated, .status-validated, [class*="validated"], :has(> [class*="validated"]) {
        background-color: rgba(0, 168, 89, 0.12) !important;
        color: #00A859 !important;
        border: 1px solid rgba(0, 168, 89, 0.25) !important;
    }
    .status-badge.processing, .badge-processing, .status-processing, [class*="processing"], :has(> [class*="processing"]) {
        background-color: rgba(232, 146, 60, 0.12) !important;
        color: var(--accent-primary) !important;
        border: 1px solid rgba(232, 146, 60, 0.25) !important;
    }
    .status-badge.error, .badge-error, .status-error, [class*="error"], :has(> [class*="error"]) {
        background-color: rgba(229, 62, 62, 0.12) !important;
        color: #E53E3E !important;
        border: 1px solid rgba(229, 62, 62, 0.25) !important;
    }
    [data-testid="stPlotlyChart"], .stPlotlyChart {
        border: var(--border-card) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        background-color: var(--bg-card) !important;
        box-shadow: var(--shadow-card) !important;
    }
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: var(--border-card) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow-card) !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: var(--border-card) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-card) !important;
        padding: 20px !important;
    }
    [data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: var(--border-card) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: var(--shadow-card) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-primary) !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">⚡ NEXT-GENERATION PREDICTIVE ENGINE</div>
    <div class="hero-title">Illuminate Your Future with FutureLens AI.</div>
    <div class="hero-subtitle">FutureLens transforms complex data into high-fidelity forecasting models. Identify emerging market shifts, mitigate risks, and seize opportunities with executive-grade precision.</div>
    <div style="margin-top: 25px; display: flex; gap: 15px;">
        <a href="#predictive-forecast-analysis" class="hero-btn-primary">Explore Active Models</a>
        <a href="#dataset-overview" class="hero-btn-secondary">View Methodology</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Session State ---
if "data"                  not in st.session_state: st.session_state.data = None
if "session_id"            not in st.session_state: st.session_state.session_id = "demo"
if "chat_history"          not in st.session_state: st.session_state.chat_history = []
if "demo_mode"             not in st.session_state: st.session_state.demo_mode = False
if "forecast_update"       not in st.session_state: st.session_state.forecast_update = None
if "forecast_notification" not in st.session_state: st.session_state.forecast_notification = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Base chart layout — does NOT include xaxis_title / yaxis_title so callers
# can safely pass those separately without hitting "multiple values" errors.
CHART_LAYOUT = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        showline=True, linewidth=1, linecolor="#EBEBEB",
        showgrid=True, gridwidth=1, gridcolor="#F3F4F6",
        title_font=dict(size=14, color="#1A1F2B", family="Arial, sans-serif"),
        tickfont=dict(size=12, color="#9CA3AF", family="Arial, sans-serif"),
    ),
    yaxis=dict(
        showline=True, linewidth=1, linecolor="#EBEBEB",
        showgrid=True, gridwidth=1, gridcolor="#F3F4F6",
        title_font=dict(size=14, color="#1A1F2B", family="Arial, sans-serif"),
        tickfont=dict(size=12, color="#9CA3AF", family="Arial, sans-serif"),
    ),
)


def _chart_layout(x_title: str = "", y_title: str = "", **extra) -> dict:
    """
    Return a layout dict that merges CHART_LAYOUT with axis titles.
    Axis titles are injected into the existing xaxis/yaxis sub-dicts so
    there is never a duplicate keyword argument.
    """
    layout = {
        "plot_bgcolor":  CHART_LAYOUT["plot_bgcolor"],
        "paper_bgcolor": CHART_LAYOUT["paper_bgcolor"],
        "xaxis": {**CHART_LAYOUT["xaxis"], "title": x_title},
        "yaxis": {**CHART_LAYOUT["yaxis"], "title": y_title},
    }
    layout.update(extra)
    return layout


def load_data(file_bytes=None, filename="data.csv"):
    with st.spinner("Analyzing data and generating forecast..."):
        try:
            if file_bytes:
                files = {"file": (filename, file_bytes, "text/csv")}
            else:
                st.error("No file provided.")
                return

            payload = {
                "selected_x_col": st.session_state.get("selected_x_col", "__auto__"),
                "selected_y_col": st.session_state.get("selected_y_col", "__auto__"),
                "selected_chart_type": st.session_state.get("selected_chart_type", "auto"),
            }

            response = requests.post(f"{API_URL}/upload", files=files, data=payload)
            if response.status_code == 200:
                st.session_state.data       = response.json()
                st.session_state.session_id = st.session_state.data.get("session_id", "demo")
                st.markdown('<div class="sidebar-success">✅ Analysis complete!</div>', unsafe_allow_html=True)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}. Ensure the FastAPI backend is running on {API_URL}.")


def _read_csv_columns(file_bytes: bytes) -> list:
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            df_preview = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, nrows=20)
            return df_preview.columns.tolist()
        except Exception:
            continue
    return []


def _render_cross_sectional_chart(chart_data: dict, chart_config: dict):
    """Render Groq-specified chart for cross-sectional datasets with proper data labels."""
    if not chart_data or not chart_data.get("rows"):
        st.info("No chart data available for this dataset.")
        return

    navy_dark = THEME_COLORS["bg_navy_dark"]
    chart_type = chart_data.get("chart_type", "horizontal_bar")
    rows       = chart_data.get("rows", [])
    x_col      = chart_data.get("x_col")
    y_col      = chart_data.get("y_col")
    color_col  = chart_data.get("color_col")
    title      = chart_data.get("chart_title", "Dataset Overview")
    x_label    = chart_data.get("x_label", x_col or "")
    y_label    = chart_data.get("y_label", y_col or "")

    df_chart = pd.DataFrame(rows)
    if df_chart.empty:
        st.info("Chart data is empty.")
        return

    fig = go.Figure()

    if chart_type == "horizontal_bar":
        xs = df_chart["x"] if "x" in df_chart.columns else df_chart.iloc[:, 0]
        ys = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 1]

        if color_col and "color" in df_chart.columns:
            colors  = df_chart["color"].unique()
            palette = px.colors.qualitative.Plotly
            for i, c in enumerate(colors):
                mask = df_chart["color"] == c
                x_vals = df_chart[mask]["y"] if "y" in df_chart.columns else []
                y_vals = df_chart[mask]["x"] if "x" in df_chart.columns else []
                fig.add_trace(go.Bar(
                    x=x_vals,
                    y=y_vals,
                    name=str(c),
                    orientation="h",
                    marker_color=palette[i % len(palette)],
                    text=[f"{v:.2f}" if isinstance(v, (int, float)) else v for v in x_vals],
                    textposition="auto",
                ))
        else:
            fig.add_trace(go.Bar(
                x=ys, 
                y=xs, 
                orientation="h", 
                marker_color=navy_dark,
                text=[f"{v:.2f}" if isinstance(v, (int, float)) else v for v in ys],
                textposition="auto",
            ))

        fig.update_layout(
            title=title,
            barmode="stack" if color_col else "relative",
            **_chart_layout(x_title=y_label, y_title=x_label),
        )

    elif chart_type == "bar":
        xs = df_chart["x"] if "x" in df_chart.columns else df_chart.iloc[:, 0]
        ys = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 1]
        fig.add_trace(go.Bar(
            x=xs, 
            y=ys, 
            marker_color=navy_dark,
            text=[f"{v:.2f}" if isinstance(v, (int, float)) else v for v in ys],
            textposition="auto",
        ))
        fig.update_layout(title=title, **_chart_layout(x_title=x_label, y_title=y_label))

    elif chart_type == "scatter":
        xs        = df_chart["x"]     if "x"     in df_chart.columns else df_chart.iloc[:, 0]
        ys        = df_chart["y"]     if "y"     in df_chart.columns else df_chart.iloc[:, 1]
        colors_col = df_chart["color"] if "color" in df_chart.columns else None
        
        # Create hover text with all available data
        hover_texts = []
        for idx, row in df_chart.iterrows():
            text = f"{x_label}: {xs.iloc[idx]}<br>{y_label}: {ys.iloc[idx]:.2f}"
            for col in df_chart.columns:
                if col not in ["x", "y", "color", "size"] and isinstance(df_chart[col].iloc[idx], (int, float)):
                    text += f"<br>{col}: {df_chart[col].iloc[idx]:.2f}"
            hover_texts.append(text)
        
        fig.add_trace(go.Scatter(
            x=xs, 
            y=ys,
            mode="markers+text",
            marker=dict(
                size=10,
                color=colors_col if colors_col is not None else navy_dark,
                colorscale="Viridis" if colors_col is not None else None,
                showscale=True if colors_col is not None else False,
            ),
            text=[f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in ys],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
        ))
        fig.update_layout(title=title, **_chart_layout(x_title=x_label, y_title=y_label))

    elif chart_type in ("pie", "donut"):
        xs   = df_chart["x"] if "x" in df_chart.columns else df_chart.iloc[:, 0]
        ys   = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 1]
        hole = 0.4 if chart_type == "donut" else 0
        fig.add_trace(go.Pie(
            labels=xs, 
            values=ys, 
            hole=hole,
            textposition="inside",
            textinfo="label+percent+value",
        ))
        fig.update_layout(title=title)

    elif chart_type == "histogram":
        ys = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 0]
        fig.add_trace(go.Histogram(
            x=ys, 
            marker_color=navy_dark,
            text=[f"{v:.0f}" if isinstance(v, (int, float)) else v for v in ys],
        ))
        fig.update_layout(title=title, **_chart_layout(x_title=y_label, y_title="Count"))

    elif chart_type == "line":
        xs = df_chart["x"] if "x" in df_chart.columns else df_chart.iloc[:, 0]
        ys = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers+text",
            marker=dict(size=8, color=navy_dark),
            line=dict(color=navy_dark, width=2),
            text=[f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in ys],
            textposition="top center",
        ))
        fig.update_layout(title=title, **_chart_layout(x_title=x_label, y_title=y_label))

    else:
        # Fallback: horizontal bar with all data
        xs = df_chart["x"] if "x" in df_chart.columns else df_chart.iloc[:, 0]
        ys = df_chart["y"] if "y" in df_chart.columns else df_chart.iloc[:, 1]
        fig.add_trace(go.Bar(
            x=ys, 
            y=xs, 
            orientation="h", 
            marker_color=navy_dark,
            text=[f"{v:.2f}" if isinstance(v, (int, float)) else v for v in ys],
            textposition="auto",
        ))
        fig.update_layout(title=title, **_chart_layout())

    st.plotly_chart(fig, use_container_width=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="sidebar-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="9" stroke-dasharray="1 3"/>
                <circle cx="12" cy="12" r="6" stroke-dasharray="1 2"/>
                <circle cx="12" cy="12" r="2" fill="#FFFFFF"/>
            </svg>
        </div>
        <div class="sidebar-logo-text-block">
            <div class="sidebar-logo-title">FutureLens AI</div>
            <div class="sidebar-logo-subtitle">STRATEGIC INSIGHTS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        columns = _read_csv_columns(uploaded_file.getvalue())
        if columns:
            st.caption("Select X-axis, Y-axis, and chart type. Leave as Auto-detect if unsure.")
            options = ["__auto__"] + columns
            st.selectbox("Chart X-axis column", options, key="selected_x_col")
            st.selectbox("Chart Y-axis column", options, key="selected_y_col")
            st.selectbox(
                "Chart type",
                ["auto", "bar", "horizontal_bar", "scatter", "pie", "donut", "histogram"],
                key="selected_chart_type",
            )

        if st.button("Run Analysis"):
            load_data(uploaded_file.getvalue(), uploaded_file.name)

    st.markdown("---")
    if st.button("🕹️ Demo Mode", width="stretch"):
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
                {"role": "user",      "content": "What will sales look like next few weeks?"},
                {"role": "assistant", "content": "Next 4 weeks: central estimate +6.2% growth. Lower: -2.1%. Upper: +12.4%. Seasonal spike expected in Week 3. Top driver: ad spend has a positive impact."},
                {"role": "user",      "content": "Are there any unusual changes?"},
                {"role": "assistant", "content": "1 unusual point detected. Sales dropped 28% on Week 67, exceeding the forecast band. Likely driver: reduced ad spend. Suggested action: increase ad spend by 8%."},
            ]
        except FileNotFoundError:
            st.error("Sample data not found. Please run generate_data.py first.")

msg_count = len(st.session_state.get("chat_history", []))
user_msgs = msg_count // 2
st.sidebar.markdown("---")
st.sidebar.caption(f"💬 {user_msgs} questions asked this session.")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
if st.session_state.data:
    data = st.session_state.data

    historical_dates   = data.get("historical_dates", [])
    historical         = data.get("historical", [])
    future_dates       = data.get("dates", [])
    forecast           = data.get("forecast", [])
    lower              = data.get("lower", [])
    upper              = data.get("upper", [])
    anomalies          = data.get("anomalies", [])
    truth_score        = data.get("truth_score", 0.0)
    truth_meter        = data.get("truth_meter", {})
    detected_freq      = data.get("detected_freq", "W")
    forecast_horizon   = data.get("forecast_horizon", len(future_dates))
    confidence_level   = data.get("confidence_level", 90)
    data_quality       = data.get("data_quality", {})
    dq_warning         = data_quality.get("warning", "")
    detected_cols      = data.get("detected_columns", {})
    target_col_name    = detected_cols.get("target", "target")
    date_col_name      = detected_cols.get("date", "date")
    feature_cols       = detected_cols.get("features", [])
    dataset_profile    = data.get("dataset_profile", {})
    group_forecasts    = data.get("group_forecasts", None)
    is_cross_sectional = data.get("is_cross_sectional", False)
    chart_data         = data.get("chart_data", None)
    chart_config       = data.get("chart_config", None)
    custom_plot_active = data.get("custom_plot_active", False)
    intelligence_card  = data.get("intelligence_card", {})

    # Info box
    if is_cross_sectional:
        st.markdown(
            f"""<div class="custom-info-box">
                 Cross-sectional dataset detected.<br/>
                Analysing <strong>'{target_col_name}'</strong> across {len(historical)} entities.<br/>
                {intelligence_card.get('groq_reason', '')}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="custom-info-box">
                 Forecasting <strong>'{target_col_name}'</strong> using <strong>'{date_col_name}'</strong> as the date.<br/>
                Detected frequency: <strong>{detected_freq}</strong>.<br/>
                {len(feature_cols)} additional numeric feature column(s) found.
            </div>""",
            unsafe_allow_html=True,
        )

    # Forecast query bar (time-series only)
    if not is_cross_sectional:
        forecast_query = st.text_input(
            label            = " Forecast query",
            placeholder      = "e.g. 'next 3 weeks', 'West region forecast', 'show me next month'",
            key              = "forecast_query_input",
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
                    json={"message": forecast_query.strip(), "session_id": st.session_state.session_id},
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

    tab1, tab2, tab3, tab4 = st.tabs([" Forecast / Chart", " Root Cause", " Scenario", " Chat"])

    # =========================================================================
    # TAB 1 — Forecast / Chart
    # =========================================================================
    with tab1:
        # ── Cross-sectional ──────────────────────────────────────────────────
        if is_cross_sectional:
            chart_title = intelligence_card.get("chart_title", "Dataset Overview")
            st.subheader(chart_title or "Dataset Overview")
            st.caption(intelligence_card.get("one_liner", ""))

            if chart_data:
                _render_cross_sectional_chart(chart_data, chart_config or {})
            else:
                st.info("No chart data returned. Try asking questions in the Chat tab.")

            if group_forecasts:
                st.subheader("Group Rankings")
                st.caption("Ranked by value within this dataset.")
                st.table(group_forecasts)

            if dataset_profile:
                with st.expander("Dataset overview"):
                    st.write("Columns:", dataset_profile.get("columns", []))
                    st.write("Numeric summary:")
                    st.json(dataset_profile.get("numeric_summary", {}))
                    st.write("Categorical summary:")
                    st.json(dataset_profile.get("categorical_summary", {}))

        # ── Time-series ──────────────────────────────────────────────────────
        else:
            fu    = st.session_state.get("forecast_update")
            notif = st.session_state.get("forecast_notification")
            if notif:
                st.success(f" Forecast chart updated — {notif}")
                st.session_state.forecast_notification = None

            st.subheader("Predictive Forecast Analysis")

            if custom_plot_active and chart_data:
                st.caption("Using selected X/Y/chart controls.")
                _render_cross_sectional_chart(chart_data, chart_config or {})
                st.markdown("---")

            plot_forecast  = forecast
            plot_lower     = lower
            plot_upper     = upper
            plot_dates     = future_dates
            chart_subtitle = None

            if fu:
                fu_type = fu.get("type")
                if fu_type == "horizon":
                    n_periods      = int(fu.get("periods", len(forecast)))
                    plot_forecast  = forecast[:n_periods]
                    plot_lower     = lower[:n_periods]
                    plot_upper     = upper[:n_periods]
                    plot_dates     = future_dates[:n_periods]
                    chart_subtitle = fu.get("label", "")
                elif fu_type == "group":
                    chart_subtitle = f"Highlighted group: {fu.get('label', fu.get('keyword','').title())}"
                elif fu_type == "default":
                    chart_subtitle = fu.get("label", "")

            if chart_subtitle:
                st.caption(f" {chart_subtitle}")

            # Group highlight callout
            if fu and fu.get("type") == "group" and group_forecasts:
                kw      = fu.get("keyword", "").lower()
                matched = [g for g in group_forecasts if kw in str(g.get("group", "")).lower()]
                if matched:
                    g      = matched[0]
                    change = g.get("expected_change_percent", 0)
                    st.markdown(
                        f"""<div style="background-color:#E6F4EA;border-left:5px solid #00A859;
                            padding:12px 16px;border-radius:8px;margin-top:12px;">
                            <strong> {g.get('group_col','Group')}: {g.get('group')}</strong><br/>
                            Current: <strong>{g.get('last_hist', 0):,.0f}</strong> →
                            Forecast: <strong>{g.get('last_forecast', 0):,.0f}</strong>
                            (<span style="color:{'green' if change>=0 else 'red'}">{change:+.1f}%</span> expected change)
                            </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(f"No group matching '{kw}' found.")

            historical_dates_dt = pd.to_datetime(historical_dates, errors="coerce") if historical_dates else []
            future_dates_dt     = pd.to_datetime(plot_dates,       errors="coerce") if plot_dates      else []

            if len(future_dates_dt) > 0:
                min_year = pd.to_datetime(future_dates_dt).min().year
                if min_year < 1980:
                    st.error("⚠️ Date parsing error — dates appear to be from 1970. Check your CSV date column.")
                    st.stop()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=historical_dates_dt if len(historical_dates_dt) else historical_dates,
                y=historical,
                line=dict(color=THEME_COLORS["bg_navy_dark"], width=2),
                mode="lines+markers", 
                name="Historical",
                text=[f"{v:.2f}" for v in historical],
                textposition="top center",
                hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
            ))

            if plot_lower and plot_upper and plot_dates:
                fig.add_trace(go.Scatter(
                    name="Lower Bound",
                    x=future_dates_dt if len(future_dates_dt) else plot_dates,
                    y=plot_lower, mode="lines", line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    name=f"{confidence_level}% Confidence Band",
                    x=future_dates_dt if len(future_dates_dt) else plot_dates,
                    y=plot_upper, mode="lines",
                    marker=dict(color=THEME_COLORS["accent_primary"]), line=dict(width=0),
                    fillcolor="rgba(232,146,60,0.15)", fill="tonexty", showlegend=True,
                ))

            if plot_forecast and plot_dates:
                fig.add_trace(go.Scatter(
                    x=future_dates_dt if len(future_dates_dt) else plot_dates,
                    y=plot_forecast,
                    line=dict(color=THEME_COLORS["accent_primary"], width=2, dash="dash"),
                    mode="lines+markers", 
                    name="Forecast",
                    text=[f"{v:.2f}" for v in plot_forecast],
                    textposition="top center",
                    hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:.2f}<extra></extra>",
                ))

            if anomalies:
                anom_dates_dt = pd.to_datetime([a.get("date") for a in anomalies], errors="coerce")
                anom_values = [a.get("actual") for a in anomalies]
                fig.add_trace(go.Scatter(
                    x=anom_dates_dt, 
                    y=anom_values,
                    mode="markers", 
                    marker=dict(color="red", size=10, symbol="x"),
                    name="Anomalies",
                    text=[f"Anomaly: {v:.2f}" for v in anom_values],
                    hovertemplate="<b>Anomaly Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                ))

            if len(historical_dates_dt) > 0 and pd.notna(historical_dates_dt[-1]):
                vline_x = historical_dates_dt[-1]
                if isinstance(vline_x, pd.Timestamp):
                    vline_x = vline_x.to_pydatetime()
                fig.add_shape(type="line", x0=vline_x, x1=vline_x, y0=0, y1=1,
                              xref="x", yref="paper", line=dict(color="gray", dash="dash"))
                fig.add_annotation(x=vline_x, y=1, xref="x", yref="paper",
                                   text="Forecast starts here", showarrow=False,
                                   xanchor="left", yanchor="top", yshift=10, font=dict(color="gray"))

            # ✅ Fixed: axis titles injected via _chart_layout, not as bare kwargs
            fig.update_layout(
                hovermode="x unified",
                **_chart_layout(x_title="Date", y_title=target_col_name),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast summary box
            if forecast and historical:
                last_hist  = float(historical[-1])
                last_fc    = float(forecast[-1])
                change_pct = (last_fc - last_hist) / abs(last_hist) * 100 if last_hist != 0 else 0.0
                lower_pct  = ((float(lower[-1]) - last_hist) / abs(last_hist) * 100) if lower and last_hist != 0 else 0.0
                upper_pct  = ((float(upper[-1]) - last_hist) / abs(last_hist) * 100) if upper and last_hist != 0 else 0.0
                direction_word = "up" if change_pct >= 0 else "down"
                summary_lines = [
                    f" Next <strong>{forecast_horizon}</strong> {detected_freq} period(s): "
                    f"central estimate <strong>{direction_word} {abs(change_pct):.1f}%</strong> from current level.",
                    f"Lower bound: <strong>{lower_pct:+.1f}%</strong> · Upper bound: <strong>{upper_pct:+.1f}%</strong>.",
                ]
                if dq_warning:
                    summary_lines.append(f"⚠️ {dq_warning}")
                st.markdown(
                    f'<div class="custom-info-box">{"<br/><br/>".join(summary_lines)}</div>',
                    unsafe_allow_html=True,
                )

            if group_forecasts:
                st.subheader("Group Growth Outlook")
                st.caption("Forecasted expected change by top groups.")
                st.table(group_forecasts)

            if dataset_profile:
                with st.expander("Dataset overview (detected columns)"):
                    st.write("Columns:", dataset_profile.get("columns", []))
                    st.write("Categorical columns (top values):")
                    st.json(dataset_profile.get("categorical_summary", {}))
                    st.write("Numeric columns summary:")
                    st.json(dataset_profile.get("numeric_summary", {}))

            st.subheader("Truth Meter")
            msg       = truth_meter.get("message", f"Truth score: {truth_score:.1f}% vs baseline")
            color     = truth_meter.get("color", "red")
            score_bar = int(min(max(float(truth_meter.get("score", truth_score)) + 50.0, 0.0), 100.0))
            st.progress(score_bar)
            st.markdown(f"**:{color}[{msg}]**")

    # =========================================================================
    # TAB 2 — Root Cause
    # =========================================================================
    with tab2:
        st.subheader("Root Cause Analysis (SHAP)")
        shap_res = data.get("shap_results", [])
        if shap_res:
            features    = [s["feature"]    for s in shap_res]
            importances = [s["importance"] for s in shap_res]
            fig_shap    = go.Figure(go.Bar(
                x=importances, y=features, orientation="h", marker_color=THEME_COLORS["bg_navy_dark"]
            ))
            fig_shap.update_layout(**_chart_layout(x_title="Importance", y_title="Feature"))
            fig_shap.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown(
                f'<div class="custom-info-box">{data.get("rca_explanation", "No explanation provided.")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write("No SHAP results available for this dataset (cross-sectional or forecast pipeline did not run).")

    # =========================================================================
    # TAB 3 — Scenario
    # =========================================================================
    with tab3:
        st.subheader("Iterative Scenario Simulation")

        if is_cross_sectional:
            st.markdown(
                '<div class="custom-info-box">⚠️ This is a cross-sectional dataset. '
                'Use the <strong>Chat tab</strong> to run what-if scenarios — e.g. '
                '"What if Dividend increases by 20%?"</div>',
                unsafe_allow_html=True,
            )
        else:
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown("####  Growth Scenario")
                growth_pct = st.slider("Adjust growth rate", min_value=-50, max_value=50, value=10, key="growth_slider")
                run_growth = st.button("Run Growth Scenario", key="btn_growth")

            with col_s2:
                st.markdown("####  Trend Scenario")
                trend_type = st.radio(
                    "Apply trend",
                    options=["flat", "recent_trend", "remove_outliers"],
                    format_func=lambda x: {
                        "flat":            "Apply flat trend",
                        "recent_trend":    "Keep recent trend",
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
                            "session_id":     st.session_state.session_id,
                            "change_percent": change_percent,
                            "scenario_type":  scenario_type,
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
                sc_dates      = sim_data.get("dates", future_dates)
                baseline_vals = sim_data.get("baseline", forecast)
                scenario_vals = sim_data.get("scenario", [])
                summary_text  = sim_data.get("summary", "")

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=pd.to_datetime(sc_dates, errors="coerce"),
                    y=baseline_vals, line=dict(color=THEME_COLORS["bg_navy_dark"], dash="dash"), name="Baseline",
                ))
                fig_sc.add_trace(go.Scatter(
                    x=pd.to_datetime(sc_dates, errors="coerce"),
                    y=scenario_vals, line=dict(color=THEME_COLORS["accent_primary"], width=3), name=label,
                ))
                # ✅ Fixed: axis titles via _chart_layout
                fig_sc.update_layout(
                    hovermode="x unified",
                    **_chart_layout(x_title="Date", y_title=target_col_name),
                )
                st.plotly_chart(fig_sc, use_container_width=True)
                if summary_text:
                    st.markdown(f'<div class="custom-info-box"> {summary_text}</div>', unsafe_allow_html=True)

            if run_growth:
                with st.spinner("Running growth scenario..."):
                    result = _call_simulate(growth_pct, "growth")
                _render_scenario(result, f"Growth ({growth_pct:+d}%)")

            if run_trend:
                with st.spinner(f"Running {trend_type} scenario..."):
                    result = _call_simulate(0.0, trend_type)
                _render_scenario(result, trend_type.replace("_", " ").title())

    # =========================================================================
    # TAB 4 — Chat
    # =========================================================================
    with tab4:
        st.subheader("Conversational Insights")

        if "pending_question" in st.session_state:
            auto_q = st.session_state.pop("pending_question")
            st.session_state.chat_history.append({"role": "user", "content": auto_q})
            try:
                auto_resp = requests.post(
                    f"{API_URL}/chat",
                    json={"message": auto_q, "session_id": st.session_state.get("session_id", "demo")},
                ).json()
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": auto_resp.get("response", ""),
                })
            except Exception:
                pass
            st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about the forecast, anomalies, scenarios, or correlations...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    rate_limited = False
                    try:
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "message":    user_input,
                                "session_id": st.session_state.get("session_id", "demo"),
                            },
                        )
                        if resp.status_code == 200:
                            response_data = resp.json()
                            bot_reply     = response_data.get("response", "No response.")
                            rate_limited  = response_data.get("rate_limited", False)
                        else:
                            bot_reply     = f"API Error {resp.status_code}: {resp.text}"
                            response_data = {}
                    except Exception as e:
                        bot_reply     = f"Error communicating with backend: {e}"
                        response_data = {}

                st.markdown(bot_reply)

                # Inline chart context
                chart_ctx = response_data.get("chart_context")
                if chart_ctx and not is_cross_sectional:
                    ctx_type = chart_ctx.get("type")
                    if ctx_type == "highlight_forecast":
                        periods  = chart_ctx.get("periods", 4)
                        label    = chart_ctx.get("label", "Forecast")
                        fc_slice = forecast[:periods]
                        lo_slice = lower[:periods]
                        hi_slice = upper[:periods]
                        dt_slice = future_dates[:periods]
                        if fc_slice and dt_slice:
                            fig_mini = go.Figure()
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=fc_slice, name=label,
                                line=dict(color=THEME_COLORS["accent_primary"], width=2), mode="lines+markers",
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=hi_slice, name="Upper", line=dict(width=0), showlegend=False,
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=pd.to_datetime(dt_slice, errors="coerce"),
                                y=lo_slice, name="Lower",
                                fill="tonexty", fillcolor="rgba(232,146,60,0.15)",
                                line=dict(width=0), showlegend=False,
                            ))
                            fig_mini.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=10, b=10),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                yaxis=dict(title=target_col_name),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_mini, use_container_width=True)

                    elif ctx_type == "highlight_group" and group_forecasts:
                        keyword = chart_ctx.get("keyword", "")
                        matched = [g for g in group_forecasts
                                   if keyword.lower() in str(g.get("group", "")).lower()]
                        if matched:
                            st.markdown(f"** Group breakdown — '{keyword}':**")
                            st.table(matched)

                if rate_limited:
                    st.markdown(
                        '<p style="color:#e53e3e;font-size:0.82rem;margin-top:4px;">'
                        "⚠️ AI response limit reached. Wait ~60 seconds and ask again.</p>",
                        unsafe_allow_html=True,
                    )

                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

                suggested = response_data.get("suggested_questions", [])
                if suggested:
                    st.markdown("**💡 Try asking:**")
                    cols = st.columns(len(suggested))
                    for i, q in enumerate(suggested):
                        with cols[i]:
                            if st.button(q, key=f"sq_{i}_{q[:15]}", width="stretch"):
                                st.session_state.pending_question = q
                                st.rerun()

else:
    st.markdown(
        '<div class="custom-info-box" style="margin-top: 50px;">'
        "Please upload a CSV or click the Demo Mode button in the sidebar to begin.</div>",
        unsafe_allow_html=True,
    )