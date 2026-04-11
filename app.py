import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(page_title="FutureLens", layout="wide")
API_URL = "http://localhost:8000"

st.title("FutureLens")
st.markdown("*FutureLens doesn't just predict the future — it tells you how to change it.*")

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
    if st.button("🕹️ Demo Mode", use_container_width=True):
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
    detected_freq = data.get("detected_freq", "W")
    forecast_horizon = data.get("forecast_horizon", len(future_dates))
    confidence_level = data.get("confidence_level", 90)
    data_quality = data.get("data_quality", {})
    dq_warning = data_quality.get("warning", "")
    detected_cols = data.get("detected_columns", {})
    target_col_name = detected_cols.get("target", "target")
    date_col_name = detected_cols.get("date", "date")
    feature_cols = detected_cols.get("features", [])

    # Show detected-columns info box right after upload
    st.info(
        f"📊 Forecasting **'{target_col_name}'** using **'{date_col_name}'** as the date. "
        f"Detected frequency: **{detected_freq}**. "
        f"{len(feature_cols)} additional feature column(s) found."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Root Cause", "Scenario", "Chat"])

    # === TAB 1: Forecast ===
    with tab1:
        st.subheader("Predictive Forecast Analysis")

        # --- Validate dates before plotting ---
        if future_dates:
            min_year = pd.to_datetime(future_dates).min().year
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
                x=historical_dates,
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
                    x=future_dates,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name=f"{confidence_level}% Confidence Band",
                    x=future_dates,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(255,165,0,0.2)",
                    fill="tonexty",
                    showlegend=True,
                )
            )

        # Forecast central line (future only)
        if forecast and future_dates:
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=forecast,
                    line=dict(color="orange", width=2, dash="dash"),
                    mode="lines",
                    name="Forecast",
                )
            )

        # Anomaly markers
        if anomalies:
            anom_dates = [a["date"] for a in anomalies]
            anom_values = [a["actual"] for a in anomalies]
            fig.add_trace(
                go.Scatter(
                    x=anom_dates,
                    y=anom_values,
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="x"),
                    name="Anomalies",
                )
            )

        # Vertical dashed line at forecast start
        if historical_dates:
            fig.add_vline(
                x=historical_dates[-1],
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast starts here",
                annotation_position="top right",
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

        # Truth Meter
        st.subheader("Truth Meter")
        if truth_score > 10:
            color = "green"
        elif truth_score > 0:
            color = "orange"
        else:
            color = "red"
        st.progress(min(int(truth_score), 100))
        st.markdown(f"**:{color}[Model is {truth_score:.1f}% better than baseline]**")

    # === TAB 2: Root Cause ===
    with tab2:
        st.subheader("Root Cause Analysis (SHAP)")
        shap_res = data.get("shap_results", [])
        if shap_res:
            features = [s["feature"] for s in shap_res]
            importances = [s["importance"] for s in shap_res]

            fig_shap = go.Figure(
                go.Bar(x=importances, y=features, orientation="h", marker_color="indigo")
            )
            fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})
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
                    x=sc_dates,
                    y=baseline_vals,
                    line=dict(color="royalblue", dash="dash"),
                    name="Baseline",
                )
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=sc_dates,
                    y=scenario_vals,
                    line=dict(color="orange", width=2),
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
