# pyrefly: ignore [missing-import]
import streamlit as st
import requests
import pandas as pd
import json
# pyrefly: ignore [missing-import]
import plotly.io as pio
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0e1117; }
.main-header {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-header { font-size: 1.1rem; color: #8B949E; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #1e2127, #262d3a);
    border-radius: 14px; padding: 22px; border: 1px solid #2d3139;
    text-align: center; transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(78,205,196,0.15); }
.metric-title { color: #8B949E; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2.2rem; font-weight: 800; color: #4ECDC4; margin-top: 8px; }
.status-ok { color: #4ECDC4; font-weight: 600; }
.status-err { color: #FF6B6B; font-weight: 600; }
.chat-user { background: #1e2127; border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 8px 0; border-left: 3px solid #4ECDC4; }
.chat-bot { background: #262d3a; border-radius: 12px 12px 12px 4px; padding: 12px 16px; margin: 8px 0; border-left: 3px solid #FF6B6B; }
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #2d3139; }
.stButton>button {
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    color: white; border: none; border-radius: 8px; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000/api/v1"

# --- Session state init ---
defaults = {
    'file_path': None, 'cleaned_file_path': None, 'schema': None,
    'active_file': None, 'summary_stats': None, 'automl_results': None,
    'ai_insights': None, 'model_insights': None, 'chat_history': [],
    'gemini_key_ok': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 Autonomous Data Analyst")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home",
        "📂 Upload & Ingestion",
        "🧹 Data Cleaning",
        "📊 EDA Dashboard",
        "🔬 Statistical Tests",
        "⚙️ AutoML Studio",
        "🧠 AI Insights",
        "💬 Chat with Data",
        "📄 Report Generator"
    ], label_visibility="collapsed")
    st.markdown("---")

    # API Key status
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your-gemini-api-key":
        st.success("✅ Gemini API Key Set")
    else:
        st.error("❌ GEMINI_API_KEY missing in .env")

    if st.session_state.active_file:
        st.info(f"📁 Active: `{os.path.basename(st.session_state.active_file)}`")
    else:
        st.warning("💡 Upload a dataset to begin")

st.markdown('<div class="main-header">Autonomous Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered End-to-End Data Science Platform</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📂", "Upload CSV/Excel/JSON"),
        ("🧹", "Auto Data Cleaning"),
        ("📊", "EDA & Visualizations"),
        ("⚙️", "AutoML Pipeline"),
    ]
    for col, (icon, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:2rem">{icon}</div><div class="metric-title" style="margin-top:10px">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    cards2 = [("🧠", "AI Insights (Gemini)"), ("💬", "Chat with Data"), ("📄", "Report Generation"), ("🔬", "Statistical Tests")]
    for col, (icon, label) in zip([c5, c6, c7, c8], cards2):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:2rem">{icon}</div><div class="metric-title" style="margin-top:10px">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.markdown("1. **Upload** your CSV/Excel/JSON dataset\n2. **Clean** the data with one click\n3. **Explore** with auto-generated charts\n4. **Train** ML models automatically\n5. **Ask** questions in plain English\n6. **Download** the full report")

# ─────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────
elif page == "📂 Upload & Ingestion":
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file is not None:
        with st.spinner("Uploading and analyzing dataset..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("✅ File uploaded successfully!")
                    st.session_state.file_path = data["file_path"]
                    st.session_state.active_file = data["file_path"]
                    st.session_state.schema = data["schema"]
                    st.session_state.cleaned_file_path = None
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Cannot connect to backend API at {API_BASE_URL}. Is the server running? Error: {e}")

    if st.session_state.schema:
        schema = st.session_state.schema
        st.subheader("📋 Dataset Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            ("Rows", f"{schema['num_rows']:,}"),
            ("Columns", schema['num_cols']),
            ("Numerical", len(schema['numerical_cols'])),
            ("Categorical", len(schema['categorical_cols'])),
            ("Duplicates", schema.get('duplicate_rows', 0)),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-title">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🗂️ Column Details")
        col_data = []
        for col_name, info in schema["columns"].items():
            col_data.append({
                "Column": col_name,
                "Type": info["dtype"],
                "Unique Values": info["unique_count"],
                "Missing": info["missing_count"],
                "Missing %": f"{info['missing_percentage']:.1f}%"
            })
        st.dataframe(pd.DataFrame(col_data), use_container_width=True)

        # Dataset preview
        st.subheader("👁️ Data Preview")
        try:
            from app.utils.data_ingestion import load_dataset
            df_preview = load_dataset(st.session_state.active_file)
            st.dataframe(df_preview.head(20), use_container_width=True)
        except Exception:
            st.info("Run backend server to see preview.")

# ─────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────
elif page == "🧹 Data Cleaning":
    st.header("🧹 Automated Data Cleaning")
    if not st.session_state.file_path:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            missing_val_strategy = st.selectbox("Missing Value Strategy", ["mean", "median", "most_frequent", "knn", "drop"])
            encode_strategy = st.selectbox("Categorical Encoding", ["label", "onehot"])
        with col2:
            scale_strategy = st.selectbox("Feature Scaling", ["standard", "minmax"])
            outlier_method = st.selectbox("Outlier Detection", ["iqr", "zscore"])
        remove_dupes = st.checkbox("Remove Duplicate Rows", value=True)

        if st.button("🚀 Run Cleaning Engine", use_container_width=True):
            with st.spinner("Cleaning dataset..."):
                payload = {
                    "file_path": st.session_state.file_path,
                    "missing_value_strategy": missing_val_strategy,
                    "encode_strategy": encode_strategy,
                    "scale_strategy": scale_strategy,
                    "outlier_method": outlier_method,
                    "remove_duplicates": remove_dupes
                }
                try:
                    resp = requests.post(f"{API_BASE_URL}/clean", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Cleaned! Rows after cleaning: **{data['rows_after_cleaning']:,}**")
                        st.session_state.cleaned_file_path = data["cleaned_file_path"]
                        st.session_state.active_file = data["cleaned_file_path"]
                        st.session_state.schema = data["schema"]
                        st.balloons()

                        # Download button
                        with open(data["cleaned_file_path"], "rb") as f:
                            st.download_button("⬇️ Download Cleaned CSV", f, file_name="cleaned_dataset.csv", mime="text/csv")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ─────────────────────────────────────────────────────────────
# EDA DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        if st.button("🔍 Generate EDA Dashboard", use_container_width=True):
            with st.spinner("Generating visualizations..."):
                try:
                    # Summary
                    resp_sum = requests.post(f"{API_BASE_URL}/eda/summary", json={"file_path": st.session_state.active_file})
                    if resp_sum.status_code == 200:
                        st.session_state.summary_stats = resp_sum.json()["summary"]

                    # Plots
                    resp_plots = requests.post(f"{API_BASE_URL}/eda/plots", json={"file_path": st.session_state.active_file})
                    if resp_plots.status_code == 200:
                        plots_data = resp_plots.json()

                        if plots_data.get("correlation_matrix"):
                            st.subheader("🔥 Correlation Matrix")
                            st.plotly_chart(pio.from_json(plots_data["correlation_matrix"]), use_container_width=True)

                        dist_plots = plots_data.get("distribution_plots", {})
                        if dist_plots:
                            st.subheader("📈 Distributions")
                            cols = st.columns(2)
                            for i, (col_name, fig_json) in enumerate(dist_plots.items()):
                                with cols[i % 2]:
                                    st.plotly_chart(pio.from_json(fig_json), use_container_width=True)

                        cat_plots = plots_data.get("categorical_plots", {})
                        if cat_plots:
                            st.subheader("📊 Categorical Features")
                            cols = st.columns(2)
                            for i, (col_name, fig_json) in enumerate(cat_plots.items()):
                                with cols[i % 2]:
                                    st.plotly_chart(pio.from_json(fig_json), use_container_width=True)

                        box_plots = plots_data.get("box_plots", {})
                        if box_plots:
                            st.subheader("📦 Boxplots (Outlier View)")
                            cols = st.columns(2)
                            for i, (col_name, fig_json) in enumerate(box_plots.items()):
                                with cols[i % 2]:
                                    st.plotly_chart(pio.from_json(fig_json), use_container_width=True)
                    else:
                        st.error(f"Failed to generate plots: {resp_plots.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        # Show summary table if available
        if st.session_state.summary_stats:
            st.subheader("📋 Numerical Summary")
            num_records = st.session_state.summary_stats.get("numerical", [])
            if num_records:
                st.dataframe(pd.DataFrame(num_records), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Statistical Tests":
    st.header("🔬 Statistical Tests")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        schema = st.session_state.schema
        target_col = None
        if schema:
            all_cols = list(schema["columns"].keys())
            target_col = st.selectbox("Optional: Select Target Column for group tests", ["(None)"] + all_cols)
            if target_col == "(None)":
                target_col = None

        if st.button("🧪 Run Statistical Tests", use_container_width=True):
            with st.spinner("Running statistical tests..."):
                try:
                    payload = {"file_path": st.session_state.active_file}
                    if target_col:
                        payload["target_column"] = target_col
                    resp = requests.post(f"{API_BASE_URL}/eda/statistical_tests", json=payload)
                    if resp.status_code == 200:
                        tests = resp.json()["statistical_tests"]

                        if tests.get("normality_tests"):
                            st.subheader("📐 Normality Tests (Shapiro-Wilk)")
                            norm_data = []
                            for col, res in tests["normality_tests"].items():
                                norm_data.append({"Column": col, "Statistic": res["statistic"],
                                    "P-Value": res["p_value"], "Normal?": "✅ Yes" if res["is_normal"] else "❌ No",
                                    "Interpretation": res["interpretation"]})
                            st.dataframe(pd.DataFrame(norm_data), use_container_width=True)

                        if tests.get("chi_square_tests"):
                            st.subheader("📊 Chi-Square Tests")
                            chi_data = []
                            for col, res in tests["chi_square_tests"].items():
                                chi_data.append({"Column": col, "Chi2": res["chi2_statistic"],
                                    "P-Value": res["p_value"], "Significant?": "✅ Yes" if res["is_significant"] else "❌ No",
                                    "Interpretation": res["interpretation"]})
                            st.dataframe(pd.DataFrame(chi_data), use_container_width=True)

                        if tests.get("t_tests"):
                            st.subheader("📏 T-Tests (Group Comparison)")
                            t_data = []
                            for col, res in tests["t_tests"].items():
                                t_data.append({"Column": col, "T-Statistic": res["t_statistic"],
                                    "P-Value": res["p_value"], "Significant?": "✅ Yes" if res["is_significant"] else "❌ No",
                                    "Interpretation": res["interpretation"]})
                            st.dataframe(pd.DataFrame(t_data), use_container_width=True)
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ─────────────────────────────────────────────────────────────
# AUTOML
# ─────────────────────────────────────────────────────────────
elif page == "⚙️ AutoML Studio":
    st.header("⚙️ Automated Machine Learning")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    elif not st.session_state.schema:
        st.warning("⚠️ Schema not loaded. Re-upload your dataset.")
    else:
        schema = st.session_state.schema
        all_cols = list(schema["columns"].keys())
        target_col = st.selectbox("🎯 Select Target Variable", all_cols)
        problem_override = st.selectbox("Problem Type Override (optional)", ["Auto Detect", "Binary Classification", "Multi-Class Classification", "Regression", "Clustering"])

        st.info("⏱️ AutoML compares multiple models. This may take 2-5 minutes.")

        if st.button("🚀 Start AutoML Pipeline", type="primary", use_container_width=True):
            with st.spinner("Training models... Please wait."):
                payload = {
                    "file_path": st.session_state.active_file,
                    "target_column": target_col,
                    "problem_type": None if problem_override == "Auto Detect" else problem_override
                }
                try:
                    resp = requests.post(f"{API_BASE_URL}/automl/train", json=payload, timeout=600)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.automl_results = data
                        st.success(f"✅ Pipeline complete! Problem Type: **{data['problem_type']}**")
                    else:
                        st.error(f"AutoML failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.automl_results:
            data = st.session_state.automl_results
            st.subheader("🏆 Model Leaderboard")
            st.dataframe(pd.DataFrame(data["leaderboard"]), use_container_width=True)
            st.info(f"💾 Best model saved: `{data['best_model_path']}`")

            if data.get("confusion_matrix"):
                st.subheader("🎯 Confusion Matrix")
                st.plotly_chart(pio.from_json(data["confusion_matrix"]), use_container_width=True)

            if data.get("roc_curve"):
                st.subheader("📈 ROC Curve")
                st.plotly_chart(pio.from_json(data["roc_curve"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# AI INSIGHTS
# ─────────────────────────────────────────────────────────────
elif page == "🧠 AI Insights":
    st.header("🧠 AI-Powered Insights")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate EDA Insights", use_container_width=True):
                if not st.session_state.summary_stats:
                    with st.spinner("Fetching summary..."):
                        try:
                            resp = requests.post(f"{API_BASE_URL}/eda/summary", json={"file_path": st.session_state.active_file})
                            if resp.status_code == 200:
                                st.session_state.summary_stats = resp.json()["summary"]
                        except Exception as e:
                            st.error(f"Error: {e}")

                if st.session_state.summary_stats:
                    with st.spinner("Analyzing with Gemini AI..."):
                        try:
                            resp = requests.post(f"{API_BASE_URL}/insights/eda", json={"summary_stats": st.session_state.summary_stats})
                            if resp.status_code == 200:
                                st.session_state.ai_insights = resp.json()["insights"]
                            else:
                                st.error(f"Error: {resp.json().get('detail', resp.text)}")
                        except Exception as e:
                            st.error(f"Error: {e}")

        with col2:
            if st.session_state.automl_results:
                if st.button("🤖 Generate Model Insights", use_container_width=True):
                    with st.spinner("Analyzing model results with Gemini..."):
                        try:
                            data = st.session_state.automl_results
                            resp = requests.post(f"{API_BASE_URL}/insights/model", json={
                                "leaderboard": data["leaderboard"],
                                "problem_type": data["problem_type"]
                            })
                            if resp.status_code == 200:
                                st.session_state.model_insights = resp.json()["insights"]
                            else:
                                st.error(f"Error: {resp.json().get('detail', resp.text)}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("Run AutoML first to generate model insights.")

        if st.session_state.ai_insights:
            st.subheader("📊 EDA Insights")
            st.markdown(st.session_state.ai_insights)

        if st.session_state.model_insights:
            st.subheader("🤖 Model Insights")
            st.markdown(st.session_state.model_insights)

# ─────────────────────────────────────────────────────────────
# CHAT WITH DATA
# ─────────────────────────────────────────────────────────────
elif page == "💬 Chat with Data":
    st.header("💬 Natural Language Querying")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        st.write("Ask questions about your dataset in plain English.")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 <strong>AI:</strong><br><pre style="white-space:pre-wrap;background:transparent;border:none;padding:0;color:#c0c0c0">{msg["content"]}</pre></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Your question:", placeholder="e.g. What is the average age? Show top 5 rows. Which column has most nulls?", key="chat_input")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if st.button("💬 Ask AI", use_container_width=True) and query:
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "file_path": st.session_state.active_file,
                        "query": query,
                        "chat_history": st.session_state.chat_history
                    }
                    resp = requests.post(f"{API_BASE_URL}/insights/query", json=payload)
                    if resp.status_code == 200:
                        answer = resp.json()["response"]
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ─────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
elif page == "📄 Report Generator":
    st.header("📄 Automated Report Generation")
    if not st.session_state.active_file:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        st.write("Generate a comprehensive HTML report with all findings.")
        st.info("The report includes: Dataset Overview, EDA Statistics, AI Insights, and AutoML Leaderboard (if available).")

        if st.button("📄 Generate Full Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                payload = {
                    "file_path": st.session_state.active_file,
                    "ai_insights": st.session_state.ai_insights or "",
                    "leaderboard": st.session_state.automl_results["leaderboard"] if st.session_state.automl_results else [],
                    "problem_type": st.session_state.automl_results["problem_type"] if st.session_state.automl_results else "",
                    "model_insights": st.session_state.model_insights or ""
                }
                try:
                    resp = requests.post(f"{API_BASE_URL}/reports/generate", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Report generated!")
                        html_bytes = data["html_content"].encode("utf-8")
                        st.download_button(
                            label="⬇️ Download HTML Report",
                            data=html_bytes,
                            file_name="autonomous_analyst_report.html",
                            mime="text/html",
                            use_container_width=True
                        )
                        with st.expander("👁️ Preview Report"):
                            st.components.v1.html(data["html_content"], height=700, scrolling=True)
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
