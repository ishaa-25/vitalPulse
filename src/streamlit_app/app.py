import sys
from pathlib import Path

# Add project root and src to python path so absolute imports work reliably
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for d in [str(ROOT_DIR), str(SRC_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

import polars as pl
import streamlit as st

from src.agentic.workflow import run_agent
from src.db.connection import get_engine

PROCESSED_DIR = SRC_DIR / "data" / "processed"
LATEST_PLOT = PROCESSED_DIR / "latest_plot.png"
LATEST_PARQUET = PROCESSED_DIR / "latest_query.parquet"

st.set_page_config(
    page_title="VitalPulse | Athletic AI",
    page_icon="🏃",
    layout="wide",
)

# Custom Sporty Font & Theme Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
    
    .sporty-title {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        color: #FAFAFA;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0px;
    }
    .sporty-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    .training-banner {
        background-color: #161B22; 
        padding: 12px 18px; 
        border-radius: 8px; 
        border-left: 5px solid #FC4C02; 
        margin-bottom: 20px;
        font-family: 'Rajdhani', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data(ttl=60)
def load_dashboard_kpis():
    try:
        engine = get_engine()
        athletes_df = pl.read_database("SELECT COUNT(*) AS cnt FROM athletes;", engine)
        act_df = pl.read_database(
            """
            SELECT 
                COUNT(*) AS total_workouts,
                ROUND(COALESCE(SUM(distance_km), 0)::numeric, 1) AS total_km
            FROM activities;
            """,
            engine,
        )
        acwr_df = pl.read_database(
            """
            SELECT ROUND(AVG(acwr)::numeric, 2) AS avg_acwr
            FROM daily_metrics
            WHERE date >= (SELECT MAX(date) - INTERVAL '7 days' FROM daily_metrics);
            """,
            engine,
        )
        return {
            "athletes": int(athletes_df["cnt"][0]),
            "workouts": int(act_df["total_workouts"][0]),
            "total_km": float(act_df["total_km"][0]),
            "avg_acwr": float(acwr_df["avg_acwr"][0] or 0.0),
        }
    except Exception:
        return None

# ------------------------------------------------------
# Sporty Header & Title
# ------------------------------------------------------
st.markdown('<div class="sporty-title"> VitalPulse // Athletic Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sporty-subtitle">Endurance Telemetry &bull; Training Stress (ACWR) &bull; Agentic Modeling</div>', unsafe_allow_html=True)
st.image("src/streamlit_app/assets/athlete_banner.png", use_container_width=True)
# Sporty Athlete Status Banner
st.markdown(
    """
    <div class="training-banner">
        <span style="font-weight: bold; color: #FC4C02; font-size: 1.1em;">⚡ ACTIVE TRAINING BLOCK: PEAK MARATHON BUILD</span><br>
        <span style="color: #8b949e; font-size: 0.95em;">Monitoring live telemetry for <b>Elena Vance</b> & <b>Marcus Kane</b> &bull; Status: <span style="color: #3fb950;">Optimal Load (ACWR 0.93)</span> &bull; Engine: Groq Llama-3.3 ReAct</span>
    </div>
    """,
    unsafe_allow_html=True
)

kpis = load_dashboard_kpis()
if kpis:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Athletes", f"{kpis['athletes']}", "Elena & Marcus")
    col2.metric("Logged Workouts", f"{kpis['workouts']:,}", "PostgreSQL Live")
    col3.metric("Cumulative Distance", f"{kpis['total_km']:,.1f} km", "All Sports")
    acwr_val = kpis["avg_acwr"]
    acwr_status = "Optimal Zone" if 0.8 <= acwr_val <= 1.3 else "High Load"
    col4.metric("7-Day Avg ACWR", f"{acwr_val:.2f}", acwr_status)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

with st.sidebar:
    st.header("⚡ Quick Analytics")
    st.caption("Click any prompt to run live SQL + Python modeling:")

    example_questions = [
        "What is the total running mileage for Elena Vance over the last 30 days?",
        "Run a linear regression between pace and heart rate for all runs.",
        "Forecast Marcus Kane's weekly training stress score for the next 4 weeks and plot it.",
    ]

    for idx, example in enumerate(example_questions):
        if st.button(example, key=f"example_btn_{idx}", use_container_width=True):
            st.session_state.queued_prompt = example

    st.divider()
    st.markdown("**Architecture Stack**")
    st.caption("• LLM: Groq (llama-3.3-70b)\n• Orchestration: LangGraph\n• Storage: PostgreSQL + Polars\n• Modeling: Statsmodels & Prophet")
    st.divider()

    if st.button("🗑️ Clear Chat & Charts", use_container_width=True):
        st.session_state.messages = []
        if LATEST_PLOT.exists():
            LATEST_PLOT.unlink()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_bytes"):
            st.image(message["image_bytes"], use_container_width=True)

typed_prompt = st.chat_input("Ask VitalPulse about mileage, regressions, or injury risk...")
user_prompt = st.session_state.queued_prompt or typed_prompt

if user_prompt:
    st.session_state.queued_prompt = None
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if LATEST_PLOT.exists():
        LATEST_PLOT.unlink()

    with st.chat_message("assistant"):
        with st.status("Querying PostgreSQL & running Python models...", expanded=True) as status:
            try:
                response = run_agent(user_prompt)
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as exc:
                response = f"Agent error: {type(exc).__name__}: {exc}"
                status.update(label="Execution failed", state="error", expanded=False)

        st.markdown(response)

        image_bytes = None
        if LATEST_PLOT.exists():
            image_bytes = LATEST_PLOT.read_bytes()
            st.image(image_bytes, use_container_width=True)

        if LATEST_PARQUET.exists():
            with st.expander("🔍 Inspect Extracted Data (latest_query.parquet)"):
                try:
                    df_preview = pl.read_parquet(LATEST_PARQUET).head(15).to_pandas()
                    st.dataframe(df_preview, use_container_width=True)
                except Exception:
                    st.caption("No tabular preview available.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "image_bytes": image_bytes,
        }
    )
