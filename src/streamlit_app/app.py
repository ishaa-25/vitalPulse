from pathlib import Path

import streamlit as st


PROCESSED_DIR = Path("src/data/processed")
LATEST_PLOT = PROCESSED_DIR / "latest_plot.png"


st.set_page_config(
    page_title="VitalPulse",
    page_icon="🏃",
    layout="wide",
)


st.title("🏃 VitalPulse")
st.caption("Fitness & Athletic Performance Analytics")


# Sidebar
st.sidebar.title("Example Questions")

example_prompts = [
    "What is the total running mileage for Elena Vance over the last 30 days?",
    "Run a linear regression between pace and heart rate for all runs.",
    "Forecast Marcus Kane's weekly training stress score (TSS) for the next 4 weeks and plot it.",
]

for prompt in example_prompts:
    st.sidebar.markdown(f"• {prompt}")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_prompt = st.chat_input(
    "Ask VitalPulse about your training data..."
)


if user_prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Temporary response until LangGraph is connected.
    response = (
        "VitalPulse received your question. "
        "The LangGraph agent will be connected next."
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)


# Latest generated plot
if LATEST_PLOT.exists():
    st.subheader("Latest Analysis")
    st.image(str(LATEST_PLOT))