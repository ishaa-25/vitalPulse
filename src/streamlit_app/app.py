import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from src.agentic.workflow import (
    run_agent,
)


PROCESSED_DIR = (
    ROOT_DIR
    / "src"
    / "data"
    / "processed"
)

LATEST_PLOT = (
    PROCESSED_DIR
    / "latest_plot.png"
)


st.set_page_config(
    page_title="VitalPulse",
    page_icon="🏃",
    layout="wide",
)


st.title(
    "🏃 VitalPulse"
)

st.caption(
    "Fitness & Athletic Performance Analytics"
)


# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------

st.sidebar.header(
    "Example Questions"
)


example_questions = [
    (
        "What is the total running mileage "
        "for Elena Vance over the last 30 days?"
    ),
    (
        "Run a linear regression between pace "
        "and heart rate for all runs."
    ),
    (
        "Forecast Marcus Kane's weekly training "
        "stress score for the next 4 weeks "
        "and plot it."
    ),
]


for example in example_questions:
    st.sidebar.markdown(
        f"• {example}"
    )


st.sidebar.divider()


if st.sidebar.button(
    "Clear chat"
):
    st.session_state.messages = []

    st.rerun()


# ------------------------------------------------------
# Session
# ------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------
# Existing messages
# ------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# ------------------------------------------------------
# Chat input
# ------------------------------------------------------

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

    with st.chat_message(
        "user"
    ):
        st.markdown(
            user_prompt
        )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analyzing training data..."
        ):

            try:
                response = run_agent(
                    user_prompt
                )

            except Exception as exc:
                response = (
                    "Agent error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        st.markdown(
            response
        )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


# ------------------------------------------------------
# Plot
# ------------------------------------------------------

if LATEST_PLOT.exists():

    st.divider()

    st.subheader(
        "Latest Analysis"
    )

    st.image(
        str(LATEST_PLOT),
        use_container_width=True,
    )