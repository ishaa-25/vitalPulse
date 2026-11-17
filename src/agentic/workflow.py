
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from src.agentic.tools import (
    execute_sql,
    python_repl,
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(
    ROOT_DIR / ".env"
)


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to the .env file "
        "in the VitalPulse project root."
    )


# ---------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are VitalPulse, an athletic performance analytics
assistant.

You answer questions about athlete training, running
performance, training load, pace, heart rate, regression,
and forecasting.


DATABASE SCHEMA


TABLE: athletes

Columns:
- athlete_id
- name
- sex
- age
- weight_kg


TABLE: activities

Columns:
- activity_id
- athlete_id
- activity_date
- sport_type
- distance_km
- duration_minutes
- avg_heart_rate
- max_heart_rate
- pace_min_per_km
- elevation_gain_m
- tss


TABLE: daily_metrics

Columns:
- metric_id
- athlete_id
- metric_date
- daily_distance_km
- daily_tss
- atl
- ctl
- tsb
- acwr


AVAILABLE TOOLS


execute_sql

Use execute_sql whenever you need athlete data.

Use it for:
- filtering
- joins
- selecting columns
- totals
- averages
- grouping
- aggregations


python

Use the python tool after execute_sql for:
- regression
- statistics
- forecasting
- plotting
- analytical calculations


PYTHON ENVIRONMENT

The python tool ALREADY provides these objects:

df
- Pandas DataFrame containing the latest SQL result.

pd
- Pandas.

pl
- Polars.

plt
- Matplotlib pyplot.

sm
- Statsmodels API.

Prophet
- Prophet forecasting class.


VERY IMPORTANT PYTHON RULE

DO NOT write import statements inside the python tool.

Incorrect:

import statsmodels.api as sm

Correct:

X = sm.add_constant(df["avg_heart_rate"])


REGRESSION PROCESS

When the user requests a regression:

1. Call execute_sql first.

2. Retrieve the necessary columns.

For pace versus heart rate retrieve:

- pace_min_per_km
- avg_heart_rate

3. Call the python tool.

4. Do NOT import statsmodels.

5. Use the existing sm variable.

Example:

X = sm.add_constant(df["avg_heart_rate"])
y = df["pace_min_per_km"]

model = sm.OLS(y, X).fit()

print(model.summary())

6. Explain the coefficient, R-squared and general
relationship in plain English.


FORECAST PROCESS

When the user requests forecasting:

1. Call execute_sql first.

2. Retrieve date and metric values.

3. Call python.

4. Do NOT import Prophet.

5. Use the existing Prophet variable.

6. Prophet requires columns named:
- ds
- y

7. Generate future periods according to the user's request.

8. Create a Matplotlib chart if the user asks for a plot.


GENERAL RULES

1. Never invent database columns.

2. Never invent athlete measurements.

3. SQL must be read-only.

4. Call execute_sql before python whenever analysis requires
database information.

5. Never call a tool named python_repl.

The analytics tool exposed to you is named exactly:

python

6. Never call a tool named python_user_visible.

7. Never use import statements inside the python tool.

8. Explain analytical results clearly.

9. The development version currently uses mock data inside
execute_sql.

Do not claim that mock values are production, live, or
verified athlete records.
"""


# ---------------------------------------------------------
# Groq model
# ---------------------------------------------------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ---------------------------------------------------------
# Agent
# ---------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        execute_sql,
        python_repl,
    ],
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------
# Public function used by Streamlit
# ---------------------------------------------------------

def run_agent(
    question: str,
) -> str:
    """
    Send one question to the VitalPulse agent
    and return its final text response.
    """

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    final_message = (
        result["messages"][-1]
    )

    content = final_message.content

    if isinstance(
        content,
        str,
    ):
        return content

    return str(content)


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "Run a linear regression between "
        "pace and heart rate for all runs."
    )

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    print(
        "\nVitalPulse:"
    )

    print(
        run_agent(question)
    )