# 🚴 VitalPulse: Agentic Fitness & Performance Analytics

VitalPulse is an end-to-end, repository-based agentic AI analytics platform designed to track athletic performance, training stress (CTL, ATL, TSB, ACWR), and recovery metrics using Strava-style data. 

Powered by **LangGraph**, **Groq (Llama-3.3-70b)**, **Polars**, **PostgreSQL**, and **Streamlit**, VitalPulse allows athletes and coaches to query complex endurance data in natural language, automatically execute SQL extractions, and run statistical models and forecasts via an isolated Python execution environment.

---

## 🏗️ System Architecture & Workflow

VitalPulse is built around a decoupled multi-track architecture:
1. **Data Engine (Track A):** Synthetic endurance data generation, time-series transformations (EWMA fatigue models), and high-performance ingestion into PostgreSQL via Polars.
2. **Agentic Interface (Track B):** A ReAct agent loop that orchestrates deterministic tool calls (`execute_sql` and `python_repl`) to inspect database schemas, cache tabular data to Parquet, and render predictive charts using Prophet and Matplotlib.

```text
 User Prompt (Streamlit UI) 
       │
       ▼
 LangGraph ReAct Agent (Groq / Llama-3.3) 
       ├───────────────────────────────┐
       ▼                               ▼
 [execute_sql Tool]              [python_repl Tool]
       │                               │
       ▼                               ▼
 PostgreSQL Database             statsmodels / Prophet
 (athletes, activities)          (Parquet Cache → Matplotlib)
       │                               │
       └──────────────┬────────────────┘
                      ▼
            Streamlit Chat Interface 
            (Markdown Tables & Plots)