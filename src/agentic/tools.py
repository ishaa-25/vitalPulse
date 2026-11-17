import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import statsmodels.api as sm
from langchain_core.tools import tool
from prophet import Prophet


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT_DIR / "src" / "data" / "processed"

LATEST_QUERY = PROCESSED_DIR / "latest_query.parquet"
LATEST_PLOT = PROCESSED_DIR / "latest_plot.png"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Helper: Polars -> Markdown
# ---------------------------------------------------------

def dataframe_to_markdown(df: pl.DataFrame) -> str:
    """
    Convert a small Polars DataFrame to a Markdown table.
    """

    if df.is_empty():
        return "No rows returned."

    columns = df.columns

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for row in df.iter_rows():
        rows.append(
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
        )

    return "\n".join(
        [header, separator, *rows]
    )


# ---------------------------------------------------------
# SQL Tool
# ---------------------------------------------------------

@tool
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against VitalPulse data.

    Development mode currently uses mock athlete data until
    the PostgreSQL pipeline from the data-engine branch is merged.

    The complete result is saved to:
    src/data/processed/latest_query.parquet
    """

    print("\n--- execute_sql called ---")
    print(query)

    normalized = query.strip().lower()

    forbidden_commands = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
        "grant ",
        "revoke ",
    ]

    if any(
        command in normalized
        for command in forbidden_commands
    ):
        return (
            "SQL rejected. "
            "VitalPulse allows read-only queries only."
        )

    # -----------------------------------------------------
    # MOCK DATA
    #
    # This section will later be replaced by PostgreSQL.
    # -----------------------------------------------------

    mock_df = pl.DataFrame(
        {
            "activity_date": [
                "2026-10-20",
                "2026-10-25",
                "2026-11-01",
                "2026-11-05",
                "2026-11-10",
                "2026-11-12",
                "2026-11-14",
                "2026-11-15",
            ],

            "athlete_name": [
                "Elena Vance",
                "Marcus Kane",
                "Elena Vance",
                "Marcus Kane",
                "Elena Vance",
                "Marcus Kane",
                "Elena Vance",
                "Marcus Kane",
            ],

            "sport_type": [
                "Run",
                "Run",
                "Run",
                "Run",
                "Run",
                "Run",
                "Run",
                "Run",
            ],

            "distance_km": [
                8.2,
                12.5,
                10.1,
                7.8,
                6.4,
                14.0,
                9.3,
                11.2,
            ],

            "avg_heart_rate": [
                148,
                152,
                145,
                150,
                143,
                155,
                147,
                151,
            ],

            "pace_min_per_km": [
                5.2,
                5.0,
                5.3,
                5.1,
                5.4,
                4.9,
                5.15,
                5.05,
            ],

            "tss": [
                55.0,
                72.0,
                61.0,
                50.0,
                42.0,
                80.0,
                58.0,
                67.0,
            ],
        }
    )

    mock_df.write_parquet(
        LATEST_QUERY
    )

    preview = mock_df.head(10)

    return (
        "Development-mode query completed.\n\n"
        f"SQL received:\n{query}\n\n"
        f"Rows available: {mock_df.height}\n\n"
        "Result preview:\n\n"
        f"{dataframe_to_markdown(preview)}\n\n"
        "The complete result is available to the Python "
        "analytics tool through latest_query.parquet."
    )


# ---------------------------------------------------------
# Python code validation
# ---------------------------------------------------------

BLOCKED_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


def validate_python_code(code: str) -> None:
    """
    Reject imports and unsafe Python operations.

    Analytics libraries are already injected into the tool,
    so model-generated code does not need imports.
    """

    if "__" in code:
        raise ValueError(
            "Dunder access is not allowed."
        )

    tree = ast.parse(code)

    for node in ast.walk(tree):

        if isinstance(
            node,
            (ast.Import, ast.ImportFrom),
        ):
            raise ValueError(
                "Imports are not allowed. "
                "Use the preloaded variables: "
                "df, pd, pl, plt, sm, Prophet."
            )

        if isinstance(node, ast.Call):

            if isinstance(
                node.func,
                ast.Name,
            ):
                if node.func.id in BLOCKED_NAMES:
                    raise ValueError(
                        f"{node.func.id} is not allowed."
                    )


# ---------------------------------------------------------
# Python analytics tool
# IMPORTANT:
# Groq sees this tool as "python"
# ---------------------------------------------------------

@tool("python")
def python_repl(code: str) -> str:
    """
    Run restricted Python analytics on the latest SQL result.

    Do NOT import packages.

    Already available:

    df
        Pandas DataFrame containing the latest SQL result.

    pd
        Pandas.

    pl
        Polars.

    plt
        Matplotlib pyplot.

    sm
        Statsmodels API.

    Prophet
        Prophet forecasting class.

    Use this tool for:
    - linear regression
    - statistics
    - forecasting
    - plotting
    - analytical calculations

    Any Matplotlib figure will automatically be saved to:
    src/data/processed/latest_plot.png
    """

    print("\n--- python tool called ---")
    print(code)

    if not LATEST_QUERY.exists():
        return (
            "No cached SQL result exists. "
            "Call execute_sql before using Python."
        )

    try:
        validate_python_code(code)

    except Exception as exc:
        return (
            f"Python code rejected: {exc}"
        )

    polars_df = pl.read_parquet(
        LATEST_QUERY
    )

    # Make dates more useful for Prophet/time-series work.
    if "activity_date" in polars_df.columns:
        try:
            polars_df = polars_df.with_columns(
                pl.col("activity_date")
                .str.to_date()
            )
        except Exception:
            pass

    df = polars_df.to_pandas()

    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
        "round": round,
        "abs": abs,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "sorted": sorted,
    }

    analysis_namespace = {
        "__builtins__": safe_builtins,
        "df": df,
        "pd": pd,
        "pl": pl,
        "plt": plt,
        "sm": sm,
        "Prophet": Prophet,
    }

    output_buffer = io.StringIO()

    try:
        # Remove previous plot so the Streamlit UI
        # does not display a stale analysis.
        if LATEST_PLOT.exists():
            LATEST_PLOT.unlink()

        plt.close("all")

        with redirect_stdout(
            output_buffer
        ):
            exec(
                code,
                analysis_namespace,
                analysis_namespace,
            )

        output = (
            output_buffer
            .getvalue()
            .strip()
        )

        if plt.get_fignums():

            plt.tight_layout()

            plt.savefig(
                LATEST_PLOT,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close("all")

        response = (
            "Python analysis completed successfully."
        )

        if output:
            response += (
                "\n\nOutput:\n"
                + output
            )

        if LATEST_PLOT.exists():
            response += (
                "\n\nA visualization was generated "
                "and saved to latest_plot.png."
            )

        return response

    except Exception as exc:

        plt.close("all")

        return (
            "Python analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# Local tests
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\n=== TOOL NAMES ==="
    )

    print(
        "SQL tool:",
        execute_sql.name,
    )

    print(
        "Python tool:",
        python_repl.name,
    )

    print(
        "\n=== SQL TEST ==="
    )

    sql_result = execute_sql.invoke(
        {
            "query": """
            SELECT
                pace_min_per_km,
                avg_heart_rate
            FROM activities
            WHERE sport_type = 'Run';
            """
        }
    )

    print(sql_result)

    print(
        "\n=== REGRESSION TEST ==="
    )

    python_result = python_repl.invoke(
        {
            "code": """
X = sm.add_constant(df['avg_heart_rate'])
y = df['pace_min_per_km']

model = sm.OLS(y, X).fit()

print(model.summary())
"""
        }
    )

    print(python_result)