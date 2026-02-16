"""
Build 1: Refactoring Functions and Simple LLM Assistant (Assignment 2)

This script refactors Build 0 functions into separate modules (src/)
and adds two LLM API calls:
  1) Summarize the dataset column names and data types
  2) Suggest research questions based on the columns and data types

Completed by Holt Young
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import refactored modules from src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utilities import ensure_dirs, read_data
from src.profiling import basic_profile, split_columns
from src.summaries import summarize_numeric, summarize_categorical
from src.analysis import missingness_table, multiple_linear_regression, correlations
from src.plots import (
    plot_missingness,
    plot_corr_heatmap,
    plot_histograms,
    plot_bar_charts,
)
from src.checks import assert_json_safe, target_check


# -----------------------------
# LLM Assistant Functions
# -----------------------------


def get_llm() -> ChatOpenAI:
    """Initialize the OpenAI LLM via LangChain."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Make sure you have a .env file with your API key."
        )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=api_key,
    )
    return llm


def summarize_columns(llm: ChatOpenAI, profile: dict) -> str:
    """
    Ask the LLM to summarize the dataset column names and data types.

    Args:
        llm: The LangChain ChatOpenAI instance
        profile: The data profile dictionary from basic_profile()

    Returns:
        A string summary from the LLM
    """
    columns_info = ""
    for col_name, dtype in profile["dtypes"].items():
        columns_info += f"  - {col_name}: {dtype}\n"

    messages = [
        SystemMessage(
            content=(
                "You are a helpful data analysis assistant. "
                "Given a dataset's column names and data types, "
                "provide a clear and concise summary of the dataset structure. "
                "Describe what each column likely represents and how the "
                "data types relate to the kind of analysis that could be done."
            )
        ),
        HumanMessage(
            content=(
                f"Here is a dataset with {profile['n_rows']} rows "
                f"and {profile['n_cols']} columns.\n\n"
                f"Column names and data types:\n{columns_info}\n"
                f"Total missing values: {profile['n_missing_total']}\n\n"
                "Please summarize this dataset's structure and what each "
                "column likely represents."
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def suggest_research_questions(llm: ChatOpenAI, profile: dict) -> str:
    """
    Ask the LLM to suggest research questions based on the column names
    and data types.

    Args:
        llm: The LangChain ChatOpenAI instance
        profile: The data profile dictionary from basic_profile()

    Returns:
        A string with suggested research questions from the LLM
    """
    columns_info = ""
    for col_name, dtype in profile["dtypes"].items():
        columns_info += f"  - {col_name}: {dtype}\n"

    messages = [
        SystemMessage(
            content=(
                "You are a helpful data analysis assistant. "
                "Given a dataset's column names and data types, "
                "suggest meaningful research questions that could be "
                "addressed using this data. Consider relationships between "
                "variables, group comparisons, and predictive modeling."
            )
        ),
        HumanMessage(
            content=(
                f"Here is a dataset with {profile['n_rows']} rows "
                f"and {profile['n_cols']} columns.\n\n"
                f"Column names and data types:\n{columns_info}\n"
                f"Total missing values: {profile['n_missing_total']}\n\n"
                "Please suggest 5 research questions that could be "
                "explored with this dataset."
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content


# -----------------------------
# Main pipeline
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build 1: Data Analysis Pipeline with LLM Assistant"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default=None, help="Target column")
    parser.add_argument(
        "--outcome", type=str, default=None, help="Outcome for regression"
    )
    parser.add_argument(
        "--predictors",
        type=str,
        default=None,
        help="Comma-separated predictors for regression",
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Output directory"
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    # --- Data Loading and Profiling ---
    print("Loading data...")
    df = read_data(Path(args.data))
    numeric_cols, cat_cols = split_columns(df)

    print("Profiling data...")
    profile = basic_profile(df)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    # --- Save Reports ---
    print("Saving reports...")
    (report_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    # --- Generate Plots ---
    print("Generating plots...")
    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    # --- Target Check ---
    if args.target:
        target_info = target_check(df, args.target)
        (report_dir / "target_overview.json").write_text(
            json.dumps(target_info, indent=2)
        )

    # --- Regression ---
    if args.outcome:
        preds: Optional[List[str]] = None
        if args.predictors:
            preds = [p.strip() for p in args.predictors.split(",") if p.strip()]

        reg_results = multiple_linear_regression(
            df, outcome=args.outcome, predictors=preds
        )
        assert_json_safe(reg_results, context="multiple_linear_regression output")
        (report_dir / "regression_results.json").write_text(
            json.dumps(reg_results, indent=2)
        )

    # --- LLM Assistant ---
    print("\nInitializing LLM assistant...")
    llm = get_llm()

    print("Asking LLM to summarize dataset columns and data types...")
    column_summary = summarize_columns(llm, profile)
    print("\n--- LLM Column Summary ---")
    print(column_summary)

    # Save the column summary
    (report_dir / "llm_column_summary.txt").write_text(column_summary)

    print("\nAsking LLM to suggest research questions...")
    research_questions = suggest_research_questions(llm, profile)
    print("\n--- LLM Research Questions ---")
    print(research_questions)

    # Save the research questions
    (report_dir / "llm_research_questions.txt").write_text(research_questions)

    print(f"\nBuild 1 pipeline complete. Outputs saved to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
