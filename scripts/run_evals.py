from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize current TARA analysis results from SQLite.")
    parser.add_argument("--db", default="data/tara_results.db", help="SQLite database path")
    parser.add_argument("--out", default="data/outputs/eval_summary.xlsx", help="Excel summary output")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        analyses = pd.read_sql_query("SELECT * FROM analyses", conn)

    verdict_counts = analyses["final_verdict"].value_counts(dropna=False).reset_index()
    verdict_counts.columns = ["Verdict", "Count"]
    by_channel = analyses.groupby("channel", dropna=False).agg(
        videos=("id", "count"),
        avg_overall=("overall_score", "mean"),
        avg_cost=("total_cost_usd", "mean"),
    ).reset_index()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        verdict_counts.to_excel(writer, sheet_name="Verdict Counts", index=False)
        by_channel.to_excel(writer, sheet_name="By Channel", index=False)
        analyses.to_excel(writer, sheet_name="Raw Analyses", index=False)
    print(f"Eval summary written: {out}")


if __name__ == "__main__":
    main()
