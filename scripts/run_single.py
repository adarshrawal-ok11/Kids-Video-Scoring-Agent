from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import analyze_video
from src.tools.persist import export_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one YouTube video with the TARA v2 pipeline.")
    parser.add_argument("url", help="YouTube URL to analyze")
    parser.add_argument("--no-db", action="store_true", help="Do not write the analysis to SQLite")
    parser.add_argument("--excel", action="store_true", help="Export an Excel workbook for this run")
    parser.add_argument("--json-out", help="Optional path to write the full state JSON")
    args = parser.parse_args()

    state = analyze_video(args.url, persist=not args.no_db)
    if args.excel:
        path = export_excel([state])
        print(f"Excel written: {path}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        print(f"JSON written: {out}")

    print(json.dumps({
        "url": state.url,
        "title": state.metadata.title if state.metadata else None,
        "verdict": state.final_verdict.verdict if state.final_verdict else None,
        "overall_score": state.final_verdict.overall_score if state.final_verdict else None,
        "errors": len(state.errors),
        "cost_usd": round(state.cost.total, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
