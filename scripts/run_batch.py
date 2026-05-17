from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import analyze_video
from src.tools.persist import export_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a batch of YouTube URLs.")
    parser.add_argument("url_file", help="Text file with one YouTube URL per line")
    parser.add_argument("--json-out", help="Optional path to write full batch states JSON")
    args = parser.parse_args()

    urls = [
        line.strip()
        for line in Path(args.url_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    states = [analyze_video(url, persist=True) for url in urls]
    excel_path = export_excel(states)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([json.loads(s.model_dump_json()) for s in states], indent=2), encoding="utf-8")
        print(f"JSON written: {out}")

    print(f"Analyzed {len(states)} videos")
    print(f"Excel written: {excel_path}")


if __name__ == "__main__":
    main()
