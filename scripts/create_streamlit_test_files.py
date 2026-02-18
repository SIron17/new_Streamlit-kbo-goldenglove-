"""Create Streamlit upload test CSV files from existing raw datasets (year-based)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/sample")

BATTING_FILE = "kbo_batting_stats_by_season_1982-2025.csv"
PITCHING_FILE = "kbo_pitching_stats_by_season_1982-2025.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Streamlit upload test files from raw KBO stats."
    )
    parser.add_argument("--year", type=int, default=2025, help="Target season year")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def _filter_year_rows(input_path: Path, year: int) -> tuple[list[str], list[dict[str, str]]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found: {input_path}")
        rows = [row for row in reader if str(row.get("Year", "")).strip() == str(year)]
        return reader.fieldnames, rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()

    batting_path = args.raw_dir / BATTING_FILE
    pitching_path = args.raw_dir / PITCHING_FILE

    batting_fields, batting_rows = _filter_year_rows(batting_path, args.year)
    pitching_fields, pitching_rows = _filter_year_rows(pitching_path, args.year)

    hitters_out = args.out_dir / f"streamlit_hitter_{args.year}.csv"
    pitchers_out = args.out_dir / f"streamlit_pitcher_{args.year}.csv"

    _write_csv(hitters_out, batting_fields, batting_rows)
    _write_csv(pitchers_out, pitching_fields, pitching_rows)

    print("[create_streamlit_test_files] complete")
    print(f"- year: {args.year}")
    print(f"- hitters: {hitters_out} ({len(batting_rows)} rows)")
    print(f"- pitchers: {pitchers_out} ({len(pitching_rows)} rows)")


if __name__ == "__main__":
    main()
