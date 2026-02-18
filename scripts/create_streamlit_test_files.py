"""Create Streamlit upload test files for a target year.

기본값은 train_table 기반(모델 피처 정합)으로 생성하여
`python -m src.train.predict --year YYYY` 결과와 높은 일관성을 보장합니다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_PATH = Path("data/processed/train_table.parquet")
DEFAULT_OUT_DIR = Path("data/sample")

BATTING_FILE = "kbo_batting_stats_by_season_1982-2025.csv"
PITCHING_FILE = "kbo_pitching_stats_by_season_1982-2025.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Streamlit upload test files from processed/raw KBO stats."
    )
    parser.add_argument("--year", type=int, default=2025, help="Target season year")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--source",
        choices=["processed", "raw"],
        default="processed",
        help="processed(train_table) is recommended for predict parity.",
    )
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
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


def _from_raw(args: argparse.Namespace) -> tuple[Path, int, Path, int]:
    batting_path = args.raw_dir / BATTING_FILE
    pitching_path = args.raw_dir / PITCHING_FILE

    batting_fields, batting_rows = _filter_year_rows(batting_path, args.year)
    pitching_fields, pitching_rows = _filter_year_rows(pitching_path, args.year)

    hitters_out = args.out_dir / f"streamlit_hitter_{args.year}.csv"
    pitchers_out = args.out_dir / f"streamlit_pitcher_{args.year}.csv"

    _write_csv(hitters_out, batting_fields, batting_rows)
    _write_csv(pitchers_out, pitching_fields, pitching_rows)
    return hitters_out, len(batting_rows), pitchers_out, len(pitching_rows)


def _from_processed(args: argparse.Namespace) -> tuple[Path, int, Path, int]:
    df = pd.read_parquet(args.processed_path)
    df = df[df["year"] == args.year].copy()
    if df.empty:
        raise ValueError(f"No rows in processed train_table for year={args.year}")

    hitters = df[df["gg_position"] != "P"].copy()
    pitchers = df[df["gg_position"] == "P"].copy()

    drop_cols = [c for c in ["label"] if c in df.columns]
    hitters = hitters.drop(columns=[c for c in drop_cols if c in hitters.columns])
    pitchers = pitchers.drop(columns=[c for c in drop_cols if c in pitchers.columns])

    hitters_out = args.out_dir / f"streamlit_hitter_{args.year}.csv"
    pitchers_out = args.out_dir / f"streamlit_pitcher_{args.year}.csv"

    hitters.to_csv(hitters_out, index=False, encoding="utf-8-sig")
    pitchers.to_csv(pitchers_out, index=False, encoding="utf-8-sig")
    return hitters_out, len(hitters), pitchers_out, len(pitchers)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "processed":
        hitters_out, hitter_n, pitchers_out, pitcher_n = _from_processed(args)
    else:
        hitters_out, hitter_n, pitchers_out, pitcher_n = _from_raw(args)

    print("[create_streamlit_test_files] complete")
    print(f"- source: {args.source}")
    print(f"- year: {args.year}")
    print(f"- hitters: {hitters_out} ({hitter_n} rows)")
    print(f"- pitchers: {pitchers_out} ({pitcher_n} rows)")


if __name__ == "__main__":
    main()
