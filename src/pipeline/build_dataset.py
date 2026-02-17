"""CLI stub for dataset building."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/dataset.parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a modeling dataset for KBO Golden Glove prediction (scaffold stub)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw data files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output dataset path.",
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=1984,
        help="First season to include.",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=2025,
        help="Last season to include.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print plan only.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("[build_dataset] scaffold command")
    print(f"- raw_dir: {args.raw_dir}")
    print(f"- output_path: {args.output_path}")
    print(f"- season_range: {args.season_start}-{args.season_end}")
    print(f"- dry_run: {args.dry_run}")
    print("Dataset build logic will be implemented in a follow-up PR.")


if __name__ == "__main__":
    main()
