"""CLI stub for model training."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/processed/dataset.parquet")
DEFAULT_MODEL_PATH = Path("results/model.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Golden Glove prediction model (scaffold stub)."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Input training dataset path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output model artifact path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("[train_model] scaffold command")
    print(f"- dataset_path: {args.dataset_path}")
    print(f"- model_path: {args.model_path}")
    print(f"- random_state: {args.random_state}")
    print("Training logic will be implemented in a follow-up PR.")


if __name__ == "__main__":
    main()
