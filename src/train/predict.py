"""CLI stub for model inference."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_PATH = Path("results/model.txt")
DEFAULT_INPUT_PATH = Path("data/processed/dataset.parquet")
DEFAULT_OUTPUT_PATH = Path("predictions/predictions.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Golden Glove winner prediction (scaffold stub)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained model artifact path.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input feature dataset path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Prediction output csv path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("[predict] scaffold command")
    print(f"- model_path: {args.model_path}")
    print(f"- input_path: {args.input_path}")
    print(f"- output_path: {args.output_path}")
    print("Prediction logic will be implemented in a follow-up PR.")


if __name__ == "__main__":
    main()
