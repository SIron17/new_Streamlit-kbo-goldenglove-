"""Run baseline model inference for a given year."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/baseline_model.pkl")
DEFAULT_FEATURES_PATH = Path("models/baseline_features.json")
DEFAULT_INPUT_PATH = Path("data/processed/train_table.parquet")
DEFAULT_OUTPUT_DIR = Path("predictions")

DISPLAY_STATS = [
    "stat_H",
    "stat_HR",
    "stat_RBI",
    "stat_R",
    "stat_WAR",
    "team_win_pct",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Golden Glove candidates by year")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with args.model_path.open("rb") as f:
        model = pickle.load(f)

    with args.features_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    features: list[str] = meta["features"]

    df = pd.read_parquet(args.input_path)
    pred_df = df[df["year"] == args.year].copy()
    if pred_df.empty:
        raise ValueError(f"No rows found for year={args.year}")

    x_pred = pred_df.reindex(columns=features, fill_value=0).fillna(0.0)
    pred_df["score"] = model.predict_proba(x_pred)[:, 1]
    pred_df["pred_rank"] = (
        pred_df.groupby(["year", "gg_position"])["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    output_cols = ["year", "gg_position", "name", "team", "score", "pred_rank"]
    output_cols += [c for c in DISPLAY_STATS if c in pred_df.columns]

    out = pred_df[output_cols].sort_values(["gg_position", "pred_rank"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"pred_{args.year}.parquet"
    out.to_parquet(out_path, index=False)

    print("[predict] complete")
    print(f"- year: {args.year}")
    print(f"- rows: {len(out)}")
    print(f"- output_path: {out_path}")


if __name__ == "__main__":
    main()
