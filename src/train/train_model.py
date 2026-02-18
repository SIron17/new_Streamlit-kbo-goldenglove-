"""Train baseline model and evaluate holdout year performance."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_TRAIN_TABLE = Path("data/processed/train_table.parquet")
DEFAULT_MODEL_PATH = Path("models/baseline_model.pkl")
DEFAULT_FEATURES_PATH = Path("models/baseline_features.json")
DEFAULT_LEADERBOARD_PATH = Path("results/leaderboard.csv")

NON_FEATURE_COLUMNS = {
    "label",
    "year",
    "team",
    "player_id",
    "name",
    "name_norm",
    "gg_position",
    "winner_name",
    "winner_name_norm",
}

FORBIDDEN_FEATURE_TOKENS = ("league", "post", "ks")
STRONG_RANK_FLAG_PREFIXES = ("is_year_top", "is_pos_year_top")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline Golden Glove model")
    parser.add_argument("--train-table", type=Path, default=DEFAULT_TRAIN_TABLE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--leaderboard-path", type=Path, default=DEFAULT_LEADERBOARD_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-rank-flags",
        action="store_true",
        help="Use strong rank-like flags(is_year_top*/is_pos_year_top*) as features.",
    )
    parser.add_argument(
        "--backtest-years",
        type=int,
        default=3,
        help="Run rolling holdout backtest for recent N years (diagnostic only).",
    )
    return parser


def select_features(df: pd.DataFrame, include_rank_flags: bool) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features: list[str] = []
    for col in numeric_cols:
        low = col.lower()
        if col in NON_FEATURE_COLUMNS:
            continue
        if any(tok in low for tok in FORBIDDEN_FEATURE_TOKENS):
            continue
        if not include_rank_flags and col.startswith(STRONG_RANK_FLAG_PREFIXES):
            continue
        features.append(col)
    return sorted(features)


def compute_group_metrics(test_df: pd.DataFrame) -> tuple[float, float, float]:
    group_results: list[dict[str, float]] = []

    for (_, gg_position), grp in test_df.groupby(["year", "gg_position"], sort=True):
        ranked = grp.sort_values("score", ascending=False).reset_index(drop=True)
        ranked["pred_rank"] = np.arange(1, len(ranked) + 1)
        winner_ranks = ranked.loc[ranked["label"] == 1, "pred_rank"].tolist()

        if gg_position == "Outfielders":
            hit = 1.0 if any(r <= 3 for r in winner_ranks) else 0.0
            best_rank = min(winner_ranks) if winner_ranks else np.inf
            group_results.append(
                {
                    "is_of": 1.0,
                    "hit": hit,
                    "mrr": 0.0 if best_rank == np.inf else 1.0 / best_rank,
                }
            )
        else:
            hit = 1.0 if any(r == 1 for r in winner_ranks) else 0.0
            best_rank = min(winner_ranks) if winner_ranks else np.inf
            group_results.append(
                {
                    "is_of": 0.0,
                    "hit": hit,
                    "mrr": 0.0 if best_rank == np.inf else 1.0 / best_rank,
                }
            )

    if not group_results:
        return 0.0, 0.0, 0.0

    result_df = pd.DataFrame(group_results)
    top1_rate = result_df.loc[result_df["is_of"] == 0.0, "hit"].mean()
    of_top3_rate = result_df.loc[result_df["is_of"] == 1.0, "hit"].mean()
    mrr = result_df["mrr"].mean()

    return float(top1_rate if not np.isnan(top1_rate) else 0.0), float(
        of_top3_rate if not np.isnan(of_top3_rate) else 0.0
    ), float(mrr if not np.isnan(mrr) else 0.0)


def evaluate_one_year(
    df: pd.DataFrame,
    holdout_year: int,
    features: list[str],
    random_state: int,
) -> dict[str, float]:
    train_df = df[df["year"] < holdout_year].copy()
    test_df = df[df["year"] == holdout_year].copy()
    if train_df.empty or test_df.empty:
        return {}

    x_train = train_df[features].fillna(0.0)
    y_train = train_df["label"].astype(int)
    x_test = test_df[features].fillna(0.0)
    y_test = test_df["label"].astype(int)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary",
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    test_df["score"] = model.predict_proba(x_test)[:, 1]

    top1_rate, of_top3_rate, mrr = compute_group_metrics(test_df)
    auc = float(roc_auc_score(y_test, test_df["score"])) if y_test.nunique() > 1 else 0.0
    pr_auc = (
        float(average_precision_score(y_test, test_df["score"]))
        if y_test.nunique() > 1
        else 0.0
    )
    return {
        "holdout_year": float(holdout_year),
        "Top1_rate": top1_rate,
        "OF_Top3_rate": of_top3_rate,
        "MRR": mrr,
        "AUC": auc,
        "PR_AUC": pr_auc,
    }


def append_leaderboard(
    leaderboard_path: Path,
    model_name: str,
    holdout_year: int,
    top1_rate: float,
    of_top3_rate: float,
    mrr: float,
    auc: float,
    pr_auc: float,
    feature_set_notes: str,
) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_name": model_name,
                "holdout_year": holdout_year,
                "Top1_rate": top1_rate,
                "OF_Top3_rate": of_top3_rate,
                "MRR": mrr,
                "AUC": auc,
                "PR_AUC": pr_auc,
                "feature_set_notes": feature_set_notes,
            }
        ]
    )

    if leaderboard_path.exists():
        prev = pd.read_csv(leaderboard_path)
        out = pd.concat([prev, row], ignore_index=True)
    else:
        out = row
    out.to_csv(leaderboard_path, index=False)


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_parquet(args.train_table)
    holdout_year = int(df["year"].max())

    features = select_features(df, include_rank_flags=args.include_rank_flags)
    if not features:
        raise RuntimeError("No numeric features found for training.")

    result = evaluate_one_year(df, holdout_year, features, args.random_state)
    if not result:
        raise RuntimeError("Train/test split is empty.")

    # train final model (same split) for artifact save
    train_df = df[df["year"] < holdout_year].copy()
    test_df = df[df["year"] == holdout_year].copy()

    x_train = train_df[features].fillna(0.0)
    y_train = train_df["label"].astype(int)
    x_test = test_df[features].fillna(0.0)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary",
        random_state=args.random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    with args.model_path.open("wb") as f:
        pickle.dump(model, f)

    args.features_path.parent.mkdir(parents=True, exist_ok=True)
    with args.features_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": features,
                "holdout_year": holdout_year,
                "include_rank_flags": args.include_rank_flags,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    recent_years = sorted(df["year"].unique())[-args.backtest_years :]
    bt_rows = []
    for y in recent_years:
        y = int(y)
        if y == int(df["year"].min()):
            continue
        r = evaluate_one_year(df, y, features, args.random_state)
        if r:
            bt_rows.append(r)

    feature_notes = f"n_features={len(features)};include_rank_flags={args.include_rank_flags}"
    append_leaderboard(
        leaderboard_path=args.leaderboard_path,
        model_name="lightgbm_baseline",
        holdout_year=holdout_year,
        top1_rate=result["Top1_rate"],
        of_top3_rate=result["OF_Top3_rate"],
        mrr=result["MRR"],
        auc=result["AUC"],
        pr_auc=result["PR_AUC"],
        feature_set_notes=feature_notes,
    )

    test_df["score"] = model.predict_proba(x_test)[:, 1]

    print("[train_model] complete")
    print(f"- holdout_year: {holdout_year}")
    print(f"- train_rows: {len(train_df)} / test_rows: {len(test_df)}")
    print(f"- n_features: {len(features)}")
    print(f"- include_rank_flags: {args.include_rank_flags}")
    print(f"- Top1_rate: {result['Top1_rate']:.4f}")
    print(f"- OF_Top3_rate: {result['OF_Top3_rate']:.4f}")
    print(f"- MRR: {result['MRR']:.4f}")
    print(f"- AUC: {result['AUC']:.4f}")
    print(f"- PR_AUC: {result['PR_AUC']:.4f}")
    if bt_rows:
        bt = pd.DataFrame(bt_rows)
        print(f"- backtest_years: {recent_years}")
        print(
            "- backtest_mean: "
            f"Top1={bt['Top1_rate'].mean():.4f}, OF_Top3={bt['OF_Top3_rate'].mean():.4f}, "
            f"MRR={bt['MRR'].mean():.4f}, AUC={bt['AUC'].mean():.4f}, PR_AUC={bt['PR_AUC'].mean():.4f}"
        )
    print(f"- model_path: {args.model_path}")
    print(f"- features_path: {args.features_path}")
    print(f"- leaderboard_path: {args.leaderboard_path}")


if __name__ == "__main__":
    main()
