"""Build dataset artifacts for KBO Golden Glove prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")

BATTING_FILE = "kbo_batting_stats_by_season_1982-2025.csv"
PITCHING_FILE = "kbo_pitching_stats_by_season_1982-2025.csv"
GOLDEN_GLOVE_FILE = "kbo_golden_glove_1984_2025.xlsx"
RANKING_FILE = "kbo_ranking_1984_2025.xlsx"

GG_POSITIONS = ["P", "C", "1B", "2B", "3B", "SS", "Outfielders", "DH"]

TEAM_NAME_MAP = {
    "OB": "두산",
    "MBC": "LG",
    "빙그레": "한화",
    "해태": "KIA",
    "태평양": "현대",
    "우리": "키움",
    "넥센": "키움",
    "히어로즈": "키움",
    "SK": "SSG",
}

POSITION_MAP = {
    "P": "P",
    "RP": "P",
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "Outfielders",
    "CF": "Outfielders",
    "RF": "Outfielders",
    "OF": "Outfielders",
    "DH": "DH",
}

NUMERIC_COLS_BATTING = [
    "G",
    "PA",
    "AB",
    "R",
    "H",
    "2B",
    "3B",
    "HR",
    "TB",
    "RBI",
    "SB",
    "CS",
    "BB",
    "HP",
    "IB",
    "SO",
    "GDP",
    "SH",
    "SF",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "wRC+",
    "WAR",
]

NUMERIC_COLS_PITCHING = [
    "G",
    "GS",
    "GR",
    "GF",
    "CG",
    "SHO",
    "W",
    "L",
    "S",
    "HD",
    "IP",
    "ER",
    "R",
    "TBF",
    "H",
    "2B",
    "3B",
    "HR",
    "BB",
    "HP",
    "IB",
    "SO",
    "ERA",
    "RA9",
    "FIP",
    "WHIP",
    "WAR",
]

KEY_NARRATIVE_METRICS = ["H", "HR", "RBI", "R"]
KEY_INTERACTION_METRICS = ["H", "HR", "RBI", "WAR"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed parquet tables for KBO Golden Glove modeling."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--season-start", type=int, default=1984)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _require_pandas():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas/numpy 가 필요합니다. `pip install -r requirements.txt`를 실행하세요."
        ) from exc
    return pd, np


def _standardize_team(s):
    return s.astype(str).str.strip().replace(TEAM_NAME_MAP)


def _coerce_numeric(df, cols, pd):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _topk_flags(
    pd,
    df,
    group_cols: list[str],
    metric: str,
    ks: Iterable[int],
    prefix: str,
) -> None:
    if metric not in df.columns:
        return
    series = pd.to_numeric(df[metric], errors="coerce")
    for k in ks:
        col = f"{prefix}top{k}_{metric}"
        if group_cols:
            ranks = series.groupby([df[g] for g in group_cols]).rank(
                method="min", ascending=False
            )
        else:
            ranks = series.groupby(df["year"]).rank(method="min", ascending=False)
        df[col] = ((ranks <= k) & series.notna()).astype("int8")


def _find_col(df, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column. candidates={candidates}")


def load_season_stats(raw_dir: Path, season_start: int, season_end: int):
    pd, np = _require_pandas()

    batting = pd.read_csv(raw_dir / BATTING_FILE)
    pitching = pd.read_csv(raw_dir / PITCHING_FILE)

    batting = batting.rename(
        columns={
            "Id": "player_id",
            "Name": "name",
            "Year": "year",
            "Team": "team",
            "Age": "age",
            "Pos.": "raw_position",
        }
    )
    pitching = pitching.rename(
        columns={
            "Id": "player_id",
            "Name": "name",
            "Year": "year",
            "Team": "team",
            "Age": "age",
            "Pos.": "raw_position",
        }
    )

    batting["source"] = "batting"
    pitching["source"] = "pitching"

    batting["gg_position"] = batting["raw_position"].map(POSITION_MAP)
    pitching["gg_position"] = "P"

    core_cols = ["year", "team", "player_id", "name", "age", "gg_position", "source"]

    for c in NUMERIC_COLS_BATTING:
        if c not in batting.columns:
            batting[c] = np.nan
    for c in NUMERIC_COLS_PITCHING:
        if c not in pitching.columns:
            pitching[c] = np.nan

    batting_metrics = batting[NUMERIC_COLS_BATTING].copy()
    pitching_metrics = pitching[NUMERIC_COLS_PITCHING].copy()

    batting_metrics.columns = [f"metric_{c}" for c in batting_metrics.columns]
    pitching_metrics.columns = [f"metric_{c}" for c in pitching_metrics.columns]

    batting = pd.concat([batting[core_cols], batting_metrics], axis=1)
    pitching = pd.concat([pitching[core_cols], pitching_metrics], axis=1)

    df = pd.concat([batting, pitching], ignore_index=True, sort=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].between(season_start, season_end)]

    df["team"] = _standardize_team(df["team"])
    df["name"] = df["name"].astype(str).str.strip()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_sq"] = df["age"] ** 2

    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    _coerce_numeric(df, metric_cols, pd)

    keep_positions = set(GG_POSITIONS)
    df = df[df["gg_position"].isin(keep_positions)].copy()

    for metric in KEY_NARRATIVE_METRICS:
        col = f"metric_{metric}"
        _topk_flags(pd, df, [], col, [1, 3, 5], prefix="is_year_")
        _topk_flags(
            pd,
            df,
            ["year", "gg_position"],
            col,
            [1, 3, 5],
            prefix="is_pos_year_",
        )

    # YoY and cumulative features
    player_group = df.groupby("player_id", dropna=False, sort=False)
    for metric in ["metric_H", "metric_HR", "metric_RBI", "metric_R", "metric_WAR", "metric_OPS", "metric_ERA"]:
        if metric not in df.columns:
            continue
        prev_col = f"prev_{metric}"
        df[prev_col] = player_group[metric].shift(1)
        df[f"delta_yoy_{metric}"] = df[metric] - df[prev_col]
        df[f"career_sum_prev_{metric}"] = (
            player_group[metric].cumsum() - df[metric].fillna(0)
        )

    return df


def load_team_standings(raw_dir: Path, season_start: int, season_end: int):
    pd, np = _require_pandas()

    rank = pd.read_excel(raw_dir / RANKING_FILE)

    year_col = _find_col(rank, ["연도", "Year", "year"])
    team_col = _find_col(rank, ["팀명", "Team", "team"])
    win_pct_col = _find_col(rank, ["승률", "Win%", "win_pct", "승율"])
    rank_col = _find_col(rank, ["순위", "Rank", "rank"])

    gb_col = None
    for c in ["게임차", "GB", "gb"]:
        if c in rank.columns:
            gb_col = c
            break

    standings = pd.DataFrame(
        {
            "year": pd.to_numeric(rank[year_col], errors="coerce"),
            "team": rank[team_col].astype(str).str.strip(),
            "team_rank": pd.to_numeric(rank[rank_col], errors="coerce"),
            "team_win_pct": pd.to_numeric(rank[win_pct_col], errors="coerce"),
        }
    )

    if gb_col is not None:
        standings["team_gb"] = pd.to_numeric(rank[gb_col], errors="coerce")
    else:
        standings["team_gb"] = np.nan

    standings = standings.dropna(subset=["year", "team", "team_rank", "team_win_pct"])
    standings = standings[standings["team"] != "팀명"].copy()
    standings["year"] = standings["year"].astype("Int64")
    standings = standings[standings["year"].between(season_start, season_end)]

    standings["team"] = _standardize_team(standings["team"])
    standings["n_teams_year"] = standings.groupby("year")["team"].transform("nunique")
    standings["rank_pct_year"] = (standings["n_teams_year"] - standings["team_rank"] + 1) / standings["n_teams_year"]

    mean_win = standings.groupby("year")["team_win_pct"].transform("mean")
    std_win = standings.groupby("year")["team_win_pct"].transform("std").replace(0, np.nan)
    standings["win_pct_z_year"] = ((standings["team_win_pct"] - mean_win) / std_win).fillna(0.0)

    return standings


def load_labels(raw_dir: Path, season_start: int, season_end: int):
    pd, _ = _require_pandas()

    gg = pd.read_excel(raw_dir / GOLDEN_GLOVE_FILE)
    year_col = _find_col(gg, ["Year", "연도", "year"])

    missing_positions = [p for p in GG_POSITIONS if p not in gg.columns]
    if missing_positions:
        raise KeyError(f"Golden glove file missing positions: {missing_positions}")

    gg = gg[[year_col] + GG_POSITIONS].rename(columns={year_col: "year"})
    gg["year"] = pd.to_numeric(gg["year"], errors="coerce").astype("Int64")
    gg = gg[gg["year"].between(season_start, season_end)]

    labels = gg.melt(id_vars=["year"], var_name="gg_position", value_name="winner_name")
    labels["winner_name"] = labels["winner_name"].astype(str).str.strip()

    # Outfielders는 3명 수상자를 long row 3개로 분해
    of_mask = labels["gg_position"] == "Outfielders"
    of = labels[of_mask].copy()
    of["winner_name"] = of["winner_name"].str.replace("·", ",", regex=False)
    of["winner_name"] = of["winner_name"].str.replace("/", ",", regex=False)
    of = of.assign(winner_name=of["winner_name"].str.split(",")).explode("winner_name")
    of["winner_name"] = of["winner_name"].astype(str).str.strip()
    of = of[of["winner_name"] != ""]

    non_of = labels[~of_mask].copy()
    non_of = non_of[non_of["winner_name"] != ""]

    out = pd.concat([non_of, of], ignore_index=True)
    out["winner_name"] = out["winner_name"].str.strip()

    return out[["year", "gg_position", "winner_name"]].drop_duplicates()


def build_train_table(season_stats, team_standings, labels):
    pd, _ = _require_pandas()

    train = season_stats.merge(team_standings, on=["year", "team"], how="left")

    winner_lookup = labels.rename(columns={"winner_name": "name"}).copy()
    winner_lookup["is_winner"] = 1
    train = train.merge(
        winner_lookup[["year", "gg_position", "name", "is_winner"]],
        on=["year", "gg_position", "name"],
        how="left",
    )
    train["label"] = train["is_winner"].fillna(0).astype("int8")
    train = train.drop(columns=["is_winner"])

    # interaction features
    for metric in KEY_INTERACTION_METRICS:
        m_col = f"metric_{metric}"
        if m_col in train.columns and "team_win_pct" in train.columns:
            train[f"interaction_team_win_pct_x_{metric}"] = (
                train["team_win_pct"].fillna(0) * train[m_col].fillna(0)
            )

    return train


def save_outputs(output_dir: Path, season_stats, team_standings, labels, train_table):
    output_dir.mkdir(parents=True, exist_ok=True)
    season_stats.to_parquet(output_dir / "season_stats.parquet", index=False)
    team_standings.to_parquet(output_dir / "team_standings.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    train_table.to_parquet(output_dir / "train_table.parquet", index=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print("[build_dataset] dry-run")
        print(f"raw_dir={args.raw_dir}")
        print(f"output_dir={args.output_dir}")
        print(f"season_range={args.season_start}-{args.season_end}")
        return

    season_stats = load_season_stats(args.raw_dir, args.season_start, args.season_end)
    team_standings = load_team_standings(args.raw_dir, args.season_start, args.season_end)
    labels = load_labels(args.raw_dir, args.season_start, args.season_end)
    train_table = build_train_table(season_stats, team_standings, labels)

    save_outputs(args.output_dir, season_stats, team_standings, labels, train_table)

    print("[build_dataset] complete")
    print(f"- season_stats: {len(season_stats):,}")
    print(f"- team_standings: {len(team_standings):,}")
    print(f"- labels: {len(labels):,}")
    print(f"- train_table: {len(train_table):,}")


if __name__ == "__main__":
    main()
