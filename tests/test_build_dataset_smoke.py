from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")
pytest.importorskip("pyarrow")

from src.pipeline.build_dataset import (  # noqa: E402
    build_train_table,
    load_labels,
    load_season_stats,
    load_team_standings,
    save_outputs,
)


def test_build_dataset_smoke(tmp_path: Path) -> None:
    raw_dir = Path("data/raw")
    output_dir = tmp_path / "processed"

    season_stats = load_season_stats(raw_dir, 1984, 2025)
    team_standings = load_team_standings(raw_dir, 1984, 2025)
    labels = load_labels(raw_dir, 1984, 2025)
    train_table = build_train_table(season_stats, team_standings, labels)

    save_outputs(output_dir, season_stats, team_standings, labels, train_table)

    expected = [
        "season_stats.parquet",
        "team_standings.parquet",
        "labels.parquet",
        "train_table.parquet",
    ]
    for fname in expected:
        assert (output_dir / fname).exists(), f"missing output file: {fname}"

    # 최소한의 스키마 체크
    season_loaded = pd.read_parquet(output_dir / "season_stats.parquet")
    standings_loaded = pd.read_parquet(output_dir / "team_standings.parquet")
    labels_loaded = pd.read_parquet(output_dir / "labels.parquet")
    train_loaded = pd.read_parquet(output_dir / "train_table.parquet")

    assert {"year", "team", "player_id", "name", "gg_position", "age"}.issubset(
        season_loaded.columns
    )
    assert {
        "year",
        "team",
        "team_win_pct",
        "team_rank",
        "team_gb",
        "n_teams_year",
        "rank_pct_year",
        "win_pct_z_year",
    }.issubset(standings_loaded.columns)
    assert {"year", "gg_position", "winner_name"}.issubset(labels_loaded.columns)
    assert "label" in train_loaded.columns
