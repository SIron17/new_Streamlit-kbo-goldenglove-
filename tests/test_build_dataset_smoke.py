from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_dataset_creates_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed"
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline.build_dataset",
        "--output-dir",
        str(out_dir),
        "--season-start",
        "2024",
        "--season-end",
        "2025",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "[build_dataset] complete" in result.stdout

    expected = [
        "season_stats.parquet",
        "team_standings.parquet",
        "labels.parquet",
        "train_table.parquet",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), f"missing output: {path}"
        assert path.stat().st_size > 0, f"empty output: {path}"
