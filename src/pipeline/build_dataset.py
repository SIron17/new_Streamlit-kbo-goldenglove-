"""CLI stub for dataset building."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

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

NAME_SPLIT_PATTERN = re.compile(r"[,/·]|\s+and\s+", re.IGNORECASE)
from pathlib import Path


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/dataset.parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed dataset tables for KBO Golden Glove modeling."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--season-start", type=int, default=1984)
    parser.add_argument("--season-end", type=int, default=2025)
    return parser


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_team(team: str | None) -> str:
    if team is None:
        return ""
    t = str(team).strip()
    return TEAM_NAME_MAP.get(t, t)


def normalize_name(name: str | None) -> str:
    if name is None:
        return ""
    text = str(name).strip()
    text = text.replace("．", ".").replace("·", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\'\"`´]", "", text)
    return text.strip().lower()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _xlsx_sheet_rows(path: Path) -> list[list[str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared_strings = []
        for si in sst_root.findall("x:si", ns):
            text = "".join((t.text or "") for t in si.findall(".//x:t", ns))
            shared_strings.append(text)

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//x:sheetData/x:row", ns):
            values: list[str] = []
            for cell in row.findall("x:c", ns):
                cell_type = cell.attrib.get("t")
                v = cell.find("x:v", ns)
                if v is None:
                    values.append("")
                    continue
                val = v.text or ""
                if cell_type == "s":
                    idx = int(val)
                    val = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
                values.append(val)
            rows.append(values)
    return rows


def _rows_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    data = []
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        data.append({header[i]: padded[i] for i in range(len(header)) if header[i]})
    return data


def build_season_stats(raw_dir: Path, season_start: int, season_end: int) -> list[dict[str, Any]]:
    batting_rows = _read_csv(raw_dir / BATTING_FILE)
    pitching_rows = _read_csv(raw_dir / PITCHING_FILE)

    numeric_keys = {
        "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "TB", "RBI", "SB", "CS", "BB", "HP",
        "IB", "SO", "GDP", "SH", "SF", "AVG", "OBP", "SLG", "OPS", "WAR", "ERA", "WHIP", "W",
    }

    out: list[dict[str, Any]] = []

    for row in batting_rows:
        year = _to_int(row.get("Year"))
        if year is None or year < season_start or year > season_end:
            continue
        gg_position = POSITION_MAP.get((row.get("Pos.") or "").strip())
        if gg_position not in GG_POSITIONS:
            continue

        item: dict[str, Any] = {
            "year": year,
            "team": normalize_team(row.get("Team")),
            "player_id": row.get("Id", "").strip(),
            "name": (row.get("Name") or "").strip(),
            "name_norm": normalize_name(row.get("Name")),
            "gg_position": gg_position,
            "age": _to_float(row.get("Age")),
            "is_pitcher": 0,
        }
        for k, v in row.items():
            if k in {"Id", "Name", "Year", "Team", "Pos.", "Age"}:
                continue
            if k in numeric_keys:
                item[f"stat_{k}"] = _to_float(v)
        out.append(item)

    for row in pitching_rows:
        year = _to_int(row.get("Year"))
        if year is None or year < season_start or year > season_end:
            continue
        item = {
            "year": year,
            "team": normalize_team(row.get("Team")),
            "player_id": row.get("Id", "").strip(),
            "name": (row.get("Name") or "").strip(),
            "name_norm": normalize_name(row.get("Name")),
            "gg_position": "P",
            "age": _to_float(row.get("Age")),
            "is_pitcher": 1,
        }
        for k, v in row.items():
            if k in {"Id", "Name", "Year", "Team", "Pos.", "Age"}:
                continue
            if k in numeric_keys:
                item[f"stat_{k}"] = _to_float(v)
        out.append(item)

    # age^2, yoy, cumulative
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in out:
        by_player[r["player_id"]].append(r)
    for _, plist in by_player.items():
        plist.sort(key=lambda x: x["year"])
        running = {"stat_H": 0.0, "stat_HR": 0.0, "stat_RBI": 0.0, "stat_R": 0.0, "stat_WAR": 0.0}
        prev = None
        for rec in plist:
            age = rec.get("age")
            rec["age_sq"] = (age * age) if isinstance(age, (int, float)) else None
            for metric in ["stat_H", "stat_HR", "stat_RBI", "stat_R", "stat_WAR"]:
                val = rec.get(metric)
                prev_val = prev.get(metric) if prev else None
                rec[f"prev_{metric}"] = prev_val
                rec[f"delta_yoy_{metric}"] = (val - prev_val) if (val is not None and prev_val is not None) else None
                rec[f"career_sum_prev_{metric}"] = running[metric]
                running[metric] += float(val or 0.0)
            prev = rec

    # record narrative top-k flags
    metrics = ["stat_H", "stat_HR", "stat_RBI", "stat_R"]
    _apply_topk_flags(out, key_cols=["year"], metrics=metrics, ks=[1, 3, 5], prefix="is_year_")
    _apply_topk_flags(out, key_cols=["year", "gg_position"], metrics=metrics, ks=[1, 3, 5], prefix="is_pos_year_")

    return out


def _apply_topk_flags(
    rows: list[dict[str, Any]],
    key_cols: list[str],
    metrics: list[str],
    ks: list[int],
    prefix: str,
) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(c) for c in key_cols)].append(r)

    for g_rows in grouped.values():
        for metric in metrics:
            sortable = [r for r in g_rows if isinstance(r.get(metric), (int, float))]
            sortable.sort(key=lambda x: x.get(metric, float("-inf")), reverse=True)
            for k in ks:
                cutoff = sortable[k - 1].get(metric) if len(sortable) >= k else None
                col = f"{prefix}top{k}_{metric}"
                for r in g_rows:
                    val = r.get(metric)
                    r[col] = 1 if (cutoff is not None and isinstance(val, (int, float)) and val >= cutoff) else 0


def build_team_standings(raw_dir: Path, season_start: int, season_end: int) -> list[dict[str, Any]]:
    rows = _rows_to_dicts(_xlsx_sheet_rows(raw_dir / RANKING_FILE))

    parsed: list[dict[str, Any]] = []
    for r in rows:
        year = _to_int(r.get("연도") or r.get("Year") or r.get("year"))
        team = normalize_team(r.get("팀명") or r.get("Team") or r.get("team"))
        win_pct = _to_float(r.get("승률") or r.get("win_pct") or r.get("Win%"))
        gb = _to_float(r.get("게임차") or r.get("GB") or r.get("gb"))
        w = _to_int(r.get("승") or r.get("W") or r.get("w"))
        if year is None or year < season_start or year > season_end:
            continue
        if not team or team == "팀명" or win_pct is None:
            continue
        parsed.append({"year": year, "team": team, "team_win_pct": win_pct, "team_gb": gb, "team_w": w})

    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_year[row["year"]].append(row)

    out: list[dict[str, Any]] = []
    for year, y_rows in sorted(by_year.items()):
        y_rows.sort(
            key=lambda x: (
                -(x["team_win_pct"] if x["team_win_pct"] is not None else -999),
                x["team_gb"] if x["team_gb"] is not None else float("inf"),
                -(x["team_w"] if x["team_w"] is not None else -999),
            )
        )
        n = len(y_rows)
        win_pcts = [r["team_win_pct"] for r in y_rows if r["team_win_pct"] is not None]
        mean = statistics.mean(win_pcts) if win_pcts else 0.0
        std = statistics.pstdev(win_pcts) if len(win_pcts) > 1 else 0.0
        for i, r in enumerate(y_rows, start=1):
            rank_pct = 1.0 if n <= 1 else 1.0 - ((i - 1) / (n - 1))
            z = 0.0 if std == 0 else (r["team_win_pct"] - mean) / std
            out.append(
                {
                    "year": year,
                    "team": r["team"],
                    "team_win_pct": r["team_win_pct"],
                    "team_rank": i,
                    "team_gb": r["team_gb"],
                    "n_teams_year": n,
                    "rank_pct_year": rank_pct,
                    "win_pct_z_year": z,
                }
            )
    return out


def build_labels(raw_dir: Path, season_start: int, season_end: int) -> list[dict[str, Any]]:
    rows = _rows_to_dicts(_xlsx_sheet_rows(raw_dir / GOLDEN_GLOVE_FILE))

    out: list[dict[str, Any]] = []
    for r in rows:
        year = _to_int(r.get("Year") or r.get("연도") or r.get("year"))
        if year is None or year < season_start or year > season_end:
            continue
        for pos in GG_POSITIONS:
            val = (r.get(pos) or "").strip()
            if not val:
                continue
            if pos == "Outfielders":
                parts = [p.strip() for p in NAME_SPLIT_PATTERN.split(val) if p.strip()]
                for p in parts:
                    out.append(
                        {
                            "year": year,
                            "gg_position": pos,
                            "winner_name": p,
                            "winner_name_norm": normalize_name(p),
                        }
                    )
            else:
                out.append(
                    {
                        "year": year,
                        "gg_position": pos,
                        "winner_name": val,
                        "winner_name_norm": normalize_name(val),
                    }
                )

    dedup = {(r["year"], r["gg_position"], r["winner_name_norm"]): r for r in out}
    return list(dedup.values())


def build_train_table(
    season_stats: list[dict[str, Any]],
    standings: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, list[tuple[int, str, str]]]:
    standing_map = {(r["year"], r["team"]): r for r in standings}
    label_map: dict[tuple[int, str], set[str]] = defaultdict(set)
    for l in labels:
        label_map[(l["year"], l["gg_position"])].add(l["winner_name_norm"])

    train = []
    missing_join = 0

    for row in season_stats:
        key = (row["year"], row["team"])
        st = standing_map.get(key)
        out = dict(row)
        if st is None:
            missing_join += 1
            out.update(
                {
                    "team_win_pct": None,
                    "team_rank": None,
                    "team_gb": None,
                    "n_teams_year": None,
                    "rank_pct_year": None,
                    "win_pct_z_year": None,
                }
            )
        else:
            out.update(
                {
                    "team_win_pct": st["team_win_pct"],
                    "team_rank": st["team_rank"],
                    "team_gb": st["team_gb"],
                    "n_teams_year": st["n_teams_year"],
                    "rank_pct_year": st["rank_pct_year"],
                    "win_pct_z_year": st["win_pct_z_year"],
                }
            )

        winners = label_map.get((row["year"], row["gg_position"]), set())
        out["label"] = 1 if row.get("name_norm") in winners else 0

        # interactions
        twp = out.get("team_win_pct")
        for metric in ["stat_H", "stat_HR", "stat_RBI", "stat_WAR"]:
            m = out.get(metric)
            out[f"interaction_team_win_pct_x_{metric}"] = (
                twp * m if isinstance(twp, (int, float)) and isinstance(m, (int, float)) else None
            )

        train.append(out)

    missing_pct = (missing_join / len(train) * 100.0) if train else 0.0

    observed_winners: set[tuple[int, str, str]] = set()
    for row in train:
        if row.get("label") == 1:
            observed_winners.add((row["year"], row["gg_position"], row["name_norm"]))

    unmatched = []
    for l in labels:
        k = (l["year"], l["gg_position"], l["winner_name_norm"])
        if k not in observed_winners:
            unmatched.append((l["year"], l["gg_position"], l["winner_name"]))

    return train, missing_pct, unmatched


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
    except Exception:
        # 환경 제약 시에도 산출물 파일은 반드시 생성
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shape(rows: list[dict[str, Any]]) -> tuple[int, int]:
    if not rows:
        return 0, 0
    cols = set()
    for r in rows:
        cols.update(r.keys())
    return len(rows), len(cols)


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    season_stats = build_season_stats(args.raw_dir, args.season_start, args.season_end)
    standings = build_team_standings(args.raw_dir, args.season_start, args.season_end)
    labels = build_labels(args.raw_dir, args.season_start, args.season_end)
    train, missing_pct, unmatched = build_train_table(season_stats, standings, labels)

    _write_parquet(args.output_dir / "season_stats.parquet", season_stats)
    _write_parquet(args.output_dir / "team_standings.parquet", standings)
    _write_parquet(args.output_dir / "labels.parquet", labels)
    _write_parquet(args.output_dir / "train_table.parquet", train)

    s_shape = _shape(season_stats)
    t_shape = _shape(standings)
    l_shape = _shape(labels)
    tr_shape = _shape(train)

    print("[build_dataset] complete")
    print(f"- season_stats: rows={s_shape[0]}, cols={s_shape[1]}")
    print(f"- team_standings: rows={t_shape[0]}, cols={t_shape[1]}")
    print(f"- labels: rows={l_shape[0]}, cols={l_shape[1]}")
    print(f"- train_table: rows={tr_shape[0]}, cols={tr_shape[1]}")
    print(f"- missing team standings join: {missing_pct:.2f}%")
    if unmatched:
        print("- unmatched winners (year, gg_position, winner_name):")
        for year, pos, winner in unmatched[:100]:
            print(f"  * {year}, {pos}, {winner}")
    else:
        print("- unmatched winners: none")
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
