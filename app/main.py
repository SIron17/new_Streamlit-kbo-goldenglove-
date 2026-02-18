"""Streamlit UI for Golden Glove candidate ranking from uploaded player stats."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/baseline_model.pkl")
FEATURES_PATH = Path("models/baseline_features.json")
REFERENCE_TABLE_PATH = Path("data/processed/train_table.parquet")

POSITION_TOPK_RULES = {
    "P": 3,
    "C": 3,
    "1B": 3,
    "2B": 3,
    "3B": 3,
    "SS": 3,
    "Outfielders": 9,
    "DH": 3,
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
    "OUTFIELDERS": "Outfielders",
    "DH": "DH",
}

DISPLAY_STATS = ["stat_H", "stat_HR", "stat_RBI", "stat_R", "stat_WAR", "stat_ERA", "stat_WHIP"]
RADAR_HITTER = ["stat_AVG", "stat_OBP", "stat_SLG", "stat_OPS", "stat_HR"]
RADAR_PITCHER = ["stat_ERA", "stat_WHIP", "stat_SO", "stat_W", "stat_HR"]


def _configure_korean_font() -> None:
    candidate_names = [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidate_names:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _looks_like_model_ready(df: pd.DataFrame, features: list[str]) -> bool:
    base_cols = {"name", "team", "gg_position"}
    return base_cols.issubset(df.columns) and len([f for f in features if f in df.columns]) >= max(10, len(features) // 3)


def _to_unified_schema(df: pd.DataFrame, is_pitcher: bool, features: list[str]) -> tuple[pd.DataFrame, str]:
    rename_candidates = {
        "Name": "name",
        "name": "name",
        "Team": "team",
        "team": "team",
        "Pos.": "position",
        "Position": "position",
        "position": "position",
        "Year": "year",
        "year": "year",
        "Age": "age",
        "age": "age",
    }
    out = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns}).copy()

    if _looks_like_model_ready(out, features):
        mode = "model_ready"
        out["name"] = out["name"].astype(str).str.strip()
        out["team"] = out["team"].astype(str).str.strip()
        if "age" not in out.columns:
            out["age"] = 0
        _safe_numeric(out, [c for c in out.columns if c.startswith("stat_")] + ["age"])
        return out, mode

    mode = "raw_approx"
    if "name" not in out.columns:
        raise ValueError("업로드 파일에 선수 이름 컬럼(Name 또는 name)이 필요합니다.")
    if "team" not in out.columns:
        out["team"] = "Unknown"
    if "age" not in out.columns:
        out["age"] = 0

    if is_pitcher:
        out["gg_position"] = "P"
        out["is_pitcher"] = 1
    else:
        if "position" not in out.columns:
            raise ValueError("타자 파일에는 포지션 컬럼(Pos. 또는 Position)이 필요합니다.")
        out["gg_position"] = out["position"].astype(str).str.upper().map(POSITION_MAP)
        out = out[out["gg_position"].isin(POSITION_TOPK_RULES.keys())].copy()
        out["is_pitcher"] = 0

    keep_non_stats = {"name", "team", "age", "gg_position", "is_pitcher"}
    for col in list(out.columns):
        if col in keep_non_stats or col.startswith("stat_"):
            continue
        if col in {"position"}:
            continue
        out.rename(columns={col: f"stat_{col}"}, inplace=True)

    stat_cols = [c for c in out.columns if c.startswith("stat_")] + ["age"]
    _safe_numeric(out, stat_cols)

    out["name"] = out["name"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip()
    if "year" not in out.columns:
        out["year"] = 9999

    return out, mode


def _load_model_and_features():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다. 먼저 `python -m src.train.train_model`을 실행하세요."
        )

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with FEATURES_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta["features"]


def _enrich_with_reference_features(input_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not REFERENCE_TABLE_PATH.exists():
        return input_df

    try:
        ref_all_cols = pd.read_parquet(REFERENCE_TABLE_PATH, columns=[]).columns.tolist()
        ref_cols = [
            c
            for c in ["year", "name", "team", "gg_position", "player_id"] + features
            if c in ref_all_cols
        ]
        ref = pd.read_parquet(REFERENCE_TABLE_PATH, columns=ref_cols)
    except Exception:
        return input_df

    if "year" in input_df.columns:
        target_year = pd.to_numeric(input_df["year"], errors="coerce").dropna()
        if not target_year.empty:
            ref = ref[ref["year"] == int(target_year.mode().iloc[0])]

    key_cols = [c for c in ["name", "team", "gg_position"] if c in input_df.columns and c in ref.columns]
    if len(key_cols) < 2:
        return input_df

    merged = input_df.merge(ref, on=key_cols, how="left", suffixes=("", "__ref"))
    for f in features:
        ref_col = f"{f}__ref"
        if ref_col in merged.columns:
            merged[f] = merged[ref_col].combine_first(merged.get(f))
            merged.drop(columns=[ref_col], inplace=True)
    return merged


def _predict_candidates(model, features: list[str], input_df: pd.DataFrame) -> pd.DataFrame:
    x = input_df.reindex(columns=features, fill_value=0).fillna(0.0)
    output = input_df.copy()
    output["score"] = model.predict_proba(x)[:, 1]
    output["pred_rank"] = (
        output.groupby("gg_position")["score"].rank(method="first", ascending=False).astype(int)
    )
    return output


def _radar_chart(top_row: pd.Series, target_row: pd.Series, features: list[str], title: str) -> None:
    cols = [c for c in features if c in top_row.index and c in target_row.index]
    if len(cols) < 3:
        st.info("레이더 차트용 지표가 부족합니다.")
        return

    labels = cols
    p1 = [float(top_row.get(c, 0) or 0) for c in labels]
    p2 = [float(target_row.get(c, 0) or 0) for c in labels]
    p1 += p1[:1]
    p2 += p2[:1]

    angles = [n / float(len(labels)) * 2 * 3.141592653589793 for n in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    ax.plot(angles, p1, linewidth=2, linestyle="-", label=f"1위: {top_row['name']}")
    ax.plot(angles, p2, linewidth=2, linestyle="-", label=f"선택: {target_row['name']}")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.1))
    st.pyplot(fig)


def main() -> None:
    _configure_korean_font()
    st.set_page_config(page_title="KBO 골든글러브 예측모델", page_icon="⚾", layout="wide")
    st.title("KBO 골든글러브 수상자 예측모델")

    st.sidebar.header("입력 데이터")
    hitter_file = st.sidebar.file_uploader("타자 성적 CSV 업로드", type=["csv"])
    pitcher_file = st.sidebar.file_uploader("투수 성적 CSV 업로드", type=["csv"])
    position_for_detail = st.sidebar.selectbox("세부 분석 포지션", list(POSITION_TOPK_RULES.keys()))

    st.caption(
        "업로드 데이터 기반 예측입니다. "
        "Outfielders Top-9 / 그 외 Top-3"
    )

    if not hitter_file and not pitcher_file:
        st.info("타자/투수 CSV를 업로드하면 모델이 후보를 예측합니다.")
        return

    try:
        model, features = _load_model_and_features()
    except Exception as exc:
        st.error(str(exc))
        return

    inputs: list[pd.DataFrame] = []
    ingest_modes: list[str] = []

    try:
        if hitter_file is not None:
            hitters = pd.read_csv(hitter_file)
            h_df, h_mode = _to_unified_schema(hitters, is_pitcher=False, features=features)
            inputs.append(h_df)
            ingest_modes.append(h_mode)
        if pitcher_file is not None:
            pitchers = pd.read_csv(pitcher_file)
            p_df, p_mode = _to_unified_schema(pitchers, is_pitcher=True, features=features)
            inputs.append(p_df)
            ingest_modes.append(p_mode)
    except Exception as exc:
        st.error(f"입력 파일 처리 오류: {exc}")
        return

    if not inputs:
        st.warning("예측 가능한 입력 데이터가 없습니다.")
        return

    mode_set = set(ingest_modes)
    if "raw_approx" in mode_set:
        st.warning(
            "현재 업로드는 raw 근사 모드입니다. `pred_2025.parquet`과 완전 동일한 결과를 원하면 "
            "train_table 기반(모델 피처 포함) 테스트 파일을 사용하세요."
        )
    else:
        st.success("모델 피처 정합 모드로 예측 중입니다 (predict 스크립트와 높은 일관성 기대).")

    input_df = pd.concat(inputs, ignore_index=True)
    input_df = _enrich_with_reference_features(input_df, features)

    available_features = [f for f in features if f in input_df.columns]
    if len(available_features) < len(features):
        st.warning(
            f"모델 피처 정합도: {len(available_features)}/{len(features)}. "
            "누락 피처는 0으로 처리되어 predict 스크립트 결과와 차이가 날 수 있습니다."
        )
    pred_df = _predict_candidates(model, features, input_df)

    final_views = []
    for position, top_k in POSITION_TOPK_RULES.items():
        pos_df = pred_df[pred_df["gg_position"] == position].copy()
        if pos_df.empty:
            continue
        pos_top = pos_df[pos_df["pred_rank"] <= top_k].sort_values("pred_rank")

        display_cols = ["pred_rank", "name", "team", "score"] + [
            c for c in DISPLAY_STATS if c in pos_top.columns
        ]
        st.subheader(f"{position} Top-{top_k}")
        st.dataframe(pos_top[display_cols], width="stretch")
        st.bar_chart(pos_top.set_index("name")["score"])

        final_views.append(pos_top.assign(top_k_rule=top_k))

    if not final_views:
        st.warning("포지션 매핑 가능한 선수 데이터가 없습니다.")
        return

    final_candidates = pd.concat(final_views, ignore_index=True)

    st.divider()
    st.subheader("세부 비교")
    detail = final_candidates[final_candidates["gg_position"] == position_for_detail].copy()
    if detail.empty:
        st.info("선택한 포지션에 대한 후보가 없습니다.")
    else:
        detail = detail.sort_values("pred_rank")
        top_player = detail.iloc[0]
        selected_name = st.selectbox(
            f"{position_for_detail} 비교 선수 선택",
            detail["name"].tolist(),
        )
        selected_player = detail[detail["name"] == selected_name].iloc[0]
        radar_features = RADAR_PITCHER if position_for_detail == "P" else RADAR_HITTER
        _radar_chart(top_player, selected_player, radar_features, f"{position_for_detail} Radar")

    st.divider()
    st.subheader("전체 예측 결과 (Top-k만)")
    export_cols = [
        c
        for c in ["gg_position", "pred_rank", "name", "team", "score", "top_k_rule"] + DISPLAY_STATS
        if c in final_candidates.columns
    ]
    st.dataframe(final_candidates[export_cols], width="stretch")
    st.download_button(
        "결과 다운로드 (CSV)",
        final_candidates[export_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="golden_glove_top_candidates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
