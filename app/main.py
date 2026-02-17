"""Streamlit app scaffold for KBO Golden Glove prediction."""

from __future__ import annotations

import streamlit as st


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


def main() -> None:
    st.set_page_config(page_title="KBO Golden Glove Predictor", layout="wide")
    st.title("KBO Golden Glove Winner Predictor (Scaffold)")
    st.markdown(
        """
        이 앱은 골든글러브 예측 프로젝트의 초기 스캐폴딩 화면입니다.

        - 기본 평가지표: 포지션별 Top-1 hit rate
        - Outfielders는 3명 수상 구조이므로 Top-3 hit rate
        - UI 후보 노출: Outfielders Top-9, 나머지 포지션 Top-3
        """
    )

    st.subheader("Position-wise UI Top-k")
    st.table(
        [{"position": p, "ui_top_k": k} for p, k in POSITION_TOPK_RULES.items()]
    )

    selected_position = st.selectbox("포지션 선택", list(POSITION_TOPK_RULES.keys()))
    st.info(
        f"선택한 포지션: {selected_position} / 표시 후보 수: Top-{POSITION_TOPK_RULES[selected_position]}"
    )


if __name__ == "__main__":
    main()
