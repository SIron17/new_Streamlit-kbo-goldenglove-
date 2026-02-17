# KBO Golden Glove Winner Prediction

KBO 골든글러브 수상자 예측을 위한 프로젝트 스캐폴딩입니다.
본 PR 단계에서는 **프로젝트 구조/문서/실행 진입점**을 우선 구성합니다.

## 프로젝트 개요
- 목표: 시즌별 포지션 단위 골든글러브 수상자 예측
- 포지션 taxonomy: `P, C, 1B, 2B, 3B, SS, Outfielders, DH`
- 포지션 매핑 규칙:
  - `LF/CF/RF/OF -> Outfielders`
  - pitching row -> `P`
- 팀 성적 변수는 `kbo_ranking_1984_2025.xlsx`를 활용하며, 리그 분리는 하지 않음
- Postseason/KS 필드는 사용하지 않음
- 기록 서사(record narrative) 특성은 시즌 누적 지표(H, HR, RBI, R 등) 중심
- 비정량 요인은 proxy로 근사:
  - 전년 대비 변화량(YoY delta)
  - 커리어 누적치
  - 나이 / 나이²
  - 팀×선수 상호작용

## 데이터 입력 (raw)
아래 4개 원천 파일을 입력으로 사용합니다.

- `data/raw/kbo_batting_stats_by_season_1982-2025.csv`
- `data/raw/kbo_pitching_stats_by_season_1982-2025.csv`
- `data/raw/kbo_golden_glove_1984_2025.xlsx`
- `data/raw/kbo_ranking_1984_2025.xlsx`

## 평가 정의
- 기본 지표: **포지션별 Top-1 hit rate**
- 단, Outfielders는 수상자 3명이므로 **Top-3 hit rate**
- Streamlit UI 노출 규칙:
  - Outfielders: Top-9 후보
  - 그 외 포지션: Top-3 후보

## 레포 구조

```text
.
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── predictions/
├── results/
└── src/
    ├── eval/
    ├── features/
    ├── pipeline/
    │   └── build_dataset.py
    └── train/
        ├── predict.py
        └── train_model.py
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 실행 방법

### 1) 데이터셋 빌드
```bash
python -m src.pipeline.build_dataset --help
python -m src.pipeline.build_dataset
```

### 2) 모델 학습
```bash
python -m src.train.train_model --help
python -m src.train.train_model
```

### 3) 예측
```bash
python -m src.train.predict --help
python -m src.train.predict
```

### 4) Streamlit 앱
```bash
streamlit run app/main.py
```

## 현재 상태
- 본 PR은 스캐폴딩/문서화 중심입니다.
- 엔트리포인트는 최소 스텁으로 동작하며, 다음 PR에서 데이터 처리/피처/모델 로직을 확장합니다.
