# AGENTS.md

## 목적
이 저장소는 KBO 골든글러브 수상자 예측 프로젝트를 위한 코드/실험 기반입니다.
모든 작업은 재현 가능한 커맨드 중심으로 진행합니다.

## 기본 컨벤션
- Python 3.11+ 기준으로 개발합니다.
- 소스 코드는 `src/` 하위 패키지로 구성합니다.
- CLI 진입점은 `python -m ...` 형태로 실행 가능해야 합니다.
- 데이터는 원본(`data/raw/`)과 가공(`data/processed/`)을 분리합니다.
- 결과물은 `results/`, 추론 산출물은 `predictions/`에 저장합니다.

## 실행/검증 규칙
- 환경 구성: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- 데이터셋 빌드: `python -m src.pipeline.build_dataset --help`
- 모델 학습: `python -m src.train.train_model --help`
- 예측: `python -m src.train.predict --help`
- 앱 실행: `streamlit run app/main.py`
- 테스트: `pytest -q`

## 작업 전달 규칙
- 모든 작업은 Git 커밋 + PR 형태로 전달합니다.
- PR 본문에는 아래를 포함합니다.
  1) 변경 요약
  2) 실행한 검증 커맨드와 결과
  3) 후속 작업 TODO(있다면)
- 재현 가능한 형태로, 필요한 명령어와 경로를 명시합니다.
