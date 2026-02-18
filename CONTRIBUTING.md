# CONTRIBUTING

## main-first 브랜치 전략 (충돌 방지)
반드시 아래 순서로 작업합니다.

```bash
git checkout main
git pull origin main   # remote가 있을 때

git checkout -b <feature-branch>
# 코드 수정/검증
./scripts/pre_pr_check.sh
pytest -q

git add -A
git commit -m "<message>"
git push -u origin <feature-branch>
```

## 필수 규칙
- 기존 PR 브랜치를 재사용하지 않습니다.
- PR 올리기 전 충돌 마커(`<<<<<<<`, `=======`, `>>>>>>>`)가 없어야 합니다.
- PR 본문에는 변경 요약/검증 커맨드/TODO를 포함합니다.
