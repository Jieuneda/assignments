# 🔍 시계열 이상 탐지 대시보드

Darts 라이브러리 기반의 다변량 시계열 이상 탐지 웹앱입니다.  
수업 핵심 이론(ForecastingAnomalyModel + NormScorer + KMeansScorer)을 완전히 구현합니다.

---

## 🚀 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit Cloud 배포 방법

1. GitHub 저장소에 이 폴더 내용을 푸시
2. [share.streamlit.io](https://share.streamlit.io) 접속 후 New App
3. Repository, Branch, Main file path (`app.py`) 지정
4. Deploy!

---

## 📂 CSV 형식

```
timestamp,feature1,feature2,...
2022-01-01 00:00:00,50.99,65.72,191.37
...
```

- 첫 번째 컬럼: 날짜/시간 (자동 감지)
- 나머지 컬럼: 수치형 피처 (다변량 지원)
- `sample_multivariate.csv` 파일로 즉시 테스트 가능

---

## 🔬 구현된 수업 이론

| 모듈 | 역할 |
|------|------|
| `SKLearnModel` (RandomForest) | 예측 기반 이상 탐지의 기반 모델 |
| `NormScorer` | 예측 오차 절댓값 기반 이상 점수 |
| `KMeansScorer(k=2)` | 클러스터링 기반 이상 점수 |
| `ForecastingAnomalyModel` | 예측 모델 + Scorer 결합 |
| Cyclic Encoders | hour/dayofweek 주기적 공변량 |
| `Scaler` | MinMax 정규화 (Data Leakage 방지) |

---

## 📊 주요 기능

- **다변량 CSV 업로드** — 파일 변경 시 자동 재탐지
- **Train/Test 분할 조절** — 사이드바 슬라이더
- **복수 Scorer 앙상블** — NormScorer + KMeansScorer 결합
- **임계값 기반 이상 판정** — 백분위수 기반
- **5개 탭 대시보드** — 시계열, 점수, 피처 분석, 분포, 원시 데이터
- **이상 유형 분류** — 전역/맥락 이상 자동 판정
- **CSV 결과 다운로드**
