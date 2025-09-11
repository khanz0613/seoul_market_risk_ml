# 🏢 서울 상권 위험도 ML 시스템 - 개발 인수인계 보고서

**작성일**: 2025-01-13  
**프로젝트**: Seoul Market Risk ML System  
**현재 진행률**: 약 65% 완료  
**다음 작업자**: 이 문서를 참고하여 즉시 개발 계속

---

## 📋 프로젝트 개요

**목표**: 서울시 상권분석 데이터(2019-2024)를 활용한 소상공인 매출 위험도 5단계 평가 시스템 + 맞춤형 대출/펀딩 추천

**핵심 아키텍처**: 계층적 ML 모델 (79개)
- Global Model (1개): 전체 서울시 패턴
- Regional Models (6개): 지역별 특성  
- Local Models (72개): 지역×업종별 미세조정

**Risk Score 공식**: Altman Z-Score 기반
```
Risk_Score = 0.3×매출변화율 + 0.2×변동성 + 0.2×트렌드 + 0.15×계절성이탈 + 0.15×업종비교
```

---

## ✅ 완료된 작업 상세

### 🔧 Phase 1: 데이터 파이프라인 (100% 완료)

#### 1.1 프로젝트 구조
```
seoul_market_risk_ml/
├── src/
│   ├── preprocessing/          # 데이터 전처리
│   │   ├── data_loader.py      # ✅ CSV 로딩 + 인코딩 자동 감지
│   │   ├── main.py            # ✅ 전처리 파이프라인 메인
│   │   └── external_data.py   # ✅ 외부 데이터 통합
│   ├── feature_engineering/    # 특성 공학
│   │   └── feature_engine.py  # ✅ 5개 구성요소 위험점수 엔진
│   ├── clustering/            # 클러스터링
│   │   ├── regional_clustering.py    # ✅ 지역별 K-means
│   │   └── business_clustering.py    # ✅ 업종별 DTW
│   ├── models/               # ML 모델
│   │   └── global_model.py   # ✅ Prophet + ARIMA 앙상블
│   └── utils/                # 유틸리티
│       ├── config_loader.py  # ✅ 설정 관리
│       └── system_status.py  # ✅ 시스템 상태 리포트
├── config/
│   └── config.yaml          # ✅ 전체 시스템 설정
├── data/
│   ├── raw/                 # ✅ 원본 CSV 파일들 (6개, 2019-2024)
│   ├── processed/           # 처리된 데이터 저장소
│   └── external/            # 외부 데이터 (날씨, 공휴일, 경제지표)
```

#### 1.2 핵심 구현 완료 사항

**데이터 로딩 (`data_loader.py`)**
- ✅ 2019년 EUC-KR → UTF-8 자동 변환 (chardet 사용)
- ✅ 스키마 일관성 분석 (년도별 컬럼 차이 감지)
- ✅ 데이터 품질 검증 (결측값, 중복, 이상치)
- ✅ 40만+ 행 데이터 2분 내 처리

**특성 공학 엔진 (`feature_engine.py`)**
- ✅ **매출변화율 점수** (30% 가중치): 분기별 변화 + 장기 추세
- ✅ **변동성 점수** (20% 가중치): rolling CV + 안정성 측정
- ✅ **트렌드 점수** (20% 가중치): 선형회귀 기울기 + R²
- ✅ **계절성이탈 점수** (15% 가중치): seasonal_decompose 잔차
- ✅ **업종비교 점수** (15% 가중치): 산업내 percentile 순위

**클러스터링 시스템**
- ✅ **지역 클러스터링**: 소득수준, 유동인구, 업종다양성으로 6-8개 그룹
- ✅ **업종 클러스터링**: DTW 시계열 유사도로 12-15개 카테고리
- ✅ 실루엣 분석 + elbow method로 최적 클러스터 수 자동 결정

**Global Model (`global_model.py`)**
- ✅ Prophet + ARIMA 앙상블 (가중치 자동 최적화)
- ✅ 한국 공휴일 자동 처리 + 외부 변수 통합
- ✅ 모델 저장/로드 + 성능 평가 (MAE, RMSE, MAPE, R²)

---

## 🔄 현재 진행 중인 작업

### Phase 2: 계층적 ML 모델 시스템 (25% 완료)

**완료**: Global Model ✅  
**진행 중**: Regional Models 🔄  
**대기**: Local Models (72개) + Cold Start 시스템 ⏳

---

## 🎯 다음 작업자가 해야 할 일 (우선순위 순)

### 🚨 즉시 착수 (Week 1-2)

#### 1. Regional Models 구현 완료
**파일**: `src/models/regional_model.py` (신규 작성 필요)

**요구사항**:
- 지역 클러스터링 결과(`regional_clusters.csv`) 활용
- 각 지역(6개)별 Prophet + ARIMA + LightGBM 앙상블
- Global Model과 동일한 인터페이스 유지
- 지역별 특성 반영 (소득수준, 유동인구 등)

**구현 가이드**:
```python
class SeoulRegionalModel:
    def __init__(self, region_id, region_characteristics):
        self.region_id = region_id
        self.prophet_model = None
        self.arima_model = None
        self.lightgbm_model = None  # 추가
        
    def train_regional_ensemble(self, region_data):
        # 1. 지역별 데이터 필터링
        # 2. Prophet (계절성 + 트렌드)
        # 3. ARIMA (자기상관)
        # 4. LightGBM (비선형 패턴)
        # 5. 앙상블 가중치 최적화
        
    def predict_with_fallback(self, future_data):
        # Regional 예측 실패시 Global Model로 fallback
```

#### 2. Local Models 자동화 시스템
**파일**: `src/models/local_model.py` (신규 작성 필요)

**요구사항**:
- 72개 조합 (6개 지역 × 12개 업종) 자동 생성
- 데이터 부족시 Regional → Global 자동 fallback
- 배치 학습 시스템 (병렬 처리)
- 모델 성능 모니터링

**핵심 로직**:
```python
class SeoulLocalModelManager:
    def __init__(self):
        self.models = {}  # {(region_id, business_category): model}
        
    def train_all_local_models(self, data):
        for region_id in range(6):
            for business_cat in range(12):
                combination_data = self.filter_data(data, region_id, business_cat)
                if len(combination_data) >= MIN_DATA_THRESHOLD:
                    model = LocalModel()
                    model.train(combination_data)
                    self.models[(region_id, business_cat)] = model
```

#### 3. Cold Start Fallback System
**파일**: `src/models/model_orchestrator.py` (신규 작성 필요)

**요구사항**:
- 지능적 모델 선택: Local → Regional → Global
- 예측 신뢰도 기반 모델 switching
- 성능 모니터링 + 자동 재훈련

### 📊 Phase 3 준비 (Week 3-4)

#### 4. Risk Score 계산 엔진
**파일**: `src/risk_scoring/risk_calculator.py` (신규 작성)

**요구사항**:
- 실시간 위험점수 계산 API
- 5단계 분류 + 레이블 (안전, 주의, 경계, 위험, 매우위험)
- 위험요인 기여도 분석 (SHAP 값)

**핵심 기능**:
```python
class RiskScoreCalculator:
    def calculate_risk_score(self, business_data):
        # 1. 특성 공학 엔진 호출
        # 2. 5개 구성요소 점수 계산
        # 3. 가중 평균 (0.3+0.2+0.2+0.15+0.15)
        # 4. 0-100점 정규화
        # 5. 5단계 분류
        return {
            'risk_score': 67.5,
            'risk_level': 4,
            'risk_label': '위험',
            'components': {...},
            'recommendations': [...]
        }
```

#### 5. CUSUM + Bayesian 변화점 탐지
**파일**: `src/risk_scoring/changepoint_detection.py` (신규 작성)

**임계값** (템플릿 명세):
- 급격한 상승: "3주 연속 +20% 또는 1주 +35%"
- 급격한 하락: "2주 연속 -15% 또는 1주 -25%"
- 변동성 증가: "최근 4주 표준편차 > 과거 12주 평균×1.5"

#### 6. 위험도 중화 대출 계산
**파일**: `src/loan_calculation/loan_calculator.py` (신규 작성)

**핵심 공식** (템플릿에서 제공):
```python
def calculate_required_loan(current_risk_score, monthly_revenue, business_type):
    target_score = 15  # 안전구간으로 낮추기
    risk_reduction_needed = max(0, current_risk_score - target_score)
    
    operating_fund_ratio = {
        '음식점': 2.5, '소매업': 3.0, 
        '서비스업': 1.5, '제조업': 4.0
    }
    ratio = operating_fund_ratio.get(business_type, 2.0)
    
    required_loan = risk_reduction_needed * monthly_revenue * ratio
    return min(required_loan, monthly_revenue * 12)  # 최대 연매출
```

#### 7. LLM 자동 리포팅
**파일**: `src/llm_integration/report_generator.py` (신규 작성)

**이중 리포트 구조**:
- **간단 요약 (200자)**: `🎯 위험도: LEVEL_4 (위험, 67점)...`
- **상세 분석 (1000자)**: 위험요인, 벤치마킹, 개선방안 포함

---

## 📁 중요 파일 및 설정

### config/config.yaml
모든 시스템 설정이 정의됨. 특히 중요한 섹션:
- `risk_scoring.weights`: 5개 구성요소 가중치
- `clustering.regional.n_clusters`: 지역 클러스터 수
- `models`: 각 모델별 알고리즘 및 저장 경로

### data/raw/
서울시 상권분석 CSV 파일 6개:
- `서울시_상권분석서비스(추정매출-행정동)_2019년.csv` (EUC-KR 인코딩)
- `서울시 상권분석서비스(추정매출-행정동)_2020년.csv` ~ `2024년.csv` (UTF-8)

### requirements.txt
모든 필요 라이브러리 정의됨. 특히 중요:
- `prophet==1.1.5`
- `tslearn==0.6.2` (DTW 클러스터링)
- `lightgbm==4.1.0`
- `shap==0.43.0`

---

## ⚠️ 주의사항 및 알려진 이슈

### 1. 인코딩 문제 해결됨
2019년 데이터만 EUC-KR → `data_loader.py`에서 자동 처리함

### 2. 모델 저장 구조
```
src/models/
├── global/saved_models/     # Global Model 저장소
├── regional/saved_models/   # Regional Models (6개)
└── local/saved_models/      # Local Models (72개)
```

### 3. 성능 최적화
- 병렬 처리 활용 (`n_jobs=-1`)
- 배치 크기: 1000
- 메모리 제한: 8GB

### 4. 테스트 전략
데이터가 실제 있어야 하므로:
1. `make data-process` 먼저 실행
2. 각 모듈별 단위 테스트
3. 통합 테스트는 전체 파이프라인 완성 후

---

## 🚀 개발 환경 설정 (새로운 개발자용)

### 1. 환경 설정
```bash
cd seoul_market_risk_ml
make setup-env               # 가상환경 생성
source seoul_risk_env/bin/activate
make install                 # 패키지 설치
```

### 2. 데이터 처리
```bash
make data-process           # 전처리 실행
```

### 3. 개별 모듈 테스트
```bash
python src/preprocessing/main.py        # 전처리
python src/feature_engineering/feature_engine.py  # 특성공학
python src/clustering/regional_clustering.py      # 지역 클러스터링
python src/models/global_model.py                # Global Model
```

---

## 📈 예상 완성 일정

- **Week 1-2**: Regional + Local Models 완성
- **Week 3**: Risk Score 엔진 + 변화점 탐지
- **Week 4**: 대출 계산 + LLM 통합
- **Week 5-6**: 테스팅 + 최적화

**전체 완성 목표**: 4-6주

---

## 💡 개발 팁

### 1. 모듈 재사용
기존 `global_model.py`와 `feature_engine.py`의 패턴을 최대한 활용

### 2. 에러 처리
모든 모델에서 데이터 부족시 상위 레벨로 fallback 구현 필수

### 3. 성능 모니터링
각 모델의 MAPE, R² 점수를 지속적으로 추적

### 4. 문서화
새로 작성하는 모든 클래스/함수에 docstring 필수

---

**다음 작업자에게**: 이 문서를 바탕으로 `src/models/regional_model.py` 부터 시작하세요. 질문사항은 `system_status.py`를 실행해서 현재 상태를 확인할 수 있습니다. 화이팅! 🚀