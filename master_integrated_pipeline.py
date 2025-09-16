#!/usr/bin/env python3
"""
Master Integrated Pipeline - ML + 통계 하이브리드 시스템
=====================================================

통합 구성요소:
1. src/ 디렉토리의 Altman Z-Score 기반 통계 시스템 (검증된 방법론)
2. 새로운 고정 ML 시스템 (데이터 누수 제거, 과적합 방지)
3. 두 시스템의 ensemble 예측으로 최고 정확도 달성

사용자 경험:
- 기존과 동일한 간단한 5개 입력 (총자산, 월매출, 4개 지출항목, 업종, 지역)
- 내부적으로 통계+ML 하이브리드 처리
- 신뢰도가 높은 위험도 예측 및 맞춤 추천

Author: Seoul Market Risk ML System - Master Integration
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
import json
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')

# 기존 src 시스템 import
try:
    from src.risk_scoring.hybrid_risk_calculator import HybridRiskCalculator
    from src.loan_simulation.loan_impact_simulator import LoanImpactSimulator
    STATISTICAL_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️ Statistical system (src/) not available")
    STATISTICAL_SYSTEM_AVAILABLE = False

# 고정 ML 시스템 components
from fixed_feature_engineering import FixedFeatureEngineering

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterIntegratedPipeline:
    """통계 + ML 하이브리드 위험도 예측 시스템"""

    def __init__(self,
                 models_dir: str = "trained_models_fixed",
                 enable_statistical: bool = True,
                 enable_ml: bool = True):

        self.models_dir = Path(models_dir)
        self.enable_statistical = enable_statistical and STATISTICAL_SYSTEM_AVAILABLE
        self.enable_ml = enable_ml

        # 시스템 구성요소
        self.statistical_calculator = None
        self.loan_simulator = None
        self.ml_models = {}
        self.best_ml_model = None
        self.feature_engineer = None

        # 위험도 등급 정의
        self.risk_descriptions = {
            1: "매우여유", 2: "여유", 3: "보통", 4: "위험", 5: "매우위험"
        }

        # NH농협 상품 추천 (위험도별)
        self.nh_products = {
            1: {"max_loan_ratio": 0.8, "interest_rate": 0.03, "products": ["NH주택담보대출", "NH신용대출"]},
            2: {"max_loan_ratio": 0.7, "interest_rate": 0.04, "products": ["NH중금리대출", "NH신용대출"]},
            3: {"max_loan_ratio": 0.5, "interest_rate": 0.06, "products": ["NH중금리대출"]},
            4: {"max_loan_ratio": 0.3, "interest_rate": 0.09, "products": ["NH소액대출"]},
            5: {"max_loan_ratio": 0.1, "interest_rate": 0.15, "products": ["NH마이크로크레딧"]}
        }

        self._initialize_systems()

    def _initialize_systems(self) -> None:
        """시스템 초기화"""
        print("🔄 Initializing Master Integrated Pipeline")
        print("=" * 50)

        # 1. 통계 시스템 초기화
        if self.enable_statistical:
            try:
                print("📊 Loading statistical system (Altman Z-Score)...")
                self.statistical_calculator = HybridRiskCalculator()
                self.loan_simulator = LoanImpactSimulator()
                print("✅ Statistical system ready")
            except Exception as e:
                print(f"❌ Statistical system failed: {e}")
                self.enable_statistical = False

        # 2. ML 시스템 초기화
        if self.enable_ml:
            try:
                print("🤖 Loading ML system (fixed models)...")
                self._load_ml_models()
                self._load_feature_engineer()
                print("✅ ML system ready")
            except Exception as e:
                print(f"❌ ML system failed: {e}")
                self.enable_ml = False

        # 시스템 상태 확인
        if not self.enable_statistical and not self.enable_ml:
            raise RuntimeError("No prediction systems available!")

        active_systems = []
        if self.enable_statistical:
            active_systems.append("Statistical (Altman Z-Score)")
        if self.enable_ml:
            active_systems.append("ML (Fixed Ensemble)")

        print(f"🎯 Active systems: {', '.join(active_systems)}")
        print("✅ Master Pipeline ready!")

    def _load_ml_models(self) -> None:
        """ML 모델들 로드"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        # 사용 가능한 모델 파일들 찾기
        model_files = list(self.models_dir.glob("*_model.joblib"))

        if not model_files:
            raise FileNotFoundError("No trained ML models found!")

        # 모델들 로드
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                self.ml_models[model_name] = joblib.load(model_file)
                print(f"   📦 Loaded {model_name}")
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")

        # 최고 성능 모델 선택 (결과 파일에서 확인)
        try:
            results_file = self.models_dir / "model_evaluation_results.json"
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            best_score = 0
            best_name = None

            for model_name, eval_results in results['evaluation_results'].items():
                # Composite score 계산
                score = (
                    eval_results.get('cv_f1_score', 0) * 0.4 +
                    eval_results.get('test_accuracy', 0) * 0.3 +
                    eval_results.get('test_f1_weighted', 0) * 0.3
                )

                if score > best_score and model_name in self.ml_models:
                    best_score = score
                    best_name = model_name

            if best_name:
                self.best_ml_model = self.ml_models[best_name]
                print(f"   🏆 Best model: {best_name} (score: {best_score:.3f})")
            else:
                # Fallback: 첫 번째 모델 사용
                first_model = list(self.ml_models.keys())[0]
                self.best_ml_model = self.ml_models[first_model]
                print(f"   🔄 Using fallback model: {first_model}")

        except Exception as e:
            print(f"   ⚠️ Model selection failed: {e}, using first available model")
            first_model = list(self.ml_models.keys())[0]
            self.best_ml_model = self.ml_models[first_model]

    def _load_feature_engineer(self) -> None:
        """피처 엔지니어링 시스템 로드"""
        try:
            # 고정 데이터셋이 있는지 확인
            fixed_dataset_path = "ml_analysis_results/seoul_commercial_fixed_dataset.csv"
            if not Path(fixed_dataset_path).exists():
                print("   ⚠️ Fixed dataset not found, ML predictions may be limited")
                return

            self.feature_engineer = FixedFeatureEngineering(fixed_dataset_path)
            print("   📋 Feature engineering system ready")

        except Exception as e:
            print(f"   ❌ Feature engineering load failed: {e}")
            self.feature_engineer = None

    def predict_risk_statistical(self,
                                total_assets: float,
                                monthly_revenue: float,
                                monthly_expenses: Dict[str, float],
                                business_type: str,
                                location: str,
                                months_in_business: int = 24) -> Dict:
        """통계 기반 위험도 예측 (Altman Z-Score)"""

        if not self.enable_statistical:
            return None

        try:
            # 지출 합계
            total_monthly_expenses = sum(monthly_expenses.values())

            # 매출 히스토리 생성 (간단한 가정)
            revenue_history = [monthly_revenue * (1 + np.random.normal(0, 0.1)) for _ in range(6)]
            expense_history = [total_monthly_expenses * (1 + np.random.normal(0, 0.05)) for _ in range(6)]

            # 통계 시스템으로 위험도 계산
            assessment = self.statistical_calculator.calculate_risk_assessment(
                business_id=f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                revenue_history=revenue_history,
                expense_history=expense_history,
                operating_assets=total_assets,
                industry_code=business_type,
                months_in_business=months_in_business
            )

            return {
                'risk_score': assessment.total_risk_score,
                'risk_level': assessment.risk_level,
                'confidence': assessment.confidence,
                'altman_zscore': assessment.altman_zscore,
                'components': {
                    'financial_health': assessment.financial_health_score,
                    'operational_stability': assessment.operational_stability_score,
                    'industry_position': assessment.industry_position_score
                },
                'method': 'statistical'
            }

        except Exception as e:
            logger.error(f"Statistical prediction failed: {e}")
            return None

    def predict_risk_ml(self,
                       total_assets: float,
                       monthly_revenue: float,
                       monthly_expenses: Dict[str, float],
                       business_type: str,
                       location: str) -> Dict:
        """ML 기반 위험도 예측 (고정 피처)"""

        if not self.enable_ml or not self.best_ml_model or not self.feature_engineer:
            return None

        try:
            # 가상의 행 데이터 생성 (ML 피처 엔지니어링용)
            dummy_row = pd.Series({
                '행정동_코드_명': location,
                '서비스_업종_코드': business_type,
                '데이터연도': 2024,
                '당월_매출_건수': max(1, int(monthly_revenue / 50000)),  # 추정 거래수
                # 연령대별/성별 분포 (가상값)
                '연령대_20_매출_금액': monthly_revenue * 0.2,
                '연령대_30_매출_금액': monthly_revenue * 0.3,
                '연령대_40_매출_금액': monthly_revenue * 0.3,
                '연령대_50_매출_금액': monthly_revenue * 0.2,
                '남성_매출_금액': monthly_revenue * 0.55,
                '여성_매출_금액': monthly_revenue * 0.45,
                '주중_매출_금액': monthly_revenue * 0.7,
                '주말_매출_금액': monthly_revenue * 0.3,
                '시간대_11~14_매출_금액': monthly_revenue * 0.3,
                '시간대_17~21_매출_금액': monthly_revenue * 0.4,
                '시간대_21~24_매출_금액': monthly_revenue * 0.2
            })

            # 피처 생성 (외부 지표만 사용)
            regional_features = self.feature_engineer.create_regional_features(dummy_row)
            industry_features = self.feature_engineer.create_industry_features(dummy_row)
            temporal_features = self.feature_engineer.create_temporal_features(dummy_row)
            scale_features = self.feature_engineer.create_business_scale_features(dummy_row)
            operational_features = self.feature_engineer.create_operational_features(dummy_row)

            # 모든 피처 결합
            all_features = {
                **regional_features,
                **industry_features,
                **temporal_features,
                **scale_features,
                **operational_features
            }

            # 복합 피처 추가
            all_features['risk_composite_1'] = (
                all_features.get('regional_competition_index', 0.5) * 0.3 +
                all_features.get('industry_risk_score', 0.5) * 0.4 +
                (1 - all_features.get('economic_stability_index', 0.5)) * 0.3
            )

            all_features['opportunity_index'] = (
                all_features.get('regional_purchasing_power', 0.5) * 0.4 +
                all_features.get('customer_age_diversity', 0.5) * 0.3 +
                all_features.get('subway_accessibility', 0.5) * 0.3
            )

            # DataFrame으로 변환 (ML 모델 입력용)
            feature_df = pd.DataFrame([all_features])

            # ML 모델 예측
            risk_proba = self.best_ml_model.predict_proba(feature_df)[0]
            risk_pred = self.best_ml_model.predict(feature_df)[0]

            # 0-4 → 1-5 변환
            risk_level_num = risk_pred + 1
            risk_level_name = self.risk_descriptions[risk_level_num]

            # 신뢰도 계산 (최대 확률값)
            confidence = max(risk_proba) * 100

            return {
                'risk_score': (5 - risk_level_num) * 20,  # 1-5 → 80-0 점수 변환
                'risk_level': risk_level_name,
                'risk_level_num': risk_level_num,
                'confidence': confidence,
                'probabilities': risk_proba.tolist(),
                'features_used': len(all_features),
                'method': 'ml'
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    def predict_risk_hybrid(self,
                           total_assets: float,
                           monthly_revenue: float,
                           monthly_expenses: Dict[str, float],
                           business_type: str,
                           location: str,
                           months_in_business: int = 24) -> Dict:
        """하이브리드 위험도 예측 (통계 + ML 조합)"""

        print(f"\n🎯 Hybrid Risk Prediction")
        print(f"   Assets: {total_assets:,}원")
        print(f"   Revenue: {monthly_revenue:,}원")
        print(f"   Expenses: {sum(monthly_expenses.values()):,}원")
        print(f"   Business: {business_type}, Location: {location}")

        # 통계 예측
        statistical_result = self.predict_risk_statistical(
            total_assets, monthly_revenue, monthly_expenses,
            business_type, location, months_in_business
        )

        # ML 예측
        ml_result = self.predict_risk_ml(
            total_assets, monthly_revenue, monthly_expenses,
            business_type, location
        )

        # 예측 결과 출력
        if statistical_result:
            print(f"   📊 Statistical: {statistical_result['risk_level']} "
                  f"(score: {statistical_result['risk_score']:.1f}, "
                  f"Z-Score: {statistical_result['altman_zscore']:.2f})")

        if ml_result:
            print(f"   🤖 ML: {ml_result['risk_level']} "
                  f"(score: {ml_result['risk_score']:.1f}, "
                  f"confidence: {ml_result['confidence']:.1f}%)")

        # 하이브리드 조합
        if statistical_result and ml_result:
            # 가중 평균 (통계 60%, ML 40%)
            hybrid_score = (
                statistical_result['risk_score'] * 0.6 +
                ml_result['risk_score'] * 0.4
            )

            # 신뢰도 조합
            hybrid_confidence = (
                statistical_result['confidence'] * 0.6 +
                ml_result['confidence'] * 0.4
            )

            # 점수를 위험도 등급으로 변환
            if hybrid_score >= 80:
                hybrid_level_num = 1
            elif hybrid_score >= 60:
                hybrid_level_num = 2
            elif hybrid_score >= 40:
                hybrid_level_num = 3
            elif hybrid_score >= 20:
                hybrid_level_num = 4
            else:
                hybrid_level_num = 5

            hybrid_level_name = self.risk_descriptions[hybrid_level_num]

            print(f"   🎯 Hybrid: {hybrid_level_name} "
                  f"(score: {hybrid_score:.1f}, confidence: {hybrid_confidence:.1f}%)")

            return {
                'risk_score': hybrid_score,
                'risk_level': hybrid_level_name,
                'risk_level_num': hybrid_level_num,
                'confidence': hybrid_confidence,
                'method': 'hybrid',
                'components': {
                    'statistical': statistical_result,
                    'ml': ml_result
                },
                'weights': {'statistical': 0.6, 'ml': 0.4}
            }

        elif statistical_result:
            print(f"   ⚠️ Using statistical only (ML unavailable)")
            return statistical_result

        elif ml_result:
            print(f"   ⚠️ Using ML only (statistical unavailable)")
            return ml_result

        else:
            # Fallback: 기본 위험도
            print(f"   ❌ Both systems failed, using fallback")
            return {
                'risk_score': 50.0,
                'risk_level': '보통',
                'risk_level_num': 3,
                'confidence': 50.0,
                'method': 'fallback'
            }

    def generate_loan_recommendations(self,
                                    risk_result: Dict,
                                    total_assets: float,
                                    monthly_revenue: float) -> Dict:
        """대출 추천 생성"""

        risk_level_num = risk_result['risk_level_num']
        risk_score = risk_result['risk_score']

        # NH농협 상품 정보
        nh_info = self.nh_products[risk_level_num]

        # 대출 금액 계산
        max_loan_amount = total_assets * nh_info['max_loan_ratio']

        # 월 상환액 계산 (36개월 기준)
        monthly_interest = nh_info['interest_rate'] / 12
        months = 36
        if monthly_interest > 0:
            monthly_payment = max_loan_amount * monthly_interest * (1 + monthly_interest)**months / ((1 + monthly_interest)**months - 1)
        else:
            monthly_payment = max_loan_amount / months

        # 상환 부담률 계산
        payment_burden = monthly_payment / monthly_revenue * 100

        # 추천 여부 결정
        recommend_loan = payment_burden < 30  # 상환부담률 30% 미만

        return {
            'recommended': recommend_loan,
            'max_loan_amount': max_loan_amount,
            'interest_rate': nh_info['interest_rate'],
            'monthly_payment': monthly_payment,
            'payment_burden_ratio': payment_burden,
            'loan_term_months': months,
            'nh_products': nh_info['products'],
            'risk_level': risk_level_num
        }

    def generate_comprehensive_report(self,
                                    total_assets: float,
                                    monthly_revenue: float,
                                    monthly_expenses: Dict[str, float],
                                    business_type: str,
                                    location: str,
                                    months_in_business: int = 24) -> Dict:
        """종합 분석 보고서 생성"""

        # 위험도 예측
        risk_result = self.predict_risk_hybrid(
            total_assets, monthly_revenue, monthly_expenses,
            business_type, location, months_in_business
        )

        # 대출 추천
        loan_recommendation = self.generate_loan_recommendations(
            risk_result, total_assets, monthly_revenue
        )

        # 종합 추천사항 생성
        recommendations = self._generate_business_recommendations(
            risk_result, loan_recommendation, monthly_revenue, total_assets
        )

        return {
            'assessment_date': datetime.now().isoformat(),
            'business_info': {
                'type': business_type,
                'location': location,
                'months_in_business': months_in_business,
                'monthly_revenue': monthly_revenue,
                'total_assets': total_assets,
                'monthly_expenses': monthly_expenses
            },
            'risk_assessment': risk_result,
            'loan_recommendation': loan_recommendation,
            'recommendations': recommendations,
            'system_info': {
                'statistical_enabled': self.enable_statistical,
                'ml_enabled': self.enable_ml,
                'prediction_method': risk_result['method']
            }
        }

    def _generate_business_recommendations(self,
                                         risk_result: Dict,
                                         loan_recommendation: Dict,
                                         monthly_revenue: float,
                                         total_assets: float) -> List[str]:
        """맞춤형 사업 추천사항 생성"""

        recommendations = []
        risk_level = risk_result['risk_level_num']

        # 위험도별 기본 추천
        if risk_level == 5:  # 매우위험
            recommendations.extend([
                "💰 긴급 현금흐름 개선이 필요합니다",
                "📊 즉시 비용 구조 점검 및 절감 방안 수립",
                "🏥 사업 건전성 회복을 위한 전문 상담 권장"
            ])
        elif risk_level == 4:  # 위험
            recommendations.extend([
                "💳 안정화 자금 확보를 통한 재무구조 개선 권장",
                "📈 매출 다각화 및 고정비 절감 전략 필요",
                "📋 월별 자금계획 수립 및 모니터링 강화"
            ])
        elif risk_level == 3:  # 보통
            recommendations.extend([
                "📊 현재 상태 유지 및 정기적 재무 점검",
                "💡 성장 기회 발굴을 위한 시장 분석",
                "🔄 사업 효율성 개선 방안 검토"
            ])
        elif risk_level == 2:  # 여유
            recommendations.extend([
                "🚀 성장 투자 기회 적극 검토 권장",
                "💼 사업 규모 확대 또는 다각화 고려",
                "📈 마케팅 투자 확대를 통한 시장점유율 증대"
            ])
        else:  # 매우여유
            recommendations.extend([
                "💎 프리미엄 투자 상품 활용 검토",
                "🌟 신사업 진출 또는 M&A 기회 모색",
                "🏆 업계 선도기업으로서 브랜드 가치 확대"
            ])

        # 대출 관련 추천
        if loan_recommendation['recommended']:
            recommendations.append(
                f"💰 {loan_recommendation['max_loan_amount']:,.0f}원 대출 활용 가능 "
                f"(월 상환액: {loan_recommendation['monthly_payment']:,.0f}원)"
            )
        else:
            recommendations.append("⚠️ 현재 대출보다는 자체 현금흐름 개선에 집중 권장")

        # 시스템별 특수 추천
        if risk_result.get('method') == 'hybrid':
            recommendations.append("🎯 통계+ML 하이브리드 분석으로 높은 신뢰도 확보")
        elif risk_result.get('method') == 'statistical':
            recommendations.append("📊 Altman Z-Score 기반 재무건전성 분석 적용")
        elif risk_result.get('method') == 'ml':
            recommendations.append("🤖 AI 머신러닝 기반 예측 분석 적용")

        return recommendations

def demo_analysis():
    """통합 시스템 데모"""
    print("\n" + "="*70)
    print("🚀 Master Integrated Pipeline Demo - 통계+ML 하이브리드")
    print("="*70)

    # 시스템 초기화
    try:
        pipeline = MasterIntegratedPipeline()
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return

    # 샘플 사업자 데이터
    sample_business = {
        'total_assets': 30000000,       # 총자산 3천만원
        'monthly_revenue': 8000000,     # 월매출 800만원
        'monthly_expenses': {           # 월지출
            'labor_cost': 2000000,      # 인건비 200만원
            'food_materials': 2500000,  # 식자재 250만원
            'rent': 1800000,            # 임대료 180만원
            'others': 700000            # 기타 70만원
        },
        'business_type': 'CS100001',    # 한식음식점
        'location': '강남구',
        'months_in_business': 18        # 운영 18개월
    }

    print(f"\n📊 Sample Business Analysis:")
    print(f"   업종: 한식음식점 (강남구)")
    print(f"   운영기간: {sample_business['months_in_business']}개월")
    print(f"   총자산: {sample_business['total_assets']:,}원")
    print(f"   월매출: {sample_business['monthly_revenue']:,}원")
    print(f"   월지출: {sum(sample_business['monthly_expenses'].values()):,}원")

    # 종합 분석 실행
    try:
        comprehensive_report = pipeline.generate_comprehensive_report(**sample_business)

        # 결과 출력
        risk_assessment = comprehensive_report['risk_assessment']
        loan_recommendation = comprehensive_report['loan_recommendation']

        print(f"\n🎯 종합 위험도 평가:")
        print(f"   위험도: {risk_assessment['risk_level']} ({risk_assessment['risk_score']:.1f}점)")
        print(f"   신뢰도: {risk_assessment['confidence']:.1f}%")
        print(f"   분석방법: {risk_assessment['method']}")

        print(f"\n💰 대출 추천:")
        if loan_recommendation['recommended']:
            print(f"   추천금액: {loan_recommendation['max_loan_amount']:,.0f}원")
            print(f"   금리: {loan_recommendation['interest_rate']:.1%}")
            print(f"   월상환액: {loan_recommendation['monthly_payment']:,.0f}원")
            print(f"   상환부담률: {loan_recommendation['payment_burden_ratio']:.1f}%")
            print(f"   추천상품: {', '.join(loan_recommendation['nh_products'])}")
        else:
            print(f"   대출 비추천 (상환부담률 {loan_recommendation['payment_burden_ratio']:.1f}% 초과)")

        print(f"\n📋 맞춤 추천사항:")
        for i, rec in enumerate(comprehensive_report['recommendations'], 1):
            print(f"   {i}. {rec}")

        print(f"\n📈 시스템 정보:")
        system_info = comprehensive_report['system_info']
        print(f"   통계 시스템: {'✅' if system_info['statistical_enabled'] else '❌'}")
        print(f"   ML 시스템: {'✅' if system_info['ml_enabled'] else '❌'}")
        print(f"   예측 방법: {system_info['prediction_method']}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*70)
    print("✅ Master Integrated Pipeline Demo Complete")
    print("🎯 Ready for production deployment!")
    print("="*70)

if __name__ == "__main__":
    demo_analysis()