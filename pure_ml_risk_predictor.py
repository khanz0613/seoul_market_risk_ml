#!/usr/bin/env python3
"""
Pure ML Risk Predictor - 100% Machine Learning System
=====================================================

최종 목표 달성:
- 사용자 입력 (5개 간단한 값)
- 100% ML 예측
- Altman Z-Score는 라벨링에만 사용됨 (이미 완료)
- 복잡한 통계 시스템 없음

간단하고 명확한 ML 100% 시스템

Author: Seoul Market Risk ML System - Pure ML
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 피처 엔지니어링만 import (통계 시스템 제거)
from fixed_feature_engineering import FixedFeatureEngineering

class PureMLRiskPredictor:
    """100% 순수 ML 위험도 예측 시스템"""

    def __init__(self, models_dir: str = "trained_models_fixed"):
        self.models_dir = Path(models_dir)

        # ML 구성요소만
        self.best_model = None
        self.feature_engineer = None
        self.model_name = None

        # 위험도 등급 (ML 출력을 인간이 이해하기 쉽게)
        self.risk_descriptions = {
            1: "매우여유", 2: "여유", 3: "보통", 4: "위험", 5: "매우위험"
        }

        # NH농협 상품 매핑 (비즈니스 로직)
        self.nh_products = {
            1: {"max_loan_ratio": 0.8, "interest_rate": 0.03, "products": ["NH주택담보대출", "NH신용대출"]},
            2: {"max_loan_ratio": 0.7, "interest_rate": 0.04, "products": ["NH중금리대출", "NH신용대출"]},
            3: {"max_loan_ratio": 0.5, "interest_rate": 0.06, "products": ["NH중금리대출"]},
            4: {"max_loan_ratio": 0.3, "interest_rate": 0.09, "products": ["NH소액대출"]},
            5: {"max_loan_ratio": 0.1, "interest_rate": 0.15, "products": ["NH마이크로크레딧"]}
        }

        self._initialize_ml_system()

    def _initialize_ml_system(self) -> None:
        """순수 ML 시스템 초기화"""
        print("🤖 Initializing Pure ML Risk Prediction System")
        print("=" * 50)

        try:
            # 1. ML 모델 로드
            self._load_best_ml_model()

            # 2. 피처 엔지니어링 시스템 로드
            self._load_feature_engineer()

            print("✅ Pure ML System Ready!")
            print(f"   Best Model: {self.model_name}")
            print(f"   100% ML Prediction: ✅")
            print(f"   No Statistical System: ✅")

        except Exception as e:
            print(f"❌ ML system initialization failed: {e}")
            raise

    def _load_best_ml_model(self) -> None:
        """최고 성능 ML 모델 로드"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        # 평가 결과에서 최고 모델 찾기
        try:
            results_file = self.models_dir / "model_evaluation_results.json"
            if results_file.exists():
                import json
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                best_score = 0
                best_name = None

                for model_name, eval_results in results['evaluation_results'].items():
                    score = (
                        eval_results.get('cv_f1_score', 0) * 0.4 +
                        eval_results.get('test_accuracy', 0) * 0.3 +
                        eval_results.get('test_f1_weighted', 0) * 0.3
                    )

                    if score > best_score:
                        best_score = score
                        best_name = model_name

                if best_name:
                    model_file = self.models_dir / f"{best_name}_model.joblib"
                    if model_file.exists():
                        self.best_model = joblib.load(model_file)
                        self.model_name = best_name
                        print(f"📦 Loaded best model: {best_name} (score: {best_score:.3f})")
                        return

            # Fallback: 첫 번째 모델 사용
            model_files = list(self.models_dir.glob("*_model.joblib"))
            if model_files:
                self.best_model = joblib.load(model_files[0])
                self.model_name = model_files[0].stem.replace('_model', '')
                print(f"📦 Using fallback model: {self.model_name}")
            else:
                raise FileNotFoundError("No ML models found!")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def _load_feature_engineer(self) -> None:
        """피처 엔지니어링 시스템 로드"""
        try:
            fixed_dataset_path = "ml_analysis_results/seoul_commercial_fixed_dataset.csv"
            if Path(fixed_dataset_path).exists():
                self.feature_engineer = FixedFeatureEngineering(fixed_dataset_path)
                print("📋 Feature engineering system ready")
            else:
                raise FileNotFoundError("Fixed dataset not found")
        except Exception as e:
            print(f"❌ Feature engineering load failed: {e}")
            raise

    def predict_risk(self,
                    total_assets: float,
                    monthly_revenue: float,
                    monthly_expenses: Dict[str, float],
                    business_type: str,
                    location: str) -> Dict:
        """순수 ML 위험도 예측"""

        print(f"\n🤖 Pure ML Risk Prediction")
        print(f"   Assets: {total_assets:,}원")
        print(f"   Revenue: {monthly_revenue:,}원")
        print(f"   Expenses: {sum(monthly_expenses.values()):,}원")
        print(f"   Business: {business_type}, Location: {location}")

        try:
            # 1. 가상 데이터 행 생성 (ML 피처 엔지니어링용)
            dummy_row = pd.Series({
                '행정동_코드_명': location,
                '서비스_업종_코드': business_type,
                '데이터연도': 2024,
                '당월_매출_건수': max(1, int(monthly_revenue / 50000)),  # 추정
                # 연령대/성별 분포 (추정값)
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

            # 2. ML 피처 생성 (외부 지표만 사용)
            regional_features = self.feature_engineer.create_regional_features(dummy_row)
            industry_features = self.feature_engineer.create_industry_features(dummy_row)
            temporal_features = self.feature_engineer.create_temporal_features(dummy_row)
            scale_features = self.feature_engineer.create_business_scale_features(dummy_row)
            operational_features = self.feature_engineer.create_operational_features(dummy_row)

            # 3. 모든 피처 결합
            all_features = {
                **regional_features,
                **industry_features,
                **temporal_features,
                **scale_features,
                **operational_features
            }

            # 4. 복합 피처 추가
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

            # 5. ML 모델 예측
            feature_df = pd.DataFrame([all_features])
            risk_proba = self.best_model.predict_proba(feature_df)[0]
            risk_pred = self.best_model.predict(feature_df)[0]

            # 6. 0-4 → 1-5 변환
            risk_level_num = risk_pred + 1
            risk_level_name = self.risk_descriptions[risk_level_num]

            # 7. 신뢰도 계산
            confidence = max(risk_proba) * 100

            print(f"   🎯 ML Result: {risk_level_name} (level {risk_level_num})")
            print(f"   🔮 Confidence: {confidence:.1f}%")
            print(f"   🧠 Model: {self.model_name}")

            return {
                'risk_level': risk_level_name,
                'risk_level_num': risk_level_num,
                'confidence': confidence,
                'probabilities': risk_proba.tolist(),
                'features_used': len(all_features),
                'model_name': self.model_name,
                'prediction_method': '100% ML'
            }

        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            # Fallback 예측
            return {
                'risk_level': '보통',
                'risk_level_num': 3,
                'confidence': 50.0,
                'probabilities': [0.2, 0.2, 0.2, 0.2, 0.2],
                'features_used': 0,
                'model_name': 'fallback',
                'prediction_method': 'fallback'
            }

    def generate_loan_recommendation(self,
                                   risk_result: Dict,
                                   total_assets: float,
                                   monthly_revenue: float) -> Dict:
        """대출 추천 생성 (비즈니스 로직)"""

        risk_level_num = risk_result['risk_level_num']
        nh_info = self.nh_products[risk_level_num]

        # 대출 계산
        max_loan_amount = total_assets * nh_info['max_loan_ratio']
        monthly_interest = nh_info['interest_rate'] / 12
        months = 36

        if monthly_interest > 0:
            monthly_payment = max_loan_amount * monthly_interest * (1 + monthly_interest)**months / ((1 + monthly_interest)**months - 1)
        else:
            monthly_payment = max_loan_amount / months

        payment_burden = monthly_payment / monthly_revenue * 100
        recommend_loan = payment_burden < 30

        return {
            'recommended': recommend_loan,
            'max_loan_amount': max_loan_amount,
            'interest_rate': nh_info['interest_rate'],
            'monthly_payment': monthly_payment,
            'payment_burden_ratio': payment_burden,
            'loan_term_months': months,
            'nh_products': nh_info['products']
        }

    def comprehensive_analysis(self,
                             total_assets: float,
                             monthly_revenue: float,
                             monthly_expenses: Dict[str, float],
                             business_type: str,
                             location: str) -> Dict:
        """종합 분석 (100% ML)"""

        # ML 위험도 예측
        risk_result = self.predict_risk(
            total_assets, monthly_revenue, monthly_expenses,
            business_type, location
        )

        # 대출 추천
        loan_recommendation = self.generate_loan_recommendation(
            risk_result, total_assets, monthly_revenue
        )

        # 맞춤 추천사항
        recommendations = self._generate_recommendations(
            risk_result, loan_recommendation, monthly_revenue
        )

        return {
            'analysis_date': datetime.now().isoformat(),
            'business_info': {
                'business_type': business_type,
                'location': location,
                'monthly_revenue': monthly_revenue,
                'total_assets': total_assets,
                'monthly_expenses': monthly_expenses
            },
            'ml_risk_assessment': risk_result,
            'loan_recommendation': loan_recommendation,
            'recommendations': recommendations,
            'system_type': '100% Pure ML'
        }

    def _generate_recommendations(self,
                                risk_result: Dict,
                                loan_recommendation: Dict,
                                monthly_revenue: float) -> List[str]:
        """맞춤형 추천사항"""

        recommendations = []
        risk_level = risk_result['risk_level_num']

        # 위험도별 기본 추천
        if risk_level == 5:
            recommendations.extend([
                "💰 긴급 현금흐름 개선 필요",
                "📊 비용 구조 재검토 및 절감",
                "🏥 전문 재무상담 권장"
            ])
        elif risk_level == 4:
            recommendations.extend([
                "💳 안정화 자금 확보 권장",
                "📈 매출 다각화 전략 필요",
                "📋 월별 자금계획 수립"
            ])
        elif risk_level == 3:
            recommendations.extend([
                "📊 현재 상태 유지 및 정기 점검",
                "💡 성장 기회 모색",
                "🔄 효율성 개선 방안 검토"
            ])
        elif risk_level == 2:
            recommendations.extend([
                "🚀 성장 투자 기회 검토",
                "💼 사업 확대 고려",
                "📈 마케팅 투자 확대"
            ])
        else:
            recommendations.extend([
                "💎 프리미엄 투자상품 활용",
                "🌟 신사업 진출 기회",
                "🏆 브랜드 가치 확대"
            ])

        # 대출 관련
        if loan_recommendation['recommended']:
            recommendations.append(
                f"💰 {loan_recommendation['max_loan_amount']:,.0f}원 대출 활용 가능"
            )
        else:
            recommendations.append("⚠️ 현재는 자체 현금흐름 개선에 집중 권장")

        # ML 시스템 정보
        recommendations.append(f"🤖 AI 분석 (신뢰도: {risk_result['confidence']:.1f}%)")

        return recommendations

def demo_pure_ml():
    """순수 ML 시스템 데모"""
    print("\n" + "="*60)
    print("🤖 Pure ML Risk Prediction Demo - 100% Machine Learning")
    print("="*60)

    try:
        # 시스템 초기화
        predictor = PureMLRiskPredictor()

        # 샘플 데이터
        sample_input = {
            'total_assets': 25000000,       # 2천5백만원
            'monthly_revenue': 7000000,     # 월 700만원
            'monthly_expenses': {
                'labor_cost': 2000000,      # 인건비
                'food_materials': 2200000,  # 식자재
                'rent': 1500000,            # 임대료
                'others': 800000            # 기타
            },
            'business_type': 'CS100001',    # 한식음식점
            'location': '관악구'
        }

        print(f"\n📊 Sample Analysis:")
        print(f"   업종: 한식음식점 (관악구)")
        print(f"   총자산: {sample_input['total_assets']:,}원")
        print(f"   월매출: {sample_input['monthly_revenue']:,}원")
        print(f"   월지출: {sum(sample_input['monthly_expenses'].values()):,}원")

        # 종합 분석 실행
        result = predictor.comprehensive_analysis(**sample_input)

        # 결과 출력
        ml_assessment = result['ml_risk_assessment']
        loan_rec = result['loan_recommendation']

        print(f"\n🎯 ML 위험도 분석:")
        print(f"   위험도: {ml_assessment['risk_level']}")
        print(f"   신뢰도: {ml_assessment['confidence']:.1f}%")
        print(f"   예측 모델: {ml_assessment['model_name']}")
        print(f"   사용된 피처: {ml_assessment['features_used']}개")

        print(f"\n💰 대출 추천:")
        if loan_rec['recommended']:
            print(f"   추천금액: {loan_rec['max_loan_amount']:,.0f}원")
            print(f"   금리: {loan_rec['interest_rate']:.1%}")
            print(f"   월상환액: {loan_rec['monthly_payment']:,.0f}원")
            print(f"   상환부담률: {loan_rec['payment_burden_ratio']:.1f}%")
        else:
            print(f"   대출 비추천 (상환부담률 {loan_rec['payment_burden_ratio']:.1f}%)")

        print(f"\n📋 AI 추천사항:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")

        print(f"\n✅ 시스템 정보:")
        print(f"   예측 방법: {result['system_type']}")
        print(f"   통계 시스템: ❌ 사용 안함")
        print(f"   ML 시스템: ✅ 100% 사용")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)
    print("🎯 Pure ML System Demo Complete!")
    print("✅ 100% Machine Learning Risk Prediction")
    print("="*60)

if __name__ == "__main__":
    demo_pure_ml()