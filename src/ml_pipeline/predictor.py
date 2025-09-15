"""
ML Prediction Pipeline
학습된 모델을 사용한 위험도 예측
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HybridRiskPredictor:
    """하이브리드 위험도 예측기"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []

        self.load_models()

    def load_models(self):
        """학습된 모델 로드"""
        try:
            # 피처 컬럼 로드
            feature_path = self.model_dir / "feature_columns.joblib"
            if feature_path.exists():
                self.feature_columns = joblib.load(feature_path)

            # 회귀 모델 로드
            reg_rf_path = self.model_dir / "regression_random_forest_regressor.joblib"
            if reg_rf_path.exists():
                self.models['regression_rf'] = joblib.load(reg_rf_path)

            reg_lr_path = self.model_dir / "regression_linear_regression.joblib"
            if reg_lr_path.exists():
                self.models['regression_lr'] = joblib.load(reg_lr_path)

            # 분류 모델 로드
            clf_rf_path = self.model_dir / "classification_random_forest_classifier.joblib"
            if clf_rf_path.exists():
                self.models['classification_rf'] = joblib.load(clf_rf_path)

            clf_lr_path = self.model_dir / "classification_logistic_regression.joblib"
            if clf_lr_path.exists():
                self.models['classification_lr'] = joblib.load(clf_lr_path)

            # 스케일러 로드
            reg_scaler_path = self.model_dir / "regression_scaler.joblib"
            if reg_scaler_path.exists():
                self.scalers['regression'] = joblib.load(reg_scaler_path)

            clf_scaler_path = self.model_dir / "classification_scaler.joblib"
            if clf_scaler_path.exists():
                self.scalers['classification'] = joblib.load(clf_scaler_path)

            logger.info(f"모델 로드 완료: {len(self.models)}개 모델")

        except Exception as e:
            logger.warning(f"모델 로드 실패: {e}")

    def prepare_features(self, business_data: Dict) -> pd.DataFrame:
        """예측용 피처 준비"""

        # 필수 피처들을 기본값으로 초기화
        features = {}

        # Altman Z-Score 기반 피처
        features['working_capital_ratio'] = business_data.get('working_capital_ratio', 0.1)
        features['retained_earnings_ratio'] = business_data.get('retained_earnings_ratio', 0.05)
        features['ebit_ratio'] = business_data.get('ebit_ratio', 0.03)
        features['equity_debt_ratio'] = business_data.get('equity_debt_ratio', 1.0)
        features['asset_turnover'] = business_data.get('asset_turnover', 0.5)
        features['altman_zscore'] = business_data.get('altman_zscore', 1.8)

        # 영업 안정성 피처
        features['avg_growth_rate'] = business_data.get('avg_growth_rate', 0.0)
        features['growth_volatility'] = business_data.get('growth_volatility', 0.1)
        features['revenue_cv'] = business_data.get('revenue_cv', 0.2)
        features['business_quarters'] = business_data.get('business_quarters', 4)
        features['revenue_consistency'] = business_data.get('revenue_consistency', 0.8)
        features['seasonality_strength'] = business_data.get('seasonality_strength', 0.1)

        # 업종 비교 피처
        features['revenue_vs_industry'] = business_data.get('revenue_vs_industry', 1.0)
        features['profit_margin_vs_industry'] = business_data.get('profit_margin_vs_industry', 1.0)
        features['industry_percentile_revenue'] = business_data.get('industry_percentile_revenue', 50.0)
        features['industry_percentile_profit'] = business_data.get('industry_percentile_profit', 50.0)

        # 기본 피처
        features['quarters_active'] = business_data.get('quarters_active', 4)
        features['latest_revenue'] = business_data.get('latest_revenue', 5000000)
        features['latest_profit_margin'] = business_data.get('latest_profit_margin', 0.1)

        # DataFrame으로 변환
        feature_df = pd.DataFrame([features])

        # 피처 순서 맞추기
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(feature_df.columns)
            for col in missing_cols:
                feature_df[col] = 0.0

            feature_df = feature_df[self.feature_columns]

        return feature_df

    def predict_risk_score(self, business_data: Dict) -> Dict:
        """위험도 점수 예측"""

        if 'regression_rf' not in self.models:
            logger.warning("회귀 모델이 로드되지 않음")
            return {'risk_score': 50.0, 'confidence': 0.0}

        # 피처 준비
        features = self.prepare_features(business_data)

        # Random Forest 회귀로 예측
        risk_score = self.models['regression_rf'].predict(features)[0]

        # 예측 신뢰도 계산 (트리들의 분산)
        if hasattr(self.models['regression_rf'], 'estimators_'):
            tree_predictions = [tree.predict(features)[0] for tree in self.models['regression_rf'].estimators_]
            confidence = max(0, min(100, 100 - np.std(tree_predictions)))
        else:
            confidence = 70.0

        # 점수 범위 제한
        risk_score = max(0, min(100, risk_score))

        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'model_used': 'Random Forest Regression'
        }

    def predict_risk_category(self, business_data: Dict) -> Dict:
        """위험 여부 분류 예측"""

        if 'classification_rf' not in self.models:
            logger.warning("분류 모델이 로드되지 않음")
            return {'is_risky': 0, 'risk_probability': 0.5}

        # 피처 준비
        features = self.prepare_features(business_data)

        # Random Forest 분류로 예측
        risk_class = self.models['classification_rf'].predict(features)[0]
        risk_proba = self.models['classification_rf'].predict_proba(features)[0]

        return {
            'is_risky': int(risk_class),
            'risk_probability': float(risk_proba[1]),  # 위험할 확률
            'safe_probability': float(risk_proba[0]),  # 안전할 확률
            'model_used': 'Random Forest Classification'
        }

    def comprehensive_prediction(self, business_data: Dict) -> Dict:
        """종합 예측"""

        # 점수 예측
        score_result = self.predict_risk_score(business_data)

        # 분류 예측
        class_result = self.predict_risk_category(business_data)

        # 위험도 등급 결정
        risk_score = score_result['risk_score']
        if risk_score >= 81:
            risk_level = "매우좋음"
        elif risk_score >= 61:
            risk_level = "좋음"
        elif risk_score >= 41:
            risk_level = "적정"
        elif risk_score >= 21:
            risk_level = "위험군"
        else:
            risk_level = "매우위험"

        # 대출 추천
        loan_recommendation = self.get_loan_recommendation(risk_score, business_data)

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'is_risky': class_result['is_risky'],
            'risk_probability': class_result['risk_probability'],
            'confidence': score_result['confidence'],
            'loan_recommendation': loan_recommendation,
            'prediction_timestamp': pd.Timestamp.now().isoformat()
        }

    def get_loan_recommendation(self, risk_score: float, business_data: Dict) -> Dict:
        """대출 추천 계산"""

        latest_revenue = business_data.get('latest_revenue', 5000000)

        if risk_score <= 20:  # 매우위험
            target_score = 40
            score_gap = target_score - risk_score
            loan_amount = latest_revenue * (score_gap / 100.0) * 3.0
            return {
                'action': 'emergency_loan',
                'amount': loan_amount,
                'reason': '긴급 자금 지원 필요'
            }

        elif risk_score <= 40:  # 위험군
            target_score = 60
            score_gap = target_score - risk_score
            loan_amount = latest_revenue * (score_gap / 100.0) * 2.0
            return {
                'action': 'stabilization_loan',
                'amount': loan_amount,
                'reason': '재무구조 안정화'
            }

        elif risk_score <= 60:  # 적정
            return {
                'action': 'monitoring',
                'amount': 0,
                'reason': '현 상태 유지'
            }

        else:  # 좋음/매우좋음
            return {
                'action': 'investment_opportunity',
                'amount': 0,
                'reason': '성장투자 검토'
            }

    def batch_predict(self, business_list: List[Dict]) -> List[Dict]:
        """여러 사업자 일괄 예측"""

        results = []
        for business_data in business_list:
            try:
                result = self.comprehensive_prediction(business_data)
                result['business_id'] = business_data.get('business_id', 'unknown')
                results.append(result)
            except Exception as e:
                logger.error(f"예측 실패: {e}")
                results.append({
                    'business_id': business_data.get('business_id', 'unknown'),
                    'error': str(e)
                })

        return results


def demo_prediction():
    """예측 데모"""
    print("=== ML 기반 위험도 예측 데모 ===")

    predictor = HybridRiskPredictor()

    # 샘플 사업자 데이터
    sample_business = {
        'business_id': 'ML_DEMO_001',
        'latest_revenue': 5000000,
        'latest_profit_margin': 0.15,
        'avg_growth_rate': 0.02,
        'revenue_cv': 0.25,
        'business_quarters': 8,
        'altman_zscore': 2.1,
        'industry_percentile_revenue': 65
    }

    # 예측 실행
    result = predictor.comprehensive_prediction(sample_business)

    print(f"사업자: {sample_business['business_id']}")
    print(f"위험도 점수: {result['risk_score']:.1f}점")
    print(f"위험도 등급: {result['risk_level']}")
    print(f"위험 확률: {result['risk_probability']:.1%}")
    print(f"예측 신뢰도: {result['confidence']:.1f}%")
    print(f"대출 추천: {result['loan_recommendation']['action']}")

    if result['loan_recommendation']['amount'] > 0:
        print(f"추천 금액: {result['loan_recommendation']['amount']:,.0f}원")


if __name__ == "__main__":
    demo_prediction()