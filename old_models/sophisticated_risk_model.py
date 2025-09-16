#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정교한 위험도 산정 모델 (Altman Z-Score 기반)
평가 영역: 재무건전성(40%) + 영업안정성(45%) + 상대적위치(15%)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from benchmark_data_processor import BenchmarkDataProcessor

class SophisticatedRiskAssessmentModel:
    def __init__(self):
        """정교한 위험도 산정 모델 초기화"""
        self.benchmark_processor = BenchmarkDataProcessor()

        # 가중치 설정 (사용자 제시 기준)
        self.weights = {
            'financial_health': 0.40,    # 재무 건전성 40%
            'operational_stability': 0.45,  # 영업 안정성 45%
            'relative_position': 0.15     # 상대적 위치 15%
        }

        # 위험도 기준점 설정
        self.risk_thresholds = {
            'altman_z': {
                'safe': 2.99,      # Z > 2.99: 안전
                'gray': 1.81,      # 1.81 < Z < 2.99: 회색지대
                'distress': 1.81   # Z < 1.81: 부실위험
            }
        }

    def calculate_altman_z_score(self, financial_metrics: Dict) -> Dict:
        """
        Altman Z-Score 계산 (수정된 Z'-Score 사용)
        Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4
        """
        try:
            # X1: 운전자본 / 총자산 (유동성)
            x1 = financial_metrics['working_capital'] / financial_metrics['total_assets'] if financial_metrics['total_assets'] > 0 else 0

            # X2: 이익잉여금 / 총자산 (누적 수익성)
            x2 = financial_metrics['retained_earnings'] / financial_metrics['total_assets'] if financial_metrics['total_assets'] > 0 else 0

            # X3: 세전이익 / 총자산 (단기 수익성)
            x3 = financial_metrics['annual_profit'] / financial_metrics['total_assets'] if financial_metrics['total_assets'] > 0 else 0

            # X4: 자기자본(장부가) / 총부채 (재무 구조 안정성)
            x4 = financial_metrics['equity_book_value'] / financial_metrics['total_debt'] if financial_metrics['total_debt'] > 0 else 10  # 부채가 0이면 매우 안전

            # Altman Z'-Score 계산
            z_score = 0.717*x1 + 0.847*x2 + 3.107*x3 + 0.420*x4

            # 각 지표별 점수
            component_scores = {
                'x1_liquidity': x1,
                'x2_cumulative_profitability': x2,
                'x3_short_term_profitability': x3,
                'x4_financial_structure': x4
            }

            # 위험도 분류
            if z_score > self.risk_thresholds['altman_z']['safe']:
                risk_level = 'safe'
                risk_description = '안전'
            elif z_score > self.risk_thresholds['altman_z']['gray']:
                risk_level = 'gray'
                risk_description = '회색지대'
            else:
                risk_level = 'distress'
                risk_description = '부실위험'

            return {
                'z_score': z_score,
                'risk_level': risk_level,
                'risk_description': risk_description,
                'component_scores': component_scores,
                'interpretation': self._interpret_altman_score(z_score, component_scores)
            }

        except Exception as e:
            print(f"❌ Altman Z-Score 계산 오류: {e}")
            return {
                'z_score': 0.0,
                'risk_level': 'distress',
                'risk_description': '계산불가',
                'component_scores': {},
                'interpretation': {'overall': '재무정보 부족으로 계산할 수 없습니다.'}
            }

    def _interpret_altman_score(self, z_score: float, components: Dict) -> Dict:
        """Altman Z-Score 해석"""
        interpretation = {}

        # 전체 점수 해석
        if z_score > 2.99:
            interpretation['overall'] = '매우 건전한 재무 상태입니다.'
        elif z_score > 1.81:
            interpretation['overall'] = '주의가 필요한 재무 상태입니다.'
        else:
            interpretation['overall'] = '재무 위험이 높은 상태입니다.'

        # 개별 지표 해석
        x1 = components.get('x1_liquidity', 0)
        if x1 > 0.1:
            interpretation['liquidity'] = '유동성이 양호합니다.'
        elif x1 > 0:
            interpretation['liquidity'] = '유동성 관리가 필요합니다.'
        else:
            interpretation['liquidity'] = '유동성이 부족합니다.'

        x2 = components.get('x2_cumulative_profitability', 0)
        if x2 > 0.1:
            interpretation['profitability'] = '누적 수익성이 우수합니다.'
        elif x2 > 0:
            interpretation['profitability'] = '누적 수익성이 보통입니다.'
        else:
            interpretation['profitability'] = '누적 손실이 발생했습니다.'

        x4 = components.get('x4_financial_structure', 0)
        if x4 > 1.0:
            interpretation['financial_structure'] = '자기자본이 부채를 초과하여 안정적입니다.'
        elif x4 > 0.5:
            interpretation['financial_structure'] = '부채 관리가 필요합니다.'
        else:
            interpretation['financial_structure'] = '부채가 과다한 상태입니다.'

        return interpretation

    def calculate_operational_stability(self,
                                      monthly_revenue: int,
                                      historical_revenue: Optional[List[int]] = None,
                                      business_months: Optional[int] = None) -> Dict:
        """
        영업 안정성 평가 (45%)
        2-1. 매출 성장성, 2-2. 매출 변동성, 2-3. 영업 지속성
        """
        scores = {}

        # 2-1. 매출 성장성 (historical_revenue가 있으면 계산, 없으면 추정)
        if historical_revenue and len(historical_revenue) >= 3:
            # 실제 성장률 계산
            recent_growth = self._calculate_growth_rate(historical_revenue)
            scores['growth_rate'] = recent_growth

            if recent_growth > 10:
                growth_score = 1.0
                growth_status = '높은 성장'
            elif recent_growth > 0:
                growth_score = 0.7
                growth_status = '안정적 성장'
            elif recent_growth > -10:
                growth_score = 0.3
                growth_status = '보합'
            else:
                growth_score = 0.1
                growth_status = '매출 감소'
        else:
            # 시계열 데이터가 없으면 업종 평균 대비로 추정
            industry_benchmark = self.benchmark_processor.get_industry_benchmark("기타")  # 기본값
            if industry_benchmark and monthly_revenue > 0:
                revenue_ratio = monthly_revenue / (industry_benchmark['revenue_mean'] / 1000000)  # 월평균으로 변환
                if revenue_ratio > 1.2:
                    growth_score = 0.8
                    growth_status = '업종 평균 이상'
                elif revenue_ratio > 0.8:
                    growth_score = 0.6
                    growth_status = '업종 평균 수준'
                else:
                    growth_score = 0.3
                    growth_status = '업종 평균 미만'
            else:
                growth_score = 0.5
                growth_status = '평가 불가'

        scores['growth_score'] = growth_score
        scores['growth_status'] = growth_status

        # 2-2. 매출 변동성 (표준편차 기반)
        if historical_revenue and len(historical_revenue) >= 3:
            volatility = np.std(historical_revenue) / np.mean(historical_revenue) if np.mean(historical_revenue) > 0 else 1.0

            if volatility < 0.1:
                volatility_score = 1.0
                volatility_status = '매우 안정'
            elif volatility < 0.2:
                volatility_score = 0.8
                volatility_status = '안정'
            elif volatility < 0.4:
                volatility_score = 0.5
                volatility_status = '보통'
            else:
                volatility_score = 0.2
                volatility_status = '불안정'

            scores['volatility'] = volatility
        else:
            # 데이터가 없으면 업종 특성으로 추정
            volatility_score = 0.6  # 중간값
            volatility_status = '추정치'

        scores['volatility_score'] = volatility_score
        scores['volatility_status'] = volatility_status

        # 2-3. 영업 지속성 (업력)
        if business_months:
            if business_months >= 36:  # 3년 이상
                continuity_score = 1.0
                continuity_status = '안정적 운영'
            elif business_months >= 24:  # 2년 이상
                continuity_score = 0.8
                continuity_status = '지속적 운영'
            elif business_months >= 12:  # 1년 이상
                continuity_score = 0.6
                continuity_status = '초기 안정화'
            else:  # 1년 미만
                continuity_score = 0.3
                continuity_status = '초기 단계'
        else:
            # 업력 정보가 없으면 보수적으로 가정
            continuity_score = 0.5
            continuity_status = '정보 부족'

        scores['continuity_score'] = continuity_score
        scores['continuity_status'] = continuity_status

        # 영업 안정성 종합 점수 (가중평균)
        operational_score = (
            scores['growth_score'] * 0.4 +      # 성장성 40%
            scores['volatility_score'] * 0.3 +   # 변동성 30%
            scores['continuity_score'] * 0.3     # 지속성 30%
        )

        return {
            'operational_score': operational_score,
            'components': scores,
            'interpretation': self._interpret_operational_stability(scores)
        }

    def _calculate_growth_rate(self, historical_revenue: List[int]) -> float:
        """최근 3개월 평균 매출 증가율 계산"""
        if len(historical_revenue) < 2:
            return 0.0

        # 최근 값과 이전 값 비교 (월별 증가율)
        recent_avg = np.mean(historical_revenue[-2:])  # 최근 2개월 평균
        previous_avg = np.mean(historical_revenue[:-2]) if len(historical_revenue) > 2 else historical_revenue[0]

        if previous_avg > 0:
            growth_rate = ((recent_avg - previous_avg) / previous_avg) * 100
            return growth_rate
        else:
            return 0.0

    def _interpret_operational_stability(self, scores: Dict) -> Dict:
        """영업 안정성 해석"""
        interpretation = {}

        # 성장성 해석
        growth_score = scores['growth_score']
        if growth_score > 0.8:
            interpretation['growth'] = '매출 성장세가 우수합니다.'
        elif growth_score > 0.5:
            interpretation['growth'] = '매출이 안정적으로 유지되고 있습니다.'
        else:
            interpretation['growth'] = '매출 성장이 필요합니다.'

        # 변동성 해석
        volatility_score = scores['volatility_score']
        if volatility_score > 0.8:
            interpretation['volatility'] = '매출이 안정적입니다.'
        elif volatility_score > 0.5:
            interpretation['volatility'] = '매출 변동성이 보통 수준입니다.'
        else:
            interpretation['volatility'] = '매출 변동성이 높아 주의가 필요합니다.'

        # 지속성 해석
        continuity_score = scores['continuity_score']
        if continuity_score > 0.8:
            interpretation['continuity'] = '사업 운영 경험이 충분합니다.'
        elif continuity_score > 0.5:
            interpretation['continuity'] = '사업 안정화 단계입니다.'
        else:
            interpretation['continuity'] = '사업 초기 단계로 안정화가 필요합니다.'

        return interpretation

    def calculate_relative_position(self,
                                  monthly_revenue: int,
                                  monthly_expenses: Dict[str, int],
                                  business_type: str,
                                  location: str) -> Dict:
        """
        상대적 위치 평가 (15%)
        업종 내 재무 비교
        """
        # 벤치마크 비교 분석
        benchmark_comparison = self.benchmark_processor.compare_user_expenses(
            monthly_revenue, monthly_expenses, business_type, location
        )

        # 영업이익률 계산
        total_expenses = sum(monthly_expenses.values())
        operating_profit = monthly_revenue - total_expenses
        operating_margin = (operating_profit / monthly_revenue * 100) if monthly_revenue > 0 else -100

        # 업종 평균 영업이익률 추정 (일반적으로 5-15%)
        industry_avg_margin = 8.0  # 기본 가정

        # 상대적 수익성 비교
        if operating_margin > industry_avg_margin * 1.5:
            profitability_score = 1.0
            profitability_status = '업종 대비 우수'
        elif operating_margin > industry_avg_margin:
            profitability_score = 0.8
            profitability_status = '업종 평균 이상'
        elif operating_margin > industry_avg_margin * 0.5:
            profitability_score = 0.6
            profitability_status = '업종 평균 수준'
        elif operating_margin > 0:
            profitability_score = 0.4
            profitability_status = '업종 평균 미만'
        else:
            profitability_score = 0.1
            profitability_status = '적자 운영'

        # 비용 효율성 평가
        total_expense_ratio = benchmark_comparison['total_comparison']['ratio_percent']
        if total_expense_ratio < 90:
            cost_efficiency_score = 1.0
            cost_efficiency_status = '비용 효율 우수'
        elif total_expense_ratio < 110:
            cost_efficiency_score = 0.8
            cost_efficiency_status = '적정 비용 수준'
        elif total_expense_ratio < 130:
            cost_efficiency_score = 0.5
            cost_efficiency_status = '비용 관리 필요'
        else:
            cost_efficiency_score = 0.2
            cost_efficiency_status = '과다 비용'

        # 상대적 위치 종합 점수
        relative_score = (profitability_score * 0.6 + cost_efficiency_score * 0.4)

        return {
            'relative_score': relative_score,
            'operating_margin': operating_margin,
            'industry_avg_margin': industry_avg_margin,
            'profitability_score': profitability_score,
            'profitability_status': profitability_status,
            'cost_efficiency_score': cost_efficiency_score,
            'cost_efficiency_status': cost_efficiency_status,
            'benchmark_comparison': benchmark_comparison,
            'interpretation': self._interpret_relative_position(relative_score, operating_margin)
        }

    def _interpret_relative_position(self, relative_score: float, operating_margin: float) -> Dict:
        """상대적 위치 해석"""
        interpretation = {}

        if relative_score > 0.8:
            interpretation['overall'] = '업종 내에서 경쟁력이 우수합니다.'
        elif relative_score > 0.6:
            interpretation['overall'] = '업종 평균 수준의 경쟁력을 보유하고 있습니다.'
        elif relative_score > 0.4:
            interpretation['overall'] = '업종 내 경쟁력 향상이 필요합니다.'
        else:
            interpretation['overall'] = '업종 내 경쟁력이 낮은 상태입니다.'

        if operating_margin > 10:
            interpretation['profitability'] = '높은 수익성을 실현하고 있습니다.'
        elif operating_margin > 5:
            interpretation['profitability'] = '적정 수준의 수익성을 유지하고 있습니다.'
        elif operating_margin > 0:
            interpretation['profitability'] = '수익성 개선이 필요합니다.'
        else:
            interpretation['profitability'] = '손실 상태로 즉시 개선이 필요합니다.'

        return interpretation

    def calculate_comprehensive_risk_score(self,
                                         total_available_assets: int,
                                         monthly_revenue: int,
                                         monthly_expenses: Dict[str, int],
                                         business_type: str,
                                         location: str,
                                         historical_revenue: Optional[List[int]] = None,
                                         business_months: Optional[int] = None) -> Dict:
        """
        종합 위험도 점수 계산
        재무건전성(40%) + 영업안정성(45%) + 상대적위치(15%)
        """
        print("🔍 정교한 위험도 분석 중...")

        # 1. 재무 지표 추정
        financial_metrics = self.benchmark_processor.estimate_financial_metrics(
            monthly_revenue, monthly_expenses, total_available_assets, business_type
        )

        # 2. 재무 건전성 평가 (40%)
        altman_analysis = self.calculate_altman_z_score(financial_metrics)
        financial_health_score = self._convert_altman_to_score(altman_analysis['z_score'])

        # 3. 영업 안정성 평가 (45%)
        operational_analysis = self.calculate_operational_stability(
            monthly_revenue, historical_revenue, business_months
        )
        operational_stability_score = operational_analysis['operational_score']

        # 4. 상대적 위치 평가 (15%)
        relative_analysis = self.calculate_relative_position(
            monthly_revenue, monthly_expenses, business_type, location
        )
        relative_position_score = relative_analysis['relative_score']

        # 5. 가중평균으로 종합 점수 계산
        comprehensive_score = (
            financial_health_score * self.weights['financial_health'] +
            operational_stability_score * self.weights['operational_stability'] +
            relative_position_score * self.weights['relative_position']
        )

        # 6. 위험도로 변환 (점수가 높을수록 위험도 낮음)
        risk_score = 1.0 - comprehensive_score  # 0~1 사이, 1에 가까울수록 위험

        # 7. 5단계 분류
        risk_level = self._classify_comprehensive_risk(risk_score)

        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comprehensive_risk_score': risk_score,
            'comprehensive_health_score': comprehensive_score,
            'risk_level': risk_level,
            'component_analyses': {
                'financial_health': {
                    'score': financial_health_score,
                    'weight': self.weights['financial_health'],
                    'weighted_score': financial_health_score * self.weights['financial_health'],
                    'altman_analysis': altman_analysis,
                    'financial_metrics': financial_metrics
                },
                'operational_stability': {
                    'score': operational_stability_score,
                    'weight': self.weights['operational_stability'],
                    'weighted_score': operational_stability_score * self.weights['operational_stability'],
                    'operational_analysis': operational_analysis
                },
                'relative_position': {
                    'score': relative_position_score,
                    'weight': self.weights['relative_position'],
                    'weighted_score': relative_position_score * self.weights['relative_position'],
                    'relative_analysis': relative_analysis
                }
            },
            'overall_interpretation': self._generate_comprehensive_interpretation(
                financial_health_score, operational_stability_score, relative_position_score, risk_score
            )
        }

    def _convert_altman_to_score(self, z_score: float) -> float:
        """Altman Z-Score를 0-1 점수로 변환"""
        if z_score > 2.99:
            return 1.0  # 안전
        elif z_score > 1.81:
            return 0.6  # 회색지대
        else:
            return 0.2  # 부실위험

    def _classify_comprehensive_risk(self, risk_score: float) -> Dict:
        """종합 위험도 5단계 분류"""
        if risk_score <= 0.2:
            return {'level': 1, 'name': '매우여유', 'emoji': '🌟', 'description': '매우 안전한 재무 상태'}
        elif risk_score <= 0.4:
            return {'level': 2, 'name': '여유', 'emoji': '🟢', 'description': '안전한 재무 상태'}
        elif risk_score <= 0.6:
            return {'level': 3, 'name': '안정', 'emoji': '🟡', 'description': '보통 수준의 재무 상태'}
        elif risk_score <= 0.8:
            return {'level': 4, 'name': '위험', 'emoji': '🟠', 'description': '주의가 필요한 재무 상태'}
        else:
            return {'level': 5, 'name': '매우위험', 'emoji': '🔴', 'description': '위험한 재무 상태'}

    def _generate_comprehensive_interpretation(self,
                                             financial_score: float,
                                             operational_score: float,
                                             relative_score: float,
                                             risk_score: float) -> Dict:
        """종합 해석 생성"""
        interpretation = {
            'overall_assessment': '',
            'key_strengths': [],
            'key_weaknesses': [],
            'recommendations': []
        }

        # 전체 평가
        if risk_score <= 0.3:
            interpretation['overall_assessment'] = '매우 우수한 재무 건전성을 보이고 있습니다.'
        elif risk_score <= 0.6:
            interpretation['overall_assessment'] = '전반적으로 안정적인 재무 상태입니다.'
        else:
            interpretation['overall_assessment'] = '재무 안정성 개선이 시급합니다.'

        # 강점과 약점 분석
        scores = {
            '재무건전성': financial_score,
            '영업안정성': operational_score,
            '상대적위치': relative_score
        }

        for area, score in scores.items():
            if score > 0.7:
                interpretation['key_strengths'].append(f'{area}이 우수합니다.')
            elif score < 0.4:
                interpretation['key_weaknesses'].append(f'{area} 개선이 필요합니다.')

        # 권장사항
        if financial_score < 0.5:
            interpretation['recommendations'].append('재무 구조 개선을 통한 안정성 확보가 필요합니다.')
        if operational_score < 0.5:
            interpretation['recommendations'].append('매출 안정성과 성장성 확보가 중요합니다.')
        if relative_score < 0.5:
            interpretation['recommendations'].append('업종 내 경쟁력 강화를 위한 전략이 필요합니다.')

        if not interpretation['recommendations']:
            interpretation['recommendations'].append('현재 상태를 유지하며 지속적인 모니터링을 권장합니다.')

        return interpretation

if __name__ == "__main__":
    # 테스트
    model = SophisticatedRiskAssessmentModel()

    # 종합 위험도 분석 테스트
    result = model.calculate_comprehensive_risk_score(
        total_available_assets=50000000,
        monthly_revenue=15000000,
        monthly_expenses={
            'labor_cost': 5000000,
            'food_materials': 4000000,
            'rent': 2000000,
            'others': 1500000
        },
        business_type="한식음식점",
        location="강남구",
        historical_revenue=[12000000, 14000000, 15000000],  # 3개월 매출
        business_months=24  # 2년 운영
    )

    print("=" * 60)
    print("🎯 정교한 위험도 분석 결과")
    print("=" * 60)

    # 종합 결과
    risk_info = result['risk_level']
    print(f"📊 종합 위험도: {risk_info['emoji']} {risk_info['name']} (점수: {result['comprehensive_risk_score']:.3f})")
    print(f"💡 평가: {risk_info['description']}")

    # 세부 분석
    components = result['component_analyses']
    print(f"\n📈 세부 분석:")
    print(f"  🏦 재무건전성: {components['financial_health']['score']:.2f} (가중치 40%)")
    print(f"  📊 영업안정성: {components['operational_stability']['score']:.2f} (가중치 45%)")
    print(f"  📍 상대적위치: {components['relative_position']['score']:.2f} (가중치 15%)")

    # Altman Z-Score
    altman = components['financial_health']['altman_analysis']
    print(f"\n🔍 Altman Z-Score: {altman['z_score']:.2f} ({altman['risk_description']})")

    # 해석
    interpretation = result['overall_interpretation']
    print(f"\n💬 종합 평가: {interpretation['overall_assessment']}")

    if interpretation['key_strengths']:
        print("✅ 주요 강점:")
        for strength in interpretation['key_strengths']:
            print(f"  • {strength}")

    if interpretation['key_weaknesses']:
        print("⚠️ 개선 필요:")
        for weakness in interpretation['key_weaknesses']:
            print(f"  • {weakness}")

    print("💡 권장사항:")
    for rec in interpretation['recommendations']:
        print(f"  • {rec}")

    print("\n✅ 정교한 위험도 분석 완료!")