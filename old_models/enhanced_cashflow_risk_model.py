#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 현금흐름 위험도 예측 모델 (Enhanced Version)
기능: 간소화된 입력 + 정교한 위험도 산정 + 업종/지역 벤치마크 비교 + 대출/투자 계산
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from benchmark_data_processor import BenchmarkDataProcessor
from sophisticated_risk_model import SophisticatedRiskAssessmentModel

class EnhancedCashFlowRiskModel:
    def __init__(self):
        """통합 현금흐름 위험도 모델 초기화"""
        self.benchmark_processor = BenchmarkDataProcessor()
        self.sophisticated_model = SophisticatedRiskAssessmentModel()

        # 5단계 위험도 분류 (위험도 스코어가 높을수록 위험함)
        self.risk_levels = {
            1: {"name": "매우여유", "range": (0.0, 0.2), "emoji": "🌟", "color": "blue"},
            2: {"name": "여유", "range": (0.2, 0.4), "emoji": "🟢", "color": "green"},
            3: {"name": "안정", "range": (0.4, 0.6), "emoji": "🟡", "color": "yellow"},
            4: {"name": "위험", "range": (0.6, 0.8), "emoji": "🟠", "color": "orange"},
            5: {"name": "매우위험", "range": (0.8, 1.0), "emoji": "🔴", "color": "red"}
        }

    def classify_risk_level(self, risk_score: float) -> int:
        """위험도 스코어를 5단계로 분류"""
        for level, info in self.risk_levels.items():
            if info["range"][0] <= risk_score < info["range"][1]:
                return level
        return 5  # 최고 위험도

    def calculate_loan_amount(self,
                            current_risk_score: float,
                            total_available_assets: int,
                            monthly_revenue: int,
                            monthly_expenses: Dict[str, int],
                            business_type: str,
                            location: str) -> Dict[str, int]:
        """
        안정권(3단계, 60% 이하) 진입에 필요한 대출 금액 계산
        """
        target_risk_score = 0.6  # 안정권 상한

        if current_risk_score <= target_risk_score:
            return {"loan_amount": 0, "reason": "이미 안정권입니다"}

        # 이진 탐색으로 필요한 대출 금액 찾기
        min_loan = 0
        max_loan = total_available_assets * 3  # 최대 현재 자산의 3배까지
        tolerance = 10000

        for _ in range(30):  # 최대 30회 반복
            mid_loan = (min_loan + max_loan) // 2
            new_assets = total_available_assets + mid_loan

            # 새로운 자산으로 위험도 재계산
            new_risk_analysis = self.sophisticated_model.calculate_comprehensive_risk_score(
                new_assets, monthly_revenue, monthly_expenses, business_type, location
            )
            new_risk_score = new_risk_analysis['comprehensive_risk_score']

            if abs(new_risk_score - target_risk_score) < 0.05:
                return {
                    "loan_amount": mid_loan,
                    "target_risk_score": target_risk_score,
                    "expected_new_risk_score": new_risk_score,
                    "expected_new_level": self.classify_risk_level(new_risk_score)
                }
            elif new_risk_score > target_risk_score:
                min_loan = mid_loan + tolerance
            else:
                max_loan = mid_loan - tolerance

            if max_loan <= min_loan:
                break

        return {
            "loan_amount": min_loan,
            "target_risk_score": target_risk_score,
            "expected_new_risk_score": self.sophisticated_model.calculate_comprehensive_risk_score(
                total_available_assets + min_loan, monthly_revenue,
                monthly_expenses, business_type, location
            )['comprehensive_risk_score'],
            "expected_new_level": self.classify_risk_level(min_loan)
        }

    def calculate_investment_amount(self,
                                  current_risk_level: int,
                                  total_available_assets: int) -> Dict[str, int]:
        """
        위험도 레벨에 따른 투자 가능 금액 계산
        """
        if current_risk_level >= 4:  # 위험, 매우위험
            return {"investment_amount": 0, "reason": "위험 단계 - 투자보다 안정화 우선"}

        if current_risk_level == 1:  # 매우여유
            investment_ratio = 0.7
            reason = "매우여유 단계 - 적극적 투자 가능"
        elif current_risk_level == 2:  # 여유
            investment_ratio = 0.5
            reason = "여유 단계 - 보수적 투자 가능"
        else:  # 안정 (레벨 3)
            investment_ratio = 0.3
            reason = "안정 단계 - 소액 투자 가능"

        return {
            "investment_amount": int(total_available_assets * investment_ratio),
            "investment_ratio": investment_ratio,
            "reason": reason
        }

    def generate_comprehensive_recommendations(self,
                                             risk_analysis: Dict,
                                             benchmark_comparison: Dict,
                                             loan_info: Dict,
                                             investment_info: Dict) -> Dict:
        """종합 권장사항 생성"""

        risk_level = risk_analysis['risk_level']['level']
        risk_name = risk_analysis['risk_level']['name']
        risk_emoji = risk_analysis['risk_level']['emoji']
        risk_score = risk_analysis['comprehensive_risk_score']

        recommendations = {
            "risk_assessment": {
                "level": risk_level,
                "name": risk_name,
                "emoji": risk_emoji,
                "score": risk_score,
                "description": risk_analysis['risk_level']['description']
            },
            "primary_message": "",
            "financial_actions": [],
            "operational_improvements": [],
            "benchmark_insights": [],
            "nh_bank_products": []
        }

        # 위험도별 주요 메시지
        if risk_level >= 4:  # 위험, 매우위험
            recommendations["primary_message"] = f"{risk_emoji} {risk_name} 상태입니다. 즉시 재무 안정화가 필요합니다."

            if loan_info.get("loan_amount", 0) > 0:
                recommendations["financial_actions"] = [
                    f"💰 긴급 대출 권장: {loan_info['loan_amount']:,}원",
                    "📞 NH농협 소상공인 긴급대출 상담 (연 4.2%~7.0%)",
                    "🏦 신용보증재단 대출 검토",
                    "📊 현금흐름 개선 계획 수립"
                ]

            recommendations["nh_bank_products"] = [
                "NH농협 소상공인 긴급운영자금",
                "신용보증재단 특별보증 대출",
                "정책자금 긴급지원 프로그램"
            ]

        elif risk_level == 3:  # 안정
            recommendations["primary_message"] = f"{risk_emoji} {risk_name} 상태입니다. 현상 유지하며 성장 기회를 모색하세요."

            recommendations["financial_actions"] = [
                "📊 현재 수준 유지 관리",
                "📈 매출 안정성 확보",
                "💼 효율성 향상 방안 검토"
            ]

            if investment_info.get("investment_amount", 0) > 0:
                recommendations["financial_actions"].append(
                    f"💎 소액 투자 검토: {investment_info['investment_amount']:,}원"
                )

            recommendations["nh_bank_products"] = [
                "NH농협 소상공인 적금 (연 3.5%~4.0%)",
                "안정형 펀드 투자",
                "정기예금 (연 3.0%~3.5%)"
            ]

        else:  # 여유, 매우여유 (레벨 1, 2)
            recommendations["primary_message"] = f"{risk_emoji} {risk_name} 상태입니다. 적극적인 자산 운용으로 수익을 극대화하세요."

            investment_amount = investment_info.get("investment_amount", 0)
            if investment_amount > 0:
                recommendations["financial_actions"] = [
                    f"💰 투자 권장: {investment_amount:,}원",
                    "📈 NH농협 펀드 투자 검토",
                    "🏢 부동산 투자 기회 탐색",
                    "💎 포트폴리오 다양화"
                ]

            if risk_level == 1:  # 매우여유
                recommendations["nh_bank_products"] = [
                    "NH농협 성장형 펀드 (기대수익 8%~12%)",
                    "주식형 펀드 투자",
                    "해외 투자 상품",
                    "부동산 투자신탁(REITs)"
                ]
            else:  # 여유
                recommendations["nh_bank_products"] = [
                    "NH농협 혼합형 펀드 (기대수익 5%~8%)",
                    "중위험 중수익 상품",
                    "채권형 펀드"
                ]

        # 벤치마크 기반 운영 개선사항
        expense_comparison = benchmark_comparison.get('expense_breakdown', {})
        recommendations["benchmark_insights"] = []

        for expense_type, data in expense_comparison.items():
            ratio = data['ratio_percent']
            expense_name = {
                'labor_cost': '인건비',
                'food_materials': '식자재비',
                'rent': '임대료',
                'others': '기타 지출'
            }.get(expense_type, expense_type)

            if ratio > 150:
                recommendations["operational_improvements"].append(
                    f"🔴 {expense_name} 절감 필요 (업종 평균 대비 {ratio:.0f}%)"
                )
                recommendations["benchmark_insights"].append(
                    f"{expense_name}가 업종 평균보다 {ratio-100:.0f}%p 높습니다"
                )
            elif ratio > 120:
                recommendations["operational_improvements"].append(
                    f"🟠 {expense_name} 관리 검토 (업종 평균 대비 {ratio:.0f}%)"
                )
            elif ratio < 80:
                recommendations["benchmark_insights"].append(
                    f"{expense_name} 관리가 우수합니다 (업종 평균 대비 {ratio:.0f}%)"
                )

        # Altman Z-Score 기반 재무 건전성 조언
        altman_analysis = risk_analysis['component_analyses']['financial_health']['altman_analysis']
        z_score = altman_analysis['z_score']

        if z_score < 1.81:
            recommendations["operational_improvements"].append(
                "🏦 재무 구조 개선을 통한 부실 위험 해소 필요"
            )
        elif z_score < 2.99:
            recommendations["operational_improvements"].append(
                "📊 재무 안정성 강화로 회색지대 탈출 권장"
            )

        # 운영 안정성 기반 조언
        operational_analysis = risk_analysis['component_analyses']['operational_stability']['operational_analysis']
        operational_score = operational_analysis['operational_score']

        if operational_score < 0.6:
            recommendations["operational_improvements"].append(
                "📈 매출 성장성과 안정성 확보가 필요"
            )

        return recommendations

    def predict_enhanced_risk(self,
                            total_available_assets: int,
                            monthly_revenue: int,
                            monthly_expenses: Dict[str, int],
                            business_type: str,
                            location: str,
                            historical_revenue: Optional[List[int]] = None,
                            business_months: Optional[int] = None) -> Dict:
        """
        통합 위험도 예측 및 종합 분석
        """
        print(f"🔍 통합 위험도 분석 시작...")
        print(f"  💰 총 운용자산: {total_available_assets:,}원")
        print(f"  📈 월 매출: {monthly_revenue:,}원")
        print(f"  📊 월 지출: {sum(monthly_expenses.values()):,}원")
        print(f"  🏪 업종: {business_type}")
        print(f"  📍 지역: {location}")

        # 1. 정교한 위험도 분석 (Altman Z-Score 기반)
        sophisticated_analysis = self.sophisticated_model.calculate_comprehensive_risk_score(
            total_available_assets, monthly_revenue, monthly_expenses,
            business_type, location, historical_revenue, business_months
        )

        # 2. 업종/지역 벤치마크 비교 (핵심 기능!)
        benchmark_comparison = self.benchmark_processor.compare_user_expenses(
            monthly_revenue, monthly_expenses, business_type, location
        )

        # 3. 위험도 레벨 분류
        risk_score = sophisticated_analysis['comprehensive_risk_score']
        risk_level = self.classify_risk_level(risk_score)

        # 4. 대출 금액 계산 (위험 단계인 경우)
        loan_info = {}
        if risk_level >= 4:
            loan_info = self.calculate_loan_amount(
                risk_score, total_available_assets, monthly_revenue,
                monthly_expenses, business_type, location
            )

        # 5. 투자 금액 계산 (안정 이상인 경우)
        investment_info = {}
        if risk_level <= 3:
            investment_info = self.calculate_investment_amount(
                risk_level, total_available_assets
            )

        # 6. 종합 권장사항 생성
        comprehensive_recommendations = self.generate_comprehensive_recommendations(
            sophisticated_analysis, benchmark_comparison, loan_info, investment_info
        )

        # 7. 최종 통합 결과
        enhanced_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_summary": {
                "total_available_assets": total_available_assets,
                "monthly_revenue": monthly_revenue,
                "monthly_expenses": monthly_expenses,
                "total_monthly_expenses": sum(monthly_expenses.values()),
                "monthly_cashflow": monthly_revenue - sum(monthly_expenses.values()),
                "business_type": business_type,
                "location": location,
                "business_months": business_months
            },
            "risk_assessment": {
                "comprehensive_risk_score": risk_score,
                "risk_level": risk_level,
                "risk_info": self.risk_levels[risk_level],
                "altman_z_score": sophisticated_analysis['component_analyses']['financial_health']['altman_analysis']['z_score'],
                "component_scores": {
                    "financial_health": sophisticated_analysis['component_analyses']['financial_health']['score'],
                    "operational_stability": sophisticated_analysis['component_analyses']['operational_stability']['score'],
                    "relative_position": sophisticated_analysis['component_analyses']['relative_position']['score']
                }
            },
            "benchmark_analysis": {
                "industry_comparison": benchmark_comparison,
                "key_insights": []
            },
            "financial_recommendations": {
                "loan_analysis": loan_info,
                "investment_analysis": investment_info
            },
            "comprehensive_recommendations": comprehensive_recommendations,
            "detailed_analysis": {
                "sophisticated_analysis": sophisticated_analysis,
                "benchmark_comparison": benchmark_comparison
            }
        }

        # 벤치마크 핵심 인사이트 추출
        for expense_type, data in benchmark_comparison.get('expense_breakdown', {}).items():
            expense_name = {'labor_cost': '인건비', 'food_materials': '식자재비', 'rent': '임대료', 'others': '기타'}.get(expense_type, expense_type)
            enhanced_result["benchmark_analysis"]["key_insights"].append(
                f"{expense_name}: {data['message']} ({data['status']})"
            )

        return enhanced_result

if __name__ == "__main__":
    # 통합 모델 테스트
    model = EnhancedCashFlowRiskModel()

    print("🚀 통합 현금흐름 위험도 예측 모델 테스트")
    print("=" * 70)

    # 테스트 케이스
    test_cases = [
        {
            "name": "위험 사례 - 인건비 과다",
            "total_available_assets": 30000000,
            "monthly_revenue": 8000000,
            "monthly_expenses": {
                "labor_cost": 6000000,    # 과다 인건비
                "food_materials": 2500000,
                "rent": 2000000,
                "others": 500000
            },
            "business_type": "한식음식점",
            "location": "관악구",
            "historical_revenue": [7000000, 7500000, 8000000],
            "business_months": 18
        },
        {
            "name": "여유 사례 - 효율적 운영",
            "total_available_assets": 80000000,
            "monthly_revenue": 20000000,
            "monthly_expenses": {
                "labor_cost": 6000000,
                "food_materials": 5000000,
                "rent": 2500000,
                "others": 1500000
            },
            "business_type": "일식음식점",
            "location": "강남구",
            "historical_revenue": [18000000, 19000000, 20000000],
            "business_months": 36
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 테스트 케이스 {i}: {case['name']}")
        print("-" * 50)

        result = model.predict_enhanced_risk(
            case["total_available_assets"],
            case["monthly_revenue"],
            case["monthly_expenses"],
            case["business_type"],
            case["location"],
            case.get("historical_revenue"),
            case.get("business_months")
        )

        # 결과 출력
        risk_info = result["risk_assessment"]["risk_info"]
        print(f"📊 위험도: {risk_info['emoji']} {risk_info['name']} (점수: {result['risk_assessment']['comprehensive_risk_score']:.3f})")
        print(f"🏦 Altman Z-Score: {result['risk_assessment']['altman_z_score']:.2f}")

        # 재무 액션
        if result["financial_recommendations"]["loan_analysis"]:
            loan = result["financial_recommendations"]["loan_analysis"]
            if loan.get("loan_amount", 0) > 0:
                print(f"💰 권장 대출: {loan['loan_amount']:,}원")

        if result["financial_recommendations"]["investment_analysis"]:
            investment = result["financial_recommendations"]["investment_analysis"]
            if investment.get("investment_amount", 0) > 0:
                print(f"💎 투자 가능: {investment['investment_amount']:,}원")

        # 벤치마크 인사이트
        print("📊 벤치마크 비교:")
        for insight in result["benchmark_analysis"]["key_insights"][:3]:
            print(f"  • {insight}")

        # 주요 권장사항
        recommendations = result["comprehensive_recommendations"]
        print(f"💬 조언: {recommendations['primary_message']}")

        # 결과 저장
        os.makedirs('enhanced_results', exist_ok=True)
        with open(f'enhanced_results/enhanced_test_case_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ 통합 모델 테스트 완료!")
    print("📁 상세 결과: enhanced_results/ 폴더 확인")