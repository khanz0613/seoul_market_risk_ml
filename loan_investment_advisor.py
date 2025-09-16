#!/usr/bin/env python3
"""
Loan & Investment Advisor System
===============================

진짜 핵심 기능:
1. 위험한 사람 → 대출 얼마 받아야 안정권 진입?
2. 흑자인 사람 → 투자 얼마까지 해도 안전권 유지?

Altman Z-Score 기반 자산 시뮬레이션
"""

import numpy as np
from typing import Dict, Tuple
import math

class LoanInvestmentAdvisor:
    """대출/투자 자문 시스템 - Altman Z-Score 기반"""

    def __init__(self):
        self.safety_threshold = 3.0    # Z-Score 안전권 기준
        self.warning_threshold = 1.8   # Z-Score 경고권 기준
        self.danger_threshold = 1.1    # Z-Score 위험권 기준

    def calculate_altman_zscore(self, financial_data: Dict) -> float:
        """Altman Z-Score 계산"""

        # 재무 데이터 추출
        total_assets = financial_data['총자산']
        available_cash = financial_data['가용자산']
        monthly_sales = financial_data['월매출']
        monthly_costs = financial_data['월비용']
        total_debt = financial_data.get('총부채', total_assets * 0.3)  # 부채 추정

        # 연간 데이터 계산
        annual_sales = monthly_sales * 12
        annual_costs = monthly_costs * 12

        # Altman Z-Score 구성 요소 계산
        working_capital = available_cash  # 가용자산을 운전자본으로 가정
        retained_earnings = total_assets * 0.15  # 이익잉여금 추정 (15%)
        ebit = annual_sales - annual_costs  # 영업이익
        market_value_equity = total_assets - total_debt  # 자기자본

        # 안전한 분모 계산 (0으로 나누기 방지)
        safe_total_assets = max(total_assets, 1000000)
        safe_total_debt = max(total_debt, 100000)

        # Altman Z-Score 공식
        # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        A = working_capital / safe_total_assets          # 운전자본/총자산
        B = retained_earnings / safe_total_assets        # 이익잉여금/총자산
        C = ebit / safe_total_assets                     # EBIT/총자산
        D = market_value_equity / safe_total_debt        # 자기자본/총부채
        E = annual_sales / safe_total_assets             # 매출/총자산

        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        return z_score

    def simulate_loan_impact(self, financial_data: Dict, loan_amount: float) -> float:
        """대출이 Z-Score에 미치는 영향 시뮬레이션"""

        # 대출 후 재무상태 시뮬레이션
        new_financial_data = financial_data.copy()
        new_financial_data['총자산'] += loan_amount      # 자산 증가 (현금)
        new_financial_data['가용자산'] += loan_amount    # 가용자산 증가
        new_financial_data['총부채'] = new_financial_data.get('총부채', financial_data['총자산'] * 0.3) + loan_amount

        return self.calculate_altman_zscore(new_financial_data)

    def simulate_investment_impact(self, financial_data: Dict, investment_amount: float) -> float:
        """투자가 Z-Score에 미치는 영향 시뮬레이션"""

        # 투자 후 재무상태 시뮬레이션 (현금 감소)
        new_financial_data = financial_data.copy()
        new_financial_data['가용자산'] -= investment_amount  # 가용자산 감소

        # 가용자산이 음수가 되면 부채로 전환
        if new_financial_data['가용자산'] < 0:
            shortage = abs(new_financial_data['가용자산'])
            new_financial_data['가용자산'] = 0
            new_financial_data['총부채'] = new_financial_data.get('총부채', financial_data['총자산'] * 0.3) + shortage

        return self.calculate_altman_zscore(new_financial_data)

    def calculate_loan_recommendation(self, financial_data: Dict) -> Dict:
        """대출 추천 계산 - 안정권 진입까지 필요한 금액"""

        current_zscore = self.calculate_altman_zscore(financial_data)

        if current_zscore >= self.safety_threshold:
            return {
                'current_zscore': current_zscore,
                'recommended_loan': 0,
                'reason': '이미 안전권입니다',
                'status': 'safe'
            }

        # 이진 탐색으로 최적 대출액 찾기
        min_loan = 0
        max_loan = financial_data['총자산'] * 2  # 총자산의 2배까지
        optimal_loan = 0

        for _ in range(50):  # 최대 50번 반복
            mid_loan = (min_loan + max_loan) / 2
            simulated_zscore = self.simulate_loan_impact(financial_data, mid_loan)

            if simulated_zscore >= self.safety_threshold:
                optimal_loan = mid_loan
                max_loan = mid_loan
            else:
                min_loan = mid_loan

            if max_loan - min_loan < 100000:  # 10만원 단위로 정밀도
                break

        final_zscore = self.simulate_loan_impact(financial_data, optimal_loan)

        return {
            'current_zscore': current_zscore,
            'recommended_loan': optimal_loan,
            'expected_zscore': final_zscore,
            'reason': f'안전권(Z-Score {self.safety_threshold:.1f}) 달성',
            'status': 'improvement_needed'
        }

    def calculate_investment_limit(self, financial_data: Dict) -> Dict:
        """투자 한도 계산 - 안전권 유지하면서 투자 가능한 최대 금액"""

        current_zscore = self.calculate_altman_zscore(financial_data)

        if current_zscore < self.safety_threshold:
            return {
                'current_zscore': current_zscore,
                'max_investment': 0,
                'reason': '현재 안전권이 아니므로 투자 비추천',
                'status': 'risky'
            }

        # 이진 탐색으로 최대 투자액 찾기
        min_investment = 0
        max_investment = financial_data['가용자산']  # 가용자산 한도
        optimal_investment = 0

        for _ in range(50):  # 최대 50번 반복
            mid_investment = (min_investment + max_investment) / 2
            simulated_zscore = self.simulate_investment_impact(financial_data, mid_investment)

            if simulated_zscore >= self.safety_threshold:
                optimal_investment = mid_investment
                min_investment = mid_investment
            else:
                max_investment = mid_investment

            if max_investment - min_investment < 100000:  # 10만원 단위로 정밀도
                break

        final_zscore = self.simulate_investment_impact(financial_data, optimal_investment)

        return {
            'current_zscore': current_zscore,
            'max_investment': optimal_investment,
            'expected_zscore': final_zscore,
            'reason': f'안전권(Z-Score {self.safety_threshold:.1f}) 유지',
            'status': 'safe_to_invest'
        }

    def comprehensive_analysis(self, 총자산: float, 월매출: float, 인건비: float,
                             임대료: float, 식자재비: float, 기타비용: float,
                             가용자산: float, 지역: str = "", 업종: str = "") -> Dict:
        """종합 분석 - 대출 추천 + 투자 한도"""

        print("💰 Loan & Investment Advisor System")
        print("=" * 50)
        print("🎯 Goal: 안정권 안에서 돈 굴리기")
        print("📊 Based on: Altman Z-Score Analysis")

        # 재무 데이터 구성
        월비용 = 인건비 + 임대료 + 식자재비 + 기타비용

        financial_data = {
            '총자산': 총자산,
            '가용자산': 가용자산,
            '월매출': 월매출,
            '월비용': 월비용,
            '총부채': 총자산 * 0.3  # 부채 추정
        }

        print(f"\n📊 현재 재무상황:")
        print(f"   총자산: {총자산:,}원")
        print(f"   가용자산: {가용자산:,}원")
        print(f"   월매출: {월매출:,}원")
        print(f"   월비용: {월비용:,}원")
        print(f"   월순익: {월매출-월비용:,}원")

        # 현재 Z-Score 계산
        current_zscore = self.calculate_altman_zscore(financial_data)

        # 위험도 등급 결정
        if current_zscore >= self.safety_threshold:
            risk_level = "안전권 ✅"
            risk_color = "🟢"
        elif current_zscore >= self.warning_threshold:
            risk_level = "경고권 ⚠️"
            risk_color = "🟡"
        elif current_zscore >= self.danger_threshold:
            risk_level = "위험권 ⚠️"
            risk_color = "🟠"
        else:
            risk_level = "매우위험 🚨"
            risk_color = "🔴"

        print(f"\n{risk_color} Altman Z-Score: {current_zscore:.2f} ({risk_level})")

        # 대출 추천 계산
        loan_recommendation = self.calculate_loan_recommendation(financial_data)

        # 투자 한도 계산
        investment_limit = self.calculate_investment_limit(financial_data)

        print(f"\n💳 대출 추천:")
        if loan_recommendation['recommended_loan'] > 0:
            print(f"   권장 대출액: {loan_recommendation['recommended_loan']:,.0f}원")
            print(f"   예상 Z-Score: {loan_recommendation['expected_zscore']:.2f}")
            print(f"   목적: {loan_recommendation['reason']}")
        else:
            print(f"   {loan_recommendation['reason']}")

        print(f"\n📈 투자 한도:")
        if investment_limit['max_investment'] > 0:
            print(f"   최대 투자액: {investment_limit['max_investment']:,.0f}원")
            print(f"   투자 후 Z-Score: {investment_limit['expected_zscore']:.2f}")
            print(f"   조건: {investment_limit['reason']}")
        else:
            print(f"   {investment_limit['reason']}")

        # 종합 결과
        result = {
            'current_status': {
                'zscore': current_zscore,
                'risk_level': risk_level,
                'monthly_profit': 월매출 - 월비용
            },
            'loan_recommendation': loan_recommendation,
            'investment_limit': investment_limit,
            'recommendations': []
        }

        # 맞춤형 추천
        if current_zscore < self.safety_threshold:
            result['recommendations'].append(f"🚨 대출 {loan_recommendation['recommended_loan']:,.0f}원으로 안정권 진입 추천")
        else:
            result['recommendations'].append(f"✅ 안전권 유지 중, 투자 최대 {investment_limit['max_investment']:,.0f}원 가능")

        if 월매출 - 월비용 > 0:
            result['recommendations'].append("💰 월흑자 달성, 투자 고려 가능")
        else:
            result['recommendations'].append("⚠️ 월적자, 비용 절감 우선 필요")

        print(f"\n🎯 맞춤형 추천:")
        for rec in result['recommendations']:
            print(f"   {rec}")

        return result

def main():
    """메인 테스트"""
    print("🚀 Loan & Investment Advisor Test")
    print("=" * 50)

    advisor = LoanInvestmentAdvisor()

    # 테스트 케이스 1: 위험한 사업자
    print("\n1️⃣ 위험한 사업자 케이스")
    result1 = advisor.comprehensive_analysis(
        총자산=20000000,      # 2천만원
        월매출=5000000,       # 500만원
        인건비=2500000,       # 250만원
        임대료=2200000,       # 220만원
        식자재비=2800000,     # 280만원
        기타비용=800000,      # 80만원
        가용자산=3000000,     # 300만원 (현금)
        지역='구로구',
        업종='한식음식점'
    )

    print("\n" + "="*60)

    # 테스트 케이스 2: 안정적인 사업자
    print("\n2️⃣ 안정적인 사업자 케이스")
    result2 = advisor.comprehensive_analysis(
        총자산=80000000,      # 8천만원
        월매출=15000000,      # 1500만원
        인건비=4000000,       # 400만원
        임대료=3000000,       # 300만원
        식자재비=4500000,     # 450만원
        기타비용=1000000,     # 100만원
        가용자산=20000000,    # 2천만원 (현금)
        지역='강남구',
        업종='카페'
    )

    print("\n✅ 핵심 기능 테스트 완료!")
    print("🎯 대출/투자 시뮬레이션 시스템 작동 확인")

if __name__ == "__main__":
    main()