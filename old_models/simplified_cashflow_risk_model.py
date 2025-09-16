#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간소화된 현금흐름 위험도 예측 모델
사용자 친화적 인터페이스 + 기존 위험도 계산 로직 활용
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SimplifiedCashFlowRiskModel:
    def __init__(self):
        """간소화된 현금흐름 위험도 예측 모델 초기화"""
        self.district_clusters = self._create_district_clusters()
        self.business_risk_mapping = self._create_business_risk_mapping()

        # 5단계 위험도 분류 (위험도 스코어가 높을수록 위험함)
        self.risk_levels = {
            1: {"name": "매우여유", "range": (0.0, 0.2), "emoji": "🌟", "color": "blue"},
            2: {"name": "여유", "range": (0.2, 0.4), "emoji": "🟢", "color": "green"},
            3: {"name": "안정", "range": (0.4, 0.6), "emoji": "🟡", "color": "yellow"},
            4: {"name": "위험", "range": (0.6, 0.8), "emoji": "🟠", "color": "orange"},
            5: {"name": "매우위험", "range": (0.8, 1.0), "emoji": "🔴", "color": "red"}
        }

    def _create_district_clusters(self) -> Dict[str, List[str]]:
        """서울 행정동 클러스터 매핑 (기존 로직 재사용)"""
        return {
            'premium': [
                '역삼1동', '논현1동', '압구정동', '청담동', '삼성1동', '대치1동', '대치2동', '대치4동',
                '역삼2동', '논현2동', '신사동', '삼성2동', '개포2동', '일원1동', '일원2동', '수서동',
                '서초1동', '서초2동', '서초3동', '서초4동', '반포1동', '반포2동', '반포3동', '반포4동',
                '잠원동', '방배1동', '방배2동', '방배3동', '양재1동', '양재2동', '내곡동',
                '종로1·2·3·4가동', '명동', '을지로동', '회현동', '여의동', '영등포동'
            ],
            'upscale': [
                '서교동', '합정동', '상수동', '연남동', '이태원1동', '이태원2동', '한남동',
                '성수1가1동', '성수1가2동', '성수2가1동', '왕십리2동',
                '화곡본동', '등촌1동', '등촌2동', '염창동', '발산1동'
            ],
            'midtier': [
                '제기동', '청운효자동', '사직동', '성북동', '삼선동', '안암동', '보문동',
                '노원1동', '노원2동', '상계1동', '상계2동', '중계본동', '중계1동',
                '은평구', '갈현1동', '불광1동', '홍제1동', '신사1동'
            ],
            'standard': [
                '강일동', '상일동', '명일1동', '고덕1동', '암사1동', '천호1동', '성내1동', '길동',
                '면목본동', '면목2동', '상봉1동', '중화1동', '묵1동', '망우본동',
                '충현동', '신촌동', '연희동', '홍은1동', '남가좌1동', '북가좌1동'
            ],
            'residential': [
                '청룡동', '청림동', '낙성대동', '서원동', '신원동', '난곡동', '상도1동',
                '흑석동', '노량진1동', '대방동', '신대방1동',
                '가산동', '독산1동', '시흥1동', '시흥2동',
                '신도림동', '구로1동', '개봉1동', '오류1동'
            ],
            'suburban': [
                '삼양동', '미아동', '번1동', '수유1동', '우이동',
                '쌍문1동', '방학1동', '창1동', '도봉1동',
                '전농1동', '답십리1동', '장안1동', '청량리동', '회기동', '이문1동'
            ]
        }

    def _create_business_risk_mapping(self) -> Dict[str, float]:
        """업종별 위험도 매핑 (기존 로직 재사용)"""
        return {
            # 고위험 업종 (0.25-0.3)
            '유흥주점': 0.3, '단란주점': 0.3, 'PC방': 0.28, '노래방': 0.25,
            '찜질방': 0.25, '게임방': 0.28, '당구장': 0.25,

            # 중위험 업종 (0.15-0.2)
            '한식음식점': 0.15, '중식음식점': 0.16, '일식음식점': 0.16,
            '양식음식점': 0.17, '카페': 0.18, '커피전문점': 0.18,
            '치킨전문점': 0.19, '분식전문점': 0.17, '호프': 0.2,
            '간이주점': 0.2, '제과점': 0.16,

            # 저위험 업종 (0.08-0.15)
            '슈퍼마켓': 0.08, '편의점': 0.09, '일반의원': 0.1,
            '약국': 0.08, '미용실': 0.12, '세탁소': 0.1,
            '문구점': 0.11, '서점': 0.12, '안경점': 0.1,
            '핸드폰판매점': 0.13, '부동산중개업소': 0.14,

            # 기타 (0.15)
            '기타': 0.15
        }

    def get_location_cluster(self, location: str) -> str:
        """지역을 클러스터로 매핑"""
        for cluster, districts in self.district_clusters.items():
            if any(district in location for district in districts):
                return cluster
        return 'standard'  # 기본값

    def get_business_risk(self, business_type: str) -> float:
        """업종별 위험도 반환"""
        # 키워드 매칭
        for business, risk in self.business_risk_mapping.items():
            if business in business_type:
                return risk
        return 0.15  # 기본값

    def calculate_risk_score(self,
                           total_available_assets: int,
                           monthly_revenue: int,
                           monthly_expenses: Dict[str, int],
                           business_type: str,
                           location: str) -> float:
        """
        기존 위험도 계산 로직 활용한 위험도 스코어 계산 (개선된 현금흐름 반영)
        """
        # 총 지출액 계산
        total_expenses = sum(monthly_expenses.values())
        monthly_cashflow = monthly_revenue - total_expenses

        # 1. 지출 규모 기반 위험도 (0-0.4) - 기존 로직 유지
        spending_score = min(0.4, total_expenses / 1e8 * 0.4)  # 1억 기준으로 조정

        # 2. 지역 클러스터 기반 위험도 (0-0.3) - 기존 로직 유지
        cluster = self.get_location_cluster(location)
        cluster_risk = {
            'premium': 0.05,     # 강남 - 낮은 위험
            'upscale': 0.1,      # 홍대 - 중간 위험
            'midtier': 0.15,     # 강북 - 중간 위험
            'standard': 0.2,     # 일반 - 높은 위험
            'residential': 0.25, # 주거 - 높은 위험
            'suburban': 0.3      # 외곽 - 최고 위험
        }
        location_score = cluster_risk.get(cluster, 0.2)

        # 3. 업종별 위험도 (0-0.2) - 기존 로직에서 약간 조정
        business_score = self.get_business_risk(business_type) * 0.67  # 0.3을 0.2로 스케일 조정

        # 4. 현금흐름 위험도 (0-0.3) - 핵심 개선사항
        if total_available_assets > 0:
            # 월 현금흐름의 자산 대비 비율
            cashflow_ratio = monthly_cashflow / total_available_assets

            if cashflow_ratio < -0.05:  # 매월 자산의 5% 이상 감소
                cashflow_score = 0.3
            elif cashflow_ratio < 0:    # 적자 (자산 감소)
                cashflow_score = 0.25
            elif cashflow_ratio < 0.02: # 매월 2% 미만 증가 (저성장)
                cashflow_score = 0.15
            elif cashflow_ratio < 0.05: # 매월 2-5% 증가 (적정성장)
                cashflow_score = 0.1
            else:                        # 매월 5% 이상 증가 (고성장)
                cashflow_score = 0.05
        else:
            cashflow_score = 0.3

        # 5. 매출 대비 지출 비율 위험도 (추가 안전장치)
        if monthly_revenue > 0:
            expense_ratio = total_expenses / monthly_revenue
            if expense_ratio >= 1.2:      # 지출이 매출의 120% 이상
                expense_penalty = 0.2
            elif expense_ratio >= 1.0:    # 지출이 매출과 같거나 큼
                expense_penalty = 0.15
            elif expense_ratio >= 0.9:    # 지출이 매출의 90% 이상
                expense_penalty = 0.1
            else:                          # 건전한 수준
                expense_penalty = 0.0
        else:
            expense_penalty = 0.3

        # 6. 노이즈 추가 (현실적 불확실성)
        noise = np.random.normal(0, 0.01)  # 더 줄어든 노이즈

        total_risk = spending_score + location_score + business_score + cashflow_score + expense_penalty + noise
        return np.clip(total_risk, 0, 1)

    def classify_risk_level(self, risk_score: float) -> int:
        """위험도 스코어를 5단계로 분류"""
        for level, info in self.risk_levels.items():
            if info["range"][0] <= risk_score < info["range"][1]:
                return level
        return 5  # 최고 위험도 (매우위험)

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
        target_risk_level = 3  # 안정
        target_risk_score = 0.6  # 안정권 상한 (위험도 스코어 0.6 이하가 안정)

        if current_risk_score <= target_risk_score:
            return {"loan_amount": 0, "reason": "이미 안정권입니다"}

        # 이진 탐색으로 필요한 대출 금액 찾기
        min_loan = 0
        max_loan = total_available_assets * 3  # 최대 현재 자산의 3배까지
        tolerance = 10000  # 1만원 단위

        for _ in range(50):  # 최대 50회 반복
            mid_loan = (min_loan + max_loan) // 2
            new_assets = total_available_assets + mid_loan

            # 새로운 자산으로 위험도 재계산
            new_risk_score = self.calculate_risk_score(
                new_assets, monthly_revenue, monthly_expenses,
                business_type, location
            )

            if abs(new_risk_score - target_risk_score) < 0.01:
                return {
                    "loan_amount": mid_loan,
                    "target_risk_score": target_risk_score,
                    "expected_new_risk_score": new_risk_score
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
            "expected_new_risk_score": self.calculate_risk_score(
                total_available_assets + min_loan, monthly_revenue,
                monthly_expenses, business_type, location
            )
        }

    def calculate_investment_amount(self,
                                  current_risk_level: int,
                                  total_available_assets: int,
                                  monthly_revenue: int,
                                  monthly_expenses: Dict[str, int],
                                  business_type: str,
                                  location: str) -> Dict[str, int]:
        """
        위험도 레벨에 따른 투자 가능 금액 계산
        """
        if current_risk_level >= 4:  # 위험, 매우위험
            return {"investment_amount": 0, "reason": "위험 단계 - 투자보다 안정화 우선"}

        if current_risk_level == 1:  # 매우여유 단계
            # 현재 자산의 70% 투자 가능
            return {
                "investment_amount": int(total_available_assets * 0.7),
                "investment_ratio": 0.7,
                "reason": "매우여유 단계 - 적극적 투자 가능"
            }
        elif current_risk_level == 2:  # 여유 단계
            # 현재 자산의 50% 투자 가능
            return {
                "investment_amount": int(total_available_assets * 0.5),
                "investment_ratio": 0.5,
                "reason": "여유 단계 - 보수적 투자 가능"
            }
        else:  # 안정 단계 (레벨 3)
            # 현재 자산의 30% 투자 가능
            return {
                "investment_amount": int(total_available_assets * 0.3),
                "investment_ratio": 0.3,
                "reason": "안정 단계 - 소액 투자 가능"
            }

    def generate_action_recommendations(self,
                                      risk_level: int,
                                      risk_score: float,
                                      loan_info: Dict,
                                      investment_info: Dict) -> Dict:
        """위험도별 액션 권장사항 생성"""

        risk_info = self.risk_levels[risk_level]

        recommendations = {
            "risk_level": risk_level,
            "risk_name": risk_info["name"],
            "risk_emoji": risk_info["emoji"],
            "risk_score": risk_score,
            "message": "",
            "primary_actions": [],
            "secondary_actions": [],
            "financial_products": []
        }

        if risk_level >= 4:  # 위험, 매우위험
            recommendations["message"] = f"{risk_info['emoji']} {risk_info['name']} 상태입니다. 즉시 자금 확보가 필요합니다."

            if loan_info.get("loan_amount", 0) > 0:
                recommendations["primary_actions"] = [
                    f"💰 긴급 대출 필요: {loan_info['loan_amount']:,}원",
                    "📞 NH농협 소상공인 긴급대출 상담",
                    "💳 신용보증재단 대출 검토",
                    "📊 비용 구조 즉시 점검"
                ]

            recommendations["secondary_actions"] = [
                "🔍 불필요한 지출 즉시 중단",
                "📈 매출 증대 방안 검토",
                "🤝 거래처 결제 조건 재협상",
                "🎯 핵심 사업에 집중"
            ]

            recommendations["financial_products"] = [
                "NH농협 소상공인 긴급운영자금 (연 4.2%~7.0%)",
                "신용보증재단 특별보증 대출",
                "정책자금 긴급지원 프로그램"
            ]

        elif risk_level == 3:  # 안정
            recommendations["message"] = f"{risk_info['emoji']} {risk_info['name']} 상태입니다. 현상 유지하며 성장 기회를 모색하세요."

            recommendations["primary_actions"] = [
                "📊 현재 수준 유지 관리",
                "📈 매출 안정성 확보",
                "💼 비즈니스 모델 최적화",
                "🎯 고객 만족도 향상"
            ]

            if investment_info.get("investment_amount", 0) > 0:
                recommendations["secondary_actions"] = [
                    f"💎 소액 투자 검토: {investment_info['investment_amount']:,}원",
                    "🏦 NH농협 적금 상품 검토",
                    "📚 사업 확장 계획 수립"
                ]

            recommendations["financial_products"] = [
                "NH농협 소상공인 적금 (연 3.5%~4.0%)",
                "안정형 펀드 투자",
                "정기예금 (연 3.0%~3.5%)"
            ]

        else:  # 여유, 매우여유 (레벨 1, 2)
            recommendations["message"] = f"{risk_info['emoji']} {risk_info['name']} 상태입니다. 적극적인 투자로 자산을 늘려보세요."

            investment_amount = investment_info.get("investment_amount", 0)
            if investment_amount > 0:
                recommendations["primary_actions"] = [
                    f"💰 투자 추천: {investment_amount:,}원",
                    "📈 NH농협 펀드 투자 검토",
                    "🏢 부동산 투자 기회 탐색",
                    "💎 주식 포트폴리오 구성"
                ]

            recommendations["secondary_actions"] = [
                "🚀 사업 확장 계획 수립",
                "🔄 추가 수익원 개발",
                "🎓 인력 교육 투자",
                "🏆 브랜드 가치 향상"
            ]

            if risk_level == 1:  # 매우여유
                recommendations["financial_products"] = [
                    "NH농협 성장형 펀드 (기대수익 8%~12%)",
                    "주식형 펀드 투자",
                    "해외 투자 상품",
                    "부동산 투자신탁(REITs)"
                ]
            else:  # 여유
                recommendations["financial_products"] = [
                    "NH농협 혼합형 펀드 (기대수익 5%~8%)",
                    "중위험 중수익 상품",
                    "채권형 펀드",
                    "금융상품 분산투자"
                ]

        return recommendations

    def predict_comprehensive_risk(self,
                                 total_available_assets: int,
                                 monthly_revenue: int,
                                 monthly_expenses: Dict[str, int],
                                 business_type: str,
                                 location: str) -> Dict:
        """
        종합적인 위험도 예측 및 권장사항 생성
        """
        print(f"🔍 위험도 분석 중...")
        print(f"  💰 총 운용자산: {total_available_assets:,}원")
        print(f"  📈 월 매출: {monthly_revenue:,}원")
        print(f"  📊 월 지출: {sum(monthly_expenses.values()):,}원")
        print(f"  🏪 업종: {business_type}")
        print(f"  📍 지역: {location}")

        # 1. 위험도 스코어 계산
        risk_score = self.calculate_risk_score(
            total_available_assets, monthly_revenue, monthly_expenses,
            business_type, location
        )

        # 2. 위험도 레벨 분류
        risk_level = self.classify_risk_level(risk_score)

        # 3. 대출 금액 계산 (위험 단계인 경우)
        loan_info = {}
        if risk_level >= 4:  # 위험, 매우위험
            loan_info = self.calculate_loan_amount(
                risk_score, total_available_assets, monthly_revenue,
                monthly_expenses, business_type, location
            )

        # 4. 투자 금액 계산 (안정 이상인 경우)
        investment_info = {}
        if risk_level <= 3:  # 매우여유, 여유, 안정
            investment_info = self.calculate_investment_amount(
                risk_level, total_available_assets, monthly_revenue,
                monthly_expenses, business_type, location
            )

        # 5. 권장사항 생성
        recommendations = self.generate_action_recommendations(
            risk_level, risk_score, loan_info, investment_info
        )

        # 6. 종합 결과
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": {
                "total_available_assets": total_available_assets,
                "monthly_revenue": monthly_revenue,
                "monthly_expenses": monthly_expenses,
                "business_type": business_type,
                "location": location,
                "location_cluster": self.get_location_cluster(location)
            },
            "risk_analysis": {
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "risk_name": self.risk_levels[risk_level]["name"],
                "risk_emoji": self.risk_levels[risk_level]["emoji"],
                "monthly_cashflow": monthly_revenue - sum(monthly_expenses.values())
            },
            "loan_analysis": loan_info,
            "investment_analysis": investment_info,
            "recommendations": recommendations
        }

        return result

if __name__ == "__main__":
    # 샘플 테스트
    model = SimplifiedCashFlowRiskModel()

    # 테스트 케이스
    test_cases = [
        {
            "name": "위험 사례 - 적자 운영",
            "total_available_assets": 30000000,  # 3천만원
            "monthly_revenue": 8000000,          # 800만원
            "monthly_expenses": {
                "labor_cost": 4000000,           # 400만원
                "food_materials": 3000000,       # 300만원
                "rent": 2000000,                 # 200만원
                "others": 1000000                # 100만원
            },
            "business_type": "한식음식점",
            "location": "관악구 신림동"
        },
        {
            "name": "안정 사례 - 균형 운영",
            "total_available_assets": 80000000,  # 8천만원
            "monthly_revenue": 15000000,         # 1500만원
            "monthly_expenses": {
                "labor_cost": 5000000,           # 500만원
                "food_materials": 4000000,       # 400만원
                "rent": 2500000,                 # 250만원
                "others": 1500000                # 150만원
            },
            "business_type": "카페",
            "location": "홍대 서교동"
        },
        {
            "name": "여유 사례 - 흑자 운영",
            "total_available_assets": 150000000, # 1억5천만원
            "monthly_revenue": 25000000,         # 2500만원
            "monthly_expenses": {
                "labor_cost": 6000000,           # 600만원
                "food_materials": 5000000,       # 500만원
                "rent": 3000000,                 # 300만원
                "others": 2000000                # 200만원
            },
            "business_type": "일식음식점",
            "location": "강남구 역삼동"
        }
    ]

    print("🚀 간소화된 현금흐름 위험도 예측 모델 테스트")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 테스트 케이스 {i}: {case['name']}")
        print("-" * 40)

        result = model.predict_comprehensive_risk(
            case["total_available_assets"],
            case["monthly_revenue"],
            case["monthly_expenses"],
            case["business_type"],
            case["location"]
        )

        # 결과 출력
        risk = result["risk_analysis"]
        print(f"📊 위험도: {risk['risk_emoji']} {risk['risk_name']} ({risk['risk_score']:.3f})")
        print(f"💰 월 현금흐름: {risk['monthly_cashflow']:,}원")

        if result.get("loan_analysis") and result["loan_analysis"].get("loan_amount", 0) > 0:
            loan = result["loan_analysis"]
            print(f"🏦 권장 대출: {loan['loan_amount']:,}원")

        if result.get("investment_analysis") and result["investment_analysis"].get("investment_amount", 0) > 0:
            investment = result["investment_analysis"]
            print(f"💎 투자 가능: {investment['investment_amount']:,}원")

        print(f"💬 조언: {result['recommendations']['message']}")

        # 결과 저장
        os.makedirs('test_results', exist_ok=True)
        with open(f'test_results/test_case_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ 테스트 완료! test_results/ 폴더에서 상세 결과 확인 가능")