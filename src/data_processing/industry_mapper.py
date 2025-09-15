"""
업종 매핑 시스템
63개 세분화 업종 → 4개 통합 카테고리 매핑
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class IndustryMapper:
    """업종 매핑 클래스"""

    def __init__(self):
        # 4개 통합 카테고리
        self.CATEGORY_MAPPING = {
            "도매소매업": {
                "description": "재료비(상품원가) 중심 - 82.3%",
                "cost_structure": {
                    "재료비": 0.823, "인건비": 0.058, "임대료": 0.039,
                    "제세공과금": 0.011, "기타": 0.069
                }
            },
            "숙박음식점업": {
                "description": "재료비+인건비 균형 - 42.6%+20.5%",
                "cost_structure": {
                    "재료비": 0.426, "인건비": 0.205, "임대료": 0.090,
                    "제세공과금": 0.035, "기타": 0.244
                }
            },
            "예술스포츠업": {
                "description": "인건비+임대료 중심 - 28.6%+19.3%",
                "cost_structure": {
                    "재료비": 0.156, "인건비": 0.286, "임대료": 0.193,
                    "제세공과금": 0.065, "기타": 0.300
                }
            },
            "개인서비스업": {
                "description": "인건비+기타 중심 - 29.7%+29.2%",
                "cost_structure": {
                    "재료비": 0.233, "인건비": 0.297, "임대료": 0.139,
                    "제세공과금": 0.039, "기타": 0.292
                }
            }
        }

        # 세분화 업종 → 통합 카테고리 매핑
        self.INDUSTRY_MAPPING = {
            # 숙박음식점업 (음식점 관련)
            "CS100001": "숙박음식점업",  # 한식음식점
            "CS100002": "숙박음식점업",  # 중식음식점
            "CS100003": "숙박음식점업",  # 일식음식점
            "CS100004": "숙박음식점업",  # 양식음식점
            "CS100005": "숙박음식점업",  # 제과점
            "CS100006": "숙박음식점업",  # 패스트푸드점
            "CS100007": "숙박음식점업",  # 치킨전문점
            "CS100008": "숙박음식점업",  # 분식전문점
            "CS100009": "숙박음식점업",  # 호프-간이주점
            "CS100010": "숙박음식점업",  # 커피-음료
            "CS200034": "숙박음식점업",  # 여관
            "CS200036": "숙박음식점업",  # 고시원

            # 도매소매업 (상품 판매 관련)
            "CS300001": "도매소매업",    # 슈퍼마켓
            "CS300002": "도매소매업",    # 편의점
            "CS300003": "도매소매업",    # 컴퓨터및주변장치판매
            "CS300004": "도매소매업",    # 핸드폰
            "CS300006": "도매소매업",    # 미곡판매
            "CS300007": "도매소매업",    # 육류판매
            "CS300008": "도매소매업",    # 수산물판매
            "CS300009": "도매소매업",    # 청과상
            "CS300010": "도매소매업",    # 반찬가게
            "CS300011": "도매소매업",    # 일반의류
            "CS300014": "도매소매업",    # 신발
            "CS300015": "도매소매업",    # 가방
            "CS300016": "도매소매업",    # 안경
            "CS300017": "도매소매업",    # 시계및귀금속
            "CS300018": "도매소매업",    # 의약품
            "CS300019": "도매소매업",    # 의료기기
            "CS300020": "도매소매업",    # 서적
            "CS300021": "도매소매업",    # 문구
            "CS300022": "도매소매업",    # 화장품
            "CS300024": "도매소매업",    # 운동/경기용품
            "CS300025": "도매소매업",    # 자전거 및 기타운송장비
            "CS300026": "도매소매업",    # 완구
            "CS300027": "도매소매업",    # 섬유제품
            "CS300028": "도매소매업",    # 화초
            "CS300029": "도매소매업",    # 애완동물
            "CS300031": "도매소매업",    # 가구
            "CS300032": "도매소매업",    # 가전제품
            "CS300033": "도매소매업",    # 철물점
            "CS300035": "도매소매업",    # 인테리어
            "CS300036": "도매소매업",    # 조명용품
            "CS300043": "도매소매업",    # 전자상거래업

            # 예술스포츠업 (교육, 스포츠, 엔터테인먼트)
            "CS200001": "예술스포츠업",  # 일반교습학원
            "CS200002": "예술스포츠업",  # 외국어학원
            "CS200003": "예술스포츠업",  # 예술학원
            "CS200005": "예술스포츠업",  # 스포츠 강습
            "CS200016": "예술스포츠업",  # 당구장
            "CS200017": "예술스포츠업",  # 골프연습장
            "CS200019": "예술스포츠업",  # PC방
            "CS200024": "예술스포츠업",  # 스포츠클럽
            "CS200037": "예술스포츠업",  # 노래방

            # 개인서비스업 (개인 대상 서비스)
            "CS200006": "개인서비스업",  # 일반의원
            "CS200007": "개인서비스업",  # 치과의원
            "CS200008": "개인서비스업",  # 한의원
            "CS200025": "개인서비스업",  # 자동차수리
            "CS200026": "개인서비스업",  # 자동차미용
            "CS200028": "개인서비스업",  # 미용실
            "CS200029": "개인서비스업",  # 네일숍
            "CS200030": "개인서비스업",  # 피부관리실
            "CS200031": "개인서비스업",  # 세탁소
            "CS200032": "개인서비스업",  # 가전제품수리
            "CS200033": "개인서비스업",  # 부동산중개업
        }

        logger.info(f"Industry Mapper 초기화: {len(self.INDUSTRY_MAPPING)}개 업종 매핑")

    def map_industry_code(self, industry_code: str) -> str:
        """업종 코드를 통합 카테고리로 매핑"""
        return self.INDUSTRY_MAPPING.get(industry_code, "기타")

    def map_industry_name(self, industry_name: str) -> str:
        """업종 명칭으로 카테고리 추론 (백업 방법)"""
        industry_name = industry_name.lower()

        # 음식점/카페/주점 관련
        food_keywords = ['음식점', '카페', '커피', '제과', '치킨', '분식', '호프', '주점',
                        '패스트푸드', '베이커리', '음료', '여관', '고시원']
        if any(keyword in industry_name for keyword in food_keywords):
            return "숙박음식점업"

        # 상품 판매 관련
        retail_keywords = ['마켓', '편의점', '판매', '상', '의류', '신발', '가방', '화장품',
                          '가구', '가전', '철물', '서적', '문구', '완구', '의약품', '전자상거래']
        if any(keyword in industry_name for keyword in retail_keywords):
            return "도매소매업"

        # 교육/스포츠/엔터테인먼트
        sports_keywords = ['학원', '교습', '스포츠', '당구', '골프', '클럽', 'pc방', '노래방']
        if any(keyword in industry_name for keyword in sports_keywords):
            return "예술스포츠업"

        # 개인서비스
        service_keywords = ['의원', '병원', '미용', '네일', '피부', '세탁', '수리', '부동산']
        if any(keyword in industry_name for keyword in service_keywords):
            return "개인서비스업"

        return "기타"

    def get_cost_structure(self, category: str) -> Dict[str, float]:
        """카테고리의 비용 구조 반환"""
        return self.CATEGORY_MAPPING.get(category, {}).get("cost_structure", {})

    def add_category_to_dataframe(self, df: pd.DataFrame,
                                 code_column: str = '서비스_업종_코드',
                                 name_column: str = '서비스_업종_코드_명') -> pd.DataFrame:
        """데이터프레임에 통합 카테고리 컬럼 추가"""

        # 업종 코드 기반 매핑
        df['통합업종카테고리'] = df[code_column].apply(self.map_industry_code)

        # 매핑되지 않은 항목은 업종명으로 재시도
        unmapped_mask = df['통합업종카테고리'] == '기타'
        if unmapped_mask.sum() > 0:
            df.loc[unmapped_mask, '통합업종카테고리'] = df.loc[unmapped_mask, name_column].apply(self.map_industry_name)

        logger.info(f"카테고리 매핑 결과:")
        category_counts = df['통합업종카테고리'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}건")

        return df

    def calculate_industry_expenses(self, revenue: float, category: str) -> Dict[str, float]:
        """카테고리별 비용 구조에 따른 예상 지출 계산"""
        cost_structure = self.get_cost_structure(category)

        if not cost_structure:
            # 기본값 (전체 평균 75.45%)
            return {
                "재료비": revenue * 0.5000,
                "인건비": revenue * 0.1500,
                "임대료": revenue * 0.0800,
                "제세공과금": revenue * 0.0245,
                "기타": revenue * 0.2000
            }

        return {
            category_name: revenue * ratio
            for category_name, ratio in cost_structure.items()
        }

    def get_mapping_summary(self) -> Dict:
        """매핑 요약 정보 반환"""
        summary = {}
        for category in self.CATEGORY_MAPPING.keys():
            mapped_codes = [code for code, cat in self.INDUSTRY_MAPPING.items() if cat == category]
            summary[category] = {
                "mapped_count": len(mapped_codes),
                "codes": mapped_codes,
                "description": self.CATEGORY_MAPPING[category]["description"]
            }

        return summary

    def validate_mapping_coverage(self, df: pd.DataFrame) -> Dict[str, int]:
        """매핑 커버리지 검증"""
        total_records = len(df)
        mapped_records = len(df[df['통합업종카테고리'] != '기타'])
        unmapped_records = total_records - mapped_records

        validation_result = {
            "total_records": total_records,
            "mapped_records": mapped_records,
            "unmapped_records": unmapped_records,
            "coverage_rate": (mapped_records / total_records) * 100 if total_records > 0 else 0
        }

        logger.info(f"매핑 커버리지: {validation_result['coverage_rate']:.1f}% "
                   f"({mapped_records}/{total_records})")

        return validation_result

# 사용 예시
def main():
    mapper = IndustryMapper()

    # 매핑 요약
    summary = mapper.get_mapping_summary()
    print("=== 업종 매핑 요약 ===")
    for category, info in summary.items():
        print(f"{category}: {info['mapped_count']}개 업종")
        print(f"  - {info['description']}")

    # 테스트
    test_codes = ["CS100001", "CS300001", "CS200001", "CS200028"]
    for code in test_codes:
        category = mapper.map_industry_code(code)
        cost_structure = mapper.get_cost_structure(category)
        print(f"{code} → {category}")
        print(f"  비용구조: {cost_structure}")

if __name__ == "__main__":
    main()