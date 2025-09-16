#!/usr/bin/env python3
"""
Test Simple ML System
====================

완성된 ML 100% 시스템 테스트
"""

from super_simple_ml_system import SuperSimpleMLSystem

def test_ml_system():
    print("🚀 Testing Super Simple ML 100% System")
    print("=" * 50)

    # ML 시스템 로드
    ml_system = SuperSimpleMLSystem()

    if not ml_system.load_model():
        print("❌ Model not found, please run super_simple_ml_system.py first")
        return

    print("✅ ML Model loaded successfully!")

    # 테스트 케이스들
    test_cases = [
        {
            'name': '안정적인 카페',
            '총자산': 50000000,      # 5천만원
            '월매출': 12000000,      # 1200만원
            '인건비': 3000000,       # 300만원
            '임대료': 2000000,       # 200만원
            '식자재비': 3500000,     # 350만원
            '기타비용': 500000,      # 50만원
            '지역': '강남구',
            '업종': '커피전문점'
        },
        {
            'name': '위험한 음식점',
            '총자산': 20000000,      # 2천만원
            '월매출': 5000000,       # 500만원
            '인건비': 2500000,       # 250만원
            '임대료': 2200000,       # 220만원
            '식자재비': 2800000,     # 280만원
            '기타비용': 800000,      # 80만원
            '지역': '구로구',
            '업종': '한식음식점'
        },
        {
            'name': '중간 규모 치킨집',
            '총자산': 30000000,      # 3천만원
            '월매출': 8000000,       # 800만원
            '인건비': 2000000,       # 200만원
            '임대료': 1800000,       # 180만원
            '식자재비': 2500000,     # 250만원
            '기타비용': 700000,      # 70만원
            '지역': '마포구',
            '업종': '치킨전문점'
        }
    ]

    print("\n🧪 ML 예측 테스트:")
    print("=" * 50)

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 30)

        result = ml_system.predict_risk(
            총자산=case['총자산'],
            월매출=case['월매출'],
            인건비=case['인건비'],
            임대료=case['임대료'],
            식자재비=case['식자재비'],
            기타비용=case['기타비용'],
            지역=case['지역'],
            업종=case['업종']
        )

        print(f"🎯 결과: {result['risk_level']} ({result['risk_name']})")
        print(f"🔬 신뢰도: {result['confidence']:.1f}%")

    print("\n" + "=" * 60)
    print("✅ ML 100% 시스템 테스트 완료!")
    print("🎉 Pure Machine Learning Risk Prediction")
    print("📋 Altman Z-Score: Used for labeling only")
    print("🤖 Prediction: 100% ML (RandomForest)")
    print("=" * 60)

if __name__ == "__main__":
    test_ml_system()