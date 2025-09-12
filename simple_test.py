#!/usr/bin/env python3
"""
간단한 모델 테스트 스크립트
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def test_models():
    """훈련된 모델들을 직접 테스트"""
    print("🧪 서울 시장 위험도 모델 테스트")
    print("=" * 50)
    
    # 모델 파일 확인
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("❌ models/ 디렉토리가 없습니다.")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    print(f"📁 발견된 모델 파일: {len(model_files)}개")
    
    # Global 모델 테스트
    global_model_path = os.path.join(model_dir, "global_model.joblib")
    if os.path.exists(global_model_path):
        print(f"\n🌍 Global 모델 테스트 중...")
        try:
            model = joblib.load(global_model_path)
            print(f"   ✅ 로딩 성공: {type(model)}")
            
            # 간단한 예측 테스트 (더미 데이터)
            # 모델이 어떤 feature를 기대하는지 확인
            if hasattr(model, 'feature_names_in_'):
                print(f"   📊 Feature 개수: {len(model.feature_names_in_)}")
                print(f"   📝 일부 Features: {list(model.feature_names_in_[:5])}")
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    # Regional 모델들 테스트  
    regional_files = [f for f in model_files if f.startswith('regional_')]
    print(f"\n🏘️ Regional 모델들: {len(regional_files)}개")
    
    for i, filename in enumerate(regional_files[:3]):  # 처음 3개만 테스트
        try:
            model_path = os.path.join(model_dir, filename)
            model = joblib.load(model_path)
            region_id = filename.split('_')[2].split('.')[0]
            print(f"   ✅ {filename}: 지역 {region_id}")
        except Exception as e:
            print(f"   ❌ {filename}: {e}")
    
    # Local 모델들 테스트
    local_files = [f for f in model_files if f.startswith('local_')]
    print(f"\n🏪 Local 모델들: {len(local_files)}개")
    
    for i, filename in enumerate(local_files[:5]):  # 처음 5개만 테스트
        try:
            model_path = os.path.join(model_dir, filename)
            model = joblib.load(model_path)
            parts = filename.split('_')
            region_id = parts[2]
            category_id = parts[3].split('.')[0]
            print(f"   ✅ {filename}: 지역 {region_id}, 업종 {category_id}")
        except Exception as e:
            print(f"   ❌ {filename}: {e}")

def create_sample_prediction():
    """샘플 비즈니스 데이터로 간단한 예측"""
    print(f"\n🎯 샘플 예측 테스트")
    print("-" * 30)
    
    # 샘플 비즈니스 데이터
    sample_businesses = [
        {
            'name': '홍대 맛집',
            'region': '11110515',
            'category': '음식점',
            'revenue_history': [12000000, 13500000, 11800000, 14200000, 13100000]
        },
        {
            'name': '중구 편의점', 
            'region': '11110540',
            'category': '소매업',
            'revenue_history': [15000000, 14200000, 13800000, 13500000, 13000000]
        },
        {
            'name': '종로 헤어살롱',
            'region': '11110530', 
            'category': '서비스업',
            'revenue_history': [8000000, 8200000, 8500000, 8800000, 9100000]
        }
    ]
    
    print("📊 간단한 위험도 분석:")
    
    for business in sample_businesses:
        revenues = business['revenue_history']
        
        # 기본 지표 계산
        revenue_change = ((revenues[-1] - revenues[0]) / revenues[0]) * 100
        volatility = np.std(revenues) / np.mean(revenues) * 100
        trend = np.polyfit(range(len(revenues)), revenues, 1)[0]
        
        # 간단한 위험도 점수 계산
        risk_score = 50  # 기본값
        
        if revenue_change < -10:
            risk_score += 25
        elif revenue_change < 0:
            risk_score += 15
        elif revenue_change > 20:
            risk_score -= 15
        elif revenue_change > 10:
            risk_score -= 10
            
        if volatility > 15:
            risk_score += 10
        elif volatility < 5:
            risk_score -= 5
            
        if trend < -500000:
            risk_score += 10
        elif trend > 500000:
            risk_score -= 10
        
        # 결과 출력
        risk_level = "매우위험" if risk_score > 80 else "위험" if risk_score > 60 else "주의" if risk_score > 40 else "안전"
        
        print(f"\n🏪 {business['name']}")
        print(f"   📍 지역: {business['region']}")
        print(f"   💼 업종: {business['category']}")
        print(f"   📈 매출변화: {revenue_change:+.1f}%")
        print(f"   📊 변동성: {volatility:.1f}%")
        print(f"   🎯 위험도: {risk_score:.0f}점 ({risk_level})")
        
        # 권장 조치
        if risk_score > 60:
            print(f"   💡 권장: 긴급 자금 지원 검토")
        elif risk_score > 40:
            print(f"   💡 권장: 안정화 대출 상담") 
        else:
            print(f"   💡 권장: 성장 투자 기회 탐색")

def main():
    """메인 함수"""
    print(f"🚀 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 모델 테스트
    test_models()
    
    # 샘플 예측
    create_sample_prediction()
    
    print(f"\n✨ 테스트 완료!")
    print(f"\n💡 다음 단계:")
    print(f"   1. python simple_test.py 로 이 스크립트 실행")
    print(f"   2. 실제 데이터가 준비되면 전체 시스템 사용 가능")
    print(f"   3. 각 모델을 개별적으로 로드해서 더 정확한 예측 가능")

if __name__ == "__main__":
    main()