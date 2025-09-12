#!/usr/bin/env python3
"""
Model Orchestrator 독립 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """모델 로딩 테스트"""
    print("🧪 모델 로딩 테스트")
    print("=" * 40)
    
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print("❌ models/ 디렉토리가 없습니다.")
        return False
    
    model_files = list(models_dir.glob("*.joblib"))
    print(f"📁 발견된 모델 파일: {len(model_files)}개")
    
    # Global 모델 테스트
    global_model_path = models_dir / "global_model.joblib"
    global_model = None
    
    if global_model_path.exists():
        try:
            global_model = joblib.load(global_model_path)
            print(f"✅ Global 모델 로딩 성공: {type(global_model)}")
            
            if hasattr(global_model, 'feature_names_in_'):
                print(f"   📊 필요 Feature 수: {len(global_model.feature_names_in_)}")
            elif hasattr(global_model, 'n_features_in_'):
                print(f"   📊 필요 Feature 수: {global_model.n_features_in_}")
                
        except Exception as e:
            print(f"❌ Global 모델 로딩 실패: {e}")
            return False
    else:
        print("❌ Global 모델 파일이 없습니다.")
        return False
    
    # Regional 모델들 테스트
    regional_files = list(models_dir.glob("regional_*.joblib"))
    print(f"\n🏘️ Regional 모델 테스트: {len(regional_files)}개")
    
    regional_models = {}
    for regional_file in regional_files[:3]:  # 처음 3개만
        try:
            model = joblib.load(regional_file)
            region_id = regional_file.stem.split('_')[2]
            regional_models[region_id] = model
            print(f"✅ {regional_file.name}: 지역 {region_id}")
        except Exception as e:
            print(f"❌ {regional_file.name}: {e}")
    
    # Local 모델들 테스트
    local_files = list(models_dir.glob("local_*.joblib"))
    print(f"\n🏪 Local 모델 테스트: {len(local_files)}개")
    
    local_models = {}
    for local_file in local_files[:5]:  # 처음 5개만
        try:
            model = joblib.load(local_file)
            parts = local_file.stem.split('_')
            region_id = parts[2]
            category_id = parts[3]
            local_models[f"{region_id}_{category_id}"] = model
            print(f"✅ {local_file.name}: 지역 {region_id}, 업종 {category_id}")
        except Exception as e:
            print(f"❌ {local_file.name}: {e}")
    
    return {
        'global': global_model,
        'regional': regional_models, 
        'local': local_models
    }

def create_sample_prediction_data():
    """샘플 예측 데이터 생성"""
    # 기본적인 feature들 (실제 모델에 맞춰 조정 필요)
    sample_data = {
        'revenue_mean': [10000000, 12000000, 11000000],
        'revenue_trend': [100000, -50000, 200000],
        'revenue_volatility': [0.1, 0.15, 0.08],
        'seasonal_factor': [1.0, 1.2, 0.9],
        'industry_avg_revenue': [9000000, 11000000, 10000000],
        'region_economic_index': [1.1, 1.0, 1.15],
        'business_age_months': [24, 36, 18],
        'employee_count': [5, 8, 3]
    }
    
    return pd.DataFrame(sample_data)

def test_model_prediction(models):
    """모델 예측 테스트"""
    print("\n🎯 모델 예측 테스트")
    print("=" * 40)
    
    if not models or not models['global']:
        print("❌ 로딩된 모델이 없습니다.")
        return
    
    # 샘플 데이터 생성
    sample_data = create_sample_prediction_data()
    print(f"📊 샘플 데이터 생성: {len(sample_data)} rows, {len(sample_data.columns)} features")
    
    # Global 모델로 예측
    try:
        global_model = models['global']
        
        # 모델이 기대하는 feature 수 확인
        if hasattr(global_model, 'feature_names_in_'):
            expected_features = len(global_model.feature_names_in_)
            expected_names = list(global_model.feature_names_in_)
        elif hasattr(global_model, 'n_features_in_'):
            expected_features = global_model.n_features_in_
            expected_names = [f'feature_{i}' for i in range(expected_features)]
        else:
            expected_features = len(sample_data.columns)
            expected_names = list(sample_data.columns)
        
        print(f"   모델이 기대하는 feature 수: {expected_features}")
        print(f"   현재 데이터 feature 수: {len(sample_data.columns)}")
        
        if expected_features != len(sample_data.columns):
            print(f"⚠️ Feature 수 불일치. 모델이 기대하는 feature들:")
            if hasattr(global_model, 'feature_names_in_'):
                for i, name in enumerate(global_model.feature_names_in_[:10]):  # 처음 10개만
                    print(f"     {i+1}. {name}")
                if len(global_model.feature_names_in_) > 10:
                    print(f"     ... 외 {len(global_model.feature_names_in_)-10}개")
            
            # 최소한의 더미 데이터로 테스트
            dummy_data = np.random.random((3, expected_features))
            predictions = global_model.predict(dummy_data)
            
        else:
            predictions = global_model.predict(sample_data)
        
        print(f"✅ Global 모델 예측 성공")
        print(f"   예측 결과 수: {len(predictions)}")
        print(f"   예측 값 범위: {np.min(predictions):,.0f} ~ {np.max(predictions):,.0f}")
        
        # 샘플 비즈니스별 결과 표시
        business_names = ['홍대 맛집', '중구 편의점', '종로 헤어살롱']
        
        for i, (name, pred) in enumerate(zip(business_names, predictions)):
            # 위험도 점수로 변환 (임시 공식)
            risk_score = max(0, min(100, 50 - (pred / 1000000) * 5))
            
            if risk_score > 70:
                risk_level = "🔴 위험"
            elif risk_score > 50:
                risk_level = "🟠 경계"
            elif risk_score > 30:
                risk_level = "🟡 주의"
            else:
                risk_level = "🟢 안전"
            
            print(f"\n   🏪 {name}")
            print(f"      예측 매출: {pred:,.0f}원")
            print(f"      위험도 점수: {risk_score:.1f}점")
            print(f"      위험 등급: {risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ 예측 실행 오류: {e}")
        print(f"   오류 타입: {type(e)}")
        return False

def demonstrate_orchestrator_concept():
    """오케스트레이터 개념 시연"""
    print("\n🎭 Model Orchestrator 개념 시연")
    print("=" * 40)
    
    print("""
📋 Model Orchestrator의 역할:

1️⃣ 지능형 모델 선택
   • Local 모델 (특정 지역+업종) → 가장 정확
   • Regional 모델 (지역별) → 중간 정확도  
   • Global 모델 (전체) → 기본 정확도

2️⃣ 자동 폴백 시스템
   • Local 모델 없음 → Regional 모델 시도
   • Regional 모델 실패 → Global 모델 사용
   • 신뢰도가 낮으면 다음 단계로 이동

3️⃣ 성능 모니터링
   • 각 모델의 예측 성능 추적
   • 실시간 신뢰도 계산
   • 자동 재훈련 권장

4️⃣ 배치 예측 지원
   • 여러 비즈니스 동시 분석
   • 효율적인 리소스 사용
   • 성능 최적화
""")

def main():
    """메인 함수"""
    print("🚀 Model Orchestrator 독립 테스트")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {project_root}")
    
    try:
        # 모델 로딩 테스트
        models = test_model_loading()
        
        if models:
            # 예측 테스트
            test_model_prediction(models)
            
            # 개념 설명
            demonstrate_orchestrator_concept()
            
            print(f"\n🎉 테스트 완료!")
            print(f"\n💡 실제 사용 방법:")
            print(f"   1. python simple_test.py - 기본 테스트")
            print(f"   2. python quick_demo.py - 상세 데모") 
            print(f"   3. python test_orchestrator.py - 오케스트레이터 테스트")
            
        else:
            print(f"\n❌ 모델 로딩에 실패했습니다.")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()