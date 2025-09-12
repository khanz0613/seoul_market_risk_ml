#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 빠른 데모
Quick demonstration of the Seoul Market Risk ML System
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def demo_single_business():
    """단일 비즈니스 위험도 계산 데모"""
    print("🏢 서울 시장 위험도 ML 시스템 - 빠른 데모")
    print("=" * 60)
    
    # 샘플 비즈니스 데이터
    sample_businesses = [
        {
            'business_id': 'DEMO_001',
            'business_name': '홍대맛집',
            'business_type': '음식점',
            'region_id': '11170520',  # 홍대 지역
            'revenue_history': [12000000, 13500000, 11800000, 14200000, 13100000],
            'description': '홍대 지역 인기 음식점'
        },
        {
            'business_id': 'DEMO_002', 
            'business_name': '편의점24',
            'business_type': '소매업',
            'region_id': '11140510',  # 중구 지역
            'revenue_history': [15000000, 14200000, 13800000, 13500000, 13000000],
            'description': '매출 감소 추세의 편의점'
        },
        {
            'business_id': 'DEMO_003',
            'business_name': '헤어살롱',
            'business_type': '서비스업', 
            'region_id': '11110530',  # 종로구 사직동
            'revenue_history': [8000000, 8200000, 8500000, 8800000, 9100000],
            'description': '꾸준한 성장세의 헤어살롱'
        }
    ]
    
    print("\n📊 샘플 비즈니스 위험도 분석:")
    print("-" * 60)
    
    results = []
    
    for business in sample_businesses:
        print(f"\n🏪 {business['business_name']} ({business['description']})")
        print(f"   📍 지역: {business['region_id']}")
        print(f"   💼 업종: {business['business_type']}")
        
        # 최근 5개월 매출 표시
        revenue_str = " → ".join([f"{rev:,}" for rev in business['revenue_history'][-3:]])
        print(f"   💰 최근 매출: {revenue_str}원")
        
        # 간단한 위험도 계산 (실제 모델 대신 데모용 로직)
        revenues = business['revenue_history']
        
        # 매출 변화율 계산
        revenue_change = ((revenues[-1] - revenues[0]) / revenues[0]) * 100
        
        # 변동성 계산
        volatility = np.std(revenues) / np.mean(revenues) * 100
        
        # 트렌드 계산 (선형 회귀)
        x = np.arange(len(revenues))
        trend = np.polyfit(x, revenues, 1)[0]
        
        # 간단한 위험도 점수 계산 (데모용)
        risk_score = 30  # 기본값
        
        if revenue_change < -10:  # 매출 10% 이상 감소
            risk_score += 20
        elif revenue_change < 0:  # 매출 감소
            risk_score += 10
        elif revenue_change > 20:  # 매출 20% 이상 증가
            risk_score -= 10
        elif revenue_change > 10:  # 매출 10% 이상 증가
            risk_score -= 5
            
        if volatility > 15:  # 높은 변동성
            risk_score += 15
        elif volatility < 5:  # 낮은 변동성
            risk_score -= 5
            
        if trend < -500000:  # 감소 트렌드
            risk_score += 10
        elif trend > 500000:  # 증가 트렌드
            risk_score -= 5
        
        # 업종별 조정
        business_multipliers = {
            "음식점": 1.0,
            "소매업": 1.1, 
            "서비스업": 0.9,
            "제조업": 0.8
        }
        
        risk_score *= business_multipliers.get(business['business_type'], 1.0)
        risk_score = max(0, min(100, risk_score))  # 0-100 범위로 제한
        
        # 위험 등급 결정
        if risk_score <= 20:
            risk_level = "🟢 안전"
            level_name = "안전"
        elif risk_score <= 40:
            risk_level = "🟡 주의"
            level_name = "주의"
        elif risk_score <= 60:
            risk_level = "🟠 경계"
            level_name = "경계"
        elif risk_score <= 80:
            risk_level = "🔴 위험"
            level_name = "위험"
        else:
            risk_level = "⚫ 매우위험"
            level_name = "매우위험"
            
        # 권장 대출 한도 계산 (평균 매출 기준)
        avg_revenue = np.mean(revenues)
        base_loan = avg_revenue * business_multipliers.get(business['business_type'], 2.0)
        risk_reduction = 1 - (risk_score / 200)  # 위험도에 따른 감소
        recommended_loan = base_loan * risk_reduction
        
        print(f"   🎯 위험도 점수: {risk_score:.1f}점")
        print(f"   📊 위험 등급: {risk_level}")
        print(f"   💰 권장 대출한도: {recommended_loan:,.0f}원")
        
        # 주요 요인 분석
        factors = []
        if revenue_change < -5:
            factors.append(f"매출 감소 ({revenue_change:.1f}%)")
        elif revenue_change > 15:
            factors.append(f"매출 증가 (+{revenue_change:.1f}%)")
            
        if volatility > 12:
            factors.append(f"높은 변동성 ({volatility:.1f}%)")
        elif volatility < 6:
            factors.append("안정적 매출")
            
        if trend < -300000:
            factors.append("하향 트렌드")
        elif trend > 300000:
            factors.append("상향 트렌드")
            
        if factors:
            print(f"   ⚠️ 주요 요인: {', '.join(factors)}")
        
        # 결과 저장
        results.append({
            'business_id': business['business_id'],
            'business_name': business['business_name'],
            'business_type': business['business_type'],
            'risk_score': round(risk_score, 1),
            'risk_level': level_name,
            'recommended_loan': int(recommended_loan),
            'revenue_change': round(revenue_change, 1),
            'volatility': round(volatility, 1),
            'trend': int(trend)
        })
    
    # 결과 요약
    print(f"\n📋 분석 결과 요약:")
    print("-" * 60)
    
    results_df = pd.DataFrame(results)
    
    print(f"총 분석 업체: {len(results)}개")
    
    # 위험도별 분포
    risk_distribution = results_df['risk_level'].value_counts()
    for level, count in risk_distribution.items():
        print(f"{level}: {count}개")
    
    # 평균 정보
    avg_risk = results_df['risk_score'].mean()
    avg_loan = results_df['recommended_loan'].mean()
    
    print(f"평균 위험도: {avg_risk:.1f}점")
    print(f"평균 권장 대출한도: {avg_loan:,.0f}원")
    
    # CSV로 저장
    results_df.to_csv('demo_results.csv', index=False, encoding='utf-8')
    print(f"\n💾 결과가 'demo_results.csv'에 저장되었습니다.")
    
    return results

def show_system_info():
    """시스템 정보 표시"""
    print("\n🔧 시스템 정보:")
    print("-" * 30)
    print(f"Python 버전: {sys.version.split()[0]}")
    print(f"작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {project_root}")
    
    # 모델 파일 존재 확인 - 실제 models/ 디렉토리 확인
    models_dir = os.path.join(project_root, 'models')
    
    print(f"\n📁 모델 파일:")
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        global_models = [f for f in model_files if f.startswith('global_')]
        regional_models = [f for f in model_files if f.startswith('regional_')]
        local_models = [f for f in model_files if f.startswith('local_')]
        
        print(f"   Global 모델: {'✅' if global_models else '❌'} ({len(global_models)}개)")
        print(f"   Regional 모델: {'✅' if regional_models else '❌'} ({len(regional_models)}개)")
        print(f"   Local 모델: {'✅' if local_models else '❌'} ({len(local_models)}개)")
        print(f"   총 모델 파일: {len(model_files)}개")
    else:
        print(f"   models/: ❌ (디렉토리 없음)")

def main():
    """메인 데모 함수"""
    print("🚀 서울 시장 위험도 ML 시스템 시작")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 시스템 정보 표시
        show_system_info()
        
        # 데모 실행
        results = demo_single_business()
        
        print(f"\n🎉 데모 완료!")
        print("\n💡 다음 단계:")
        print("   1. 사용법_가이드.md 파일을 확인하세요")
        print("   2. config/config.yaml에서 설정을 조정하세요") 
        print("   3. 실제 데이터로 python src/training/model_trainer.py를 실행하세요")
        print("   4. python src/benchmarks/system_benchmark.py로 성능을 확인하세요")
        
    except Exception as e:
        print(f"\n❌ 데모 실행 중 오류가 발생했습니다:")
        print(f"   {str(e)}")
        print(f"\n🔧 해결 방법:")
        print(f"   1. pip install pandas numpy scikit-learn")
        print(f"   2. 프로젝트 디렉토리에서 실행했는지 확인")

if __name__ == "__main__":
    main()