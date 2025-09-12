#!/usr/bin/env python3
"""
독립 실행 가능한 Model Orchestrator 스크립트
Import 문제 해결된 버전
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ModelLevel(Enum):
    """Model hierarchy levels."""
    LOCAL = "local"
    REGIONAL = "regional" 
    GLOBAL = "global"

class PredictionConfidence(Enum):
    """Prediction confidence levels."""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"        # <70% confidence

@dataclass
class PredictionRequest:
    """Request structure for model predictions."""
    business_id: str
    region_id: Optional[int] = None
    business_category: Optional[int] = None
    historical_data: List[Dict] = field(default_factory=list)
    prediction_horizon: int = 30
    required_confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    prefer_local: bool = True

@dataclass
class PredictionResult:
    """Result structure for model predictions."""
    predictions: pd.DataFrame
    model_used: str
    model_level: ModelLevel
    confidence_score: float
    prediction_timestamp: str
    processing_time_seconds: float
    fallback_chain: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimpleModelOrchestrator:
    """
    간단한 Model Orchestrator 구현
    Import 의존성 없이 독립 실행 가능
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_performance = {}
        
        # 설정 값들
        self.confidence_thresholds = {
            PredictionConfidence.HIGH: 0.9,
            PredictionConfidence.MEDIUM: 0.7,
            PredictionConfidence.LOW: 0.5
        }
        
        print(f"🎭 Simple Model Orchestrator 초기화")
        print(f"   모델 디렉토리: {self.models_dir}")
        
        # 모델 로딩
        self._load_available_models()
    
    def _load_available_models(self):
        """사용 가능한 모델들 로딩"""
        if not self.models_dir.exists():
            print(f"❌ 모델 디렉토리가 없습니다: {self.models_dir}")
            return
        
        model_files = list(self.models_dir.glob("*.joblib"))
        print(f"📁 발견된 모델 파일: {len(model_files)}개")
        
        # Global 모델 로딩
        global_model_path = self.models_dir / "global_model.joblib"
        if global_model_path.exists():
            try:
                self.loaded_models['global'] = joblib.load(global_model_path)
                print(f"✅ Global 모델 로딩 성공")
            except Exception as e:
                print(f"❌ Global 모델 로딩 실패: {e}")
        
        # Regional 모델들 로딩
        regional_files = list(self.models_dir.glob("regional_*.joblib"))
        regional_count = 0
        for regional_file in regional_files:
            try:
                model = joblib.load(regional_file)
                region_id = regional_file.stem.split('_')[2]
                self.loaded_models[f'regional_{region_id}'] = model
                regional_count += 1
            except Exception as e:
                print(f"⚠️ Regional 모델 로딩 실패 {regional_file.name}: {e}")
        
        if regional_count > 0:
            print(f"✅ Regional 모델 {regional_count}개 로딩 성공")
        
        # Local 모델들 로딩 (처음 10개만)
        local_files = list(self.models_dir.glob("local_*.joblib"))
        local_count = 0
        for local_file in local_files[:10]:  # 너무 많으면 처음 10개만
            try:
                model = joblib.load(local_file)
                parts = local_file.stem.split('_')
                region_id = parts[2]
                category_id = parts[3]
                self.loaded_models[f'local_{region_id}_{category_id}'] = model
                local_count += 1
            except Exception as e:
                print(f"⚠️ Local 모델 로딩 실패 {local_file.name}: {e}")
        
        if local_count > 0:
            print(f"✅ Local 모델 {local_count}개 로딩 성공")
        
        print(f"📊 총 로딩된 모델: {len(self.loaded_models)}개")
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        메인 예측 메소드
        """
        start_time = datetime.now()
        
        result = PredictionResult(
            predictions=pd.DataFrame(),
            model_used="none",
            model_level=ModelLevel.GLOBAL,
            confidence_score=0.0,
            prediction_timestamp=start_time.isoformat(),
            processing_time_seconds=0.0
        )
        
        try:
            # 모델 선택 전략 결정
            strategy = self._determine_model_strategy(request)
            result.fallback_chain.append(f"전략: {' → '.join(strategy)}")
            
            # 예측 실행
            prediction_result = self._execute_prediction_with_fallback(request, strategy)
            
            if prediction_result:
                result.predictions = prediction_result['predictions']
                result.model_used = prediction_result['model_used']
                result.model_level = self._get_model_level(prediction_result['model_used'])
                result.confidence_score = prediction_result.get('confidence', 0.7)
                result.metadata = prediction_result.get('metadata', {})
                result.fallback_chain.extend(prediction_result.get('fallback_chain', []))
            else:
                result.warnings.append("모든 예측 방법이 실패했습니다.")
        
        except Exception as e:
            result.warnings.append(f"예측 오류: {str(e)}")
        
        # 처리 시간 계산
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _determine_model_strategy(self, request: PredictionRequest) -> List[str]:
        """모델 선택 전략 결정"""
        strategy = []
        
        # Local 모델 시도
        if request.region_id and request.business_category:
            local_key = f"local_{request.region_id}_{request.business_category}"
            if local_key in self.loaded_models:
                strategy.append(local_key)
        
        # Regional 모델 시도
        if request.region_id:
            regional_key = f"regional_{request.region_id}"
            if regional_key in self.loaded_models:
                strategy.append(regional_key)
        
        # Global 모델 (항상 사용 가능)
        if 'global' in self.loaded_models:
            strategy.append('global')
        
        return strategy
    
    def _execute_prediction_with_fallback(self, request: PredictionRequest, 
                                        strategy: List[str]) -> Optional[Dict[str, Any]]:
        """폴백을 사용한 예측 실행"""
        fallback_chain = []
        
        for model_key in strategy:
            try:
                model = self.loaded_models[model_key]
                
                # 더미 데이터로 예측 (실제로는 request.historical_data 사용)
                sample_data = self._create_sample_features(request)
                
                predictions = model.predict(sample_data)
                confidence = self._calculate_confidence(predictions, model_key)
                
                # 예측 결과 구성
                result = {
                    'predictions': pd.DataFrame({
                        'prediction': predictions,
                        'timestamp': pd.date_range('2024-01-01', periods=len(predictions), freq='M')
                    }),
                    'model_used': model_key,
                    'confidence': confidence,
                    'fallback_chain': fallback_chain + [f"성공: {model_key}"],
                    'metadata': {
                        'prediction_count': len(predictions),
                        'model_type': model_key.split('_')[0]
                    }
                }
                
                return result
                
            except Exception as e:
                fallback_chain.append(f"실패 {model_key}: {str(e)}")
                continue
        
        return None
    
    def _create_sample_features(self, request: PredictionRequest) -> np.ndarray:
        """샘플 feature 데이터 생성 (실제로는 historical_data에서 추출)"""
        # 모델이 기대하는 feature 수를 맞춰야 함
        # 여기서는 간단히 더미 데이터 생성
        
        # 예: [매출평균, 매출트렌드, 변동성, 계절성, 업종평균, 지역지수]
        sample_features = [
            [10000000, 100000, 0.1, 1.0, 9500000, 1.1],  # 비즈니스 1
            [12000000, -50000, 0.15, 1.2, 11000000, 1.0],  # 비즈니스 2  
            [8000000, 200000, 0.08, 0.9, 8500000, 1.15]   # 비즈니스 3
        ]
        
        return np.array(sample_features)
    
    def _calculate_confidence(self, predictions: np.ndarray, model_key: str) -> float:
        """예측 신뢰도 계산"""
        # 간단한 신뢰도 계산
        base_confidence = 0.6  # 기본 신뢰도
        
        # 모델 타입별 보정
        if model_key.startswith('local'):
            base_confidence = 0.9
        elif model_key.startswith('regional'):
            base_confidence = 0.8
        elif model_key.startswith('global'):
            base_confidence = 0.7
        
        # 예측 일관성 검사
        if len(predictions) > 1:
            cv = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 1.0
            consistency_bonus = max(0, 0.2 * (1 - cv))
            base_confidence += consistency_bonus
        
        return min(1.0, base_confidence)
    
    def _get_model_level(self, model_used: str) -> ModelLevel:
        """모델 레벨 결정"""
        if model_used.startswith('local'):
            return ModelLevel.LOCAL
        elif model_used.startswith('regional'):
            return ModelLevel.REGIONAL
        else:
            return ModelLevel.GLOBAL
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 리포트"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_count': len(self.loaded_models),
            'global_available': 'global' in self.loaded_models,
            'regional_count': len([k for k in self.loaded_models if k.startswith('regional')]),
            'local_count': len([k for k in self.loaded_models if k.startswith('local')])
        }

def demo_orchestrator():
    """오케스트레이터 데모"""
    print("🎭 Model Orchestrator 데모 실행")
    print("=" * 50)
    
    # 오케스트레이터 초기화
    orchestrator = SimpleModelOrchestrator()
    
    # 시스템 상태 확인
    status = orchestrator.get_system_status()
    print(f"\n📊 시스템 상태:")
    print(f"   로딩된 모델 수: {status['model_count']}개")
    print(f"   Global 모델: {'✅' if status['global_available'] else '❌'}")
    print(f"   Regional 모델: {status['regional_count']}개")
    print(f"   Local 모델: {status['local_count']}개")
    
    if status['model_count'] == 0:
        print("❌ 사용 가능한 모델이 없습니다.")
        return
    
    # 샘플 예측 요청들
    sample_requests = [
        PredictionRequest(
            business_id="DEMO_001",
            region_id=11110515,
            business_category=4,
            historical_data=[],
            required_confidence=PredictionConfidence.MEDIUM
        ),
        PredictionRequest(
            business_id="DEMO_002", 
            region_id=11110530,
            business_category=8,
            historical_data=[],
            required_confidence=PredictionConfidence.HIGH
        ),
        PredictionRequest(
            business_id="DEMO_003",
            region_id=None,  # Global 모델 사용
            business_category=None,
            historical_data=[],
            required_confidence=PredictionConfidence.LOW
        )
    ]
    
    business_names = ["홍대 맛집", "종로 헤어살롱", "일반 비즈니스"]
    
    print(f"\n🎯 샘플 예측 실행:")
    print("-" * 50)
    
    for i, (request, business_name) in enumerate(zip(sample_requests, business_names)):
        print(f"\n🏪 {business_name} (ID: {request.business_id})")
        
        # 예측 실행
        result = orchestrator.predict(request)
        
        print(f"   사용된 모델: {result.model_used}")
        print(f"   모델 레벨: {result.model_level.value}")
        print(f"   신뢰도: {result.confidence_score:.2f}")
        print(f"   처리 시간: {result.processing_time_seconds:.3f}초")
        
        if not result.predictions.empty:
            avg_prediction = result.predictions['prediction'].mean()
            print(f"   평균 예측값: {avg_prediction:,.0f}원")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"   ⚠️ {warning}")
        
        if result.fallback_chain:
            print(f"   폴백 체인: {' → '.join(result.fallback_chain)}")

def main():
    """메인 함수"""
    print("🚀 Standalone Model Orchestrator")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_orchestrator()
        
        print(f"\n🎉 데모 완료!")
        print(f"\n💡 다른 테스트 스크립트들:")
        print(f"   • python simple_test.py - 기본 모델 테스트")
        print(f"   • python quick_demo.py - 비즈니스 위험도 데모") 
        print(f"   • python test_orchestrator.py - 오케스트레이터 상세 테스트")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()