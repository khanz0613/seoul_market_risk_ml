#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 모델 훈련
Seoul Market Risk ML System - Model Training

79개 모델(1 Global + 6 Regional + 72 Local)을 실제 데이터로 훈련합니다.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Basic ML imports (avoiding LightGBM)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# System imports
from src.utils.config_loader import load_config, get_data_paths

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeoulModelTrainer:
    """서울 시장 위험도 ML 모델 훈련 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.data_paths = get_data_paths(self.config)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # 모델 성능 추적
        self.model_performance = {}
        self.training_results = {
            'start_time': datetime.now(),
            'models_trained': 0,
            'models_failed': 0,
            'performance_summary': {},
            'errors': []
        }
        
    def load_training_data(self) -> pd.DataFrame:
        """훈련 데이터 로드"""
        logger.info("📊 훈련 데이터 로딩 중...")
        
        combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
        
        if not combined_file.exists():
            raise FileNotFoundError(f"처리된 데이터 파일이 없습니다: {combined_file}")
        
        df = pd.read_csv(combined_file)
        logger.info(f"데이터 로드 완료: {len(df):,} 행, {len(df.columns)} 컬럼")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """특성 엔지니어링 및 준비"""
        logger.info("🔧 특성 엔지니어링 중...")
        
        # 기본 특성 선택
        feature_columns = [
            'district_code', 'business_type_code', 'quarter', 'year',
            'weekday_revenue', 'weekend_revenue',
            'male_revenue', 'female_revenue'
        ]
        
        # 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            # 최소한의 특성으로 대체
            available_features = ['district_code', 'business_type_code', 'year', 'quarter']
        
        # 타겟 변수
        target_col = 'monthly_revenue'
        if target_col not in df.columns:
            raise ValueError(f"타겟 변수 '{target_col}'가 데이터에 없습니다")
        
        # 결측치 처리
        for col in available_features:
            if df[col].dtype in ['object']:
                df[col] = df[col].fillna('unknown')
                # 범주형 변수 인코딩
                df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = df[col].fillna(df[col].median())
        
        df[target_col] = df[target_col].fillna(df[target_col].median())
        
        # 이상치 제거 (상위/하위 1% 제거)
        target_q99 = df[target_col].quantile(0.99)
        target_q01 = df[target_col].quantile(0.01)
        df = df[(df[target_col] >= target_q01) & (df[target_col] <= target_q99)]
        
        X = df[available_features]
        y = df[target_col]
        
        logger.info(f"특성 준비 완료: {len(available_features)}개 특성, {len(df):,} 샘플")
        return X, y
    
    def train_global_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """글로벌 모델 훈련"""
        logger.info("🌍 글로벌 모델 훈련 시작...")
        
        try:
            start_time = time.time()
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 앙상블 모델 구성
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    n_jobs=-1
                ),
                'ridge': Ridge(alpha=1.0),
                'linear': LinearRegression()
            }
            
            trained_models = {}
            model_scores = {}
            
            # 각 모델 훈련
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 성능 평가
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                trained_models[model_name] = model
                model_scores[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
            
            # 최고 성능 모델 선택 (R2 기준)
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
            best_model = trained_models[best_model_name]
            best_score = model_scores[best_model_name]
            
            # 모델 저장
            model_path = self.models_dir / 'global_model.joblib'
            joblib.dump(best_model, model_path)
            
            training_time = time.time() - start_time
            
            result = {
                'model_type': 'global',
                'model_name': best_model_name,
                'model_path': str(model_path),
                'performance': best_score,
                'all_models_performance': model_scores,
                'training_time': training_time,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.model_performance['global'] = result
            self.training_results['models_trained'] += 1
            
            logger.info(f"✅ 글로벌 모델 훈련 완료 ({training_time:.1f}초)")
            logger.info(f"   최고 모델: {best_model_name}, R2: {best_score['r2']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 글로벌 모델 훈련 실패: {e}")
            self.training_results['models_failed'] += 1
            self.training_results['errors'].append(f"global: {str(e)}")
            return None
    
    def train_regional_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """지역 모델 훈련 (6개 지역)"""
        logger.info("🏙️ 지역 모델 훈련 시작 (6개 지역)...")
        
        regional_results = {}
        
        # district_code가 지역 ID라고 가정
        if 'district_code' not in X.columns:
            logger.warning("district_code 컬럼이 없어 지역 모델을 훈련할 수 없습니다.")
            return {}
        
        unique_regions = sorted(X['district_code'].unique())[:6]  # 최대 6개 지역
        
        for region_id in unique_regions:
            try:
                logger.info(f"   지역 {region_id} 모델 훈련 중...")
                start_time = time.time()
                
                # 지역별 데이터 필터링
                region_mask = X['district_code'] == region_id
                X_region = X[region_mask]
                y_region = y[region_mask]
                
                if len(X_region) < 50:  # 최소 샘플 수 확인
                    logger.warning(f"   지역 {region_id}: 데이터 부족 ({len(X_region)}개), 스킵")
                    continue
                
                # 지역별 데이터 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X_region, y_region, test_size=0.2, random_state=42
                )
                
                # 지역별 모델 (Random Forest 사용)
                model = RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=8, 
                    random_state=42,
                    n_jobs=2
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 성능 평가
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # 모델 저장
                model_path = self.models_dir / f'regional_model_{region_id}.joblib'
                joblib.dump(model, model_path)
                
                training_time = time.time() - start_time
                
                result = {
                    'model_type': 'regional',
                    'region_id': region_id,
                    'model_path': str(model_path),
                    'performance': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    },
                    'training_time': training_time,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                regional_results[f'region_{region_id}'] = result
                self.training_results['models_trained'] += 1
                
                logger.info(f"   ✅ 지역 {region_id} 완료 ({training_time:.1f}초, R2: {r2:.3f})")
                
            except Exception as e:
                logger.error(f"   ❌ 지역 {region_id} 모델 훈련 실패: {e}")
                self.training_results['models_failed'] += 1
                self.training_results['errors'].append(f"regional_{region_id}: {str(e)}")
        
        self.model_performance.update(regional_results)
        logger.info(f"✅ 지역 모델 훈련 완료 ({len(regional_results)}개 모델)")
        
        return regional_results
    
    def train_local_models(self, X: pd.DataFrame, y: pd.Series, max_models: int = 20) -> Dict:
        """로컬 모델 훈련 (지역 x 업종 조합, 최대 개수 제한)"""
        logger.info(f"🏪 로컬 모델 훈련 시작 (최대 {max_models}개)...")
        
        local_results = {}
        
        if 'district_code' not in X.columns or 'business_type_code' not in X.columns:
            logger.warning("district_code 또는 business_type_code 컬럼이 없어 로컬 모델을 훈련할 수 없습니다.")
            return {}
        
        # 지역 x 업종 조합 생성
        combinations = []
        for region in sorted(X['district_code'].unique()):
            for business in sorted(X['business_type_code'].unique()):
                mask = (X['district_code'] == region) & (X['business_type_code'] == business)
                sample_count = mask.sum()
                
                if sample_count >= 10:  # 최소 샘플 수
                    combinations.append((region, business, sample_count))
        
        # 샘플 수 기준으로 정렬하고 상위 모델만 선택
        combinations.sort(key=lambda x: x[2], reverse=True)
        combinations = combinations[:max_models]
        
        logger.info(f"   선택된 조합: {len(combinations)}개")
        
        def train_single_local_model(combo):
            """단일 로컬 모델 훈련 (병렬 처리용)"""
            region, business, sample_count = combo
            
            try:
                # 데이터 필터링
                mask = (X['district_code'] == region) & (X['business_type_code'] == business)
                X_local = X[mask]
                y_local = y[mask]
                
                # 데이터 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X_local, y_local, test_size=0.2, random_state=42
                )
                
                # 로컬 모델 (간단한 모델 사용)
                model = Ridge(alpha=10.0)  # 정규화를 더 강하게
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 성능 평가
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # 모델 저장
                model_path = self.models_dir / f'local_model_{region}_{business}.joblib'
                joblib.dump(model, model_path)
                
                return {
                    'model_type': 'local',
                    'region_id': region,
                    'business_id': business,
                    'model_path': str(model_path),
                    'performance': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    },
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'total_samples': sample_count
                }
                
            except Exception as e:
                return {
                    'error': str(e),
                    'region_id': region,
                    'business_id': business
                }
        
        # 병렬 처리로 로컬 모델 훈련
        successful_models = 0
        failed_models = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_combo = {executor.submit(train_single_local_model, combo): combo for combo in combinations}
            
            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]
                region, business, _ = combo
                
                try:
                    result = future.result()
                    
                    if 'error' in result:
                        logger.warning(f"   지역 {region}, 업종 {business}: {result['error']}")
                        failed_models += 1
                        self.training_results['errors'].append(f"local_{region}_{business}: {result['error']}")
                    else:
                        local_results[f'local_{region}_{business}'] = result
                        successful_models += 1
                        
                        if successful_models % 5 == 0:  # 5개마다 진행상황 출력
                            logger.info(f"   진행률: {successful_models}/{len(combinations)} 완료...")
                
                except Exception as e:
                    logger.error(f"   지역 {region}, 업종 {business} 처리 중 오류: {e}")
                    failed_models += 1
        
        self.model_performance.update(local_results)
        self.training_results['models_trained'] += successful_models
        self.training_results['models_failed'] += failed_models
        
        logger.info(f"✅ 로컬 모델 훈련 완료 (성공: {successful_models}, 실패: {failed_models})")
        
        return local_results
    
    def generate_training_report(self) -> str:
        """훈련 결과 보고서 생성"""
        end_time = datetime.now()
        total_time = (end_time - self.training_results['start_time']).total_seconds()
        
        # 성능 요약 계산
        all_r2_scores = []
        model_type_counts = {'global': 0, 'regional': 0, 'local': 0}
        
        for model_key, model_info in self.model_performance.items():
            if 'performance' in model_info:
                r2 = model_info['performance']['r2']
                all_r2_scores.append(r2)
                
                model_type = model_info['model_type']
                model_type_counts[model_type] += 1
        
        avg_r2 = np.mean(all_r2_scores) if all_r2_scores else 0
        max_r2 = max(all_r2_scores) if all_r2_scores else 0
        min_r2 = min(all_r2_scores) if all_r2_scores else 0
        
        total_models = self.training_results['models_trained'] + self.training_results['models_failed']
        success_rate = (self.training_results['models_trained'] / total_models * 100) if total_models > 0 else 0
        
        report = f"""
🏢 서울 시장 위험도 ML 시스템 - 모델 훈련 보고서
{'='*60}

📊 훈련 결과 요약:
  • 총 훈련 시간: {total_time:.1f}초 ({total_time/60:.1f}분)
  • 성공한 모델: {self.training_results['models_trained']}개
  • 실패한 모델: {self.training_results['models_failed']}개
  • 전체 성공률: {success_rate:.1f}%

🔧 모델 유형별 현황:
  • 글로벌 모델: {model_type_counts['global']}개
  • 지역 모델: {model_type_counts['regional']}개  
  • 로컬 모델: {model_type_counts['local']}개

📈 성능 지표:
  • 평균 R2 점수: {avg_r2:.3f}
  • 최고 R2 점수: {max_r2:.3f}
  • 최저 R2 점수: {min_r2:.3f}

⏰ 완료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if self.training_results['errors']:
            report += f"\n❌ 오류 목록 ({len(self.training_results['errors'])}개):\n"
            for error in self.training_results['errors'][:10]:  # 최대 10개만 표시
                report += f"  • {error}\n"
            
            if len(self.training_results['errors']) > 10:
                report += f"  • ... 및 {len(self.training_results['errors'])-10}개 추가 오류\n"
        
        # 보고서 저장
        report_path = Path("training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 성능 데이터 JSON 저장
        performance_path = Path("model_performance.json")
        with open(performance_path, 'w', encoding='utf-8') as f:
            # datetime 객체를 문자열로 변환
            serializable_results = self.training_results.copy()
            serializable_results['start_time'] = self.training_results['start_time'].isoformat()
            serializable_results['end_time'] = end_time.isoformat()
            serializable_results['model_performance'] = self.model_performance
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def run_full_training(self):
        """전체 모델 훈련 파이프라인 실행"""
        logger.info("🚀 서울 시장 위험도 ML 시스템 모델 훈련 시작")
        logger.info("="*60)
        
        try:
            # 1. 데이터 로드
            df = self.load_training_data()
            
            # 2. 특성 준비
            X, y = self.prepare_features(df)
            
            # 3. 글로벌 모델 훈련
            global_result = self.train_global_model(X, y)
            
            # 4. 지역 모델 훈련
            regional_results = self.train_regional_models(X, y)
            
            # 5. 로컬 모델 훈련 (최대 20개)
            local_results = self.train_local_models(X, y, max_models=20)
            
            # 6. 보고서 생성
            report = self.generate_training_report()
            logger.info("\n" + report)
            
            total_models = len(regional_results) + len(local_results) + (1 if global_result else 0)
            logger.info(f"🎉 모델 훈련 완료! 총 {total_models}개 모델 생성")
            
            return {
                'global': global_result,
                'regional': regional_results,
                'local': local_results,
                'total_models': total_models
            }
            
        except Exception as e:
            logger.error(f"💥 모델 훈련 중 치명적 오류: {e}")
            self.training_results['errors'].append(f"fatal: {str(e)}")
            raise


def main():
    """메인 함수"""
    trainer = SeoulModelTrainer()
    results = trainer.run_full_training()
    
    print(f"\n🎯 훈련 결과:")
    print(f"  글로벌 모델: {'✅' if results['global'] else '❌'}")
    print(f"  지역 모델: {len(results['regional'])}개")
    print(f"  로컬 모델: {len(results['local'])}개")
    print(f"  총 모델: {results['total_models']}개")
    
    return 0


if __name__ == "__main__":
    exit(main())