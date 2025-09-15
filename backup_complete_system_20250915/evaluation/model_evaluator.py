#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 모델 성능 평가
Seoul Market Risk ML System - Model Performance Evaluation

훈련된 모델들의 성능을 종합적으로 평가하고 분석합니다.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

# System imports
from src.utils.config_loader import load_config, get_data_paths

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeoulModelEvaluator:
    """서울 시장 위험도 ML 모델 성능 평가 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.data_paths = get_data_paths(self.config)
        self.models_dir = Path("models")
        self.evaluation_results = {
            'timestamp': datetime.now(),
            'model_performance': {},
            'comparative_analysis': {},
            'recommendations': []
        }
        
    def load_trained_models(self) -> Dict:
        """훈련된 모델들 로드"""
        logger.info("📚 훈련된 모델들 로딩 중...")
        
        models = {}
        
        # 글로벌 모델
        global_path = self.models_dir / 'global_model.joblib'
        if global_path.exists():
            models['global'] = joblib.load(global_path)
            logger.info("   ✅ 글로벌 모델 로드 완료")
        
        # 지역 모델들
        regional_models = {}
        for model_file in self.models_dir.glob('regional_model_*.joblib'):
            region_id = model_file.stem.split('_')[-1]
            regional_models[f'region_{region_id}'] = joblib.load(model_file)
        
        if regional_models:
            models['regional'] = regional_models
            logger.info(f"   ✅ 지역 모델 {len(regional_models)}개 로드 완료")
        
        # 로컬 모델들
        local_models = {}
        for model_file in self.models_dir.glob('local_model_*.joblib'):
            parts = model_file.stem.split('_')
            region_id = parts[2]
            business_id = parts[3]
            local_models[f'local_{region_id}_{business_id}'] = joblib.load(model_file)
        
        if local_models:
            models['local'] = local_models
            logger.info(f"   ✅ 로컬 모델 {len(local_models)}개 로드 완료")
        
        total_models = 1 if 'global' in models else 0
        total_models += len(regional_models)
        total_models += len(local_models)
        
        logger.info(f"총 {total_models}개 모델 로드 완료")
        return models
    
    def prepare_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """테스트 데이터 준비"""
        logger.info("🧪 테스트 데이터 준비 중...")
        
        # 훈련에 사용된 동일한 전처리 적용
        combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
        df = pd.read_csv(combined_file)
        
        # 특성 선택 (훈련과 동일하게)
        feature_columns = [
            'district_code', 'business_type_code', 'quarter', 'year',
            'weekday_revenue', 'weekend_revenue',
            'male_revenue', 'female_revenue'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        target_col = 'monthly_revenue'
        
        # 결측치 처리
        for col in available_features:
            if df[col].dtype in ['object']:
                df[col] = df[col].fillna('unknown')
                df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = df[col].fillna(df[col].median())
        
        df[target_col] = df[target_col].fillna(df[target_col].median())
        
        # 이상치 제거
        target_q99 = df[target_col].quantile(0.99)
        target_q01 = df[target_col].quantile(0.01)
        df = df[(df[target_col] >= target_q01) & (df[target_col] <= target_q99)]
        
        # 테스트 샘플 추출 (전체 데이터의 10%)
        test_sample = df.sample(n=min(10000, len(df) // 10), random_state=42)
        
        X_test = test_sample[available_features]
        y_test = test_sample[target_col]
        
        logger.info(f"테스트 데이터 준비 완료: {len(X_test):,} 샘플")
        return X_test, y_test
    
    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                            model_name: str) -> Dict:
        """단일 모델 성능 평가"""
        try:
            # 예측 실행
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # 성능 지표 계산
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # MAPE 계산 (0 값 때문에 안전하게 처리)
            mask = y_test != 0
            mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) if mask.sum() > 0 else np.inf
            
            # 추가 통계
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            
            return {
                'model_name': model_name,
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'rmse': float(rmse),
                'mape': float(mape) if mape != np.inf else None,
                'residual_std': float(residual_std),
                'prediction_time': float(prediction_time),
                'predictions_per_second': float(len(X_test) / prediction_time) if prediction_time > 0 else 0,
                'sample_count': len(X_test)
            }
            
        except Exception as e:
            logger.warning(f"모델 {model_name} 평가 실패: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'success': False
            }
    
    def evaluate_all_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """모든 모델 성능 평가"""
        logger.info("📊 모든 모델 성능 평가 중...")
        
        results = {}
        
        # 글로벌 모델 평가
        if 'global' in models:
            logger.info("   글로벌 모델 평가 중...")
            global_result = self.evaluate_single_model(models['global'], X_test, y_test, 'global')
            results['global'] = global_result
        
        # 지역 모델 평가
        if 'regional' in models:
            logger.info(f"   지역 모델 {len(models['regional'])}개 평가 중...")
            regional_results = {}
            
            for region_key, model in models['regional'].items():
                region_id = region_key.split('_')[1]
                
                # 해당 지역 데이터만 필터링
                region_mask = X_test['district_code'] == int(region_id)
                if region_mask.sum() < 10:  # 최소 샘플 수 확인
                    continue
                
                X_region = X_test[region_mask]
                y_region = y_test[region_mask]
                
                result = self.evaluate_single_model(model, X_region, y_region, region_key)
                regional_results[region_key] = result
            
            results['regional'] = regional_results
        
        # 로컬 모델 평가 (샘플링하여 일부만)
        if 'local' in models:
            logger.info(f"   로컬 모델 {len(models['local'])}개 평가 중...")
            local_results = {}
            evaluated_count = 0
            max_local_evaluations = 10  # 시간 절약을 위해 일부만 평가
            
            for local_key, model in models['local'].items():
                if evaluated_count >= max_local_evaluations:
                    break
                
                parts = local_key.split('_')
                region_id = int(parts[1])
                business_id = int(parts[2])
                
                # 해당 지역+업종 데이터 필터링
                local_mask = (X_test['district_code'] == region_id) & (X_test['business_type_code'] == business_id)
                if local_mask.sum() < 5:  # 최소 샘플 수 확인
                    continue
                
                X_local = X_test[local_mask]
                y_local = y_test[local_mask]
                
                result = self.evaluate_single_model(model, X_local, y_local, local_key)
                local_results[local_key] = result
                evaluated_count += 1
            
            results['local'] = local_results
            logger.info(f"   로컬 모델 {evaluated_count}개 평가 완료")
        
        self.evaluation_results['model_performance'] = results
        return results
    
    def analyze_performance_patterns(self, results: Dict) -> Dict:
        """성능 패턴 분석"""
        logger.info("🔍 성능 패턴 분석 중...")
        
        analysis = {
            'model_tier_comparison': {},
            'performance_distribution': {},
            'speed_analysis': {},
            'best_worst_models': {}
        }
        
        # 모델 계층별 성능 비교
        tier_stats = {}
        
        for tier in ['global', 'regional', 'local']:
            if tier in results and results[tier]:
                if tier == 'global':
                    models_data = [results[tier]]
                else:
                    models_data = list(results[tier].values())
                
                # 성공한 모델들만 필터링
                successful_models = [m for m in models_data if 'error' not in m and 'r2' in m]
                
                if successful_models:
                    r2_scores = [m['r2'] for m in successful_models]
                    mae_scores = [m['mae'] for m in successful_models]
                    prediction_speeds = [m['predictions_per_second'] for m in successful_models]
                    
                    tier_stats[tier] = {
                        'model_count': len(successful_models),
                        'avg_r2': float(np.mean(r2_scores)),
                        'std_r2': float(np.std(r2_scores)),
                        'avg_mae': float(np.mean(mae_scores)),
                        'avg_speed': float(np.mean(prediction_speeds)),
                        'best_r2': float(max(r2_scores)),
                        'worst_r2': float(min(r2_scores))
                    }
        
        analysis['model_tier_comparison'] = tier_stats
        
        # 최고/최악 모델 찾기
        all_models = []
        for tier, tier_results in results.items():
            if tier == 'global':
                if 'r2' in tier_results:
                    all_models.append({**tier_results, 'tier': tier})
            else:
                for model_name, model_result in tier_results.items():
                    if 'r2' in model_result:
                        all_models.append({**model_result, 'tier': tier})
        
        if all_models:
            # R2 점수 기준 정렬
            all_models.sort(key=lambda x: x['r2'], reverse=True)
            
            analysis['best_worst_models'] = {
                'best_3': all_models[:3],
                'worst_3': all_models[-3:] if len(all_models) >= 3 else all_models
            }
        
        self.evaluation_results['comparative_analysis'] = analysis
        return analysis
    
    def generate_recommendations(self, results: Dict, analysis: Dict) -> List[str]:
        """성능 분석 기반 추천사항 생성"""
        logger.info("💡 추천사항 생성 중...")
        
        recommendations = []
        
        # 모델 계층별 성능 분석
        tier_stats = analysis.get('model_tier_comparison', {})
        
        if 'global' in tier_stats and 'regional' in tier_stats:
            global_r2 = tier_stats['global']['avg_r2']
            regional_r2 = tier_stats['regional']['avg_r2']
            
            if regional_r2 > global_r2:
                diff = regional_r2 - global_r2
                recommendations.append(f"✅ 지역 모델이 글로벌 모델보다 {diff:.3f}만큼 우수한 성능을 보입니다. 지역별 특성을 잘 반영하고 있습니다.")
            else:
                recommendations.append("⚠️  글로벌 모델이 지역 모델과 비슷하거나 더 나은 성능을 보입니다. 지역별 특성 반영을 개선해야 합니다.")
        
        if 'local' in tier_stats:
            local_r2_avg = tier_stats['local']['avg_r2']
            local_r2_std = tier_stats['local']['std_r2']
            
            if local_r2_std > 0.1:
                recommendations.append("⚠️  로컬 모델들 간 성능 편차가 큽니다. 데이터 부족한 조합들의 모델을 개선하거나 fallback 전략을 강화하세요.")
            
            if local_r2_avg < 0.8:
                recommendations.append("🔧 일부 로컬 모델의 성능이 낮습니다. 특성 엔지니어링이나 모델 복잡도를 조정해보세요.")
        
        # 속도 분석
        if 'global' in tier_stats:
            global_speed = tier_stats['global']['avg_speed']
            if global_speed < 1000:  # 초당 1000개 미만
                recommendations.append("⚡ 글로벌 모델의 예측 속도가 느립니다. 모델 최적화나 더 간단한 알고리즘 고려해보세요.")
        
        # 최고/최악 모델 분석
        best_worst = analysis.get('best_worst_models', {})
        if 'worst_3' in best_worst:
            worst_models = best_worst['worst_3']
            worst_r2_avg = np.mean([m['r2'] for m in worst_models])
            
            if worst_r2_avg < 0.7:
                recommendations.append(f"🚨 최하위 모델들의 평균 R2가 {worst_r2_avg:.3f}로 매우 낮습니다. 해당 모델들을 재훈련하거나 제거를 고려하세요.")
        
        # 전체적인 권장사항
        if len(recommendations) == 0:
            recommendations.append("🎉 모든 모델이 양호한 성능을 보이고 있습니다! 현재 설정을 유지하세요.")
        
        self.evaluation_results['recommendations'] = recommendations
        return recommendations
    
    def generate_evaluation_report(self, results: Dict, analysis: Dict, recommendations: List[str]) -> str:
        """종합 평가 보고서 생성"""
        logger.info("📋 종합 평가 보고서 생성 중...")
        
        # 전체 통계
        total_models = 0
        successful_evaluations = 0
        
        for tier, tier_results in results.items():
            if tier == 'global':
                if 'error' not in tier_results:
                    successful_evaluations += 1
                total_models += 1
            else:
                tier_total = len(tier_results)
                tier_success = sum(1 for r in tier_results.values() if 'error' not in r)
                total_models += tier_total
                successful_evaluations += tier_success
        
        success_rate = (successful_evaluations / total_models * 100) if total_models > 0 else 0
        
        report = f"""
🏢 서울 시장 위험도 ML 시스템 - 모델 성능 평가 보고서
{'='*65}

📊 평가 결과 요약:
  • 평가 대상 모델: {total_models}개
  • 성공적 평가: {successful_evaluations}개
  • 평가 성공률: {success_rate:.1f}%
  • 평가 완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 모델 계층별 성능:
"""
        
        tier_stats = analysis.get('model_tier_comparison', {})
        for tier, stats in tier_stats.items():
            tier_name = {'global': '글로벌', 'regional': '지역', 'local': '로컬'}[tier]
            report += f"""  📍 {tier_name} 모델:
    - 모델 수: {stats['model_count']}개
    - 평균 R2: {stats['avg_r2']:.3f} (±{stats['std_r2']:.3f})
    - 평균 MAE: {stats['avg_mae']:,.0f}
    - 평균 속도: {stats['avg_speed']:,.0f} 예측/초
    - 최고 성능: R2 {stats['best_r2']:.3f}
    - 최저 성능: R2 {stats['worst_r2']:.3f}

"""
        
        # 최고/최악 모델
        best_worst = analysis.get('best_worst_models', {})
        if 'best_3' in best_worst:
            report += "🏆 최고 성능 모델 TOP 3:\n"
            for i, model in enumerate(best_worst['best_3'], 1):
                tier_name = {'global': '글로벌', 'regional': '지역', 'local': '로컬'}[model['tier']]
                report += f"  {i}. {model['model_name']} ({tier_name}) - R2: {model['r2']:.3f}\n"
            report += "\n"
        
        if 'worst_3' in best_worst:
            report += "⚠️  개선 필요 모델:\n"
            for i, model in enumerate(best_worst['worst_3'], 1):
                tier_name = {'global': '글로벌', 'regional': '지역', 'local': '로컬'}[model['tier']]
                report += f"  {i}. {model['model_name']} ({tier_name}) - R2: {model['r2']:.3f}\n"
            report += "\n"
        
        # 추천사항
        report += "💡 추천사항:\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        # 보고서 파일 저장
        report_path = Path("model_evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 상세 결과 JSON 저장
        detailed_results = {
            'timestamp': self.evaluation_results['timestamp'].isoformat(),
            'model_performance': results,
            'comparative_analysis': analysis,
            'recommendations': recommendations,
            'summary': {
                'total_models': total_models,
                'successful_evaluations': successful_evaluations,
                'success_rate': success_rate
            }
        }
        
        json_path = Path("detailed_evaluation_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def run_full_evaluation(self):
        """전체 평가 파이프라인 실행"""
        logger.info("🚀 모델 성능 평가 시작")
        logger.info("="*65)
        
        try:
            # 1. 모델 로드
            models = self.load_trained_models()
            
            if not models:
                raise ValueError("평가할 모델이 없습니다.")
            
            # 2. 테스트 데이터 준비
            X_test, y_test = self.prepare_test_data()
            
            # 3. 모든 모델 평가
            results = self.evaluate_all_models(models, X_test, y_test)
            
            # 4. 성능 패턴 분석
            analysis = self.analyze_performance_patterns(results)
            
            # 5. 추천사항 생성
            recommendations = self.generate_recommendations(results, analysis)
            
            # 6. 종합 보고서 생성
            report = self.generate_evaluation_report(results, analysis, recommendations)
            logger.info("\n" + report)
            
            return {
                'results': results,
                'analysis': analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"💥 모델 평가 중 오류 발생: {e}")
            raise


def main():
    """메인 함수"""
    evaluator = SeoulModelEvaluator()
    evaluation = evaluator.run_full_evaluation()
    
    print(f"\n🎯 평가 완료!")
    print(f"  분석된 모델: {len(evaluation['results'])} 계층")
    print(f"  추천사항: {len(evaluation['recommendations'])}개")
    print(f"  상세 결과: detailed_evaluation_results.json")
    
    return 0


if __name__ == "__main__":
    exit(main())