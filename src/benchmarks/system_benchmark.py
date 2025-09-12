#!/usr/bin/env python3
"""
서울 시장 위험도 ML 시스템 - 최종 시스템 벤치마크
Seoul Market Risk ML System - Final System Benchmark

전체 시스템의 성능, 안정성, 배포 준비 상태를 종합적으로 검증합니다.
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
import psutil
import gc

# System imports
from src.utils.config_loader import load_config, get_data_paths

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemBenchmark:
    """시스템 종합 벤치마크 클래스"""
    
    def __init__(self):
        self.config = load_config()
        self.data_paths = get_data_paths(self.config)
        self.start_time = datetime.now()
        
        self.benchmark_results = {
            'timestamp': self.start_time,
            'system_info': {},
            'data_validation': {},
            'model_inventory': {},
            'performance_benchmarks': {},
            'resource_usage': {},
            'deployment_readiness': {},
            'final_recommendations': []
        }
        
    def collect_system_info(self):
        """시스템 정보 수집"""
        logger.info("💻 시스템 정보 수집 중...")
        
        try:
            system_info = {
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory': psutil.virtual_memory()._asdict(),
                'disk_usage': psutil.disk_usage('/')._asdict(),
                'platform': os.name,
                'working_directory': os.getcwd(),
                'project_structure': self._analyze_project_structure()
            }
            
            self.benchmark_results['system_info'] = system_info
            
            # 메모리 정보 로깅
            memory_gb = system_info['memory']['total'] / (1024**3)
            available_gb = system_info['memory']['available'] / (1024**3)
            
            logger.info(f"   CPU: {system_info['cpu_count']}코어")
            logger.info(f"   메모리: {memory_gb:.1f}GB 전체, {available_gb:.1f}GB 사용 가능")
            
        except Exception as e:
            logger.warning(f"시스템 정보 수집 실패: {e}")
    
    def _analyze_project_structure(self) -> Dict:
        """프로젝트 구조 분석"""
        structure = {}
        
        important_dirs = [
            'src', 'config', 'data/processed', 'models', 
            'test_outputs', 'logs'
        ]
        
        for dir_name in important_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                if dir_path.is_dir():
                    file_count = len(list(dir_path.rglob('*')))
                    structure[dir_name] = {'exists': True, 'file_count': file_count}
                else:
                    structure[dir_name] = {'exists': True, 'type': 'file'}
            else:
                structure[dir_name] = {'exists': False}
        
        return structure
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        logger.info("📊 데이터 무결성 검증 중...")
        
        validation_results = {
            'processed_data_exists': False,
            'data_quality_metrics': {},
            'data_consistency': {}
        }
        
        try:
            # 처리된 데이터 확인
            combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
            
            if combined_file.exists():
                validation_results['processed_data_exists'] = True
                
                # 데이터 품질 메트릭
                df = pd.read_csv(combined_file, nrows=10000)  # 샘플 확인
                
                validation_results['data_quality_metrics'] = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'missing_values_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'duplicate_rows': df.duplicated().sum(),
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns)
                }
                
                # 주요 컬럼 존재 확인
                required_columns = ['monthly_revenue', 'district_code', 'business_type_code']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                validation_results['data_consistency'] = {
                    'required_columns_present': len(missing_columns) == 0,
                    'missing_columns': missing_columns,
                    'revenue_stats': df['monthly_revenue'].describe().to_dict() if 'monthly_revenue' in df.columns else None
                }
                
                logger.info(f"   ✅ 데이터 파일 존재: {len(df):,} 행")
                logger.info(f"   결측값 비율: {validation_results['data_quality_metrics']['missing_values_pct']:.2f}%")
                
            else:
                logger.warning("   ❌ 처리된 데이터 파일이 없음")
                
        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            validation_results['error'] = str(e)
        
        self.benchmark_results['data_validation'] = validation_results
    
    def inventory_models(self):
        """모델 인벤토리 및 상태 확인"""
        logger.info("🤖 모델 인벤토리 확인 중...")
        
        models_dir = Path("models")
        inventory = {
            'models_directory_exists': models_dir.exists(),
            'global_models': [],
            'regional_models': [],
            'local_models': [],
            'total_model_count': 0,
            'total_model_size_mb': 0
        }
        
        if models_dir.exists():
            # 모델 파일들 탐색
            for model_file in models_dir.glob('*.joblib'):
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                
                model_info = {
                    'filename': model_file.name,
                    'size_mb': round(file_size_mb, 2),
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
                
                if 'global' in model_file.name:
                    inventory['global_models'].append(model_info)
                elif 'regional' in model_file.name:
                    inventory['regional_models'].append(model_info)
                elif 'local' in model_file.name:
                    inventory['local_models'].append(model_info)
                
                inventory['total_model_size_mb'] += file_size_mb
                inventory['total_model_count'] += 1
            
            logger.info(f"   글로벌 모델: {len(inventory['global_models'])}개")
            logger.info(f"   지역 모델: {len(inventory['regional_models'])}개")
            logger.info(f"   로컬 모델: {len(inventory['local_models'])}개")
            logger.info(f"   총 모델 크기: {inventory['total_model_size_mb']:.1f}MB")
            
        else:
            logger.warning("   ❌ 모델 디렉토리가 없음")
        
        self.benchmark_results['model_inventory'] = inventory
    
    def run_performance_benchmarks(self):
        """성능 벤치마크 실행"""
        logger.info("⚡ 성능 벤치마크 실행 중...")
        
        benchmarks = {
            'data_loading_speed': {},
            'model_loading_speed': {},
            'prediction_throughput': {},
            'memory_efficiency': {}
        }
        
        try:
            # 데이터 로딩 속도 테스트
            logger.info("   데이터 로딩 속도 측정...")
            start_time = time.time()
            
            combined_file = self.data_paths['processed'] / 'seoul_sales_combined.csv'
            if combined_file.exists():
                df = pd.read_csv(combined_file, nrows=50000)  # 5만 행 테스트
                loading_time = time.time() - start_time
                
                benchmarks['data_loading_speed'] = {
                    'rows_loaded': len(df),
                    'loading_time_seconds': loading_time,
                    'rows_per_second': len(df) / loading_time if loading_time > 0 else 0
                }
                
            # 모델 로딩 속도 테스트 (샘플)
            logger.info("   모델 로딩 속도 측정...")
            models_dir = Path("models")
            
            if models_dir.exists():
                model_files = list(models_dir.glob('*.joblib'))[:3]  # 샘플 3개
                loading_times = []
                
                for model_file in model_files:
                    start_time = time.time()
                    try:
                        import joblib
                        _ = joblib.load(model_file)
                        loading_time = time.time() - start_time
                        loading_times.append(loading_time)
                    except:
                        pass
                
                if loading_times:
                    benchmarks['model_loading_speed'] = {
                        'models_tested': len(loading_times),
                        'avg_loading_time': np.mean(loading_times),
                        'total_time': sum(loading_times)
                    }
            
            # 메모리 사용량 모니터링
            process = psutil.Process()
            memory_info = process.memory_info()
            
            benchmarks['memory_efficiency'] = {
                'current_memory_mb': memory_info.rss / (1024 * 1024),
                'peak_memory_mb': memory_info.vms / (1024 * 1024),
                'memory_percent': process.memory_percent()
            }
            
            logger.info(f"   현재 메모리 사용량: {benchmarks['memory_efficiency']['current_memory_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"성능 벤치마크 실패: {e}")
            benchmarks['error'] = str(e)
        
        self.benchmark_results['performance_benchmarks'] = benchmarks
    
    def assess_deployment_readiness(self):
        """배포 준비 상태 평가"""
        logger.info("🚀 배포 준비 상태 평가 중...")
        
        readiness = {
            'configuration_status': {},
            'required_files_check': {},
            'dependencies_check': {},
            'security_check': {},
            'scalability_assessment': {},
            'overall_readiness_score': 0
        }
        
        score = 0
        max_score = 100
        
        # 1. 설정 파일 상태 (20점)
        config_file = Path("config/config.yaml")
        if config_file.exists():
            readiness['configuration_status'] = {
                'config_file_exists': True,
                'config_valid': True  # 이미 로드되었으므로 유효
            }
            score += 20
        
        # 2. 필수 파일 확인 (25점)
        essential_files = [
            'src/models/global_model.py',
            'src/risk_scoring/risk_calculator.py',
            'src/loan_calculation/loan_calculator.py',
            'models/global_model.joblib'
        ]
        
        existing_files = [f for f in essential_files if Path(f).exists()]
        file_score = (len(existing_files) / len(essential_files)) * 25
        score += file_score
        
        readiness['required_files_check'] = {
            'required_files': essential_files,
            'existing_files': existing_files,
            'completion_rate': len(existing_files) / len(essential_files)
        }
        
        # 3. 의존성 확인 (20점)
        try:
            import pandas, numpy, sklearn, joblib
            dependencies_ok = True
            score += 20
        except ImportError as e:
            dependencies_ok = False
            readiness['dependencies_check']['missing'] = str(e)
        
        readiness['dependencies_check'] = {
            'core_dependencies_available': dependencies_ok,
            'import_test_passed': dependencies_ok
        }
        
        # 4. 보안 체크 (15점) - 기본적인 확인
        secret_files = ['.env', 'secrets.yaml', 'api_keys.txt']
        exposed_secrets = [f for f in secret_files if Path(f).exists()]
        
        if not exposed_secrets:
            score += 15
            readiness['security_check'] = {
                'no_exposed_secrets': True,
                'basic_security_ok': True
            }
        else:
            readiness['security_check'] = {
                'no_exposed_secrets': False,
                'exposed_files': exposed_secrets
            }
        
        # 5. 확장성 평가 (20점)
        models_count = self.benchmark_results.get('model_inventory', {}).get('total_model_count', 0)
        data_size = self.benchmark_results.get('data_validation', {}).get('data_quality_metrics', {}).get('total_rows', 0)
        
        scalability_score = 0
        if models_count > 20:
            scalability_score += 10
        if data_size > 100000:
            scalability_score += 10
        
        score += scalability_score
        
        readiness['scalability_assessment'] = {
            'model_count': models_count,
            'data_scale': data_size,
            'scalability_ready': scalability_score >= 15
        }
        
        readiness['overall_readiness_score'] = score
        
        # 배포 권장사항
        if score >= 90:
            readiness_level = "🟢 배포 준비 완료"
        elif score >= 70:
            readiness_level = "🟡 배포 가능 (일부 개선 권장)"
        else:
            readiness_level = "🔴 배포 전 개선 필요"
        
        readiness['deployment_recommendation'] = readiness_level
        
        logger.info(f"   배포 준비 점수: {score}/100점")
        logger.info(f"   상태: {readiness_level}")
        
        self.benchmark_results['deployment_readiness'] = readiness
    
    def generate_final_recommendations(self):
        """최종 권장사항 생성"""
        logger.info("💡 최종 권장사항 생성 중...")
        
        recommendations = []
        
        # 배포 준비도 기반 권장사항
        deployment_score = self.benchmark_results.get('deployment_readiness', {}).get('overall_readiness_score', 0)
        
        if deployment_score >= 90:
            recommendations.append("🎉 시스템이 프로덕션 배포 준비가 완료되었습니다!")
        elif deployment_score >= 70:
            recommendations.append("⚠️  일부 개선 후 배포를 권장합니다.")
        else:
            recommendations.append("🔧 배포 전 주요 이슈들을 해결해야 합니다.")
        
        # 성능 기반 권장사항
        memory_usage = self.benchmark_results.get('performance_benchmarks', {}).get('memory_efficiency', {}).get('current_memory_mb', 0)
        
        if memory_usage > 1000:  # 1GB 초과
            recommendations.append(f"💾 메모리 사용량이 높습니다 ({memory_usage:.0f}MB). 메모리 최적화를 고려하세요.")
        
        # 모델 관련 권장사항
        model_count = self.benchmark_results.get('model_inventory', {}).get('total_model_count', 0)
        
        if model_count > 0:
            recommendations.append(f"✅ {model_count}개 모델이 성공적으로 준비되었습니다.")
            
            if model_count < 20:
                recommendations.append("📈 더 많은 로컬 모델을 생성하여 정확도를 향상시킬 수 있습니다.")
        
        # 데이터 관련 권장사항
        data_validation = self.benchmark_results.get('data_validation', {})
        if data_validation.get('processed_data_exists', False):
            missing_pct = data_validation.get('data_quality_metrics', {}).get('missing_values_pct', 0)
            
            if missing_pct < 5:
                recommendations.append("✅ 데이터 품질이 우수합니다.")
            elif missing_pct < 10:
                recommendations.append("⚠️  일부 데이터 결측값이 있습니다. 정기적인 데이터 점검을 권장합니다.")
            else:
                recommendations.append("🔧 데이터 결측값 비율이 높습니다. 데이터 정제를 강화하세요.")
        
        # 일반적인 운영 권장사항
        recommendations.extend([
            "🔄 정기적인 모델 재훈련 스케줄을 수립하세요.",
            "📊 실시간 모니터링 시스템을 구축하여 모델 성능을 추적하세요.",
            "💾 중요 데이터와 모델의 백업 전략을 수립하세요.",
            "📈 사용량 증가에 대비한 확장 계획을 준비하세요."
        ])
        
        self.benchmark_results['final_recommendations'] = recommendations
        return recommendations
    
    def generate_comprehensive_report(self):
        """종합 벤치마크 보고서 생성"""
        logger.info("📋 종합 벤치마크 보고서 생성 중...")
        
        end_time = datetime.now()
        total_benchmark_time = (end_time - self.start_time).total_seconds()
        
        # 시스템 정보
        system_info = self.benchmark_results.get('system_info', {})
        memory_gb = system_info.get('memory', {}).get('total', 0) / (1024**3)
        
        # 배포 준비도
        deployment = self.benchmark_results.get('deployment_readiness', {})
        readiness_score = deployment.get('overall_readiness_score', 0)
        readiness_status = deployment.get('deployment_recommendation', 'Unknown')
        
        # 모델 인벤토리
        models = self.benchmark_results.get('model_inventory', {})
        total_models = models.get('total_model_count', 0)
        total_size_mb = models.get('total_model_size_mb', 0)
        
        # 성능 메트릭
        performance = self.benchmark_results.get('performance_benchmarks', {})
        current_memory = performance.get('memory_efficiency', {}).get('current_memory_mb', 0)
        
        report = f"""
🏢 서울 시장 위험도 ML 시스템 - 최종 벤치마크 보고서
{'='*70}

📊 시스템 개요:
  • 벤치마크 완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
  • 총 소요 시간: {total_benchmark_time:.1f}초
  • 시스템 메모리: {memory_gb:.1f}GB
  • 현재 메모리 사용량: {current_memory:.1f}MB

🤖 모델 현황:
  • 총 모델 개수: {total_models}개
  • 총 모델 크기: {total_size_mb:.1f}MB
  • 글로벌 모델: {len(models.get('global_models', []))}개
  • 지역 모델: {len(models.get('regional_models', []))}개  
  • 로컬 모델: {len(models.get('local_models', []))}개

🚀 배포 준비도:
  • 배포 준비 점수: {readiness_score}/100점
  • 배포 상태: {readiness_status}
  
📈 데이터 상태:
"""
        
        # 데이터 검증 결과
        data_validation = self.benchmark_results.get('data_validation', {})
        if data_validation.get('processed_data_exists', False):
            quality_metrics = data_validation.get('data_quality_metrics', {})
            total_rows = quality_metrics.get('total_rows', 0)
            missing_pct = quality_metrics.get('missing_values_pct', 0)
            
            report += f"""  • 처리된 데이터: ✅ {total_rows:,} 행
  • 데이터 품질: 결측값 {missing_pct:.2f}%
  • 필수 컬럼: {'✅ 모두 존재' if data_validation.get('data_consistency', {}).get('required_columns_present', False) else '❌ 일부 누락'}
"""
        else:
            report += "  • 처리된 데이터: ❌ 없음\n"
        
        # 최종 권장사항
        recommendations = self.benchmark_results.get('final_recommendations', [])
        report += "\n💡 최종 권장사항:\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        # 다음 단계
        report += f"""
🎯 다음 단계:
  1. 실제 비즈니스 데이터로 시스템 테스트
  2. API 엔드포인트 구축 및 통합 테스트  
  3. 사용자 인터페이스 개발
  4. 운영 모니터링 시스템 구축
  5. 프로덕션 환경 배포

⏰ 벤치마크 완료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 보고서 저장
        report_path = Path("final_system_benchmark_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 상세 결과 JSON 저장
        detailed_results = self.benchmark_results.copy()
        detailed_results['timestamp'] = self.benchmark_results['timestamp'].isoformat()
        detailed_results['completion_time'] = end_time.isoformat()
        detailed_results['total_benchmark_time'] = total_benchmark_time
        
        json_path = Path("detailed_system_benchmark.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def run_full_benchmark(self):
        """전체 벤치마크 실행"""
        logger.info("🚀 서울 시장 위험도 ML 시스템 최종 벤치마크 시작")
        logger.info("="*70)
        
        try:
            # 1. 시스템 정보 수집
            self.collect_system_info()
            
            # 2. 데이터 무결성 검증
            self.validate_data_integrity()
            
            # 3. 모델 인벤토리 확인
            self.inventory_models()
            
            # 4. 성능 벤치마크 실행
            self.run_performance_benchmarks()
            
            # 5. 배포 준비 상태 평가
            self.assess_deployment_readiness()
            
            # 6. 최종 권장사항 생성
            recommendations = self.generate_final_recommendations()
            
            # 7. 종합 보고서 생성
            report = self.generate_comprehensive_report()
            logger.info("\n" + report)
            
            return {
                'benchmark_results': self.benchmark_results,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"💥 벤치마크 중 오류 발생: {e}")
            raise


def main():
    """메인 함수"""
    benchmark = SystemBenchmark()
    results = benchmark.run_full_benchmark()
    
    deployment_score = results['benchmark_results'].get('deployment_readiness', {}).get('overall_readiness_score', 0)
    
    print(f"\n🎯 최종 벤치마크 완료!")
    print(f"  배포 준비 점수: {deployment_score}/100점")
    print(f"  권장사항: {len(results['recommendations'])}개")
    print(f"  상세 보고서: final_system_benchmark_report.txt")
    
    return 0


if __name__ == "__main__":
    exit(main())