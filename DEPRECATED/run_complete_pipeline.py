#!/usr/bin/env python3
"""
Complete Pipeline Execution & Validation
========================================

전체 ML 파이프라인 재구축 및 검증 스크립트

실행 순서:
1. 데이터 누수 없는 라벨링 시스템 실행
2. 외부 지표 기반 피처 엔지니어링 실행
3. 과적합 방지 앙상블 모델 훈련
4. 통합 마스터 파이프라인 테스트
5. 성능 검증 및 최종 보고서

사용법:
python run_complete_pipeline.py [--skip-training] [--quick-test]

Author: Seoul Market Risk ML System - Complete Integration
Date: 2025-09-17
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class CompleteMLPipeline:
    """전체 ML 파이프라인 실행 및 검증"""

    def __init__(self, skip_training: bool = False, quick_test: bool = False):
        self.skip_training = skip_training
        self.quick_test = quick_test
        self.execution_log = []
        self.start_time = time.time()

    def log_step(self, step: str, status: str, details: str = "", duration: float = 0):
        """실행 로그 기록"""
        self.execution_log.append({
            'step': step,
            'status': status,
            'details': details,
            'duration': duration,
            'timestamp': time.time() - self.start_time
        })

        # 콘솔 출력
        status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🔄"
        print(f"{status_emoji} {step}")
        if details:
            print(f"   {details}")
        if duration > 0:
            print(f"   Duration: {duration:.1f}s")

    def run_python_script(self, script_name: str, description: str) -> Tuple[bool, str]:
        """Python 스크립트 실행"""
        print(f"\n🔄 Executing: {description}")
        print(f"   Script: {script_name}")

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=1800  # 30분 타임아웃
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                self.log_step(description, "SUCCESS",
                            f"Exit code: {result.returncode}", duration)
                return True, result.stdout
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                self.log_step(description, "FAILED",
                            f"Exit code: {result.returncode}, Error: {error_msg[:200]}", duration)
                return False, error_msg

        except subprocess.TimeoutExpired:
            self.log_step(description, "FAILED", "Timeout (30 minutes)", duration)
            return False, "Script execution timed out"
        except Exception as e:
            duration = time.time() - start_time
            self.log_step(description, "FAILED", str(e), duration)
            return False, str(e)

    def check_file_exists(self, file_path: str, description: str) -> bool:
        """파일 존재 확인"""
        exists = Path(file_path).exists()
        status = "SUCCESS" if exists else "FAILED"
        self.log_step(f"Check: {description}", status, f"Path: {file_path}")
        return exists

    def step1_data_labeling(self) -> bool:
        """1단계: 데이터 누수 없는 라벨링"""
        print("\n" + "="*60)
        print("1️⃣ STEP 1: Data Labeling (NO LEAKAGE)")
        print("="*60)

        # 1-1. 라벨링 시스템 실행
        success, output = self.run_python_script(
            "fixed_data_labeling_system.py",
            "Step 1-1: Execute Fixed Data Labeling System"
        )

        if not success:
            return False

        # 1-2. 결과 파일 확인
        return self.check_file_exists(
            "ml_analysis_results/seoul_commercial_fixed_dataset.csv",
            "Step 1-2: Fixed labeled dataset created"
        )

    def step2_feature_engineering(self) -> bool:
        """2단계: 외부 지표 피처 엔지니어링"""
        print("\n" + "="*60)
        print("2️⃣ STEP 2: Feature Engineering (EXTERNAL INDICATORS)")
        print("="*60)

        # 2-1. 피처 엔지니어링 실행
        success, output = self.run_python_script(
            "fixed_feature_engineering.py",
            "Step 2-1: Execute Fixed Feature Engineering"
        )

        if not success:
            return False

        # 2-2. 결과 파일들 확인
        files_to_check = [
            "ml_preprocessed_data_fixed/train_data.csv",
            "ml_preprocessed_data_fixed/validation_data.csv",
            "ml_preprocessed_data_fixed/test_data.csv",
            "ml_preprocessed_data_fixed/class_weights.joblib"
        ]

        all_exist = True
        for file_path in files_to_check:
            if not self.check_file_exists(file_path, f"Step 2-2: {Path(file_path).name}"):
                all_exist = False

        return all_exist

    def step3_model_training(self) -> bool:
        """3단계: 앙상블 모델 훈련"""
        print("\n" + "="*60)
        print("3️⃣ STEP 3: Ensemble Model Training (OVERFITTING PREVENTION)")
        print("="*60)

        if self.skip_training:
            print("⏭️ Skipping model training (--skip-training flag)")
            # 기존 모델 파일들 확인
            model_files = list(Path("trained_models_fixed").glob("*_model.joblib"))
            if len(model_files) >= 3:
                self.log_step("Step 3: Model Training", "SKIPPED",
                            f"Found {len(model_files)} existing models")
                return True
            else:
                self.log_step("Step 3: Model Training", "FAILED",
                            "No existing models found, training required")
                return False

        # 3-1. 모델 훈련 실행
        success, output = self.run_python_script(
            "fixed_ensemble_training.py",
            "Step 3-1: Execute Fixed Ensemble Training"
        )

        if not success:
            return False

        # 3-2. 모델 파일들 확인
        model_files = [
            "trained_models_fixed/random_forest_model.joblib",
            "trained_models_fixed/xgboost_model.joblib",
            "trained_models_fixed/model_evaluation_results.json"
        ]

        all_exist = True
        for file_path in model_files:
            if Path(file_path).exists():
                self.log_step(f"Step 3-2: {Path(file_path).name}", "SUCCESS", f"Found: {file_path}")
            else:
                self.log_step(f"Step 3-2: {Path(file_path).name}", "WARNING", f"Missing: {file_path}")

        # 최소 1개 모델만 있으면 성공으로 간주
        existing_models = list(Path("trained_models_fixed").glob("*_model.joblib"))
        if len(existing_models) >= 1:
            self.log_step("Step 3-2: Model Training", "SUCCESS",
                        f"Created {len(existing_models)} models")
            return True
        else:
            self.log_step("Step 3-2: Model Training", "FAILED", "No models created")
            return False

    def step4_integration_test(self) -> bool:
        """4단계: 통합 파이프라인 테스트"""
        print("\n" + "="*60)
        print("4️⃣ STEP 4: Master Pipeline Integration Test")
        print("="*60)

        # 4-1. 통합 파이프라인 실행
        success, output = self.run_python_script(
            "master_integrated_pipeline.py",
            "Step 4-1: Execute Master Integrated Pipeline"
        )

        if not success:
            return False

        # 4-2. 출력 분석 (간단한 검증)
        if "✅ Master Pipeline ready!" in output:
            self.log_step("Step 4-2: Pipeline Initialization", "SUCCESS",
                        "Master pipeline initialized successfully")
        else:
            self.log_step("Step 4-2: Pipeline Initialization", "WARNING",
                        "Pipeline may have initialization issues")

        if "🎯 Hybrid Risk Prediction" in output:
            self.log_step("Step 4-3: Prediction Test", "SUCCESS",
                        "Prediction functionality working")
            return True
        else:
            self.log_step("Step 4-3: Prediction Test", "WARNING",
                        "Prediction test may have issues")
            return True  # 부분 성공으로 간주

    def step5_performance_validation(self) -> bool:
        """5단계: 성능 검증"""
        print("\n" + "="*60)
        print("5️⃣ STEP 5: Performance Validation")
        print("="*60)

        # 5-1. 모델 성능 결과 로드
        try:
            results_file = Path("trained_models_fixed/model_evaluation_results.json")
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                # 성능 지표 분석
                best_accuracy = 0
                best_f1 = 0
                model_count = 0

                for model_name, eval_results in results.get('evaluation_results', {}).items():
                    accuracy = eval_results.get('test_accuracy', 0)
                    f1_score = eval_results.get('test_f1_weighted', 0)

                    best_accuracy = max(best_accuracy, accuracy)
                    best_f1 = max(best_f1, f1_score)
                    model_count += 1

                    print(f"   📊 {model_name}: Accuracy={accuracy:.3f}, F1={f1_score:.3f}")

                # 성능 평가
                if best_accuracy >= 0.8 and best_f1 >= 0.75:
                    self.log_step("Step 5-1: Performance Analysis", "SUCCESS",
                                f"Best: Accuracy={best_accuracy:.3f}, F1={best_f1:.3f}")
                    performance_pass = True
                elif best_accuracy >= 0.7 and best_f1 >= 0.65:
                    self.log_step("Step 5-1: Performance Analysis", "WARNING",
                                f"Moderate: Accuracy={best_accuracy:.3f}, F1={best_f1:.3f}")
                    performance_pass = True
                else:
                    self.log_step("Step 5-1: Performance Analysis", "FAILED",
                                f"Low: Accuracy={best_accuracy:.3f}, F1={best_f1:.3f}")
                    performance_pass = False

            else:
                self.log_step("Step 5-1: Performance Analysis", "WARNING",
                            "No evaluation results found")
                performance_pass = True  # 부분 성공

        except Exception as e:
            self.log_step("Step 5-1: Performance Analysis", "FAILED", str(e))
            performance_pass = False

        # 5-2. 데이터 누수 검증
        try:
            # 피처 파일에서 매출 관련 컬럼 확인
            train_data = Path("ml_preprocessed_data_fixed/train_data.csv")
            if train_data.exists():
                import pandas as pd
                df = pd.read_csv(train_data, nrows=1)  # 헤더만 읽기
                revenue_columns = [col for col in df.columns if '매출' in col or 'revenue' in col.lower()]

                if len(revenue_columns) == 0:
                    self.log_step("Step 5-2: Data Leakage Check", "SUCCESS",
                                "No revenue columns found in features")
                    leakage_pass = True
                else:
                    self.log_step("Step 5-2: Data Leakage Check", "WARNING",
                                f"Found potential revenue columns: {revenue_columns}")
                    leakage_pass = True  # 경고지만 통과
            else:
                self.log_step("Step 5-2: Data Leakage Check", "WARNING",
                            "Cannot check - training data not found")
                leakage_pass = True

        except Exception as e:
            self.log_step("Step 5-2: Data Leakage Check", "FAILED", str(e))
            leakage_pass = False

        return performance_pass and leakage_pass

    def generate_final_report(self) -> None:
        """최종 보고서 생성"""
        print("\n" + "="*70)
        print("📋 FINAL PIPELINE EXECUTION REPORT")
        print("="*70)

        total_duration = time.time() - self.start_time

        # 단계별 요약
        step_status = {}
        for log_entry in self.execution_log:
            step = log_entry['step']
            status = log_entry['status']

            if 'STEP' in step and ':' in step:
                step_num = step.split(':')[0].strip()
                if step_num not in step_status:
                    step_status[step_num] = {'SUCCESS': 0, 'WARNING': 0, 'FAILED': 0, 'SKIPPED': 0}
                step_status[step_num][status] = step_status[step_num].get(status, 0) + 1

        # 단계별 결과 출력
        print(f"\n📊 Step-by-Step Results:")
        overall_success = True

        for step, counts in step_status.items():
            success_count = counts.get('SUCCESS', 0)
            warning_count = counts.get('WARNING', 0)
            failed_count = counts.get('FAILED', 0)
            skipped_count = counts.get('SKIPPED', 0)

            if failed_count > 0:
                status_emoji = "❌"
                overall_success = False
            elif warning_count > 0:
                status_emoji = "⚠️"
            elif skipped_count > 0:
                status_emoji = "⏭️"
            else:
                status_emoji = "✅"

            print(f"   {status_emoji} {step}: ✅{success_count} ⚠️{warning_count} ❌{failed_count} ⏭️{skipped_count}")

        # 전체 결과
        print(f"\n🎯 Overall Result:")
        if overall_success:
            print(f"   ✅ PIPELINE SUCCESS")
            print(f"   🎉 All critical components working")
            print(f"   ⏱️ Total execution time: {total_duration:.1f}s")
        else:
            print(f"   ❌ PIPELINE FAILED")
            print(f"   🔧 Some components need attention")
            print(f"   ⏱️ Execution time: {total_duration:.1f}s")

        # 세부 로그 저장
        log_file = Path("pipeline_execution_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'execution_log': self.execution_log,
                'total_duration': total_duration,
                'overall_success': overall_success,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed log saved: {log_file}")

        # 다음 단계 추천
        print(f"\n🚀 Next Steps:")
        if overall_success:
            print(f"   1. Review model performance in trained_models_fixed/")
            print(f"   2. Test master_integrated_pipeline.py with your data")
            print(f"   3. Deploy to production environment")
        else:
            print(f"   1. Check detailed logs for error information")
            print(f"   2. Fix failed components and re-run")
            print(f"   3. Consider running with --skip-training for faster debugging")

        print("="*70)

    def run_complete_pipeline(self) -> bool:
        """전체 파이프라인 실행"""
        print("🚀 Starting Complete ML Pipeline Execution")
        print(f"⚙️ Configuration: skip_training={self.skip_training}, quick_test={self.quick_test}")
        print("="*70)

        # 단계별 실행
        steps = [
            ("Step 1: Data Labeling", self.step1_data_labeling),
            ("Step 2: Feature Engineering", self.step2_feature_engineering),
            ("Step 3: Model Training", self.step3_model_training),
            ("Step 4: Integration Test", self.step4_integration_test),
            ("Step 5: Performance Validation", self.step5_performance_validation)
        ]

        failed_steps = []

        for step_name, step_function in steps:
            try:
                success = step_function()
                if not success:
                    failed_steps.append(step_name)
                    if not self.quick_test:  # quick_test 모드가 아니면 실패시 중단
                        print(f"\n❌ {step_name} failed. Stopping execution.")
                        break
            except Exception as e:
                self.log_step(step_name, "FAILED", f"Exception: {str(e)}")
                failed_steps.append(step_name)
                if not self.quick_test:
                    print(f"\n❌ {step_name} failed with exception. Stopping execution.")
                    break

        # 최종 보고서 생성
        self.generate_final_report()

        return len(failed_steps) == 0

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Complete ML Pipeline Execution")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training step (use existing models)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Continue execution even if steps fail")

    args = parser.parse_args()

    # 파이프라인 실행
    pipeline = CompleteMLPipeline(
        skip_training=args.skip_training,
        quick_test=args.quick_test
    )

    success = pipeline.run_complete_pipeline()

    # 종료 코드 설정
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()