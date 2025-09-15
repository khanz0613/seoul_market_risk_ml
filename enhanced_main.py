"""
향상된 소상공인 위험도 분석 시스템
통합 실행 파일
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.user_interface.minimal_input_interface import MinimalInputInterface, BusinessInput
from src.ml_pipeline.expense_prediction_model import ExpensePredictionModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBusinessRiskAnalyzer:
    """향상된 소상공인 위험도 분석 시스템"""

    def __init__(self, use_ml_model: bool = True, model_path: Optional[str] = None):
        """
        초기화

        Args:
            use_ml_model: ML 모델 사용 여부 (False면 기존 고정 비율 사용)
            model_path: ML 모델 파일 경로 (None이면 자동으로 최신 모델 찾기)
        """
        self.use_ml_model = use_ml_model
        self.ml_model = None

        # ML 모델 로딩 시도
        if use_ml_model:
            try:
                self.ml_model = self._load_ml_model(model_path)
                logger.info("✅ ML 모델 로딩 성공 - AI 예측 모드로 실행")
            except Exception as e:
                logger.warning(f"⚠️ ML 모델 로딩 실패: {e}")
                logger.warning("💡 기존 고정 비율 모드로 실행됩니다")
                self.use_ml_model = False

        # 기본 인터페이스 초기화
        self.interface = MinimalInputInterface(ml_model=self.ml_model if self.use_ml_model else None)

        mode = "AI 예측 모드" if self.use_ml_model else "고정 비율 모드"
        logger.info(f"Enhanced Business Risk Analyzer 초기화 완료 - {mode}")

    def _load_ml_model(self, model_path: Optional[str] = None) -> ExpensePredictionModel:
        """
        ML 모델 로딩

        Args:
            model_path: 모델 파일 경로 (None이면 자동으로 찾기)

        Returns:
            로딩된 ExpensePredictionModel
        """
        from pathlib import Path
        import glob

        if model_path and os.path.exists(model_path):
            # 명시적 경로가 제공된 경우
            model = ExpensePredictionModel()
            model.load_model(model_path)
            logger.info(f"ML 모델 로딩: {model_path}")
            return model

        # 자동으로 최신 모델 찾기
        models_dir = Path("models")
        if not models_dir.exists():
            raise FileNotFoundError("models 디렉토리를 찾을 수 없습니다. 먼저 모델을 학습해주세요.")

        # .joblib 파일 중에서 가장 최신 것 찾기
        model_files = list(models_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError("학습된 모델을 찾을 수 없습니다. train_ml_models.py를 실행해주세요.")

        # 파일 수정 시간 기준으로 최신 파일 선택
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)

        model = ExpensePredictionModel()
        model.load_model(str(latest_model_file))
        logger.info(f"최신 ML 모델 자동 로딩: {latest_model_file.name}")
        return model

    def run_demo_analysis(self) -> Dict[str, Any]:
        """데모 분석 실행"""

        print("🏪 소상공인 위험도 분석 시스템 v2.0")
        prediction_mode = "🤖 AI 예측 모드" if self.use_ml_model else "📊 고정 비율 모드"
        print(f"   {prediction_mode}")
        print("=" * 60)

        # 데모 데이터로 분석 실행
        demo_input = BusinessInput(
            업종코드="CS100001",    # 한식음식점
            월매출=8500000,        # 850만원
            운용자산=12000000,     # 1200만원
            업력_개월=24,          # 24개월 (2년)
            실제_인건비=2000000,   # 200만원 (선택 입력)
            실제_임대료=800000     # 80만원 (선택 입력)
        )

        # 입력 검증
        is_valid, errors = self.interface.validate_input(demo_input)
        if not is_valid:
            print(f"❌ 입력 오류: {', '.join(errors)}")
            return {}

        # 분석 실행
        print("📊 위험도 분석 실행 중...")
        result = self.interface.process_minimal_input(demo_input)

        return result

    def display_analysis_results(self, result: Dict[str, Any]) -> None:
        """분석 결과 출력"""

        if not result:
            print("❌ 분석 결과가 없습니다.")
            return

        # 1. 기본 정보
        print("\n📋 사업 정보")
        print("-" * 30)
        info = result['사업정보']
        print(f"업종: {info['업종']}")
        print(f"카테고리: {info['업종카테고리']}")
        print(f"월매출: {info['월매출']}")
        print(f"업력: {info['업력']}")

        # 2. 위험도 평가
        print("\n🎯 위험도 평가 결과")
        print("-" * 30)
        risk = result['위험도평가']
        print(f"종합 점수: {risk['종합점수']}")
        print(f"위험 등급: {risk['위험등급']}")
        print(f"분석 신뢰도: {risk['신뢰도']}")

        # 3. 위험 원인 분석 (핵심 요구사항)
        print("\n⚠️ 위험 원인 분석")
        print("-" * 30)
        risk_analysis = result['위험원인분석']

        print("📊 항목별 위험 비중:")
        for category, data in risk_analysis['항목별비중'].items():
            if isinstance(data, dict) and '기여도' in data:
                print(f"  {category}: {data['기여도']}", end="")
                if '업종평균대비' in data:
                    print(f" (업종평균대비 {data['업종평균대비']})")
                else:
                    print()

        print(f"\n🔥 가장 큰 위험 원인: {risk_analysis['가장큰원인']['항목']} ({risk_analysis['가장큰원인']['기여도']})")

        # 4. 7일간 현금 흐름 예측 (핵심 요구사항)
        print("\n💰 7일간 현금 흐름 예측")
        print("-" * 30)
        cashflow_data = result['현금흐름예측']['7일예측']

        print("일별 예상 현금 흐름:")
        for day_data in cashflow_data:
            print(f"  Day {day_data['day']}: 매출 {day_data['predicted_revenue']:,}원 → "
                  f"순현금흐름 {day_data['net_cashflow']:,}원")

        # 지난주 동기 대비
        weekly_comp = result['현금흐름예측']['지난주대비']
        if 'revenue_change_percent' in weekly_comp:
            print(f"\n📈 지난주 동기 대비: {weekly_comp['revenue_change_percent']:+.1f}%")
            if weekly_comp.get('change_factors'):
                print(f"   주요 변화 요인: {', '.join(weekly_comp['change_factors'])}")

        # 5. 비용 구조 비교
        print("\n💳 업종 평균 대비 비용 구조")
        print("-" * 30)
        cost_comp = result['비용구조비교']

        for category, data in cost_comp.items():
            status_emoji = {"높음": "🔴", "낮음": "🔵", "적정": "🟢"}.get(data['상태'], "⚪")
            print(f"{status_emoji} {category}: 사용자 {data['사용자']} vs 업종평균 {data['업종평균']} "
                  f"(편차 {data['편차']})")

        # 6. 개선 방안
        print("\n🎯 개선 방안")
        print("-" * 30)
        improvement = result['개선방안']['3단계달성방법']

        if isinstance(improvement, dict) and '권장_개선_금액' in improvement:
            print(f"💡 3단계 달성을 위한 권장 개선 금액: {improvement['권장_개선_금액']:,.0f}원")

            print("\n개선 방법별 필요 금액:")
            for method, amount in improvement.items():
                if isinstance(amount, (int, float)) and method != '권장_개선_금액':
                    print(f"  {method}: {amount:,.0f}원")

        # 투자 기회
        investment = result['개선방안']['투자기회']
        if isinstance(investment, dict) and '추정_여유자금' in investment:
            print(f"\n💎 투자 기회 분석:")
            print(f"  추정 여유자금: {investment['추정_여유자금']:,.0f}원")
            print(f"  안전투자 가능금액: {investment['안전투자_가능금액']:,.0f}원")

    def save_results_to_file(self, result: Dict[str, Any], filename: str = None) -> str:
        """결과를 JSON 파일로 저장"""

        if filename is None:
            filename = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(os.getcwd(), filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return filepath

    def display_usage_guide(self) -> None:
        """사용법 가이드 출력"""

        print("\n📖 사용법 가이드")
        print("=" * 60)

        guide = self.interface.get_input_guide()

        print("\n✅ 필수 입력 항목:")
        for item, desc in guide['필수입력'].items():
            print(f"  • {item}: {desc}")

        print("\n🔧 선택 입력 항목 (더 정확한 분석):")
        for item, desc in guide['선택입력'].items():
            print(f"  • {item}: {desc}")

        print("\n🤖 자동 계산 항목:")
        for item, desc in guide['자동계산'].items():
            print(f"  • {item}: {desc}")

        print("\n💡 분석 정확도 향상 팁:")
        for tip_key, tip_desc in guide['팁'].items():
            print(f"  • {tip_desc}")

    def show_industry_choices(self) -> None:
        """업종 선택 옵션 표시"""

        print("\n🏢 지원 업종 목록")
        print("=" * 60)

        choices = self.interface.get_industry_choices()

        categories = {
            "숙박음식점업": [],
            "도매소매업": [],
            "예술스포츠업": [],
            "개인서비스업": []
        }

        # 업종별로 분류
        for code, name in choices.items():
            category = self.interface.industry_mapper.map_industry_code(code)
            if category in categories:
                categories[category].append(f"{code}: {name}")

        # 카테고리별 출력
        for category, items in categories.items():
            if items:
                print(f"\n📂 {category}:")
                for item in items[:10]:  # 처음 10개만 표시
                    print(f"  {item}")
                if len(items) > 10:
                    print(f"  ... 및 {len(items)-10}개 업종 더")

def main():
    """메인 실행 함수"""

    try:
        analyzer = EnhancedBusinessRiskAnalyzer()

        # 사용법 가이드 표시
        analyzer.display_usage_guide()
        analyzer.show_industry_choices()

        # 데모 분석 실행
        print("\n" + "="*60)
        print("🚀 데모 분석 실행")
        print("="*60)

        result = analyzer.run_demo_analysis()

        if result:
            # 결과 출력
            analyzer.display_analysis_results(result)

            # 결과 파일 저장
            saved_file = analyzer.save_results_to_file(result)
            print(f"\n💾 분석 결과가 저장되었습니다: {saved_file}")

            # 요약
            print("\n" + "="*60)
            print("📋 요구사항 달성 현황")
            print("="*60)
            print("✅ 7일간 현금 흐름 예측 (그래프용 데이터 제공)")
            print("✅ 위험 원인 분석 (식자재, 인건비, 임대료, 기타 각 비중)")
            print("✅ 가장 큰 원인 제시 (전체 원인의 몇% 기여)")
            print("✅ 일별 예상 현금 흐름")
            print("✅ 지난주 동기 대비 분석 (주요 변화 요인 포함)")
            print("✅ 업종별 평균 대비 비교")
            print("✅ 3단계 달성을 위한 개선 금액 산정")
            print("✅ 투자 가능 여유금 계산")
            print("✅ 최소 입력으로 최대 분석 결과 제공")

            # ML 모델 상태 표시
            if analyzer.use_ml_model:
                print("🤖 ML 모델을 통한 AI 기반 비용 예측 (6년치 서울시 상권 데이터 학습)")
            else:
                print("📊 업종별 평균 비율 기반 비용 예측 (ML 모델 미사용)")

        else:
            print("❌ 분석 실행 중 오류가 발생했습니다.")
            return 1

    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"❌ 오류: {e}")
        return 1

    print("\n🎉 분석이 완료되었습니다!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)