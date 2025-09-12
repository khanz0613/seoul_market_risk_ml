# Seoul Market Opportunity ML System

A proactive financial services platform for small businesses in Seoul using revenue trend monitoring and opportunity scoring (2019-2024).

## Project Overview

This system continuously monitors small business revenue trends to provide proactive financial services. Using a 5-tier opportunity scoring system (0-100 points), it recommends:

- **Preemptive loans** when revenue declines to prevent cash flow crises
- **Investment opportunities** when revenue grows to maximize surplus funds
- **NH Bank integration** for personalized loan and investment product matching

## Key Features

- **Hierarchical ML Models**: 79 models (1 Global + 6 Regional + 72 Local)
- **Opportunity Score Engine**: 5-tier opportunity assessment (매우위험, 위험군, 적정, 좋음, 매우좋음)
- **Proactive Financial Services**: Automatic loan/investment recommendations based on revenue trends
- **NH Bank API Integration**: Real-time access to loan and investment products
- **Revenue Monitoring**: Continuous tracking and early intervention system
- **LLM Integration**: Automated opportunity reports and financial guidance

## Architecture

```
Revenue Monitoring System
         ↓
ML Prediction Models (79)
         ↓
Opportunity Score Calculator (5-tier)
         ↓
Financial Recommendation Engine
    ↓            ↓
Loan Services   Investment Services
    ↓            ↓
NH Bank API Integration
```

### 5-Tier Opportunity System
- **매우위험 (0-20)**: 긴급 대출 필요 → Emergency loan products
- **위험군 (21-40)**: 안정화 대출 추천 → Stabilization funding  
- **적정 (41-60)**: 모니터링 지속 → Continuous monitoring
- **좋음 (61-80)**: 성장 투자 기회 → Growth investment options
- **매우좋음 (81-100)**: 고수익 투자 추천 → Premium investment products

## Data Sources

- Seoul Commercial Area Analysis Data (2019-2024, 6 CSV files)
- External APIs: Weather, holidays, economic indicators
- Business categorization: 12-15 major categories
- Regional clustering: 6-8 groups based on demographics

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run data preprocessing: `python src/preprocessing/main.py`
3. Train models: `python src/models/train_hierarchical.py`
4. Calculate risk scores: `python src/risk_scoring/calculate.py`

## Project Structure

```
seoul_market_risk_ml/
├── data/                    # Data storage
│   ├── raw/                # Original CSV files
│   ├── processed/          # Cleaned and processed data
│   └── external/           # External API data
├── src/                    # Source code
│   ├── preprocessing/      # Data cleaning and encoding
│   ├── feature_engineering/# Feature creation
│   ├── clustering/         # Regional and business clustering
│   ├── models/            # ML models
│   ├── risk_scoring/      # Risk calculation engine
│   ├── loan_calculation/  # Loan recommendation system
│   └── llm_integration/   # LLM report generation
├── config/                # Configuration files
├── tests/                 # Test suites
└── notebooks/             # Jupyter notebooks
```

## Risk Score Formula

Based on Altman Z-Score methodology:
```
Risk_Score = 0.3×매출변화율 + 0.2×변동성 + 0.2×트렌드 + 0.15×계절성이탈 + 0.15×업종비교
```

## Development Timeline

- **Month 1**: Data pipeline and Global model
- **Month 2**: Regional/Local models and Risk scoring
- **Month 3**: LLM integration and system testing

## License

MIT License