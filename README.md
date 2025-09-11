# Seoul Market Risk ML System

A hierarchical ML system for predicting small business revenue risk in Seoul using commercial area data (2019-2024).

## Project Overview

This system uses Seoul's commercial area analysis data to evaluate revenue risk for small businesses using a 5-level scoring system (0-100 points) and provides personalized loan/funding recommendations.

## Key Features

- **Hierarchical ML Models**: 79 models (1 Global + 6 Regional + 72 Local)
- **Risk Score Engine**: 5-level risk assessment based on Altman Z-Score methodology
- **Loan Calculator**: Risk-neutralization based loan amount calculation
- **LLM Integration**: Automated report generation and financial recommendations
- **Real-time Processing**: Fast risk calculation for new data inputs

## Architecture

```
Global Model (1) → Regional Models (6) → Local Models (72)
     ↓                    ↓                     ↓
Seoul-wide patterns → Regional patterns → Business-specific patterns
```

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