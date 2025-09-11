# Seoul Market Risk ML System Makefile

.PHONY: help install install-dev test lint format clean setup-env data-process train predict

# Default target
help:
	@echo "Seoul Market Risk ML System - Available Commands:"
	@echo ""
	@echo "  Setup & Installation:"
	@echo "    make setup-env     - Set up Python virtual environment"
	@echo "    make install       - Install package and dependencies"
	@echo "    make install-dev   - Install with development dependencies"
	@echo ""
	@echo "  Data Processing:"
	@echo "    make data-process  - Run data preprocessing pipeline"
	@echo "    make data-validate - Validate processed data"
	@echo ""
	@echo "  Model Training:"
	@echo "    make train-global  - Train global model"
	@echo "    make train-regional - Train regional models"
	@echo "    make train-local   - Train local models"
	@echo "    make train-all     - Train all models"
	@echo ""
	@echo "  Risk Prediction:"
	@echo "    make predict       - Run risk prediction"
	@echo "    make calculate-loan - Calculate loan recommendations"
	@echo ""
	@echo "  Development:"
	@echo "    make test          - Run all tests"
	@echo "    make lint          - Run code linting"
	@echo "    make format        - Format code"
	@echo "    make clean         - Clean build artifacts"
	@echo ""

# Environment setup
setup-env:
	python -m venv seoul_risk_env
	@echo "Virtual environment created. Activate with: source seoul_risk_env/bin/activate"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Data processing
data-process:
	python src/preprocessing/main.py

data-validate:
	python src/preprocessing/validate.py

# Model training
train-global:
	python src/models/train_global.py

train-regional:
	python src/models/train_regional.py

train-local:
	python src/models/train_local.py

train-all:
	python src/models/train_hierarchical.py

# Prediction and analysis
predict:
	python src/risk_scoring/calculate.py

calculate-loan:
	python src/loan_calculation/main.py

# Development tools
test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Check system requirements
check-requirements:
	@python --version
	@echo "Checking required Python packages..."
	@pip list | grep -E "(pandas|numpy|scikit-learn|prophet|lightgbm)"

# Quick start - sets up everything for new users
quickstart: setup-env install data-process
	@echo ""
	@echo "🎉 Seoul Market Risk ML System is ready!"
	@echo "Run 'make help' to see available commands."
	@echo "Start with 'make train-global' to train your first model."