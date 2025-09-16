#!/usr/bin/env python3
"""
Fast Ensemble ML Model Training for Seoul Market Risk
====================================================

Optimized version for faster training on large datasets.
Uses smaller model configurations and early stopping.

Author: Seoul Market Risk ML System
Date: 2025-09-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Try to import XGBoost and LightGBM, use alternatives if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

class FastEnsembleTrainer:
    """Fast ensemble trainer optimized for large datasets"""

    def __init__(self, data_dir: str = "ml_preprocessed_data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self._load_data()

    def _load_data(self):
        """Load preprocessed data"""
        print("📂 Loading data...")

        self.train_data = pd.read_csv(self.data_dir / "train_data.csv")
        self.val_data = pd.read_csv(self.data_dir / "validation_data.csv")
        self.test_data = pd.read_csv(self.data_dir / "test_data.csv")

        # Use subset for faster training
        train_size = min(50000, len(self.train_data))  # Limit training size
        self.train_sample = self.train_data.sample(n=train_size, random_state=42)

        self.X_train = self.train_sample.drop('risk_label', axis=1)
        self.y_train = self.train_sample['risk_label']

        self.X_val = self.val_data.drop('risk_label', axis=1)
        self.y_val = self.val_data['risk_label']

        self.X_test = self.test_data.drop('risk_label', axis=1)
        self.y_test = self.test_data['risk_label']

        print(f"✅ Training on {len(self.X_train):,} samples (subset for speed)")
        print(f"✅ Validation: {len(self.X_val):,}, Test: {len(self.X_test):,}")

    def train_random_forest(self):
        """Train Random Forest with optimized settings"""
        print("🌳 Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        # Evaluate
        val_pred = model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred, average='weighted')

        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'training_time': training_time,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        print(f"✅ RF: Acc={val_acc:.3f}, F1={val_f1:.3f}, Time={training_time:.1f}s")

    def train_logistic_regression(self):
        """Train Logistic Regression (fast baseline)"""
        print("📈 Training Logistic Regression...")

        model = LogisticRegression(
            max_iter=200,
            random_state=42,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        val_pred = model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred, average='weighted')

        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'training_time': training_time,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        print(f"✅ LR: Acc={val_acc:.3f}, F1={val_f1:.3f}, Time={training_time:.1f}s")

    def train_neural_network(self):
        """Train smaller Neural Network"""
        print("🧠 Training Neural Network...")

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Smaller architecture
            max_iter=200,  # Reduced iterations
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        val_pred = model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred, average='weighted')

        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'training_time': training_time,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        print(f"✅ NN: Acc={val_acc:.3f}, F1={val_f1:.3f}, Time={training_time:.1f}s")

    def train_xgboost(self):
        """Train XGBoost if available"""
        if not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost not available, skipping...")
            return

        print("🚀 Training XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        val_pred = model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred, average='weighted')

        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'training_time': training_time,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        print(f"✅ XGB: Acc={val_acc:.3f}, F1={val_f1:.3f}, Time={training_time:.1f}s")

    def train_lightgbm(self):
        """Train LightGBM if available"""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available, skipping...")
            return

        print("💡 Training LightGBM...")

        model = lgb.LGBMClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        val_pred = model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred, average='weighted')

        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'training_time': training_time,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }

        print(f"✅ LGB: Acc={val_acc:.3f}, F1={val_f1:.3f}, Time={training_time:.1f}s")

    def evaluate_on_test(self):
        """Evaluate all models on test set"""
        print("\n📊 Test Set Evaluation")
        print("=" * 30)

        test_results = {}

        for name, model in self.models.items():
            print(f"\n🔍 Testing {name}...")

            # Test predictions
            start_time = time.time()
            test_pred = model.predict(self.X_test)
            pred_time = time.time() - start_time

            # Calculate metrics
            test_acc = accuracy_score(self.y_test, test_pred)
            test_f1 = f1_score(self.y_test, test_pred, average='weighted')

            # Prediction speed
            samples_per_sec = len(self.X_test) / pred_time

            test_results[name] = {
                'test_accuracy': test_acc,
                'test_f1_weighted': test_f1,
                'prediction_time': pred_time,
                'samples_per_second': samples_per_sec
            }

            print(f"   Accuracy: {test_acc:.3f}")
            print(f"   F1-Score: {test_f1:.3f}")
            print(f"   Speed: {samples_per_sec:.0f} samples/sec")

            # Check targets
            meets_acc = test_acc > 0.85
            meets_f1 = test_f1 > 0.80
            fast_pred = samples_per_sec > 1000

            print(f"   🎯 Targets: Acc{'✅' if meets_acc else '❌'} F1{'✅' if meets_f1 else '❌'} Speed{'✅' if fast_pred else '❌'}")

        return test_results

    def get_best_model(self, test_results):
        """Find best performing model"""
        best_score = 0
        best_model = None
        best_name = None

        for name, results in test_results.items():
            # Composite score
            score = (results['test_accuracy'] * 0.6 + results['test_f1_weighted'] * 0.4)

            if score > best_score:
                best_score = score
                best_model = self.models[name]
                best_name = name

        print(f"\n🏆 Best Model: {best_name} (Score: {best_score:.3f})")
        return best_name, best_model

    def save_models(self, test_results, output_dir: str = "trained_models"):
        """Save trained models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving to {output_dir}/...")

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, output_path / f"{name}.joblib")

        # Save results
        all_results = {
            'training_results': self.results,
            'test_results': test_results,
            'feature_count': len(self.X_train.columns),
            'training_samples': len(self.X_train)
        }

        import json
        with open(output_path / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"✅ Saved {len(self.models)} models and results")

    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🎯 Starting Fast Ensemble Training")
        print("=" * 40)

        # Train all available models
        training_methods = [
            self.train_random_forest,
            self.train_logistic_regression,
            self.train_neural_network,
            self.train_xgboost,
            self.train_lightgbm
        ]

        total_start = time.time()

        for method in training_methods:
            try:
                method()
            except Exception as e:
                print(f"❌ Training method failed: {e}")

        total_training_time = time.time() - total_start

        print(f"\n✅ Trained {len(self.models)} models in {total_training_time:.1f}s")

        # Evaluate on test set
        test_results = self.evaluate_on_test()

        # Find best model
        best_name, best_model = self.get_best_model(test_results)

        # Save everything
        self.save_models(test_results)

        print(f"\n🎯 Training Complete!")
        print(f"   Models: {list(self.models.keys())}")
        print(f"   Best: {best_name}")
        print(f"   Total time: {total_training_time:.1f}s")

        return self.models, test_results, best_name

def main():
    trainer = FastEnsembleTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()