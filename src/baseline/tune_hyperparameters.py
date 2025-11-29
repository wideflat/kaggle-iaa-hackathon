"""
Hyperparameter tuning script for Ames Housing
Run this to find optimal LightGBM hyperparameters
"""

import pandas as pd
import numpy as np
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.preprocessor import AmesPreprocessor
from src.baseline.model import LGBMTuner


def main():
    print("=" * 70)
    print("LightGBM Hyperparameter Tuning for Ames Housing")
    print("=" * 70)

    # Load data
    print("\n1. Loading and preprocessing data...")
    loader = AmesDataLoader('data')
    train = loader.load_train()

    # Extract target
    target = train['SalePrice']
    target_log = np.log1p(target)

    # Preprocess
    preprocessor = AmesPreprocessor()
    X_train_raw = train.drop(['Id', 'SalePrice'], axis=1)
    X_train = preprocessor.fit_transform(X_train_raw)

    print(f"   Data shape: {X_train.shape}")
    print(f"   Target: log-transformed SalePrice")

    # Run hyperparameter tuning
    print("\n2. Running hyperparameter optimization...")
    print("   This will take several minutes...")

    tuner = LGBMTuner(
        X=X_train,
        y=target_log,
        n_folds=5,
        n_trials=30,  # Reduced for faster completion (~5-7 minutes)
        random_state=42
    )

    best_params = tuner.tune(verbose=True)

    # Save best parameters
    print("\n3. Saving best parameters...")
    output_path = 'outputs/models/best_lgbm_params.json'
    os.makedirs('outputs/models', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"   Saved to: {output_path}")

    # Print comparison
    print("\n" + "=" * 70)
    print("TUNING COMPLETE!")
    print("=" * 70)
    print(f"Best CV RMSE (log scale): {tuner.study.best_value:.5f}")
    print(f"Parameters saved to: {output_path}")
    print("\nTo use these parameters, run:")
    print("  venv/bin/python src/baseline/train.py --use-tuned")
    print("=" * 70)


if __name__ == '__main__':
    main()
