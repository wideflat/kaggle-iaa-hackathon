"""
Hyperparameter tuning script for Ames Housing

Run this to find optimal LightGBM hyperparameters.

Usage:
    # Tune on minimal baseline (default)
    python -m src.baseline.tune_hyperparameters

    # Tune on data with agent's discovered features
    python -m src.baseline.tune_hyperparameters --with-features

    # More trials for better tuning (slower)
    python -m src.baseline.tune_hyperparameters --n-trials 100
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.minimal_preprocessor import MinimalPreprocessor
from src.baseline.model import LGBMTuner


def load_and_apply_features(df: pd.DataFrame, memory_path: str) -> pd.DataFrame:
    """Apply discovered features from agent memory"""
    if not os.path.exists(memory_path):
        print(f"   No memory file found at {memory_path}")
        return df

    with open(memory_path, 'r') as f:
        memory = json.load(f)

    successful_features = memory.get('successful_features', [])
    if not successful_features:
        print("   No successful features in memory")
        return df

    print(f"   Applying {len(successful_features)} discovered features...")

    # Safe execution environment
    safe_globals = {'pd': pd, 'np': np}

    for feat in successful_features:
        code = feat.get('code', '')
        try:
            exec(code, safe_globals, {'df': df})
            print(f"      Applied: {feat.get('columns', ['unknown'])}")
        except Exception as e:
            print(f"      Failed: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Tune LightGBM hyperparameters')
    parser.add_argument('--with-features', action='store_true',
                        help='Apply agent-discovered features before tuning')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--memory-path', type=str,
                        default='outputs/logs/agent_memory.json',
                        help='Path to agent memory file')
    args = parser.parse_args()

    print("=" * 70)
    print("LightGBM Hyperparameter Tuning for Ames Housing")
    if args.with_features:
        print("Mode: WITH agent-discovered features")
    else:
        print("Mode: Minimal baseline only")
    print("=" * 70)

    # Load data
    print("\n1. Loading and preprocessing data...")
    loader = AmesDataLoader('data')
    train = loader.load_train()

    # Extract target
    target = train['SalePrice']
    target_log = np.log1p(target)

    # Preprocess with minimal preprocessor
    preprocessor = MinimalPreprocessor()
    X_train = preprocessor.fit_transform(train)

    # Drop target and ID
    X_train = X_train.drop(['Id', 'SalePrice'], axis=1, errors='ignore')

    print(f"   Base shape: {X_train.shape}")

    # Optionally apply discovered features
    if args.with_features:
        X_train = load_and_apply_features(X_train, args.memory_path)
        print(f"   Final shape: {X_train.shape}")

    print(f"   Target: log-transformed SalePrice")

    # Run hyperparameter tuning
    print("\n2. Running hyperparameter optimization...")
    print(f"   Trials: {args.n_trials}")
    print("   This will take several minutes...")

    tuner = LGBMTuner(
        X=X_train,
        y=target_log,
        n_folds=5,
        n_trials=args.n_trials,
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
