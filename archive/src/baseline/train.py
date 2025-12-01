"""
Phase A1: Standard Preprocessing
Goal: Improve baseline with proper preprocessing and feature engineering
Features:
- RMSLE metric with log-transformed target
- K-Fold Cross-Validation
- Semantic imputation for missing values
- Ordinal encoding for quality features
- One-hot encoding for categorical features
- Engineered features (TotalSF, HouseAge, RemodAge, TotalBath, etc.)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import os
import sys
import json
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.preprocessor import AmesPreprocessor


def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_tuned_params():
    """Load tuned hyperparameters if available"""
    params_path = 'outputs/models/best_lgbm_params.json'
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return json.load(f)
    return None


def main(use_tuned=False):
    phase = "A3" if use_tuned else "A1"
    print("=" * 60)
    print(f"Phase {phase}: {'Hyperparameter Tuned' if use_tuned else 'Standard Preprocessing'} Training")
    print("=" * 60)

    # Define paths
    data_dir = 'data'
    output_dir = 'outputs/predictions'

    # Load data
    print("\n1. Loading data...")
    loader = AmesDataLoader(data_dir)
    train = loader.load_train()
    test = loader.load_test()
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")

    # Extract target and IDs
    print("\n2. Extracting target and IDs...")
    target = train['SalePrice']
    target_log = np.log1p(target)  # Log transformation for RMSLE
    train_ids = train['Id']
    test_ids = test['Id']
    print(f"   Target (SalePrice) stats:")
    print(f"   - Mean: ${target.mean():,.2f}")
    print(f"   - Median: ${target.median():,.2f}")
    print(f"   - Std: ${target.std():,.2f}")
    print(f"   Target (log-transformed) stats:")
    print(f"   - Mean: {target_log.mean():.4f}")
    print(f"   - Std: {target_log.std():.4f}")

    # Preprocessing with AmesPreprocessor
    print("\n3. Preprocessing with feature engineering...")
    preprocessor = AmesPreprocessor()

    # Separate features from target/ID
    X_train_raw = train.drop(['Id', 'SalePrice'], axis=1)
    X_test_raw = test.drop(['Id'], axis=1)

    # Fit and transform training data
    print("   - Fitting preprocessor on training data...")
    X_train = preprocessor.fit_transform(X_train_raw)
    print(f"   - Training data shape after preprocessing: {X_train.shape}")

    # Transform test data
    print("   - Transforming test data...")
    X_test = preprocessor.transform(X_test_raw)
    print(f"   - Test data shape after preprocessing: {X_test.shape}")

    print(f"   - Total features after preprocessing: {X_train.shape[1]}")
    print(f"   - Features include:")
    print(f"     * Engineered: TotalSF, HouseAge, RemodAge, TotalBath, PorchArea")
    print(f"     * Ordinal encoded quality features")
    print(f"     * One-hot encoded categorical features")

    # Model parameters
    print("\n4. Configuring model parameters...")
    if use_tuned:
        tuned_params = load_tuned_params()
        if tuned_params is None:
            print("   ⚠ Warning: No tuned parameters found. Using defaults.")
            print("   Run: venv/bin/python src/baseline/tune_hyperparameters.py")
            model_params = {'n_estimators': 100, 'random_state': 42, 'verbosity': -1}
        else:
            model_params = tuned_params
            print("   ✓ Using tuned hyperparameters from Optuna")
            print(f"   Key params: n_estimators={model_params.get('n_estimators')}, "
                  f"learning_rate={model_params.get('learning_rate', 'N/A'):.4f}")
    else:
        model_params = {'n_estimators': 100, 'random_state': 42, 'verbosity': -1}
        print("   Using default parameters (Phase A1 baseline)")

    # Cross-Validation
    print("\n5. Running K-Fold Cross-Validation...")
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores_rmsle = []
    cv_scores_rmse_log = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\n   Fold {fold}/{n_folds}:")

        # Split data
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = target_log.iloc[train_idx]
        y_val_fold = target_log.iloc[val_idx]
        y_val_original = target.iloc[val_idx]

        # Train model on log-transformed target
        model = lgb.LGBMRegressor(**model_params)

        # With tuned params, use early stopping
        if use_tuned and 'n_estimators' in model_params:
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model.fit(X_train_fold, y_train_fold)
        fold_models.append(model)

        # Predict (in log space)
        val_pred_log = model.predict(X_val_fold)

        # Transform back to original scale
        val_pred = np.expm1(val_pred_log)

        # Calculate RMSLE (on original scale)
        fold_rmsle = rmsle(y_val_original, val_pred)
        cv_scores_rmsle.append(fold_rmsle)

        # Calculate RMSE on log scale (equivalent to RMSLE)
        fold_rmse_log = rmse(y_val_fold, val_pred_log)
        cv_scores_rmse_log.append(fold_rmse_log)

        print(f"      RMSLE: {fold_rmsle:.5f}")
        print(f"      RMSE (log): {fold_rmse_log:.5f}")

    # Summary statistics
    mean_rmsle = np.mean(cv_scores_rmsle)
    std_rmsle = np.std(cv_scores_rmsle)

    print(f"\n   {'='*50}")
    print(f"   Cross-Validation Results ({n_folds}-Fold):")
    print(f"   {'='*50}")
    print(f"   Mean RMSLE: {mean_rmsle:.5f} (± {std_rmsle:.5f})")
    print(f"   Individual Folds: {[f'{s:.5f}' for s in cv_scores_rmsle]}")
    print(f"   {'='*50}")

    # Train final model on full training data
    print("\n6. Training final model on full training data...")
    final_model = lgb.LGBMRegressor(**model_params)
    final_model.fit(X_train, target_log)
    print("   Final model trained successfully")

    # Make predictions on test set
    print("\n7. Making predictions on test set...")
    test_pred_log = final_model.predict(X_test)
    test_pred = np.expm1(test_pred_log)  # Transform back from log space
    print(f"   Predictions stats:")
    print(f"   - Mean: ${test_pred.mean():,.2f}")
    print(f"   - Median: ${np.median(test_pred):,.2f}")
    print(f"   - Min: ${test_pred.min():,.2f}")
    print(f"   - Max: ${test_pred.max():,.2f}")

    # Save predictions
    print("\n8. Saving predictions...")
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_pred
    })

    output_filename = f'baseline_phase_{phase.lower()}.csv'
    output_path = os.path.join(output_dir, output_filename)
    submission.to_csv(output_path, index=False)
    print(f"   Predictions saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"PHASE {phase} COMPLETE!")
    print("=" * 60)
    print(f"Metric: RMSLE (Root Mean Squared Logarithmic Error)")
    print(f"Cross-Validation: {n_folds}-Fold")
    print(f"Mean CV RMSLE: {mean_rmsle:.5f} (± {std_rmsle:.5f})")
    print(f"Features used: {X_train.shape[1]} (with preprocessing & engineering)")
    if use_tuned:
        print(f"Hyperparameters: Tuned with Optuna" if tuned_params else "Default")
    print(f"Predictions saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Ames Housing baseline model')
    parser.add_argument('--use-tuned', action='store_true',
                        help='Use tuned hyperparameters from Optuna')
    args = parser.parse_args()

    main(use_tuned=args.use_tuned)
