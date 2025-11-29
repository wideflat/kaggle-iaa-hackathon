"""
Phase A0: Minimal Baseline (Updated)
Goal: Get a working end-to-end pipeline with proper CV and RMSLE metric.
Minimum Requirements:
- Use RMSLE (Root Mean Squared Logarithmic Error) metric
- Use K-Fold Cross-Validation for robust evaluation
- Log-transform target variable
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import os


def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    print("=" * 60)
    print("Phase A0: Minimal Baseline Training")
    print("=" * 60)

    # Define paths
    data_dir = 'data'
    output_dir = 'outputs/predictions'

    # Load data
    print("\n1. Loading data...")
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
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

    # Minimal preprocessing: Select only numeric features
    print("\n3. Preprocessing (minimal)...")
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [f for f in numeric_features if f not in ['Id', 'SalePrice']]
    print(f"   Found {len(numeric_features)} numeric features")

    # Create feature matrices
    X_train = train[numeric_features].copy()
    X_test = test[numeric_features].copy()

    # Fill missing values with median
    print("\n4. Handling missing values...")
    missing_counts = X_train.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    if len(features_with_missing) > 0:
        print(f"   Features with missing values: {len(features_with_missing)}")
        for feature, count in features_with_missing.head(5).items():
            print(f"   - {feature}: {count} missing")
        if len(features_with_missing) > 5:
            print(f"   ... and {len(features_with_missing) - 5} more")

    # Fill with median
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    print(f"   Filled missing values with median")

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
        model = lgb.LGBMRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )
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
    final_model = lgb.LGBMRegressor(
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
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

    output_path = os.path.join(output_dir, 'baseline_phase_a0.csv')
    submission.to_csv(output_path, index=False)
    print(f"   Predictions saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE A0 COMPLETE!")
    print("=" * 60)
    print(f"Metric: RMSLE (Root Mean Squared Logarithmic Error)")
    print(f"Cross-Validation: {n_folds}-Fold")
    print(f"Mean CV RMSLE: {mean_rmsle:.5f} (± {std_rmsle:.5f})")
    print(f"Features used: {len(numeric_features)} (numeric only)")
    print(f"Predictions saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
