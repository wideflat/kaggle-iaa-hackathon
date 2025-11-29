"""
Phase A0: Minimal Baseline
Goal: Get a working end-to-end pipeline that produces a submission file.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os


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
    train_ids = train['Id']
    test_ids = test['Id']
    print(f"   Target (SalePrice) stats:")
    print(f"   - Mean: ${target.mean():,.2f}")
    print(f"   - Median: ${target.median():,.2f}")
    print(f"   - Std: ${target.std():,.2f}")

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

    # Train model
    print("\n5. Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, target)
    print("   Model trained successfully")

    # Evaluate on training set (sanity check)
    print("\n6. Evaluating on training set...")
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(target, train_pred)
    train_rmse = np.sqrt(train_mse)
    print(f"   Training RMSE: ${train_rmse:,.2f}")

    # Check if within expected range
    if train_rmse < 50000:
        print("   ✓ Success: RMSE < $50,000 (Phase A0 goal met)")
    else:
        print("   ⚠ Warning: RMSE >= $50,000 (higher than expected)")

    # Make predictions on test set
    print("\n7. Making predictions on test set...")
    test_pred = model.predict(X_test)
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
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Features used: {len(numeric_features)}")
    print(f"Predictions saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
