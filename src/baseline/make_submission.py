"""
End-to-end submission pipeline

Uses agent-discovered features + model stacking for final prediction.

Workflow:
1. Load train/test data
2. Remove outliers from training
3. Apply minimal preprocessing
4. Apply agent-discovered features (from memory)
5. Select best features
6. Run model stacking
7. Generate submission.csv

Usage:
    python -m src.baseline.make_submission [OPTIONS]

    --skip-features     Skip applying agent features (use raw features only)
    --skip-selection    Skip feature selection
    --output PATH       Output path for submission (default: outputs/submission.csv)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.minimal_preprocessor import MinimalPreprocessor
from src.baseline.outlier_remover import OutlierRemover
from src.baseline.model_stacker import ModelStacker
from src.baseline.feature_selector import FeatureSelector
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import get_features_and_target


def load_agent_features(memory_path: str = 'outputs/logs/agent_memory.json') -> list:
    """
    Load successful features from agent memory.

    Args:
        memory_path: Path to agent memory JSON

    Returns:
        List of successful feature code snippets
    """
    if not os.path.exists(memory_path):
        print(f"   Warning: Agent memory not found at {memory_path}")
        return []

    with open(memory_path, 'r') as f:
        memory = json.load(f)

    successful = memory.get('successful_features', [])
    codes = [f['code'] for f in successful]
    print(f"   Loaded {len(codes)} agent-discovered features")

    return codes


def apply_features(df: pd.DataFrame, feature_codes: list, executor: CodeExecutor) -> pd.DataFrame:
    """
    Apply feature engineering code to dataframe.

    Args:
        df: Input dataframe
        feature_codes: List of code snippets
        executor: Code executor instance

    Returns:
        DataFrame with new features
    """
    for i, code in enumerate(feature_codes, 1):
        result_df, error = executor.execute(code, df)
        if error:
            print(f"   Warning: Feature {i} failed: {error[:50]}...")
        else:
            df = result_df

    return df


def main(
    skip_features: bool = False,
    skip_selection: bool = False,
    output_path: str = 'outputs/submission.csv'
):
    """
    Run end-to-end submission pipeline.

    Args:
        skip_features: Skip applying agent features
        skip_selection: Skip feature selection
        output_path: Output path for submission CSV
    """
    print("=" * 60)
    print("Submission Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = AmesDataLoader()
    train_df = loader.load_train()
    test_df = loader.load_test()
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")

    # Store test IDs for submission
    test_ids = test_df['Id'].values

    # 2. Remove outliers from training
    print("\n2. Removing outliers...")
    outlier_remover = OutlierRemover()
    train_df = outlier_remover.remove_known_outliers(train_df)
    print(f"   Train shape after outlier removal: {train_df.shape}")

    # 3. Preprocess
    print("\n3. Preprocessing...")
    preprocessor = MinimalPreprocessor()
    target = train_df['SalePrice'].copy()

    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    print(f"   Processed train shape: {train_processed.shape}")
    print(f"   Processed test shape: {test_processed.shape}")

    # 4. Apply agent features
    if not skip_features:
        print("\n4. Applying agent-discovered features...")
        feature_codes = load_agent_features()

        if feature_codes:
            executor = CodeExecutor()
            train_processed = apply_features(train_processed, feature_codes, executor)
            test_processed = apply_features(test_processed, feature_codes, executor)
            print(f"   Train shape after features: {train_processed.shape}")
            print(f"   Test shape after features: {test_processed.shape}")
    else:
        print("\n4. Skipping agent features (--skip-features)")

    # 5. Extract features
    print("\n5. Extracting features...")
    X_train, _ = get_features_and_target(train_processed)
    X_test, _ = get_features_and_target(test_processed, target_col=None)

    # Align columns (test may be missing some features)
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"   Features: {len(common_cols)}")

    # Log-transform target
    y_train = np.log1p(target)

    # 6. Feature selection
    if not skip_selection:
        print("\n6. Feature selection...")
        selector = FeatureSelector(threshold=0)
        selected = selector.select_by_importance(X_train, y_train)

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        print(f"   Selected features: {len(selected)}")
    else:
        print("\n6. Skipping feature selection (--skip-selection)")

    # 7. Three-layer model stacking
    print("\n7. Three-layer model stacking...")
    stacker = ModelStacker(n_folds=5)
    predictions, layer_preds, oof_scores = stacker.fit_predict(X_train, y_train, X_test)

    # Transform predictions back from log scale
    predictions = np.expm1(predictions)

    # Clip negative predictions (shouldn't happen, but just in case)
    predictions = np.maximum(predictions, 0)

    # 8. Create submission
    print("\n8. Creating submission...")
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"   Submission saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Predictions: {len(predictions)}")
    print(f"   Mean: ${predictions.mean():,.0f}")
    print(f"   Median: ${np.median(predictions):,.0f}")
    print(f"   Min: ${predictions.min():,.0f}")
    print(f"   Max: ${predictions.max():,.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Kaggle submission')
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip applying agent features'
    )
    parser.add_argument(
        '--skip-selection',
        action='store_true',
        help='Skip feature selection'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/submission.csv',
        help='Output path for submission'
    )
    args = parser.parse_args()

    main(
        skip_features=args.skip_features,
        skip_selection=args.skip_selection,
        output_path=args.output
    )
