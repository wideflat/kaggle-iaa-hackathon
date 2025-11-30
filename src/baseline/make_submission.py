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
import shutil
from datetime import datetime
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


def apply_features(
    df: pd.DataFrame,
    feature_codes: list,
    executor: CodeExecutor,
    dataset_name: str = "data"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply feature engineering code to dataframe.

    Args:
        df: Input dataframe
        feature_codes: List of code snippets
        executor: Code executor instance
        dataset_name: Name for logging (e.g., "train", "test")

    Returns:
        Tuple of (DataFrame with new features, list of new column names)
    """
    applied_cols = []
    original_cols = set(df.columns)

    for i, code in enumerate(feature_codes, 1):
        result_df, error = executor.execute(code, df)
        if error:
            print(f"   Warning: Feature {i} failed on {dataset_name}: {error[:50]}...")
        else:
            new_cols = list(set(result_df.columns) - original_cols)
            applied_cols.extend(new_cols)
            original_cols = set(result_df.columns)
            df = result_df

    return df, applied_cols


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

    # Store test IDs for output files
    test_ids = test_df['Id'].values

    # 2. Remove outliers from training
    print("\n2. Removing outliers...")
    outlier_remover = OutlierRemover()
    train_df = outlier_remover.remove_known_outliers(train_df)
    print(f"   Train shape after outlier removal: {train_df.shape}")

    # Store train IDs after outlier removal (for OOF predictions)
    train_ids = train_df['Id'].values

    # 3. Preprocess
    print("\n3. Preprocessing...")
    preprocessor = MinimalPreprocessor()
    target = train_df['SalePrice'].copy()

    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    print(f"   Processed train shape: {train_processed.shape}")
    print(f"   Processed test shape: {test_processed.shape}")

    # Store original columns before applying features (for verification later)
    original_cols = set(train_processed.columns)

    # 4. Apply agent features
    if not skip_features:
        print("\n4. Applying agent-discovered features...")
        feature_codes = load_agent_features()

        if feature_codes:
            executor = CodeExecutor()
            train_processed, train_new_cols = apply_features(
                train_processed, feature_codes, executor, "train"
            )
            test_processed, test_new_cols = apply_features(
                test_processed, feature_codes, executor, "test"
            )
            print(f"   Train shape after features: {train_processed.shape}")
            print(f"   Test shape after features: {test_processed.shape}")
            print(f"   Train new columns: {len(train_new_cols)}")
            print(f"   Test new columns: {len(test_new_cols)}")

            # Show which features are in both
            common_new = set(train_new_cols) & set(test_new_cols)
            print(f"   Common new features: {len(common_new)}")
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

    # Verify which engineered features made it to the stacker
    engineered_in_final = [c for c in common_cols if c not in original_cols]
    if engineered_in_final:
        print(f"\n   === ENGINEERED FEATURES PASSED TO STACKER ({len(engineered_in_final)}) ===")
        for col in sorted(engineered_in_final):
            print(f"   - {col}")
    else:
        print("\n   WARNING: No engineered features made it to stacker!")

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

    # 8. Create submission and save all outputs
    print("\n8. Creating submission...")

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/submissions/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output folder: {output_dir}")

    # Save submission.csv
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    submission.to_csv(f"{output_dir}/submission.csv", index=False)
    print(f"   Saved: submission.csv")

    # Save OOF predictions - Layer 1
    oof_layer1 = pd.DataFrame({'Id': train_ids})
    for model_name, oof_pred in layer_preds['oof_layer1'].items():
        oof_layer1[model_name] = oof_pred
    oof_layer1.to_csv(f"{output_dir}/oof_layer1.csv", index=False)
    print(f"   Saved: oof_layer1.csv (5 base models)")

    # Save OOF predictions - Layer 2 & 3
    oof_layer2 = pd.DataFrame({
        'Id': train_ids,
        'weighted': layer_preds['oof_layer2_weighted'],
        'stacking': layer_preds['oof_layer2_stacking'],
        'final': layer_preds['oof_layer3_final']
    })
    oof_layer2.to_csv(f"{output_dir}/oof_layer2.csv", index=False)
    print(f"   Saved: oof_layer2.csv (weighted, stacking, final)")

    # Save test predictions - Layer 1
    test_layer1 = pd.DataFrame({'Id': test_ids})
    for model_name, test_pred in layer_preds['layer1'].items():
        test_layer1[model_name] = test_pred
    test_layer1.to_csv(f"{output_dir}/test_layer1.csv", index=False)
    print(f"   Saved: test_layer1.csv (5 base models)")

    # Save test predictions - Layer 2 & 3
    test_layer2 = pd.DataFrame({
        'Id': test_ids,
        'weighted': layer_preds['layer2_weighted'],
        'stacking': layer_preds['layer2_stacking'],
        'final': layer_preds['layer3_final']
    })
    test_layer2.to_csv(f"{output_dir}/test_layer2.csv", index=False)
    print(f"   Saved: test_layer2.csv (weighted, stacking, final)")

    # Copy supporting files
    files_to_copy = [
        ('outputs/models/best_lgbm_params.json', 'best_lgbm_params.json'),
        ('outputs/models/best_xgb_params.json', 'best_xgb_params.json'),
        ('outputs/logs/agent_memory.json', 'agent_memory.json'),
        ('outputs/logs/agent.log', 'agent.log'),
        ('outputs/logs/progress_plot.png', 'progress_plot.png'),
    ]

    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, f"{output_dir}/{dst}")
            print(f"   Copied: {dst}")

    # Also save to the legacy output path for compatibility
    submission.to_csv(output_path, index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Output folder: {output_dir}")
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
