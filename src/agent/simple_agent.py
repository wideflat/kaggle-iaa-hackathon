"""
Simple Feature Engineering Agent - Single Iteration
Phase B0-B3: Minimal viable agent

Usage:
    python -m src.agent.simple_agent
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.preprocessor import AmesPreprocessor
from src.agent.gemini_client import GeminiClient
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import FeatureEvaluator, get_features_and_target


def main():
    """Run single iteration of feature engineering agent"""
    print("=" * 60)
    print("Feature Engineering Agent - Single Iteration")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = AmesDataLoader()
    train_df = loader.load_train()
    data_desc = loader.load_description()
    print(f"   Train shape: {train_df.shape}")
    print(f"   Data description: {len(data_desc)} characters")

    # 2. Preprocess with baseline preprocessor
    print("\n2. Preprocessing with AmesPreprocessor...")
    preprocessor = AmesPreprocessor()

    # Keep target separate
    target = train_df['SalePrice'].copy()

    # Preprocess
    train_processed = preprocessor.fit_transform(train_df)
    print(f"   Processed shape: {train_processed.shape}")

    # 3. Get baseline features and target
    print("\n3. Extracting features...")
    X_baseline, _ = get_features_and_target(train_processed)
    print(f"   Baseline features: {X_baseline.shape[1]}")

    # 4. Calculate baseline RMSLE
    print("\n4. Calculating baseline RMSLE...")
    evaluator = FeatureEvaluator(n_folds=5)
    baseline_rmsle = evaluator.evaluate(X_baseline, target, verbose=True)
    print(f"\n   >>> Baseline RMSLE: {baseline_rmsle:.5f}")

    # 5. Generate feature with Gemini
    print("\n5. Generating feature with Gemini...")
    try:
        gemini = GeminiClient()
    except ValueError as e:
        print(f"   ERROR: {e}")
        print("\n   To fix: Create a .env file in the project root with:")
        print("   GEMINI_API_KEY=your_api_key_here")
        return

    column_info = ', '.join(train_df.columns.tolist())
    print("   Calling Gemini API...")

    generated_code = gemini.generate_feature(data_desc, column_info)
    print(f"\n   Generated Code:")
    print("   " + "-" * 50)
    for line in generated_code.split('\n'):
        print(f"   {line}")
    print("   " + "-" * 50)

    # 6. Execute generated code
    print("\n6. Executing generated code...")
    executor = CodeExecutor()

    # Execute on the processed dataframe
    new_df, error = executor.execute(generated_code, train_processed)

    if error:
        print(f"\n   EXECUTION ERROR:")
        print(f"   {error}")
        print("\n   Agent failed to create a valid feature.")
        return

    # Check what new columns were added
    new_cols = executor.get_new_columns(train_processed, new_df)
    print(f"   New columns created: {new_cols}")

    # 7. Evaluate new features
    print("\n7. Evaluating with new feature...")
    X_new, _ = get_features_and_target(new_df)
    print(f"   New feature count: {X_new.shape[1]}")

    new_rmsle = evaluator.evaluate(X_new, target, verbose=True)
    print(f"\n   >>> New RMSLE: {new_rmsle:.5f}")

    # 8. Compare results
    print("\n8. Comparison Results:")
    print("   " + "=" * 50)
    result = evaluator.compare(baseline_rmsle, new_rmsle)

    print(f"   Baseline RMSLE:  {result['baseline_rmsle']:.5f}")
    print(f"   New RMSLE:       {result['new_rmsle']:.5f}")
    print(f"   Delta:           {result['delta']:.5f}")
    print(f"   Change:          {result['percent_change']:+.3f}%")

    if result['is_improvement']:
        print(f"\n   >>> IMPROVEMENT! The new feature helped.")
    else:
        print(f"\n   >>> No improvement. The feature did not help.")

    print("   " + "=" * 50)

    # Summary
    print("\n" + "=" * 60)
    print("AGENT RUN COMPLETE")
    print("=" * 60)
    print(f"Feature generated: {new_cols}")
    print(f"Result: {'IMPROVED' if result['is_improvement'] else 'No improvement'}")
    print(f"RMSLE change: {result['percent_change']:+.3f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
