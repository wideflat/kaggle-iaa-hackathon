"""
Iterative Feature Engineering Agent
Phase B4-B5: Memory system + iteration loop

Usage:
    python -m src.agent.iterative_agent [--iterations N] [--clear]
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.preprocessor import AmesPreprocessor
from src.agent.gemini_client import GeminiClient
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import FeatureEvaluator, get_features_and_target
from src.agent.memory import AgentMemory


def run_iteration(
    iteration: int,
    df: 'pd.DataFrame',
    target: 'pd.Series',
    data_desc: str,
    column_info: str,
    gemini: GeminiClient,
    executor: CodeExecutor,
    evaluator: FeatureEvaluator,
    memory: AgentMemory,
    current_best_rmsle: float
) -> tuple[float, bool]:
    """
    Run a single iteration of feature generation

    Returns:
        Tuple of (new_rmsle, is_improvement)
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")

    # Get already tried features to avoid duplicates
    tried_features = memory.get_all_tried_codes()
    print(f"   Previously tried features: {len(tried_features)}")

    # Generate new feature
    print("   Generating feature with Gemini...")
    try:
        generated_code = gemini.generate_feature(
            data_desc,
            column_info,
            existing_features=tried_features
        )
    except Exception as e:
        print(f"   ERROR calling Gemini: {e}")
        memory.add_failed_feature("", str(e))
        return current_best_rmsle, False

    print(f"   Generated code:")
    for line in generated_code.split('\n')[:5]:  # Show first 5 lines
        print(f"      {line}")
    if len(generated_code.split('\n')) > 5:
        print(f"      ...")

    # Execute the code
    print("   Executing code...")
    new_df, error = executor.execute(generated_code, df)

    if error:
        print(f"   EXECUTION FAILED: {error[:100]}...")
        memory.add_failed_feature(generated_code, error)
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False)
        return current_best_rmsle, False

    # Check new columns
    new_cols = executor.get_new_columns(df, new_df)
    if not new_cols:
        print("   No new columns created (feature already exists)")
        memory.add_failed_feature(generated_code, "No new columns created")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False)
        return current_best_rmsle, False

    print(f"   New columns: {new_cols}")

    # Evaluate
    print("   Evaluating...")
    X_new, _ = get_features_and_target(new_df)
    new_rmsle = evaluator.evaluate(X_new, target, verbose=False)

    # Compare
    improvement = current_best_rmsle - new_rmsle
    is_better = improvement > 0

    print(f"   Current best RMSLE: {current_best_rmsle:.5f}")
    print(f"   New RMSLE:          {new_rmsle:.5f}")
    print(f"   Delta:              {improvement:+.5f}")

    if is_better:
        print(f"   >>> IMPROVEMENT! +{improvement/current_best_rmsle*100:.2f}%")
        memory.add_successful_feature(generated_code, new_cols, new_rmsle, improvement)
        memory.log_iteration(iteration, new_rmsle, generated_code, success=True)
        return new_rmsle, True
    else:
        print(f"   >>> No improvement")
        memory.add_failed_feature(generated_code, f"No improvement: {new_rmsle:.5f} vs {current_best_rmsle:.5f}")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False)
        return current_best_rmsle, False


def main(n_iterations: int = 10, clear_memory: bool = False):
    """Run iterative feature engineering agent"""
    print("=" * 60)
    print(f"Iterative Feature Engineering Agent")
    print(f"Iterations: {n_iterations}")
    print("=" * 60)

    # Initialize memory
    memory = AgentMemory()
    if clear_memory:
        print("\n   Clearing previous memory...")
        memory.clear()

    # Load data
    print("\n1. Loading data...")
    loader = AmesDataLoader()
    train_df = loader.load_train()
    data_desc = loader.load_description()
    column_info = ', '.join(train_df.columns.tolist())
    print(f"   Train shape: {train_df.shape}")

    # Preprocess
    print("\n2. Preprocessing...")
    preprocessor = AmesPreprocessor()
    target = train_df['SalePrice'].copy()
    train_processed = preprocessor.fit_transform(train_df)
    print(f"   Processed shape: {train_processed.shape}")

    # Get baseline
    print("\n3. Calculating baseline RMSLE...")
    evaluator = FeatureEvaluator(n_folds=5)
    X_baseline, _ = get_features_and_target(train_processed)
    baseline_rmsle = evaluator.evaluate(X_baseline, target, verbose=True)
    print(f"\n   >>> Baseline RMSLE: {baseline_rmsle:.5f}")

    memory.set_baseline(baseline_rmsle)

    # Initialize Gemini
    print("\n4. Initializing Gemini client...")
    try:
        gemini = GeminiClient()
        print("   Gemini client ready")
    except ValueError as e:
        print(f"   ERROR: {e}")
        return

    # Initialize executor
    executor = CodeExecutor()

    # Get current best (may be from previous session)
    current_best_rmsle = memory.memory['best_rmsle'] or baseline_rmsle
    print(f"\n   Starting from best RMSLE: {current_best_rmsle:.5f}")

    # Apply any previously successful features
    accumulated_df = train_processed.copy()
    successful_codes = memory.get_successful_codes()
    if successful_codes:
        print(f"\n   Applying {len(successful_codes)} previously successful features...")
        for code in successful_codes:
            result_df, error = executor.execute(code, accumulated_df)
            if not error:
                accumulated_df = result_df

    # Run iterations
    print("\n5. Running iterations...")
    successes = 0
    failures = 0

    for i in range(1, n_iterations + 1):
        new_rmsle, is_success = run_iteration(
            iteration=i,
            df=accumulated_df,
            target=target,
            data_desc=data_desc,
            column_info=column_info,
            gemini=gemini,
            executor=executor,
            evaluator=evaluator,
            memory=memory,
            current_best_rmsle=current_best_rmsle
        )

        if is_success:
            current_best_rmsle = new_rmsle
            successes += 1
            # Apply successful feature to accumulated df for next iteration
            successful_code = memory.memory['successful_features'][-1]['code']
            result_df, _ = executor.execute(successful_code, accumulated_df)
            if result_df is not None:
                accumulated_df = result_df
        else:
            failures += 1

    # Final summary
    print("\n" + "=" * 60)
    print("ITERATIVE AGENT COMPLETE")
    print("=" * 60)
    memory.print_summary()

    print(f"\nThis session:")
    print(f"   Iterations run:    {n_iterations}")
    print(f"   Successful:        {successes}")
    print(f"   Failed:            {failures}")
    print(f"   Success rate:      {successes/n_iterations*100:.1f}%")

    # Show successful features
    if memory.memory['successful_features']:
        print(f"\nSuccessful features found:")
        for i, f in enumerate(memory.memory['successful_features'], 1):
            print(f"   {i}. {f['columns']} (improvement: {f['improvement']:.5f})")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run iterative feature engineering agent')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear previous memory and start fresh')
    args = parser.parse_args()

    main(n_iterations=args.iterations, clear_memory=args.clear)
