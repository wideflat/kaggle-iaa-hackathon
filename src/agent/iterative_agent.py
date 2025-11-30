"""
Iterative Feature Engineering Agent
Phase B4-B5: Memory system + iteration loop
Phase B7: SHAP-based feedback loop
Phase B9: Batch mode + advanced strategies

Usage:
    python -m src.agent.iterative_agent [OPTIONS]

Arguments:
    --iterations, -n INT    Number of iterations/batches to run (default: 10)
    --clear                 Clear previous memory and start fresh
    --visualize, -v         Generate visualization after completion
                            (plot, table, and HTML report)
    --feedback, -f          Enable SHAP-based feedback loop (B7)
                            Uses feature importance to guide generation
    --batch, -b             Enable batch mode (generate multiple features per iteration)
    --batch-size INT        Number of features per batch (default: 5)

Examples:
    # Run 10 iterations (single feature mode)
    python -m src.agent.iterative_agent

    # Run 5 iterations with visualization
    python -m src.agent.iterative_agent -n 5 --visualize

    # Run with SHAP feedback (recommended)
    python -m src.agent.iterative_agent -n 10 --feedback

    # Batch mode: 3 batches x 5 features = 15 features total
    python -m src.agent.iterative_agent -n 3 --batch --clear

    # Batch mode with custom size: 2 batches x 10 features = 20 features
    python -m src.agent.iterative_agent -n 2 --batch --batch-size 10

    # Full featured run
    python -m src.agent.iterative_agent -n 5 -v -f --batch --clear

Output:
    - Memory saved to: outputs/logs/agent_memory.json
    - Plot saved to: outputs/logs/progress_plot.png (with --visualize)
    - Report saved to: outputs/logs/progress_report.html (with --visualize)
"""

import os
import sys
import argparse

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.baseline.data_loader import AmesDataLoader
from src.baseline.minimal_preprocessor import MinimalPreprocessor
from src.baseline.outlier_remover import OutlierRemover
from src.agent.gemini_client import GeminiClient
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import FeatureEvaluator, get_features_and_target
from src.agent.memory import AgentMemory


def run_iteration(
    iteration: int,
    df: pd.DataFrame,
    target: pd.Series,
    data_desc: str,
    column_info: str,
    gemini: GeminiClient,
    executor: CodeExecutor,
    evaluator: FeatureEvaluator,
    memory: AgentMemory,
    current_best_rmsle: float,
    use_feedback: bool = False,
    accumulated_df: pd.DataFrame = None
) -> tuple[float, bool]:
    """
    Run a single iteration of feature generation

    Args:
        iteration: Current iteration number
        df: Preprocessed dataframe
        target: Target series (SalePrice)
        data_desc: Data description text
        column_info: Comma-separated column names
        gemini: Gemini client instance
        executor: Code executor instance
        evaluator: Feature evaluator instance
        memory: Agent memory instance
        current_best_rmsle: Current best RMSLE score
        use_feedback: If True, use SHAP-based feedback (B7)
        accumulated_df: The current accumulated dataframe with all successful features
                       (used to recompute SHAP after rejection)

    Returns:
        Tuple of (new_rmsle, is_improvement)
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}" + (" [SHAP Feedback]" if use_feedback else ""))
    print(f"{'='*60}")

    # Get already tried features to avoid duplicates
    tried_features = memory.get_all_tried_codes()
    print(f"   Previously tried features: {len(tried_features)}")

    # Generate new feature (with or without SHAP feedback)
    print("   Generating feature with Gemini...")
    try:
        if use_feedback:
            # Get SHAP insights from evaluator
            shap_summary = evaluator.get_shap_summary(top_n=10)
            feature_insights = evaluator.get_feature_insights(top_n=5)
            print(f"   Using SHAP feedback (top feature: {feature_insights.get('top_features', ['N/A'])[0]})")

            # Store SHAP insights in memory
            memory.log_shap_insights(
                iteration=iteration,
                top_features=feature_insights.get('top_features', []),
                importance_scores=feature_insights.get('importance_scores', {}),
                total_features=feature_insights.get('total_features', 0)
            )

            generated_code = gemini.generate_feature_with_feedback(
                data_desc,
                column_info,
                shap_summary=shap_summary,
                feature_insights=feature_insights,
                existing_features=tried_features
            )
        else:
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
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=None)
        return current_best_rmsle, False

    # Check new columns
    new_cols = executor.get_new_columns(df, new_df)
    if not new_cols:
        print("   No new columns created (feature already exists)")
        memory.add_failed_feature(generated_code, "No new columns created")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=None)
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
        memory.log_iteration(iteration, new_rmsle, generated_code, success=True, columns=new_cols, prev_rmsle=current_best_rmsle)
        return new_rmsle, True
    else:
        print(f"   >>> No improvement")
        memory.add_failed_feature(generated_code, f"No improvement: {new_rmsle:.5f} vs {current_best_rmsle:.5f}")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=new_cols, prev_rmsle=current_best_rmsle)

        # CRITICAL FIX: Recompute SHAP on accumulated_df (not rejected features)
        # This ensures next iteration's SHAP feedback only references existing columns
        if use_feedback and accumulated_df is not None:
            print("   Recomputing SHAP on accumulated features...")
            X_current, _ = get_features_and_target(accumulated_df)
            evaluator.recompute_shap_for_features(X_current, target)

        return current_best_rmsle, False


def run_batch_iteration(
    batch_num: int,
    df: pd.DataFrame,
    target: pd.Series,
    data_desc: str,
    column_info: str,
    gemini: GeminiClient,
    executor: CodeExecutor,
    evaluator: FeatureEvaluator,
    memory: AgentMemory,
    current_best_rmsle: float,
    accumulated_df: pd.DataFrame,
    batch_size: int = 5
) -> tuple[float, int, pd.DataFrame]:
    """
    Run a batch iteration: generate multiple features, evaluate each, keep all that improve.

    Args:
        batch_num: Current batch number
        df: Preprocessed dataframe
        target: Target series (SalePrice)
        data_desc: Data description text
        column_info: Comma-separated column names
        gemini: Gemini client instance
        executor: Code executor instance
        evaluator: Feature evaluator instance
        memory: Agent memory instance
        current_best_rmsle: Current best RMSLE score
        accumulated_df: Current accumulated dataframe
        batch_size: Number of features to generate per batch

    Returns:
        Tuple of (new_best_rmsle, num_successes, updated_accumulated_df)
    """
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num} [Generating {batch_size} features]")
    print(f"{'='*60}")

    # Get already tried features to avoid duplicates
    tried_features = memory.get_all_tried_codes()
    print(f"   Previously tried features: {len(tried_features)}")

    # Generate batch of features
    print(f"   Generating {batch_size} features with Gemini...")
    try:
        feature_codes = gemini.generate_feature_batch(
            data_desc,
            column_info,
            existing_features=tried_features,
            n_features=batch_size
        )
        print(f"   Generated {len(feature_codes)} features")
    except Exception as e:
        print(f"   ERROR calling Gemini: {e}")
        return current_best_rmsle, 0, accumulated_df

    # Evaluate each feature independently
    successes = 0
    best_rmsle = current_best_rmsle
    current_df = accumulated_df.copy()

    for i, code in enumerate(feature_codes, 1):
        print(f"\n   --- Feature {i}/{len(feature_codes)} ---")

        # Show first few lines of code
        code_preview = code.split('\n')[0]
        print(f"   Code: {code_preview}")

        # Execute the code
        new_df, error = executor.execute(code, current_df)

        if error:
            print(f"   FAILED: {error[:80]}...")
            memory.add_failed_feature(code, error)
            continue

        # Check new columns
        new_cols = executor.get_new_columns(current_df, new_df)
        if not new_cols:
            print("   FAILED: No new columns created")
            memory.add_failed_feature(code, "No new columns created")
            continue

        print(f"   New columns: {new_cols}")

        # Evaluate
        X_new, _ = get_features_and_target(new_df)
        new_rmsle = evaluator.evaluate(X_new, target, verbose=False)

        # Compare against current best
        improvement = best_rmsle - new_rmsle
        is_better = improvement > 0

        print(f"   RMSLE: {new_rmsle:.5f} (delta: {improvement:+.5f})")

        if is_better:
            print(f"   >>> KEPT! Improvement: {improvement/best_rmsle*100:.2f}%")
            memory.add_successful_feature(code, new_cols, new_rmsle, improvement)
            memory.log_iteration(
                batch_num * 100 + i,  # Unique iteration ID
                new_rmsle,
                code,
                success=True,
                columns=new_cols,
                prev_rmsle=best_rmsle
            )
            best_rmsle = new_rmsle
            current_df = new_df
            successes += 1
        else:
            print(f"   >>> Rejected")
            memory.add_failed_feature(code, f"No improvement: {new_rmsle:.5f} vs {best_rmsle:.5f}")

    print(f"\n   Batch {batch_num} complete: {successes}/{len(feature_codes)} features kept")
    return best_rmsle, successes, current_df


def main(
    n_iterations: int = 10,
    clear_memory: bool = False,
    visualize: bool = False,
    use_feedback: bool = False,
    batch_mode: bool = False,
    batch_size: int = 5
):
    """Run iterative feature engineering agent

    Args:
        n_iterations: Number of iterations (or batches in batch mode) to run
        clear_memory: If True, clear previous memory and start fresh
        visualize: If True, generate visualization after completion
        use_feedback: If True, use SHAP-based feedback loop (B7)
        batch_mode: If True, generate multiple features per iteration
        batch_size: Number of features per batch (only used in batch mode)
    """
    print("=" * 60)
    print(f"Iterative Feature Engineering Agent")
    if batch_mode:
        print(f"Mode: BATCH ({batch_size} features x {n_iterations} batches)")
    else:
        print(f"Iterations: {n_iterations}")
    if use_feedback:
        print(f"SHAP Feedback: Enabled")
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
    print(f"   Train shape: {train_df.shape}")

    # Remove outliers (famous Ames Housing outliers)
    print("\n1b. Removing outliers...")
    outlier_remover = OutlierRemover()
    train_df = outlier_remover.remove_known_outliers(train_df)
    print(f"   Train shape after outlier removal: {train_df.shape}")

    column_info = ', '.join(train_df.columns.tolist())

    # Preprocess (ultra-minimal: impute + label encode only)
    print("\n2. Preprocessing (ultra-minimal mode)...")
    preprocessor = MinimalPreprocessor()
    target = train_df['SalePrice'].copy()
    train_processed = preprocessor.fit_transform(train_df)
    print(f"   Processed shape: {train_processed.shape}")
    print("   NO advanced preprocessing - agent will discover features")

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
    total_features = 0

    if batch_mode:
        # Batch mode: generate multiple features per iteration
        for batch_num in range(1, n_iterations + 1):
            new_rmsle, batch_successes, accumulated_df = run_batch_iteration(
                batch_num=batch_num,
                df=accumulated_df,
                target=target,
                data_desc=data_desc,
                column_info=column_info,
                gemini=gemini,
                executor=executor,
                evaluator=evaluator,
                memory=memory,
                current_best_rmsle=current_best_rmsle,
                accumulated_df=accumulated_df,
                batch_size=batch_size
            )
            current_best_rmsle = new_rmsle
            successes += batch_successes
            total_features += batch_size
            failures = total_features - successes
    else:
        # Single feature mode
        total_features = n_iterations
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
                current_best_rmsle=current_best_rmsle,
                use_feedback=use_feedback,
                accumulated_df=accumulated_df  # Pass for SHAP recompute after rejection
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
    if batch_mode:
        print(f"   Batches run:       {n_iterations}")
        print(f"   Features tried:    {total_features}")
    else:
        print(f"   Iterations run:    {n_iterations}")
    print(f"   Successful:        {successes}")
    print(f"   Failed:            {failures}")
    print(f"   Success rate:      {successes/total_features*100:.1f}%")

    # Show successful features
    if memory.memory['successful_features']:
        print(f"\nSuccessful features found:")
        for i, f in enumerate(memory.memory['successful_features'], 1):
            print(f"   {i}. {f['columns']} (improvement: {f['improvement']:.5f})")

    print("=" * 60)

    # Generate visualization if requested
    if visualize:
        print("\nGenerating visualization...")
        from src.agent.visualizer import ProgressVisualizer
        viz = ProgressVisualizer()
        viz.plot_progress()
        viz.print_table()
        viz.generate_report()
        print("Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run iterative feature engineering agent')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Number of iterations or batches (default: 10)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear previous memory and start fresh')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization after completion')
    parser.add_argument('--feedback', '-f', action='store_true',
                        help='Enable SHAP-based feedback loop (B7)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Enable batch mode (generate multiple features per iteration)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of features per batch (default: 5, only used with --batch)')
    args = parser.parse_args()

    main(
        n_iterations=args.iterations,
        clear_memory=args.clear,
        visualize=args.visualize,
        use_feedback=args.feedback,
        batch_mode=args.batch,
        batch_size=args.batch_size
    )
