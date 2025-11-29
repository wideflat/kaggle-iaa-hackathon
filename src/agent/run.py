#!/usr/bin/env python
"""
Production CLI for the Feature Engineering Agent

Phase B8: Enhanced CLI with config files, logging, and error recovery.

Usage:
    python -m src.agent.run [OPTIONS]

Options:
    --iterations, -n INT    Number of iterations to run
    --config, -c PATH       Path to YAML config file
    --feedback / --no-feedback
                            Enable/disable SHAP feedback (default: from config)
    --model MODEL           Gemini model name (default: gemini-2.5-flash)
    --log-level LEVEL       Log level: DEBUG, INFO, WARNING, ERROR
    --output-dir PATH       Output directory for logs and memory
    --visualize, -v         Generate visualization after completion
    --clear                 Clear previous memory and start fresh
    --dry-run               Show config and exit without running
    --resume                Resume from last failed iteration

Examples:
    # Run with defaults from config
    python -m src.agent.run

    # Run 20 iterations with feedback
    python -m src.agent.run -n 20 --feedback

    # Use custom config file
    python -m src.agent.run --config config/local.yaml

    # Debug mode with verbose logging
    python -m src.agent.run -n 5 --log-level DEBUG

    # Show config without running
    python -m src.agent.run --dry-run

Output:
    - Memory: outputs/logs/agent_memory.json
    - Log:    outputs/logs/agent.log
    - Plot:   outputs/logs/progress_plot.png (with --visualize)
    - Report: outputs/logs/progress_report.html (with --visualize)
"""

import argparse
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd

from src.baseline.data_loader import AmesDataLoader
from src.baseline.preprocessor import AmesPreprocessor
from src.agent.gemini_client import GeminiClient
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import FeatureEvaluator, get_features_and_target
from src.agent.memory import AgentMemory
from src.agent.config import Config
from src.agent.logger import AgentLogger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Feature Engineering Agent - Production CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Core options
    parser.add_argument('--iterations', '-n', type=int,
                        help='Number of iterations to run')
    parser.add_argument('--config', '-c', type=str,
                        help='Path to YAML config file')

    # Feedback options
    feedback_group = parser.add_mutually_exclusive_group()
    feedback_group.add_argument('--feedback', '-f', action='store_true',
                                help='Enable SHAP feedback')
    feedback_group.add_argument('--no-feedback', action='store_true',
                                help='Disable SHAP feedback')

    # Model options
    parser.add_argument('--model', type=str,
                        help='Gemini model name')

    # Output options
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization after completion')

    # Control options
    parser.add_argument('--clear', action='store_true',
                        help='Clear previous memory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show config and exit')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last session')

    return parser.parse_args()


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
    logger: AgentLogger,
    current_best_rmsle: float,
    use_feedback: bool = False
) -> tuple[float, bool, pd.DataFrame]:
    """
    Run a single iteration of feature generation

    Returns:
        Tuple of (new_rmsle, is_improvement, updated_df)
    """
    logger.iteration_start(iteration, iteration, use_feedback)

    # Get already tried features
    tried_features = memory.get_all_tried_codes()
    logger.debug(f"Previously tried features: {len(tried_features)}")

    # Generate new feature
    try:
        if use_feedback:
            shap_summary = evaluator.get_shap_summary(top_n=10)
            feature_insights = evaluator.get_feature_insights(top_n=5)
            top_feature = feature_insights.get('top_features', ['N/A'])[0]
            logger.shap_summary(feature_insights.get('top_features', []), top_feature)

            # Store SHAP insights
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
        logger.gemini_error(str(e))
        memory.add_failed_feature("", str(e))
        return current_best_rmsle, False, df

    logger.generated_code(generated_code)

    # Execute the code
    new_df, error = executor.execute(generated_code, df)

    if error:
        logger.execution_error(error)
        memory.add_failed_feature(generated_code, error)
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=None)
        return current_best_rmsle, False, df

    # Check new columns
    new_cols = executor.get_new_columns(df, new_df)
    if not new_cols:
        logger.warning("No new columns created (feature already exists)")
        memory.add_failed_feature(generated_code, "No new columns created")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=None)
        return current_best_rmsle, False, df

    logger.debug(f"New columns: {new_cols}")

    # Evaluate
    X_new, _ = get_features_and_target(new_df)
    new_rmsle = evaluator.evaluate(X_new, target, verbose=False)

    # Compare
    improvement = current_best_rmsle - new_rmsle
    is_better = improvement > 0

    feature_name = new_cols[0] if new_cols else "unknown"
    logger.iteration_result(iteration, feature_name, current_best_rmsle, new_rmsle, is_better)

    if is_better:
        memory.add_successful_feature(generated_code, new_cols, new_rmsle, improvement)
        memory.log_iteration(iteration, new_rmsle, generated_code, success=True, columns=new_cols)
        return new_rmsle, True, new_df
    else:
        memory.add_failed_feature(generated_code, f"No improvement: {new_rmsle:.5f} vs {current_best_rmsle:.5f}")
        memory.log_iteration(iteration, current_best_rmsle, generated_code, success=False, columns=new_cols)
        return current_best_rmsle, False, df


def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    config = Config(config_path=args.config)
    config.override_from_args(args)

    # Handle feedback flag
    if args.feedback:
        config.set('agent', 'use_feedback', True)
    elif args.no_feedback:
        config.set('agent', 'use_feedback', False)

    # Dry run: show config and exit
    if args.dry_run:
        config.print_config()
        return

    # Initialize logger
    log_config = {
        'level': config.get('logging', 'level'),
        'log_path': config.get('output', 'log_path'),
        'file': config.get('logging', 'file'),
        'console': config.get('logging', 'console'),
    }
    logger = AgentLogger(log_config)

    # Print banner
    logger.info("=" * 60)
    logger.info("Feature Engineering Agent")
    logger.info(f"Iterations: {config.get('agent', 'iterations')}")
    logger.info(f"Feedback: {'Enabled' if config.get('agent', 'use_feedback') else 'Disabled'}")
    logger.info(f"Model: {config.get('gemini', 'model')}")
    logger.info("=" * 60)

    # Initialize memory
    memory = AgentMemory(filepath=config.get('output', 'memory_path'))
    if args.clear:
        logger.info("Clearing previous memory...")
        memory.clear()

    # Load data
    logger.info("Loading data...")
    loader = AmesDataLoader()
    train_df = loader.load_train()
    data_desc = loader.load_description()
    column_info = ', '.join(train_df.columns.tolist())
    logger.info(f"Train shape: {train_df.shape}")

    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = AmesPreprocessor()
    target = train_df['SalePrice'].copy()
    train_processed = preprocessor.fit_transform(train_df)
    logger.info(f"Processed shape: {train_processed.shape}")

    # Get baseline
    logger.info("Calculating baseline RMSLE...")
    evaluator = FeatureEvaluator(
        n_folds=config.get('evaluation', 'n_folds'),
        params_path=config.get('evaluation', 'params_path')
    )
    X_baseline, _ = get_features_and_target(train_processed)
    baseline_rmsle = evaluator.evaluate(X_baseline, target, verbose=True)
    logger.info(f"Baseline RMSLE: {baseline_rmsle:.5f}")

    memory.set_baseline(baseline_rmsle)

    # Initialize Gemini
    logger.info("Initializing Gemini client...")
    try:
        gemini = GeminiClient(model_name=config.get('gemini', 'model'))
        logger.info("Gemini client ready")
    except ValueError as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return

    # Initialize executor
    executor = CodeExecutor()

    # Get current best (may be from previous session)
    current_best_rmsle = memory.memory.get('best_rmsle') or baseline_rmsle
    logger.info(f"Starting from best RMSLE: {current_best_rmsle:.5f}")

    # Apply any previously successful features
    accumulated_df = train_processed.copy()
    successful_codes = memory.get_successful_codes()
    if successful_codes:
        logger.info(f"Applying {len(successful_codes)} previously successful features...")
        for code in successful_codes:
            result_df, error = executor.execute(code, accumulated_df)
            if not error:
                accumulated_df = result_df

    # Run iterations
    logger.info("Starting iterations...")
    n_iterations = config.get('agent', 'iterations')
    use_feedback = config.get('agent', 'use_feedback')
    successes = 0
    failures = 0

    for i in range(1, n_iterations + 1):
        try:
            new_rmsle, is_success, accumulated_df = run_iteration(
                iteration=i,
                df=accumulated_df,
                target=target,
                data_desc=data_desc,
                column_info=column_info,
                gemini=gemini,
                executor=executor,
                evaluator=evaluator,
                memory=memory,
                logger=logger,
                current_best_rmsle=current_best_rmsle,
                use_feedback=use_feedback
            )

            if is_success:
                current_best_rmsle = new_rmsle
                successes += 1
            else:
                failures += 1

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")
            logger.debug(traceback.format_exc())
            failures += 1
            continue

    # Session summary
    logger.session_summary(
        iterations=successes + failures,
        successes=successes,
        baseline_rmsle=baseline_rmsle,
        best_rmsle=current_best_rmsle
    )

    # Show successful features
    if memory.memory.get('successful_features'):
        logger.info("Successful features:")
        for i, f in enumerate(memory.memory['successful_features'], 1):
            logger.info(f"  {i}. {f['columns']} (improvement: {f['improvement']:.5f})")

    # Generate visualization if requested
    if args.visualize or config.get('visualization', 'auto_generate'):
        logger.info("Generating visualization...")
        from src.agent.visualizer import ProgressVisualizer
        viz = ProgressVisualizer(memory_path=config.get('output', 'memory_path'))
        viz.plot_progress(
            save_path=config.get('output', 'plot_path'),
            show=config.get('visualization', 'show_plot')
        )
        viz.print_table()
        viz.generate_report(output_path=config.get('output', 'report_path'))
        logger.info("Visualization complete!")


if __name__ == '__main__':
    main()
