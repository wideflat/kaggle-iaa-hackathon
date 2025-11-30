"""
Parallel Feature Engineering Agent

Runs multiple workers in parallel to speed up feature discovery.
Each worker independently generates and evaluates features.

Usage:
    python -m src.agent.parallel_agent [OPTIONS]

Arguments:
    --iterations, -n INT    Total number of iterations (default: 10)
    --workers, -w INT       Number of parallel workers (default: 2)
    --clear                 Clear previous memory and start fresh
    --feedback, -f          Enable SHAP-based feedback loop
    --dashboard, -d         Open real-time dashboard in browser

Examples:
    # Run 10 iterations with 2 workers
    python -m src.agent.parallel_agent -n 10 -w 2

    # Run with dashboard
    python -m src.agent.parallel_agent -n 10 -w 2 --dashboard

    # With SHAP feedback
    python -m src.agent.parallel_agent -n 10 -w 2 --feedback --dashboard
"""

import os
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

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
from src.agent.shared_state import SharedState
from src.agent.event_emitter import emit


class ParallelFeatureAgent:
    """
    Parallel feature engineering agent.

    Runs N workers concurrently, each generating and evaluating features.
    Uses thread-safe shared state for coordination.
    """

    def __init__(
        self,
        n_workers: int = 2,
        use_feedback: bool = False,
        memory: AgentMemory = None
    ):
        """
        Initialize parallel agent.

        Args:
            n_workers: Number of parallel workers
            use_feedback: Enable SHAP-based feedback
            memory: Agent memory instance
        """
        self.n_workers = n_workers
        self.use_feedback = use_feedback
        self.memory = memory or AgentMemory()

        # Will be set during run()
        self.shared_state: Optional[SharedState] = None
        self.data_desc: str = ""
        self.column_info: str = ""
        self.target: pd.Series = None

    def run(
        self,
        train_df: pd.DataFrame,
        target: pd.Series,
        data_desc: str,
        total_iterations: int
    ) -> dict:
        """
        Run parallel feature engineering.

        Args:
            train_df: Preprocessed training dataframe
            target: Target series
            data_desc: Data description for LLM
            total_iterations: Total iterations across all workers

        Returns:
            Summary statistics
        """
        # Initialize shared state
        evaluator = FeatureEvaluator(n_folds=5)
        X_baseline, _ = get_features_and_target(train_df)
        baseline_rmsle = evaluator.evaluate(X_baseline, target, verbose=True)

        print(f"\n   >>> Baseline RMSLE: {baseline_rmsle:.5f}")
        self.memory.set_baseline(baseline_rmsle)

        # Apply any previously successful features
        accumulated_df = train_df.copy()
        executor = CodeExecutor()
        successful_codes = self.memory.get_successful_codes()
        if successful_codes:
            print(f"\n   Applying {len(successful_codes)} previously successful features...")
            for code in successful_codes:
                result_df, error = executor.execute(code, accumulated_df)
                if not error:
                    accumulated_df = result_df

        # Recalculate best RMSLE with previously successful features
        if successful_codes:
            X_current, _ = get_features_and_target(accumulated_df)
            current_rmsle = evaluator.evaluate(X_current, target, verbose=False)
            print(f"   Current best RMSLE: {current_rmsle:.5f}")
        else:
            current_rmsle = baseline_rmsle

        self.shared_state = SharedState(
            initial_df=accumulated_df,
            baseline_rmsle=current_rmsle,
            total_iterations=total_iterations
        )

        self.data_desc = data_desc
        self.column_info = ', '.join(train_df.columns.tolist())
        self.target = target

        # Emit agent start event
        emit('agent_start', {
            'total_iterations': total_iterations,
            'baseline_rmsle': baseline_rmsle,
            'current_rmsle': current_rmsle,
            'n_workers': self.n_workers
        })

        print(f"\n5. Running {total_iterations} iterations with {self.n_workers} workers...")

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._worker_loop, worker_id)
                for worker_id in range(self.n_workers)
            ]

            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")

        # Get final statistics
        stats = self.shared_state.get_stats()

        # Save successful features to memory
        for result in self.shared_state.get_results():
            if result.accepted:
                self.memory.add_successful_feature(
                    result.code,
                    result.columns,
                    result.rmsle,
                    result.improvement
                )
                self.memory.log_iteration(
                    result.iteration,
                    result.rmsle,
                    result.code,
                    success=True,
                    columns=result.columns
                )
            else:
                self.memory.add_failed_feature(result.code, "No improvement")

        # Emit agent complete event
        emit('agent_complete', {
            'final_rmsle': stats['best_rmsle'],
            'baseline_rmsle': baseline_rmsle,
            'total_improvement': baseline_rmsle - stats['best_rmsle'],
            'successes': stats['successes'],
            'failures': stats['failures']
        })

        return stats

    def _worker_loop(self, worker_id: int):
        """
        Worker loop: generate and evaluate features until done.

        Args:
            worker_id: Unique worker identifier
        """
        # Each worker has its own Gemini client and executor
        try:
            gemini = GeminiClient()
        except Exception as e:
            print(f"Worker {worker_id}: Failed to initialize Gemini: {e}")
            return

        executor = CodeExecutor()
        evaluator = FeatureEvaluator(n_folds=5)

        self.shared_state.register_worker(worker_id)

        try:
            while True:
                # Get next iteration
                iteration = self.shared_state.get_next_iteration()
                if iteration is None:
                    break

                # Run one iteration
                self._run_iteration(
                    worker_id=worker_id,
                    iteration=iteration,
                    gemini=gemini,
                    executor=executor,
                    evaluator=evaluator
                )
        finally:
            self.shared_state.unregister_worker(worker_id)

    def _run_iteration(
        self,
        worker_id: int,
        iteration: int,
        gemini: GeminiClient,
        executor: CodeExecutor,
        evaluator: FeatureEvaluator
    ):
        """Run a single iteration for a worker"""
        print(f"\n[Worker {worker_id}] Iteration {iteration}")

        # Emit iteration start
        emit('iteration_start', {
            'worker_id': worker_id,
            'iteration': iteration
        })

        self.shared_state.update_worker_status(worker_id, 'generating')

        # Get current state
        current_df = self.shared_state.get_current_df()
        tried_features = self.shared_state.get_tried_codes()
        current_best = self.shared_state.get_best_rmsle()

        # Generate feature
        try:
            if self.use_feedback:
                # Get SHAP insights
                X_current, _ = get_features_and_target(current_df)
                evaluator.evaluate(X_current, self.target, verbose=False)
                shap_summary = evaluator.get_shap_summary(top_n=10)
                feature_insights = evaluator.get_feature_insights(top_n=5)

                generated_code = gemini.generate_feature_with_feedback(
                    self.data_desc,
                    self.column_info,
                    shap_summary=shap_summary,
                    feature_insights=feature_insights,
                    existing_features=tried_features
                )
            else:
                generated_code = gemini.generate_feature(
                    self.data_desc,
                    self.column_info,
                    existing_features=tried_features
                )
        except Exception as e:
            print(f"[Worker {worker_id}] Gemini error: {e}")
            self.shared_state.reject_feature(
                worker_id, iteration, "", current_best, [], str(e)
            )
            return

        # Emit feature generated
        emit('feature_generated', {
            'worker_id': worker_id,
            'code': generated_code
        })

        self.shared_state.update_worker_status(worker_id, 'evaluating', generated_code)

        # Execute code
        new_df, error = executor.execute(generated_code, current_df)

        if error:
            print(f"[Worker {worker_id}] Execution error: {error[:50]}...")
            self.shared_state.reject_feature(
                worker_id, iteration, generated_code, current_best, [], error
            )
            emit('feature_rejected', {
                'worker_id': worker_id,
                'code': generated_code,
                'rmsle': current_best,
                'reason': f"Execution error: {error[:100]}"
            })
            return

        # Check new columns
        new_cols = executor.get_new_columns(current_df, new_df)
        if not new_cols:
            print(f"[Worker {worker_id}] No new columns created")
            self.shared_state.reject_feature(
                worker_id, iteration, generated_code, current_best, [], "No new columns"
            )
            emit('feature_rejected', {
                'worker_id': worker_id,
                'code': generated_code,
                'rmsle': current_best,
                'columns': [],
                'reason': "No new columns created"
            })
            return

        # Evaluate
        X_new, _ = get_features_and_target(new_df)
        new_rmsle = evaluator.evaluate(X_new, self.target, verbose=False)

        print(f"[Worker {worker_id}] {new_cols}: RMSLE={new_rmsle:.5f} (best={current_best:.5f})")

        # Try to accept feature
        accepted = self.shared_state.try_accept_feature(
            worker_id=worker_id,
            iteration=iteration,
            code=generated_code,
            new_df=new_df,
            new_rmsle=new_rmsle,
            columns=new_cols
        )

        if accepted:
            improvement = current_best - new_rmsle
            print(f"[Worker {worker_id}] >>> ACCEPTED! Improvement: {improvement:.5f}")
            emit('feature_accepted', {
                'worker_id': worker_id,
                'columns': new_cols,
                'code': generated_code,
                'new_rmsle': new_rmsle,
                'improvement': improvement
            })
        else:
            print(f"[Worker {worker_id}] >>> Rejected (no improvement)")
            emit('feature_rejected', {
                'worker_id': worker_id,
                'columns': new_cols,
                'code': generated_code,
                'rmsle': new_rmsle,
                'reason': f"No improvement: {new_rmsle:.5f} vs {current_best:.5f}"
            })

        self.shared_state.update_worker_status(worker_id, 'idle')


def main(
    n_iterations: int = 10,
    n_workers: int = 2,
    clear_memory: bool = False,
    use_feedback: bool = False,
    dashboard: bool = False
):
    """Run parallel feature engineering agent"""

    # Start dashboard server if requested
    if dashboard:
        import webbrowser
        from src.agent.dashboard_server import app
        import uvicorn

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")

        dashboard_thread = threading.Thread(target=run_server, daemon=True)
        dashboard_thread.start()
        print("Dashboard started at http://localhost:8765")
        webbrowser.open("http://localhost:8765")

    print("=" * 60)
    print(f"Parallel Feature Engineering Agent")
    print(f"Workers: {n_workers}")
    print(f"Iterations: {n_iterations}")
    if use_feedback:
        print(f"SHAP Feedback: Enabled")
    if dashboard:
        print(f"Dashboard: http://localhost:8765")
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

    # Remove outliers
    print("\n1b. Removing outliers...")
    outlier_remover = OutlierRemover()
    train_df = outlier_remover.remove_known_outliers(train_df)
    print(f"   Train shape after outlier removal: {train_df.shape}")

    # Preprocess
    print("\n2. Preprocessing (ultra-minimal mode)...")
    preprocessor = MinimalPreprocessor()
    target = train_df['SalePrice'].copy()
    train_processed = preprocessor.fit_transform(train_df)
    print(f"   Processed shape: {train_processed.shape}")

    # Initialize Gemini (test connection)
    print("\n3. Testing Gemini connection...")
    try:
        test_gemini = GeminiClient()
        print("   Gemini client ready")
        del test_gemini
    except ValueError as e:
        print(f"   ERROR: {e}")
        return

    # Initialize and run parallel agent
    print("\n4. Initializing parallel agent...")
    agent = ParallelFeatureAgent(
        n_workers=n_workers,
        use_feedback=use_feedback,
        memory=memory
    )

    stats = agent.run(
        train_df=train_processed,
        target=target,
        data_desc=data_desc,
        total_iterations=n_iterations
    )

    # Final summary
    print("\n" + "=" * 60)
    print("PARALLEL AGENT COMPLETE")
    print("=" * 60)
    memory.print_summary()

    print(f"\nThis session:")
    print(f"   Workers:           {n_workers}")
    print(f"   Iterations:        {n_iterations}")
    print(f"   Successful:        {stats['successes']}")
    print(f"   Failed:            {stats['failures']}")
    print(f"   Success rate:      {stats['success_rate']:.1f}%")
    print(f"   Final RMSLE:       {stats['best_rmsle']:.5f}")

    # Show successful features
    if memory.memory['successful_features']:
        print(f"\nSuccessful features found:")
        for i, f in enumerate(memory.memory['successful_features'], 1):
            print(f"   {i}. {f['columns']} (improvement: {f['improvement']:.5f})")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run parallel feature engineering agent')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Total number of iterations (default: 10)')
    parser.add_argument('--workers', '-w', type=int, default=2,
                        help='Number of parallel workers (default: 2)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear previous memory and start fresh')
    parser.add_argument('--feedback', '-f', action='store_true',
                        help='Enable SHAP-based feedback loop')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Open real-time dashboard in browser')
    args = parser.parse_args()

    main(
        n_iterations=args.iterations,
        n_workers=args.workers,
        clear_memory=args.clear,
        use_feedback=args.feedback,
        dashboard=args.dashboard
    )
