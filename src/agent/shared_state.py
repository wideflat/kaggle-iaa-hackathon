"""
Thread-safe shared state for parallel feature engineering agent.

Manages coordination between multiple worker threads.
"""

import threading
from typing import Set, List, Optional
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureResult:
    """Result of a feature evaluation"""
    worker_id: int
    iteration: int
    code: str
    columns: List[str]
    rmsle: float
    improvement: float
    accepted: bool


class SharedState:
    """
    Thread-safe shared state for parallel workers.

    Manages:
    - Tried features (to avoid duplicates)
    - Current best RMSLE
    - Accumulated dataframe with successful features
    - Iteration counter (shared work queue)
    """

    def __init__(
        self,
        initial_df: pd.DataFrame,
        baseline_rmsle: float,
        total_iterations: int
    ):
        """
        Initialize shared state.

        Args:
            initial_df: Preprocessed dataframe to start with
            baseline_rmsle: Baseline RMSLE score
            total_iterations: Total number of iterations to run
        """
        self._lock = threading.RLock()
        self._tried_features: Set[str] = set()
        self._tried_codes: List[str] = []
        self._best_rmsle = baseline_rmsle
        self._accumulated_df = initial_df.copy()
        self._results: List[FeatureResult] = []

        # Shared work queue
        self._total_iterations = total_iterations
        self._current_iteration = 0

        # Worker status tracking
        self._worker_status: dict = {}
        self._active_workers = 0

    def get_next_iteration(self) -> Optional[int]:
        """
        Get the next iteration number (atomic).

        Returns:
            Next iteration number, or None if all iterations complete
        """
        with self._lock:
            if self._current_iteration >= self._total_iterations:
                return None
            self._current_iteration += 1
            return self._current_iteration

    def get_tried_codes(self) -> List[str]:
        """Get list of all tried feature codes (for LLM context)"""
        with self._lock:
            return self._tried_codes.copy()

    def get_current_df(self) -> pd.DataFrame:
        """Get current accumulated dataframe"""
        with self._lock:
            return self._accumulated_df.copy()

    def get_best_rmsle(self) -> float:
        """Get current best RMSLE"""
        with self._lock:
            return self._best_rmsle

    def try_accept_feature(
        self,
        worker_id: int,
        iteration: int,
        code: str,
        new_df: pd.DataFrame,
        new_rmsle: float,
        columns: List[str]
    ) -> bool:
        """
        Attempt to accept a new feature.

        Thread-safe: Only accepts if it improves over current best.

        Args:
            worker_id: ID of the worker
            iteration: Iteration number
            code: Feature generation code
            new_df: DataFrame with new feature applied
            new_rmsle: New RMSLE score
            columns: New column names

        Returns:
            True if feature was accepted (improved RMSLE)
        """
        with self._lock:
            # Add to tried codes regardless of acceptance
            self._tried_codes.append(code)

            improvement = self._best_rmsle - new_rmsle
            is_improvement = improvement > 0

            result = FeatureResult(
                worker_id=worker_id,
                iteration=iteration,
                code=code,
                columns=columns,
                rmsle=new_rmsle,
                improvement=improvement,
                accepted=is_improvement
            )
            self._results.append(result)

            if is_improvement:
                self._best_rmsle = new_rmsle
                self._accumulated_df = new_df.copy()
                return True

            return False

    def reject_feature(
        self,
        worker_id: int,
        iteration: int,
        code: str,
        rmsle: float,
        columns: List[str],
        error: str = None
    ):
        """
        Record a rejected/failed feature.

        Args:
            worker_id: ID of the worker
            iteration: Iteration number
            code: Feature generation code
            rmsle: RMSLE score (or current best if failed)
            columns: Column names (if any)
            error: Error message if failed
        """
        with self._lock:
            self._tried_codes.append(code)

            result = FeatureResult(
                worker_id=worker_id,
                iteration=iteration,
                code=code,
                columns=columns or [],
                rmsle=rmsle,
                improvement=0,
                accepted=False
            )
            self._results.append(result)

    def update_worker_status(self, worker_id: int, status: str, code: str = None):
        """Update a worker's current status"""
        with self._lock:
            self._worker_status[worker_id] = {
                'status': status,
                'code': code
            }

    def register_worker(self, worker_id: int):
        """Register a worker as active"""
        with self._lock:
            self._active_workers += 1
            self._worker_status[worker_id] = {'status': 'idle', 'code': None}

    def unregister_worker(self, worker_id: int):
        """Unregister a worker"""
        with self._lock:
            self._active_workers -= 1
            if worker_id in self._worker_status:
                self._worker_status[worker_id] = {'status': 'done', 'code': None}

    def get_stats(self) -> dict:
        """Get current statistics"""
        with self._lock:
            successes = sum(1 for r in self._results if r.accepted)
            total = len(self._results)

            return {
                'current_iteration': self._current_iteration,
                'total_iterations': self._total_iterations,
                'successes': successes,
                'failures': total - successes,
                'success_rate': successes / total * 100 if total > 0 else 0,
                'best_rmsle': self._best_rmsle,
                'active_workers': self._active_workers,
                'worker_status': dict(self._worker_status)
            }

    def get_results(self) -> List[FeatureResult]:
        """Get all results"""
        with self._lock:
            return self._results.copy()
