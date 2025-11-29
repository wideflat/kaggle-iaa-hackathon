"""
Memory system for tracking feature engineering attempts
"""

import json
import os
import re
from datetime import datetime
from typing import Optional


class AgentMemory:
    """
    Persistent memory for the feature engineering agent

    Stores:
    - Successful features (code + improvement)
    - Failed features (code + error)
    - Iteration history (RMSLE over time)
    """

    def __init__(self, filepath: str = 'outputs/logs/agent_memory.json'):
        """
        Initialize memory system

        Args:
            filepath: Path to JSON file for persistence
        """
        self.filepath = filepath
        self._ensure_directory()
        self.memory = self._load()

    def _ensure_directory(self):
        """Create directory if it doesn't exist"""
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _load(self) -> dict:
        """Load memory from disk"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"   Warning: Could not load {self.filepath}, starting fresh")

        return {
            'successful_features': [],
            'failed_features': [],
            'iteration_history': [],
            'shap_history': [],  # SHAP insights per iteration
            'baseline_rmsle': None,
            'best_rmsle': None,
            'session_start': datetime.now().isoformat()
        }

    def save(self):
        """Save memory to disk"""
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def set_baseline(self, rmsle: float):
        """Set the baseline RMSLE"""
        self.memory['baseline_rmsle'] = rmsle
        if self.memory['best_rmsle'] is None:
            self.memory['best_rmsle'] = rmsle
        self.save()

    def add_successful_feature(
        self,
        code: str,
        new_columns: list[str],
        rmsle: float,
        improvement: float
    ):
        """
        Record a successful feature

        Args:
            code: The Python code that created the feature
            new_columns: List of new column names created
            rmsle: The new RMSLE after adding this feature
            improvement: The improvement over previous best
        """
        self.memory['successful_features'].append({
            'code': code,
            'columns': new_columns,
            'rmsle': rmsle,
            'improvement': improvement,
            'timestamp': datetime.now().isoformat()
        })

        # Update best RMSLE
        if rmsle < self.memory['best_rmsle']:
            self.memory['best_rmsle'] = rmsle

        self.save()

    def add_failed_feature(self, code: str, error: str):
        """
        Record a failed feature attempt

        Args:
            code: The Python code that failed
            error: The error message
        """
        self.memory['failed_features'].append({
            'code': code,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def log_iteration(
        self,
        iteration: int,
        rmsle: float,
        feature_code: str,
        success: bool,
        columns: list[str] | None = None
    ):
        """
        Log an iteration result

        Args:
            iteration: Iteration number
            rmsle: RMSLE at this iteration
            feature_code: The feature code attempted
            success: Whether the feature improved the model
            columns: List of new column names created (optional)
        """
        feature_summary = self._extract_feature_summary(feature_code, columns)
        self.memory['iteration_history'].append({
            'iteration': iteration,
            'rmsle': rmsle,
            'success': success,
            'feature_summary': feature_summary,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def _extract_feature_summary(
        self,
        code: str,
        columns: list[str] | None = None
    ) -> str:
        """
        Extract a human-readable summary from feature code

        Tries to extract "# Feature: XXX" comment, falls back to column names
        """
        if not code:
            return "(no code)"

        # Try to extract "# Feature: XXX" comment
        match = re.search(r'#\s*Feature:\s*(.+)', code, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try to extract column name from df['ColumnName'] = ...
        match = re.search(r"df\['(\w+)'\]\s*=", code)
        if match:
            col_name = match.group(1)
            # Try to get the right side of the assignment for context
            rhs_match = re.search(r"df\['\w+'\]\s*=\s*(.+?)(?:\n|$)", code)
            if rhs_match:
                rhs = rhs_match.group(1).strip()[:50]  # Truncate
                return f"{col_name} ({rhs})"
            return col_name

        # Fall back to column names if provided
        if columns:
            return ', '.join(columns)

        # Last resort: first line of code
        first_line = code.strip().split('\n')[0][:50]
        return first_line if first_line else "(unknown)"

    def log_shap_insights(
        self,
        iteration: int,
        top_features: list[str],
        importance_scores: dict[str, float],
        total_features: int
    ):
        """
        Log SHAP insights for an iteration

        Args:
            iteration: Iteration number
            top_features: List of top feature names
            importance_scores: Dict of feature name to SHAP importance
            total_features: Total number of features in the model
        """
        # Ensure shap_history exists (for backwards compatibility)
        if 'shap_history' not in self.memory:
            self.memory['shap_history'] = []

        self.memory['shap_history'].append({
            'iteration': iteration,
            'top_features': top_features,
            'importance_scores': importance_scores,
            'total_features': total_features,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def get_shap_history(self) -> list[dict]:
        """Get all SHAP history entries"""
        return self.memory.get('shap_history', [])

    def get_successful_codes(self) -> list[str]:
        """Get list of all successful feature codes"""
        return [f['code'] for f in self.memory['successful_features']]

    def get_all_tried_codes(self) -> list[str]:
        """Get list of all tried feature codes (success + failed)"""
        successful = [f['code'] for f in self.memory['successful_features']]
        failed = [f['code'] for f in self.memory['failed_features']]
        return successful + failed

    def get_successful_columns(self) -> list[str]:
        """Get list of all successful feature column names"""
        columns = []
        for f in self.memory['successful_features']:
            columns.extend(f.get('columns', []))
        return columns

    def get_stats(self) -> dict:
        """Get summary statistics"""
        return {
            'total_attempts': len(self.memory['iteration_history']),
            'successful_features': len(self.memory['successful_features']),
            'failed_features': len(self.memory['failed_features']),
            'baseline_rmsle': self.memory['baseline_rmsle'],
            'best_rmsle': self.memory['best_rmsle'],
            'improvement': (
                self.memory['baseline_rmsle'] - self.memory['best_rmsle']
                if self.memory['baseline_rmsle'] and self.memory['best_rmsle']
                else 0
            )
        }

    def clear(self):
        """Clear all memory (start fresh)"""
        self.memory = {
            'successful_features': [],
            'failed_features': [],
            'iteration_history': [],
            'shap_history': [],
            'baseline_rmsle': None,
            'best_rmsle': None,
            'session_start': datetime.now().isoformat()
        }
        self.save()

    def print_summary(self):
        """Print a summary of the memory state"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("AGENT MEMORY SUMMARY")
        print("=" * 50)
        print(f"Total attempts:      {stats['total_attempts']}")
        print(f"Successful features: {stats['successful_features']}")
        print(f"Failed attempts:     {stats['failed_features']}")
        print(f"Baseline RMSLE:      {stats['baseline_rmsle']:.5f}" if stats['baseline_rmsle'] else "Baseline RMSLE:      N/A")
        print(f"Best RMSLE:          {stats['best_rmsle']:.5f}" if stats['best_rmsle'] else "Best RMSLE:          N/A")
        print(f"Total improvement:   {stats['improvement']:.5f} ({stats['improvement']/stats['baseline_rmsle']*100:.2f}%)" if stats['baseline_rmsle'] else "")
        print("=" * 50)
