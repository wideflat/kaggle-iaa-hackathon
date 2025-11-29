"""
Feature evaluator for measuring improvement over baseline
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Dict, Optional


class FeatureEvaluator:
    """
    Evaluate feature quality using K-Fold Cross-Validation

    Uses LightGBM with tuned hyperparameters (if available)
    Measures RMSLE on log-transformed target
    """

    def __init__(
        self,
        n_folds: int = 5,
        params_path: str = 'outputs/models/best_lgbm_params.json'
    ):
        """
        Initialize evaluator

        Args:
            n_folds: Number of CV folds
            params_path: Path to tuned hyperparameters JSON
        """
        self.n_folds = n_folds
        self.model_params = self._load_params(params_path)
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _load_params(self, params_path: str) -> dict:
        """Load tuned hyperparameters if available"""
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                print(f"   Loaded tuned params from {params_path}")
                return params
        else:
            print(f"   Using default params (no tuned params found)")
            return {
                'n_estimators': 100,
                'random_state': 42,
                'verbosity': -1
            }

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = False
    ) -> float:
        """
        Evaluate features using K-Fold CV

        Args:
            X: Feature matrix
            y: Target (original scale, will be log-transformed)
            verbose: Print per-fold scores

        Returns:
            Mean CV RMSLE score
        """
        # Log-transform target
        y_log = np.log1p(y)

        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X), 1):
            # Split data
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y_log.iloc[train_idx]
            y_val = y_log.iloc[val_idx]
            y_val_original = y.iloc[val_idx]

            # Train model
            model = lgb.LGBMRegressor(**self.model_params)

            # Use early stopping if n_estimators is high
            if self.model_params.get('n_estimators', 100) > 200:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)

            # Predict and transform back
            val_pred_log = model.predict(X_val)
            val_pred = np.expm1(val_pred_log)

            # Calculate RMSLE
            # RMSLE = sqrt(mean((log(pred+1) - log(actual+1))^2))
            # Since we're working in log space already:
            fold_rmsle = np.sqrt(mean_squared_error(y_val, val_pred_log))
            cv_scores.append(fold_rmsle)

            if verbose:
                print(f"      Fold {fold}: RMSLE = {fold_rmsle:.5f}")

        mean_rmsle = np.mean(cv_scores)

        if verbose:
            print(f"      Mean RMSLE: {mean_rmsle:.5f} (± {np.std(cv_scores):.5f})")

        return mean_rmsle

    def compare(
        self,
        baseline_rmsle: float,
        new_rmsle: float
    ) -> Dict[str, float | bool]:
        """
        Compare new RMSLE against baseline

        Args:
            baseline_rmsle: Baseline CV RMSLE
            new_rmsle: New CV RMSLE after feature addition

        Returns:
            Dictionary with comparison results
        """
        delta = baseline_rmsle - new_rmsle  # Positive = improvement
        percent_change = (delta / baseline_rmsle) * 100

        return {
            'is_improvement': delta > 0,
            'delta': delta,
            'percent_change': percent_change,
            'baseline_rmsle': baseline_rmsle,
            'new_rmsle': new_rmsle
        }


def get_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'SalePrice',
    drop_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from preprocessed dataframe

    Args:
        df: Preprocessed dataframe
        target_col: Name of target column
        drop_cols: Additional columns to drop

    Returns:
        Tuple of (X, y)
    """
    cols_to_drop = ['Id']
    if target_col in df.columns:
        cols_to_drop.append(target_col)
    if drop_cols:
        cols_to_drop.extend(drop_cols)

    # Only drop columns that exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(cols_to_drop, axis=1)

    # Select only numeric columns (including uint8/bool from one-hot encoding)
    X = X.select_dtypes(include=['number', 'bool'])

    y = df[target_col] if target_col in df.columns else None

    return X, y
