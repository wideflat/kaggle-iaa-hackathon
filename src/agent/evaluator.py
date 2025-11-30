"""
Feature evaluator for measuring improvement over baseline

Enhanced with SHAP-based feature importance for feedback loop.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Dict, Optional


class FeatureEvaluator:
    """
    Evaluate feature quality using K-Fold Cross-Validation

    Uses LightGBM with tuned hyperparameters (if available)
    Measures RMSLE on log-transformed target

    Enhanced with SHAP-based feature importance for feedback loop.
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
        self.last_model = None
        self.last_X = None

    def _load_params(self, params_path: str) -> dict:
        """Load tuned hyperparameters if available"""
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                print(f"   Loaded tuned params from {params_path}")
                return params
        else:
            print(f"   Using default params (moderate strength for feature discovery)")
            return {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
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

        # Store last model and data for SHAP computation
        # Train a final model on full data for SHAP
        self.last_X = X
        self.last_model = lgb.LGBMRegressor(**self.model_params)
        self.last_model.fit(X, y_log)

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

    def get_shap_importance(
        self,
        top_n: int = 10,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Get SHAP-based feature importance from last evaluated model

        Args:
            top_n: Number of top features to return
            sample_size: Number of samples to use for SHAP (for speed)

        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        if self.last_model is None or self.last_X is None:
            return {}

        # Sample data for faster SHAP computation
        if len(self.last_X) > sample_size:
            X_sample = self.last_X.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.last_X

        # Compute SHAP values using TreeExplainer (fast for LightGBM)
        explainer = shap.TreeExplainer(self.last_model)
        shap_values = explainer.shap_values(X_sample)

        # Get mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance dict
        importance = dict(zip(self.last_X.columns, mean_abs_shap))

        # Sort and return top N
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        return sorted_importance

    def get_shap_summary(self, top_n: int = 10) -> str:
        """
        Get human-readable summary of SHAP feature importance

        Args:
            top_n: Number of top features to include

        Returns:
            Formatted string summary for prompting
        """
        importance = self.get_shap_importance(top_n=top_n)

        if not importance:
            return "No feature importance available yet."

        lines = ["Top Features by SHAP Importance:"]
        for i, (feature, value) in enumerate(importance.items(), 1):
            lines.append(f"  {i}. {feature}: {value:.4f}")

        return "\n".join(lines)

    def get_feature_insights(self, top_n: int = 5) -> Dict[str, any]:
        """
        Get insights about features for the feedback loop

        Returns:
            Dictionary with top features, their importance, and suggestions
        """
        importance = self.get_shap_importance(top_n=top_n)

        if not importance:
            return {'top_features': [], 'suggestions': []}

        top_features = list(importance.keys())

        # Generate suggestions based on top features
        suggestions = []
        for feature in top_features[:3]:
            suggestions.append(f"Consider interactions with '{feature}'")
            suggestions.append(f"Try polynomial transforms of '{feature}'")

        return {
            'top_features': top_features,
            'importance_scores': importance,
            'suggestions': suggestions,
            'total_features': len(self.last_X.columns) if self.last_X is not None else 0
        }

    def recompute_shap_for_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Retrain model on given features and update SHAP state.

        This is used after a feature is rejected to ensure SHAP feedback
        only references features that actually exist in the accumulated dataframe.

        Args:
            X: Feature matrix (should be the current accumulated features)
            y: Target variable (original scale, will be log-transformed)
        """
        y_log = np.log1p(y)
        self.last_X = X
        self.last_model = lgb.LGBMRegressor(**self.model_params)
        self.last_model.fit(X, y_log)


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
