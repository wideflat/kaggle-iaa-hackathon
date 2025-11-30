"""
Feature selection to reduce noise

Methods:
- LightGBM importance threshold
- Correlation-based filtering
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import lightgbm as lgb


class FeatureSelector:
    """
    Select best features to reduce noise and improve model performance.

    Methods:
    1. importance: Remove features with importance <= threshold
    2. correlation: Remove features with low correlation to target

    Features are selected based on training data, then applied to test data.
    """

    def __init__(self, threshold: float = 0, verbose: bool = True):
        """
        Initialize feature selector.

        Args:
            threshold: Minimum importance/correlation to keep feature
            verbose: Print selection information
        """
        self.threshold = threshold
        self.verbose = verbose
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[pd.Series] = None

    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 500
    ) -> List[str]:
        """
        Select features based on LightGBM feature importance.

        Args:
            X: Feature matrix
            y: Target variable (log-transformed recommended)
            n_estimators: Number of estimators for importance calculation

        Returns:
            List of selected feature names
        """
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        self.feature_importance = importance

        # Keep features with importance > threshold
        mask = importance > self.threshold
        self.selected_features = importance[mask].index.tolist()

        if self.verbose:
            removed = len(X.columns) - len(self.selected_features)
            print(f"   Feature selection (importance > {self.threshold}):")
            print(f"      Kept: {len(self.selected_features)} features")
            print(f"      Removed: {removed} features")
            if removed > 0:
                removed_features = importance[~mask].index.tolist()[:5]
                print(f"      Removed (top 5): {removed_features}")

        return self.selected_features

    def select_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_correlation: float = 0.05
    ) -> List[str]:
        """
        Select features based on correlation with target.

        Args:
            X: Feature matrix
            y: Target variable
            min_correlation: Minimum absolute correlation to keep

        Returns:
            List of selected feature names
        """
        correlations = X.corrwith(y).abs()

        # Keep features with correlation >= min_correlation
        mask = correlations >= min_correlation
        self.selected_features = correlations[mask].index.tolist()
        self.feature_importance = correlations.sort_values(ascending=False)

        if self.verbose:
            removed = len(X.columns) - len(self.selected_features)
            print(f"   Feature selection (correlation >= {min_correlation}):")
            print(f"      Kept: {len(self.selected_features)} features")
            print(f"      Removed: {removed} features")

        return self.selected_features

    def select_combined(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        importance_threshold: float = 0,
        correlation_threshold: float = 0.01
    ) -> List[str]:
        """
        Select features using both importance AND correlation.

        Features must pass BOTH thresholds to be kept.

        Args:
            X: Feature matrix
            y: Target variable
            importance_threshold: Minimum importance
            correlation_threshold: Minimum correlation

        Returns:
            List of selected feature names
        """
        # Get importance-based selection
        model = lgb.LGBMRegressor(n_estimators=500, random_state=42, verbosity=-1)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance_mask = importance > importance_threshold

        # Get correlation-based selection
        correlations = X.corrwith(y).abs()
        correlation_mask = correlations >= correlation_threshold

        # Combine: feature must pass both
        combined_mask = importance_mask & correlation_mask
        self.selected_features = X.columns[combined_mask].tolist()

        self.feature_importance = pd.DataFrame({
            'importance': importance,
            'correlation': correlations
        }).sort_values('importance', ascending=False)

        if self.verbose:
            removed = len(X.columns) - len(self.selected_features)
            print(f"   Combined feature selection:")
            print(f"      Kept: {len(self.selected_features)} features")
            print(f"      Removed: {removed} features")

        return self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only selected features.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with only selected features
        """
        if self.selected_features is None:
            return X

        # Handle case where some selected features may not exist
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'importance'
    ) -> pd.DataFrame:
        """
        Select features and return transformed DataFrame.

        Args:
            X: Feature matrix
            y: Target variable
            method: 'importance', 'correlation', or 'combined'

        Returns:
            DataFrame with selected features only
        """
        if method == 'importance':
            self.select_by_importance(X, y)
        elif method == 'correlation':
            self.select_by_correlation(X, y)
        elif method == 'combined':
            self.select_combined(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.transform(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from last selection."""
        return self.feature_importance

    def get_selected_features(self) -> Optional[List[str]]:
        """Get list of selected features."""
        return self.selected_features
