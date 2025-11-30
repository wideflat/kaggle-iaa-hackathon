"""
Model stacking for final prediction

Combines multiple models using weighted averaging:
- Ridge, Lasso, ElasticNet (linear models, 30% weight)
- XGBoost, LightGBM (tree models, 70% weight)

Based on top Kaggle Ames Housing solutions.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb


class ModelStacker:
    """
    Stack multiple models for improved predictions.

    Models:
    - Ridge (alpha=10) - Linear, robust to multicollinearity
    - Lasso (alpha=0.0005) - Linear, feature selection built-in
    - ElasticNet (alpha=0.0005, l1_ratio=0.9) - Linear, balanced L1/L2
    - XGBoost - Tree-based, different from LightGBM
    - LightGBM - Tree-based, fast and accurate

    Linear models use RobustScaler for feature scaling.
    Tree models use raw features.

    Blending: Weighted average with configurable weights.
    """

    def __init__(
        self,
        n_folds: int = 5,
        lgb_params_path: Optional[str] = 'outputs/models/best_lgbm_params.json',
        xgb_params_path: Optional[str] = 'outputs/models/best_xgb_params.json',
        verbose: bool = True
    ):
        """
        Initialize model stacker.

        Args:
            n_folds: Number of CV folds for stacking
            lgb_params_path: Path to tuned LightGBM params (optional)
            xgb_params_path: Path to tuned XGBoost params (optional)
            verbose: Print progress
        """
        self.n_folds = n_folds
        self.verbose = verbose
        self.scaler = RobustScaler()

        # Load tuned params or use defaults
        self.lgb_params = self._load_params(lgb_params_path, 'lgb')
        self.xgb_params = self._load_params(xgb_params_path, 'xgb')

        # Model weights (sum to 1.0)
        self.weights = {
            'ridge': 0.10,
            'lasso': 0.10,
            'elasticnet': 0.10,
            'xgboost': 0.35,
            'lightgbm': 0.35
        }

        # Store trained models and OOF predictions
        self.models = {}
        self.oof_predictions = {}

    def _load_params(self, path: Optional[str], model_type: str) -> dict:
        """Load tuned parameters from file or return defaults."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                params = json.load(f)
            if self.verbose:
                print(f"   Loaded tuned {model_type} params from {path}")
            return params

        # Default parameters
        if model_type == 'lgb':
            return {
                'n_estimators': 3000,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': -1
            }
        else:  # xgb
            return {
                'n_estimators': 3000,
                'learning_rate': 0.01,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': 0
            }

    def _get_base_models(self) -> Dict:
        """Get base model instances."""
        return {
            'ridge': Ridge(alpha=10, random_state=42),
            'lasso': Lasso(alpha=0.0005, random_state=42, max_iter=10000),
            'elasticnet': ElasticNet(
                alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000
            ),
            'xgboost': xgb.XGBRegressor(**self.xgb_params),
            'lightgbm': lgb.LGBMRegressor(**self.lgb_params)
        }

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fit all models using CV and generate stacked predictions.

        Args:
            X_train: Training features
            y_train: Training target (log-transformed recommended)
            X_test: Test features

        Returns:
            Tuple of (final_predictions, individual_predictions_dict)
        """
        # Scale features for linear models
        X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test

        X_train_scaled = self.scaler.fit_transform(X_train_arr)
        X_test_scaled = self.scaler.transform(X_test_arr)

        predictions = {}
        base_models = self._get_base_models()

        for name, model in base_models.items():
            if self.verbose:
                print(f"   Training {name}...")

            # Use scaled data for linear models, raw for tree models
            if name in ['ridge', 'lasso', 'elasticnet']:
                train_data = X_train_scaled
                test_data = X_test_scaled
            else:
                train_data = X_train_arr
                test_data = X_test_arr

            # Fit with CV and predict
            test_pred, oof_pred = self._fit_predict_cv(
                model, train_data, y_train, test_data
            )

            predictions[name] = test_pred
            self.oof_predictions[name] = oof_pred

            if self.verbose:
                oof_rmse = np.sqrt(np.mean((oof_pred - y_train) ** 2))
                print(f"      {name} OOF RMSE: {oof_rmse:.5f}")

        # Weighted blend
        final_pred = sum(
            self.weights[name] * pred
            for name, pred in predictions.items()
        )

        if self.verbose:
            # Calculate blended OOF score
            blended_oof = sum(
                self.weights[name] * self.oof_predictions[name]
                for name in self.weights.keys()
            )
            blended_rmse = np.sqrt(np.mean((blended_oof - y_train) ** 2))
            print(f"\n   Blended OOF RMSE: {blended_rmse:.5f}")

        return final_pred, predictions

    def _fit_predict_cv(
        self,
        model,
        X: np.ndarray,
        y: pd.Series,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model with CV, generate OOF and test predictions.

        Args:
            model: sklearn-compatible model
            X: Training features
            y: Training target
            X_test: Test features

        Returns:
            Tuple of (test_predictions, oof_predictions)
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        test_preds = np.zeros(len(X_test))
        oof_preds = np.zeros(len(X))

        y_arr = y.values if hasattr(y, 'values') else y

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # Clone model for each fold
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)

            # OOF predictions
            oof_preds[val_idx] = fold_model.predict(X_val)

            # Test predictions (average across folds)
            test_preds += fold_model.predict(X_test) / self.n_folds

        return test_preds, oof_preds

    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom model weights.

        Args:
            weights: Dict mapping model names to weights (should sum to 1.0)
        """
        assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights must sum to 1.0"
        self.weights = weights

    def get_oof_predictions(self) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for each model."""
        return self.oof_predictions
