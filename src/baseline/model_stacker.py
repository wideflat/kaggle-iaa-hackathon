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
        verbose: bool = True,
        enable_stacking: bool = True,
        meta_learner_type: str = 'ridge',
        meta_learner_alpha: float = 1.0,
        layer3_weights: Optional[Dict[str, float]] = None,
        verbose_layers: bool = True
    ):
        """
        Initialize model stacker with three-layer ensemble architecture.

        Args:
            n_folds: Number of CV folds for stacking
            lgb_params_path: Path to tuned LightGBM params (optional)
            xgb_params_path: Path to tuned XGBoost params (optional)
            verbose: Print progress
            enable_stacking: Enable Layer 2b stacking meta-learner (default: True)
            meta_learner_type: Type of meta-learner ('ridge', 'lasso', 'elasticnet')
            meta_learner_alpha: Regularization strength for meta-learner
            layer3_weights: Custom weights for Layer 3 ensemble (default: 50-50)
            verbose_layers: Show OOF scores for each layer
        """
        self.n_folds = n_folds
        self.verbose = verbose
        self.scaler = RobustScaler()

        # Three-layer ensemble parameters
        self.enable_stacking = enable_stacking
        self.meta_learner_type = meta_learner_type
        self.meta_learner_alpha = meta_learner_alpha
        self.verbose_layers = verbose_layers

        # Layer 3 weights (default: equal weighting)
        if layer3_weights is None:
            self.layer3_weights = {'weighted': 0.5, 'stacking': 0.5}
        else:
            assert abs(sum(layer3_weights.values()) - 1.0) < 0.001, \
                "Layer 3 weights must sum to 1.0"
            self.layer3_weights = layer3_weights

        # Load tuned params or use defaults
        self.lgb_params = self._load_params(lgb_params_path, 'lgb')
        self.xgb_params = self._load_params(xgb_params_path, 'xgb')

        # Layer 2a: Model weights for weighted average (sum to 1.0)
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
        self.meta_learner = None

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

    def _create_meta_features(
        self,
        predictions_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Create meta-features from base model predictions.

        Args:
            predictions_dict: Dict mapping model names to predictions

        Returns:
            np.ndarray of shape (n_samples, n_models) with predictions stacked column-wise
        """
        # Consistent ordering of base models
        model_order = ['ridge', 'lasso', 'elasticnet', 'xgboost', 'lightgbm']

        # Stack predictions column-wise
        meta_features = np.column_stack([
            predictions_dict[name] for name in model_order
        ])

        return meta_features

    def _train_meta_learner(self, y_train: pd.Series):
        """
        Train meta-learner on OOF predictions from Layer 1.

        Args:
            y_train: Training target (log-transformed)

        Returns:
            Trained meta-learner model
        """
        # Create meta-features from OOF predictions
        X_meta = self._create_meta_features(self.oof_predictions)

        # Select meta-learner type
        if self.meta_learner_type == 'ridge':
            meta_model = Ridge(alpha=self.meta_learner_alpha, random_state=42)
        elif self.meta_learner_type == 'lasso':
            meta_model = Lasso(
                alpha=self.meta_learner_alpha,
                random_state=42,
                max_iter=10000
            )
        elif self.meta_learner_type == 'elasticnet':
            meta_model = ElasticNet(
                alpha=self.meta_learner_alpha,
                l1_ratio=0.9,
                random_state=42,
                max_iter=10000
            )
        else:
            raise ValueError(
                f"Unknown meta_learner_type: {self.meta_learner_type}. "
                f"Choose from: 'ridge', 'lasso', 'elasticnet'"
            )

        # Train on OOF predictions (no leakage)
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train
        meta_model.fit(X_meta, y_arr)

        return meta_model

    def _get_stacking_predictions(
        self,
        test_predictions_dict: Dict[str, np.ndarray],
        meta_learner
    ) -> np.ndarray:
        """
        Get stacking predictions using trained meta-learner.

        Args:
            test_predictions_dict: Dict mapping model names to test predictions
            meta_learner: Trained meta-learner model

        Returns:
            Stacking predictions
        """
        # Create meta-features from test predictions
        X_meta_test = self._create_meta_features(test_predictions_dict)

        # Apply meta-learner
        stacking_pred = meta_learner.predict(X_meta_test)

        return stacking_pred

    def _calculate_layer_scores(
        self,
        y_train: pd.Series,
        weighted_pred_oof: np.ndarray,
        stacking_pred_oof: Optional[np.ndarray] = None,
        final_pred_oof: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate OOF RMSE for all layers.

        Args:
            y_train: Training target (log-transformed)
            weighted_pred_oof: Layer 2a OOF predictions
            stacking_pred_oof: Layer 2b OOF predictions (optional)
            final_pred_oof: Layer 3 OOF predictions (optional)

        Returns:
            Dict mapping layer names to OOF RMSE scores
        """
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        scores = {}

        # Layer 1: Individual base model scores
        scores['layer1'] = {}
        for name, oof_pred in self.oof_predictions.items():
            rmse = np.sqrt(np.mean((oof_pred - y_arr) ** 2))
            scores['layer1'][name] = rmse

        # Layer 2a: Weighted average
        scores['layer2_weighted'] = np.sqrt(
            np.mean((weighted_pred_oof - y_arr) ** 2)
        )

        # Layer 2b: Stacking (if enabled)
        if stacking_pred_oof is not None:
            scores['layer2_stacking'] = np.sqrt(
                np.mean((stacking_pred_oof - y_arr) ** 2)
            )

        # Layer 3: Final ensemble
        if final_pred_oof is not None:
            scores['layer3_final'] = np.sqrt(
                np.mean((final_pred_oof - y_arr) ** 2)
            )

        return scores

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """
        Fit all models using CV and generate three-layer ensemble predictions.

        Architecture:
            Layer 1: 5 base models (Ridge, Lasso, ElasticNet, XGBoost, LightGBM)
            Layer 2a: Weighted average of Layer 1
            Layer 2b: Stacking meta-learner trained on Layer 1 OOF predictions
            Layer 3: Ensemble of Layer 2a and 2b

        Args:
            X_train: Training features
            y_train: Training target (log-transformed recommended)
            X_test: Test features

        Returns:
            Tuple of (final_predictions, layer_predictions_dict, oof_scores_dict)
        """
        # ===== LAYER 1: Train base models =====
        if self.verbose:
            print("\n   === LAYER 1: Base Models ===")

        # Convert to arrays for scaling (linear models only)
        X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test

        X_train_scaled = self.scaler.fit_transform(X_train_arr)
        X_test_scaled = self.scaler.transform(X_test_arr)

        predictions = {}
        base_models = self._get_base_models()

        for name, model in base_models.items():
            if self.verbose:
                print(f"   Training {name}...")

            # Use scaled arrays for linear models, DataFrames for tree models
            if name in ['ridge', 'lasso', 'elasticnet']:
                train_data = X_train_scaled
                test_data = X_test_scaled
            else:
                # Keep as DataFrame to preserve feature names
                train_data = X_train
                test_data = X_test

            # Fit with CV and predict
            test_pred, oof_pred = self._fit_predict_cv(
                model, train_data, y_train, test_data
            )

            predictions[name] = test_pred
            self.oof_predictions[name] = oof_pred

            if self.verbose:
                oof_rmse = np.sqrt(np.mean((oof_pred - y_train) ** 2))
                print(f"      {name} OOF RMSE: {oof_rmse:.5f}")

        # ===== LAYER 2a: Weighted Average =====
        if self.verbose:
            print("\n   === LAYER 2a: Weighted Average ===")

        weighted_pred = sum(
            self.weights[name] * pred
            for name, pred in predictions.items()
        )

        weighted_oof = sum(
            self.weights[name] * self.oof_predictions[name]
            for name in self.weights.keys()
        )

        if self.verbose:
            weighted_rmse = np.sqrt(np.mean((weighted_oof - y_train) ** 2))
            print(f"   OOF RMSE: {weighted_rmse:.5f}")

        # ===== LAYER 2b: Stacking Meta-Learner =====
        stacking_pred = None
        stacking_oof = None

        if self.enable_stacking:
            if self.verbose:
                print("\n   === LAYER 2b: Stacking Meta-Learner ===")
                print(f"   Training meta-learner ({self.meta_learner_type}, "
                      f"alpha={self.meta_learner_alpha})...")

            # Train meta-learner on OOF predictions
            self.meta_learner = self._train_meta_learner(y_train)

            # Get stacking predictions
            stacking_pred = self._get_stacking_predictions(
                predictions, self.meta_learner
            )

            # Get stacking OOF predictions (meta-learner on OOF features)
            X_meta_oof = self._create_meta_features(self.oof_predictions)
            stacking_oof = self.meta_learner.predict(X_meta_oof)

            if self.verbose:
                stacking_rmse = np.sqrt(np.mean((stacking_oof - y_train) ** 2))
                print(f"   OOF RMSE: {stacking_rmse:.5f}")

        # ===== LAYER 3: Final Ensemble =====
        if self.enable_stacking:
            if self.verbose:
                print("\n   === LAYER 3: Final Ensemble ===")
                print(f"   Weights: weighted={self.layer3_weights['weighted']:.2f}, "
                      f"stacking={self.layer3_weights['stacking']:.2f}")

            final_pred = (
                self.layer3_weights['weighted'] * weighted_pred +
                self.layer3_weights['stacking'] * stacking_pred
            )

            final_oof = (
                self.layer3_weights['weighted'] * weighted_oof +
                self.layer3_weights['stacking'] * stacking_oof
            )

            if self.verbose:
                final_rmse = np.sqrt(np.mean((final_oof - y_train) ** 2))
                print(f"   OOF RMSE: {final_rmse:.5f}")
        else:
            # If stacking disabled, use weighted average as final
            final_pred = weighted_pred
            final_oof = weighted_oof

        # ===== Calculate all layer scores =====
        oof_scores = self._calculate_layer_scores(
            y_train,
            weighted_oof,
            stacking_oof,
            final_oof if self.enable_stacking else None
        )

        # ===== Prepare return values =====
        layer_predictions = {
            'layer1': predictions,
            'layer2_weighted': weighted_pred,
            'layer2_stacking': stacking_pred,
            'layer3_final': final_pred
        }

        return final_pred, layer_predictions, oof_scores

    def _fit_predict_cv(
        self,
        model,
        X,
        y: pd.Series,
        X_test
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model with CV, generate OOF and test predictions.

        Args:
            model: sklearn-compatible model
            X: Training features (DataFrame or ndarray)
            y: Training target
            X_test: Test features (DataFrame or ndarray)

        Returns:
            Tuple of (test_predictions, oof_predictions)
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        test_preds = np.zeros(len(X_test))
        oof_preds = np.zeros(len(X))

        y_arr = y.values if hasattr(y, 'values') else y

        # Check if X is DataFrame or array
        is_dataframe = isinstance(X, pd.DataFrame)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            # Use iloc for DataFrames, direct indexing for arrays
            if is_dataframe:
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
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
