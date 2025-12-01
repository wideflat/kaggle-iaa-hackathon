"""
Kaggle-style 3-Layer Model Blending + Stacking using mlxtend

Based on top Kaggle Ames Housing solutions.

Architecture:
- Layer 1: 7 individual models (Ridge, Lasso, ElasticNet, SVR, GBR, XGBoost, LightGBM)
- Layer 2:
    a) Weighted average of Layer 1 predictions
    b) StackingCVRegressor with XGBoost meta-regressor
- Layer 3: 50/50 blend of Layer 2a and Layer 2b
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
import xgboost as xgb


class ModelStacker:
    """
    Kaggle-style 3-layer model stacking with mlxtend StackingCVRegressor.

    Architecture:
    - Layer 1: 7 individual models
    - Layer 2a: Weighted average of Layer 1
    - Layer 2b: StackingCVRegressor
    - Layer 3: 50/50 blend of Layer 2a and Layer 2b
    """

    def __init__(
        self,
        n_folds: int = 10,
        lgb_params_path: Optional[str] = 'outputs/models/best_lgbm_params.json',
        xgb_params_path: Optional[str] = 'outputs/models/best_xgb_params.json',
        verbose: bool = True
    ):
        """
        Initialize model stacker.

        Args:
            n_folds: Number of CV folds
            lgb_params_path: Path to tuned LightGBM params
            xgb_params_path: Path to tuned XGBoost params
            verbose: Print progress
        """
        self.n_folds = n_folds
        self.verbose = verbose
        self.kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Load tuned params or use defaults
        self.lgb_params = self._load_params(lgb_params_path, 'lgb')
        self.xgb_params = self._load_params(xgb_params_path, 'xgb')

        # Layer 1: Blending weights for individual models (sum to 1.0)
        self.layer1_weights = {
            'ridge': 0.10,
            'lasso': 0.10,
            'elasticnet': 0.10,
            'svr': 0.05,  # Lower weight due to higher error
            'gbr': 0.20,
            'xgboost': 0.20,
            'lightgbm': 0.25,
        }

        # Layer 3: Blend weights for Layer 2 outputs
        self.layer3_weights = {
            'weighted': 0.75,
            'stacking': 0.25
        }

        # Trained models storage
        self.models = {}
        self.stack_gen = None
        self.oof_predictions = {}

    def _load_params(self, path: Optional[str], model_type: str) -> dict:
        """Load tuned parameters from file or return defaults."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                params = json.load(f)
            if self.verbose:
                print(f"   Loaded tuned {model_type} params from {path}")
            return params

        # Default parameters (from Kaggle notebook)
        if model_type == 'lgb':
            return {
                'objective': 'regression',
                'num_leaves': 4,
                'learning_rate': 0.01,
                'n_estimators': 5000,
                'max_bin': 200,
                'bagging_fraction': 0.75,
                'bagging_freq': 5,
                'bagging_seed': 7,
                'feature_fraction': 0.2,
                'feature_fraction_seed': 7,
                'verbose': -1,
                'random_state': 42
            }
        else:  # xgb
            return {
                'learning_rate': 0.01,
                'n_estimators': 3460,
                'max_depth': 3,
                'min_child_weight': 0,
                'gamma': 0,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.00006,
                'random_state': 42,
                'verbosity': 0
            }

    def _create_base_models(self) -> Dict:
        """Create all base model instances with Kaggle hyperparameters."""
        # Alpha ranges for CV-tuned linear models
        alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
        alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
        alphas_elastic = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
        l1_ratios = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

        return {
            'ridge': make_pipeline(
                RobustScaler(),
                RidgeCV(alphas=alphas_ridge, cv=self.kfolds)
            ),
            'lasso': make_pipeline(
                RobustScaler(),
                LassoCV(max_iter=int(1e7), alphas=alphas_lasso, random_state=42, cv=self.kfolds)
            ),
            'elasticnet': make_pipeline(
                RobustScaler(),
                ElasticNetCV(max_iter=int(1e7), alphas=alphas_elastic, cv=self.kfolds, l1_ratio=l1_ratios)
            ),
            'svr': make_pipeline(
                RobustScaler(),
                SVR(C=20, epsilon=0.008, gamma=0.0003)
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=3000,
                learning_rate=0.05,
                max_depth=4,
                max_features='sqrt',
                min_samples_leaf=15,
                min_samples_split=10,
                loss='huber',
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(**self.xgb_params),
            'lightgbm': lgb.LGBMRegressor(**self.lgb_params)
        }

    def _create_stacking_regressor(self) -> StackingCVRegressor:
        """Create mlxtend StackingCVRegressor with XGBoost meta-regressor."""
        # Alpha ranges for CV-tuned linear models
        alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
        alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
        alphas_elastic = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
        l1_ratios = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

        # Base estimators for stacking
        ridge = make_pipeline(
            RobustScaler(),
            RidgeCV(alphas=alphas_ridge, cv=self.kfolds)
        )
        lasso = make_pipeline(
            RobustScaler(),
            LassoCV(max_iter=int(1e7), alphas=alphas_lasso, random_state=42, cv=self.kfolds)
        )
        elasticnet = make_pipeline(
            RobustScaler(),
            ElasticNetCV(max_iter=int(1e7), alphas=alphas_elastic, cv=self.kfolds, l1_ratio=l1_ratios)
        )
        gbr = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features='sqrt',
            min_samples_leaf=15,
            min_samples_split=10,
            loss='huber',
            random_state=42
        )
        xgboost = xgb.XGBRegressor(**self.xgb_params)
        lightgbm = lgb.LGBMRegressor(**self.lgb_params)

        # Meta-regressor (XGBoost)
        meta_xgb = xgb.XGBRegressor(
            learning_rate=0.01,
            n_estimators=2000,
            max_depth=3,
            min_child_weight=0,
            gamma=0,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.00006,
            random_state=42,
            verbosity=0
        )

        # Use mlxtend StackingCVRegressor
        return StackingCVRegressor(
            regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
            meta_regressor=meta_xgb,
            use_features_in_secondary=True  # Include original features for meta-regressor
        )

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """
        Fit all models and generate 3-layer blended predictions.

        Args:
            X_train: Training features
            y_train: Training target (log-transformed recommended)
            X_test: Test features

        Returns:
            Tuple of (final_predictions, layer_predictions_dict, oof_scores_dict)
        """
        # Convert to numpy arrays for mlxtend compatibility
        X_arr = np.array(X_train) if hasattr(X_train, 'values') else np.array(X_train)
        y_arr = np.array(y_train) if hasattr(y_train, 'values') else np.array(y_train)
        X_test_arr = np.array(X_test) if hasattr(X_test, 'values') else np.array(X_test)

        layer1_preds = {}
        layer1_oof = {}

        # ===== LAYER 1: Train individual models =====
        if self.verbose:
            print("\n   === Layer 1: Training Individual Models ===")

        base_models = self._create_base_models()

        for name, model in base_models.items():
            if self.verbose:
                print(f"   Training {name}...")

            # Fit on full data
            model.fit(X_arr, y_arr)
            self.models[name] = model

            # Generate test predictions
            layer1_preds[name] = model.predict(X_test_arr)

            # Calculate OOF predictions for scoring
            oof_pred = self._get_oof_predictions(model, X_arr, y_arr)
            layer1_oof[name] = oof_pred
            self.oof_predictions[name] = oof_pred

            if self.verbose:
                oof_rmse = np.sqrt(np.mean((oof_pred - y_arr) ** 2))
                print(f"      {name} OOF RMSE: {oof_rmse:.5f}")

        # ===== LAYER 2a: Weighted average of Layer 1 =====
        if self.verbose:
            print("\n   === Layer 2a: Weighted Average ===")
            print(f"   Weights: {self.layer1_weights}")

        # Test predictions
        layer2_weighted = np.zeros(len(X_test_arr))
        for name, weight in self.layer1_weights.items():
            layer2_weighted += weight * layer1_preds[name]

        # OOF predictions
        layer2_weighted_oof = np.zeros(len(X_arr))
        for name, weight in self.layer1_weights.items():
            layer2_weighted_oof += weight * layer1_oof[name]

        weighted_oof_rmse = np.sqrt(np.mean((layer2_weighted_oof - y_arr) ** 2))
        if self.verbose:
            print(f"   Layer 2a (Weighted) OOF RMSE: {weighted_oof_rmse:.5f}")

        # ===== LAYER 2b: StackingCVRegressor =====
        if self.verbose:
            print("\n   === Layer 2b: StackingCVRegressor (mlxtend) ===")
            print("   Base models: ridge, lasso, elasticnet, gbr, xgboost, lightgbm")
            print("   Meta-regressor: XGBoost")
            print("   use_features_in_secondary: True")

        self.stack_gen = self._create_stacking_regressor()
        self.stack_gen.fit(X_arr, y_arr)

        # Test predictions
        layer2_stacking = self.stack_gen.predict(X_test_arr)

        # OOF predictions (train data predictions - note: this is optimistic)
        layer2_stacking_oof = self.stack_gen.predict(X_arr)

        stacking_oof_rmse = np.sqrt(np.mean((layer2_stacking_oof - y_arr) ** 2))
        if self.verbose:
            print(f"   Layer 2b (Stacking) train RMSE: {stacking_oof_rmse:.5f}")

        # ===== LAYER 3: Final blend (50/50) =====
        if self.verbose:
            print("\n   === Layer 3: Final Blend ===")
            print(f"   Weights: {self.layer3_weights}")

        # Test predictions
        layer3_final = (
            self.layer3_weights['weighted'] * layer2_weighted +
            self.layer3_weights['stacking'] * layer2_stacking
        )

        # OOF predictions
        layer3_final_oof = (
            self.layer3_weights['weighted'] * layer2_weighted_oof +
            self.layer3_weights['stacking'] * layer2_stacking_oof
        )

        final_oof_rmse = np.sqrt(np.mean((layer3_final_oof - y_arr) ** 2))
        if self.verbose:
            print(f"   Layer 3 (Final) OOF RMSE: {final_oof_rmse:.5f}")

        # ===== Calculate scores =====
        oof_scores = {'layer1': {}}
        for name, oof_pred in layer1_oof.items():
            oof_scores['layer1'][name] = np.sqrt(np.mean((oof_pred - y_arr) ** 2))

        oof_scores['layer2_weighted'] = weighted_oof_rmse
        oof_scores['layer2_stacking'] = stacking_oof_rmse
        oof_scores['layer3_final'] = final_oof_rmse

        # ===== Prepare return values =====
        layer_predictions = {
            # Test predictions
            'layer1': layer1_preds,
            'layer2_weighted': layer2_weighted,
            'layer2_stacking': layer2_stacking,
            'layer3_final': layer3_final,
            # OOF predictions
            'oof_layer1': layer1_oof,
            'oof_layer2_weighted': layer2_weighted_oof,
            'oof_layer2_stacking': layer2_stacking_oof,
            'oof_layer3_final': layer3_final_oof
        }

        return layer3_final, layer_predictions, oof_scores

    def _get_oof_predictions(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Generate out-of-fold predictions for a model."""
        oof_preds = np.zeros(len(X))

        for train_idx, val_idx in self.kfolds.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            oof_preds[val_idx] = fold_model.predict(X_val)

        return oof_preds

    def set_layer1_weights(self, weights: Dict[str, float]):
        """
        Set custom Layer 1 blending weights.

        Args:
            weights: Dict mapping model names to weights (should sum to 1.0)
        """
        assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights must sum to 1.0"
        self.layer1_weights = weights

    def set_layer3_weights(self, weights: Dict[str, float]):
        """
        Set custom Layer 3 blending weights.

        Args:
            weights: Dict with 'weighted' and 'stacking' keys (should sum to 1.0)
        """
        assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights must sum to 1.0"
        self.layer3_weights = weights

    def get_oof_predictions(self) -> Dict[str, np.ndarray]:
        """Get out-of-fold predictions for each model."""
        return self.oof_predictions
