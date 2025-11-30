"""
Hyperparameter tuning using Optuna

Tunes LightGBM, XGBoost, and linear models separately.
Saves best parameters to JSON for use in model stacking.
"""

import os
import json
import numpy as np
from typing import Dict, Optional
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
import pandas as pd


class HyperparameterTuner:
    """
    Tune model hyperparameters using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) for efficient search.
    Saves best parameters to JSON files for reproducibility.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize tuner.

        Args:
            n_folds: Number of CV folds
            random_state: Random seed for reproducibility
            verbose: Show progress
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose

        # Optuna settings
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )

    def tune_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        output_path: str = 'outputs/models/best_lgbm_params.json'
    ) -> Dict:
        """
        Tune LightGBM hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable (log-transformed)
            n_trials: Number of Optuna trials
            output_path: Path to save best params

        Returns:
            Best parameters dict
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.005, 0.1, log=True
                ),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbosity': -1
            }

            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(
                model, X, y,
                cv=self.n_folds,
                scoring='neg_root_mean_squared_error'
            )
            return -scores.mean()

        if self.verbose:
            print(f"\nTuning LightGBM ({n_trials} trials)...")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=self.verbose
        )

        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['verbosity'] = -1

        # Save best params
        self._save_params(best_params, output_path)

        if self.verbose:
            print(f"   Best LightGBM RMSE: {study.best_value:.5f}")
            print(f"   Params saved to: {output_path}")

        return best_params

    def tune_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        output_path: str = 'outputs/models/best_xgb_params.json'
    ) -> Dict:
        """
        Tune XGBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable (log-transformed)
            n_trials: Number of Optuna trials
            output_path: Path to save best params

        Returns:
            Best parameters dict
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.005, 0.1, log=True
                ),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbosity': 0
            }

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(
                model, X, y,
                cv=self.n_folds,
                scoring='neg_root_mean_squared_error'
            )
            return -scores.mean()

        if self.verbose:
            print(f"\nTuning XGBoost ({n_trials} trials)...")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=self.verbose
        )

        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['verbosity'] = 0

        # Save best params
        self._save_params(best_params, output_path)

        if self.verbose:
            print(f"   Best XGBoost RMSE: {study.best_value:.5f}")
            print(f"   Params saved to: {output_path}")

        return best_params

    def tune_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100
    ) -> Dict[str, Dict]:
        """
        Tune all models.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of trials per model

        Returns:
            Dict mapping model name to best params
        """
        results = {}

        results['lightgbm'] = self.tune_lightgbm(X, y, n_trials)
        results['xgboost'] = self.tune_xgboost(X, y, n_trials)

        return results

    def _save_params(self, params: Dict, path: str):
        """Save parameters to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)


def main():
    """Run hyperparameter tuning as standalone script."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.baseline.data_loader import AmesDataLoader
    from src.baseline.minimal_preprocessor import MinimalPreprocessor
    from src.baseline.outlier_remover import OutlierRemover
    from src.agent.evaluator import get_features_and_target

    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)

    # Load and preprocess data
    print("\n1. Loading data...")
    loader = AmesDataLoader()
    train_df = loader.load_train()

    print("\n2. Removing outliers...")
    outlier_remover = OutlierRemover()
    train_df = outlier_remover.remove_known_outliers(train_df)

    print("\n3. Preprocessing...")
    preprocessor = MinimalPreprocessor()
    target = train_df['SalePrice'].copy()
    train_processed = preprocessor.fit_transform(train_df)

    print("\n4. Extracting features...")
    X, _ = get_features_and_target(train_processed)
    y = np.log1p(target)  # Log-transform target

    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")

    # Run tuning
    print("\n5. Tuning hyperparameters...")
    tuner = HyperparameterTuner(n_folds=5)
    tuner.tune_all(X, y, n_trials=50)

    print("\n" + "=" * 60)
    print("Tuning complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
