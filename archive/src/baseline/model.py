"""
Model selection and hyperparameter tuning for Ames Housing
Phase A3: Hyperparameter Tuning
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import optuna
from optuna.samplers import TPESampler


def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class LGBMTuner:
    """
    Hyperparameter tuning for LightGBM using Optuna
    """

    def __init__(self, X, y, n_folds=5, n_trials=50, random_state=42):
        """
        Initialize tuner

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target (log-transformed)
        n_folds : int
            Number of CV folds
        n_trials : int
            Number of Optuna trials
        random_state : int
            Random seed
        """
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.study = None

    def objective(self, trial):
        """
        Optuna objective function

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float : Mean CV RMSE (on log scale)
        """

        # Define hyperparameter search space
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        # Cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train_fold = self.X.iloc[train_idx]
            X_val_fold = self.X.iloc[val_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_val_fold = self.y.iloc[val_idx]

            # Train model
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Predict
            val_pred = model.predict(X_val_fold)

            # Calculate RMSE on log scale
            fold_rmse = rmse(y_val_fold, val_pred)
            cv_scores.append(fold_rmse)

        mean_cv_rmse = np.mean(cv_scores)
        return mean_cv_rmse

    def tune(self, verbose=True):
        """
        Run hyperparameter tuning

        Parameters:
        -----------
        verbose : bool
            Print progress

        Returns:
        --------
        dict : Best hyperparameters
        """
        if verbose:
            print(f"Starting hyperparameter tuning with {self.n_trials} trials...")

        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='lgbm_tuning'
        )

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_params['objective'] = 'regression'
        self.best_params['metric'] = 'rmse'
        self.best_params['verbosity'] = -1
        self.best_params['random_state'] = self.random_state

        if verbose:
            print(f"\nBest CV RMSE (log): {self.study.best_value:.5f}")
            print(f"\nBest hyperparameters:")
            for key, value in self.best_params.items():
                if key not in ['objective', 'metric', 'verbosity', 'random_state']:
                    print(f"  {key}: {value}")

        return self.best_params

    def get_best_model(self):
        """
        Get model with best hyperparameters

        Returns:
        --------
        lgb.LGBMRegressor : Model with best params
        """
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        return lgb.LGBMRegressor(**self.best_params)


def get_default_params():
    """
    Get default LightGBM parameters (Phase A1 baseline)

    Returns:
    --------
    dict : Default parameters
    """
    return {
        'n_estimators': 100,
        'random_state': 42,
        'verbosity': -1
    }


def get_tuned_params():
    """
    Get pre-tuned LightGBM parameters (if available)

    Returns:
    --------
    dict : Tuned parameters or None
    """
    # Will be populated after first tuning run
    return None
