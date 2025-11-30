# Top Kaggle Solutions Analysis & Improvements

## Executive Summary

Based on analysis of top Kaggle Ames Housing solutions (Top 1%, Top 0.3%), the following key gaps exist between our current agent and winning solutions:

| Gap | Current State | Top Solutions | Priority |
|-----|---------------|---------------|----------|
| **Model Stacking** | LightGBM only | 5-10 models blended | HIGH |
| **Outlier Removal** | None | Remove 2+ extreme outliers | HIGH |
| **Feature Selection** | Use all features | Boruta/RFE/correlation | MEDIUM |
| **Hyperparameter Tuning** | Basic defaults | Optuna/Bayesian per model | MEDIUM |
| **Polynomial Degree** | Squared only | 2nd and 3rd degree | LOW |

---

## Gap 1: Model Stacking/Ensembling (HIGHEST IMPACT)

### What Top Solutions Do
Top 1% solutions typically blend 5-10 diverse models:

```python
# Linear models (regularized)
- Ridge (alpha tuned)
- Lasso (alpha tuned)
- ElasticNet (l1_ratio tuned)

# Gradient boosting
- XGBoost
- LightGBM
- CatBoost
- GradientBoostingRegressor

# Other
- SVR (Support Vector Regression)
- RandomForest
- KernelRidge
```

### Blending Strategy
```python
# Simple averaging (common baseline)
final_pred = 0.70 * stacked_pred + 0.30 * xgboost_pred

# Or weighted average of all models
weights = [0.1, 0.1, 0.1, 0.2, 0.25, 0.25]  # tuned on CV
final_pred = sum(w * pred for w, pred in zip(weights, all_preds))

# Or meta-stacking (train meta-model on OOF predictions)
meta_model = Ridge()
meta_model.fit(oof_predictions, y_train)
final_pred = meta_model.predict(test_predictions)
```

### Recommended Implementation

**New File**: `src/baseline/model_stacker.py`

```python
class ModelStacker:
    """
    Ensemble multiple models using stacking or weighted averaging.

    Models:
    1. Ridge (linear, good for log-transformed target)
    2. Lasso (linear, feature selection built-in)
    3. ElasticNet (linear, balanced L1/L2)
    4. XGBoost (tree-based, different from LightGBM)
    5. LightGBM (current default)

    Blending: Weighted average with weights tuned on CV
    """
```

---

## Gap 2: Outlier Removal (HIGH IMPACT, SIMPLE)

### What Top Solutions Do
Remove well-known outliers in Ames Housing data:

```python
# Famous outliers: Large houses with very low prices
# These 2 points are consistently removed by top solutions
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]

# Remove them (usually index 523 and 1298)
train = train.drop(outliers.index)
```

### Recommended Implementation

**New File**: `src/baseline/outlier_remover.py`

```python
class OutlierRemover:
    """
    Remove known outliers from Ames Housing dataset.

    Strategies:
    1. Known outliers: GrLivArea > 4000 with low SalePrice
    2. Optional: IQR-based removal for selected features
    """

    def remove_known_outliers(self, df):
        """Remove the 2 famous Ames outliers"""
        mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        return df[~mask]
```

---

## Gap 3: Feature Selection (MEDIUM IMPACT)

### What Top Solutions Do
Reduce noise by selecting best features:

```python
# Method 1: Correlation-based
corr_with_target = X.corrwith(y).abs()
selected = corr_with_target[corr_with_target > 0.05].index

# Method 2: LightGBM importance threshold
lgb_model.fit(X, y)
importance = lgb_model.feature_importances_
selected = X.columns[importance > 0]
```

### Recommended Implementation

**New File**: `src/baseline/feature_selector.py`

---

## Gap 4: Hyperparameter Tuning (MEDIUM IMPACT)

### What Top Solutions Do
Tune each model separately using Optuna:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        # ...
    }
    cv_score = cross_val_score(LGBMRegressor(**params), X, y, cv=5)
    return -cv_score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Recommended Implementation

**New File**: `src/baseline/hyperparameter_tuner.py`

---

## User Choices
- **Scope**: All 4 improvements (Outlier removal + Stacking + Tuning + Feature selection)
- **Integration**: Agent uses LightGBM for feature discovery, stacking only for final submission

---

## Implementation Order

### Step 1: Add Dependencies
**File**: `requirements.txt` - Add:
```
xgboost>=2.0.0
optuna>=3.0.0
```

### Step 2: Create Outlier Remover (15 min)
- Create `src/baseline/outlier_remover.py`
- Integrate into `src/agent/iterative_agent.py`

### Step 3: Create Model Stacker (1 hour)
- Create `src/baseline/model_stacker.py`
- Create `src/baseline/make_submission.py` (submission script)

### Step 4: Create Hyperparameter Tuner (30 min)
- Create `src/baseline/hyperparameter_tuner.py`

### Step 5: Create Feature Selector (30 min)
- Create `src/baseline/feature_selector.py`
- Integrate into stacker pipeline

---

## Final Workflow

```
1. Agent discovers features (LightGBM only, fast)
   python -m src.agent.iterative_agent -n 20 --feedback --batch

2. Tune hyperparameters (optional, run once)
   python -m src.baseline.hyperparameter_tuner

3. Generate submission (stacking + feature selection)
   python -m src.baseline.make_submission

   This script:
   - Loads train/test data
   - Removes outliers from training
   - Applies discovered features (from agent memory)
   - Selects best features
   - Runs model stacking
   - Generates submission.csv
```

---

## Files Summary

| File | Purpose | Priority |
|------|---------|----------|
| `src/baseline/outlier_remover.py` | Remove known outliers | 1 |
| `src/baseline/model_stacker.py` | Stack 5 models | 2 |
| `src/baseline/hyperparameter_tuner.py` | Tune with Optuna | 3 |
| `src/baseline/feature_selector.py` | Select best features | 4 |
| `src/baseline/make_submission.py` | End-to-end pipeline | 5 |
| `requirements.txt` | Add xgboost, optuna | 1 |

---

## Expected Impact

| Improvement | Expected RMSLE Reduction |
|-------------|--------------------------|
| Outlier Removal | 0.005 - 0.010 |
| Model Stacking | 0.010 - 0.020 |
| Hyperparameter Tuning | 0.005 - 0.010 |
| Feature Selection | 0.002 - 0.005 |
| **Total** | **0.022 - 0.045** |

If current RMSLE is ~0.12, this could bring it down to ~0.08-0.10, which is competitive with top Kaggle solutions.

---

## Success Metrics

- [ ] Outlier removal reduces training set by 2-3 rows
- [ ] Model stacking CV score better than LightGBM alone
- [ ] Hyperparameter tuning finds better params
- [ ] Feature selection removes noisy features
- [ ] Final submission RMSLE < 0.12 (competitive)
