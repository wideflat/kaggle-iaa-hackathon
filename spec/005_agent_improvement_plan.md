# Agent Improvement Plan: Zero to Hero

## Problem Statement

The agent ran 10 iterations with **ZERO improvements**. All features were rejected.

## Root Cause Analysis

### Bug 1: SHAP Feedback References Non-Existent Features (CRITICAL)

**Location**: `evaluator.py:122-128` and `iterative_agent.py:99-119`

After evaluation, `self.last_model` is trained on the NEW data **including the proposed feature**. When a feature is rejected, `accumulated_df` is NOT updated, but `evaluator.last_model` still references the rejected feature.

Next iteration: SHAP shows `QualTotalSFInteraction` as the #1 important feature, but it doesn't exist in `accumulated_df`. Gemini then generates code using this non-existent column, causing "Column not found" errors.

**Evidence from logs**: Iterations 2-10 all failed with "Column not found: 'QualTotalSFInteraction'" because SHAP incorrectly showed this rejected feature.

### Bug 2: Import Statements Blocked

**Location**: `code_executor.py:29-46`

The sandbox blocks `import` but `np` is already available in `safe_globals`. However, `FEW_SHOT_EXAMPLES` in `gemini_client.py` includes examples with `import numpy as np` (e.g., `LogLotArea`), which causes Gemini to generate code with import statements that fail.

### Bug 3: Fundamental Approach Issue

The baseline already has 212 features with engineered features (TotalSF, HouseAge, etc.). LightGBM automatically learns feature interactions, so simple `A * B` features add no value.

**From Top Kaggle Solutions** (user's reference: 1% solution):
1. Outlier removal (GrLivArea > 4500)
2. Skewness correction (BoxCox for features with skewness > 0.75)
3. Feature selection (Lasso-based)
4. Model stacking/ensembling

---

## Implementation Plan

### Phase 1: Fix Critical Bugs (Priority: HIGH) - CURRENT

#### 1.1 Fix SHAP Feedback Bug

**File**: `src/agent/evaluator.py`

Add method to re-compute SHAP on a specific dataframe:

```python
def recompute_shap_for_features(self, X: pd.DataFrame, y: pd.Series):
    """Retrain model on given features and update SHAP"""
    y_log = np.log1p(y)
    self.last_X = X
    self.last_model = lgb.LGBMRegressor(**self.model_params)
    self.last_model.fit(X, y_log)
```

**File**: `src/agent/iterative_agent.py`

After rejection, re-compute SHAP on accumulated_df (line ~177):

```python
if not is_better:
    # Re-compute SHAP on actual accumulated features, not rejected ones
    X_current, _ = get_features_and_target(accumulated_df)
    evaluator.recompute_shap_for_features(X_current, target)
```

#### 1.2 Fix Import Issue in Few-Shot Examples

**File**: `src/agent/gemini_client.py`

Remove `import numpy as np` from examples - just use `np.` directly since it's pre-loaded:

```python
# BEFORE
'code': "import numpy as np\ndf['LogLotArea'] = np.log1p(df['LotArea'])"

# AFTER
'code': "df['LogLotArea'] = np.log1p(df['LotArea'])"
```

Add explicit instruction in prompt to NOT use import statements.

---

### Phase 2: Add Data Preprocessing Improvements (Priority: HIGH)

#### 2.1 Outlier Removal

**File**: `src/baseline/preprocessor.py` (or new `src/agent/data_cleaner.py`)

```python
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known outliers from Ames dataset"""
    # Houses with GrLivArea > 4500 are outliers (from EDA)
    df = df[df['GrLivArea'] < 4500]
    return df
```

Apply before training in `iterative_agent.py`.

#### 2.2 Skewness Correction

**File**: `src/agent/feature_transforms.py` (new)

```python
from scipy.stats import skew
from scipy.special import boxcox1p

def fix_skewness(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Apply BoxCox transform to highly skewed features"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    skewed = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    skewed_features = skewed[abs(skewed) > threshold].index

    for col in skewed_features:
        df[col] = boxcox1p(df[col], 0.15)  # Lambda from top solutions
    return df
```

---

### Phase 3: Improve Feature Generation Strategy (Priority: MEDIUM)

#### 3.1 Batch Feature Generation

**File**: `src/agent/gemini_client.py`

Add method to generate multiple features at once:

```python
def generate_feature_batch(self, ..., n_features: int = 5) -> list[str]:
    """Generate N features in a single prompt"""
    # Modify prompt to ask for N features
    # Return list of code snippets
```

**File**: `src/agent/iterative_agent.py`

Add batch evaluation mode:
- Generate 5 features at once
- Evaluate each independently
- Keep all that improve

#### 3.2 Add Domain-Specific Strategies

**File**: `src/agent/gemini_client.py`

Add new strategies based on top Kaggle solutions:

```python
'neighborhood': [
    {
        'name': 'NeighborhoodQual',
        'code': "df['NeighborhoodQual'] = df.groupby('Neighborhood')['SalePrice'].transform('median')",
        'rationale': 'Neighborhood median price as quality proxy'
    }
],
'target_encoding': [
    {
        'name': 'OverallQual_MeanPrice',
        'code': "df['OverallQual_MeanPrice'] = df.groupby('OverallQual')['SalePrice'].transform('mean')",
        'rationale': 'Target encoding for OverallQual'
    }
]
```

---

### Phase 4: Model Improvements (Priority: MEDIUM)

#### 4.1 Add Model Stacking Option

**File**: `src/agent/evaluator.py`

Add ensemble evaluation:

```python
def evaluate_stacked(self, X, y) -> float:
    """Evaluate with stacked models (Ridge + Lasso + LightGBM)"""
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import StackingRegressor

    base_models = [
        ('ridge', Ridge(alpha=10)),
        ('lasso', Lasso(alpha=0.0005)),
        ('lgbm', lgb.LGBMRegressor(**self.model_params))
    ]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
    # ... CV evaluation
```

#### 4.2 Feature Selection

**File**: `src/agent/feature_selector.py` (new)

```python
from sklearn.linear_model import Lasso

def select_features(X, y, alpha=0.0005):
    """Use Lasso for feature selection"""
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, np.log1p(y))

    # Features with non-zero coefficients
    selected = X.columns[lasso.coef_ != 0].tolist()
    return selected
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/agent/evaluator.py` | Add `recompute_shap_for_features()` method |
| `src/agent/iterative_agent.py` | Fix SHAP recompute after rejection, add outlier removal |
| `src/agent/gemini_client.py` | Remove imports from examples, add "no import" instruction |
| `src/baseline/preprocessor.py` | Add outlier removal option |
| `src/agent/feature_transforms.py` | NEW: Skewness correction |
| `src/agent/feature_selector.py` | NEW: Lasso-based feature selection |

---

## Execution Order

1. **Immediate Fixes** (Phase 1): Fix the bugs first - this alone may significantly improve results
2. **Quick Wins** (Phase 2.1): Add outlier removal - minimal code, proven to help
3. **Skewness** (Phase 2.2): Add BoxCox transforms
4. **Better Generation** (Phase 3): Improve prompts and add batch mode
5. **Model Stacking** (Phase 4): Add as optional enhancement

---

## Expected Outcomes

After Phase 1 (Bug Fixes):
- SHAP will only show features that actually exist
- Features with numpy functions will execute correctly
- Success rate should increase from 0% to at least 20-30%

After Phase 2 (Data Preprocessing):
- Outlier removal: ~0.5-1% RMSLE improvement (from top solutions)
- Skewness correction: ~0.3-0.5% additional improvement

After Phases 3-4:
- Better feature generation strategies
- Model stacking for final boost

---

## Success Metrics

- [ ] At least 1 successful feature in 10 iterations (vs 0 currently)
- [ ] RMSLE improvement over baseline (vs 0% currently)
- [ ] No "Column not found" errors from SHAP feedback
- [ ] No import-related execution failures
