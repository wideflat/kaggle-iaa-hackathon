# True Minimal Preprocessor Plan (v2)

**Status: IMPLEMENTED**

## Goal

Make the preprocessor truly minimal so agent can discover:
1. **Semantic NA handling** → Binary presence features (HasGarage, HasPool)
2. **Quality ordinal encoding** → Agent reads data_description, creates mappings
3. **Grouped imputation** → Agent creates neighborhood-based features

## Key Design Decision

**Keep categorical columns as original strings** (not label encoded), so agent can:
- Read `data_description.txt` to understand what "Ex", "Gd", "TA" mean
- Create ordinal mappings: `df['ExterQual'].map({'Ex': 5, 'Gd': 4, ...})`
- Create binary features: `df['HasPool'] = (df['PoolQC'] != 'None').astype(int)`

Use LightGBM's native categorical support to handle string columns directly.

---

## Implementation Status

| File | Action | Status |
|------|--------|--------|
| `src/baseline/minimal_preprocessor.py` | REWRITE: True minimal (keep strings, use pd.Categorical) | ✅ Done |
| `src/agent/evaluator.py` | Update get_features_and_target() for categorical handling | ✅ Done |
| `src/agent/gemini_client.py` | Add presence_detection, ordinal_from_description, grouped_features strategies | ✅ Done |
| `src/agent/gemini_client.py` | Add categorical guidance and target leakage warning to prompts | ✅ Done |

---

## What MinimalPreprocessor Does

```python
class MinimalPreprocessor:
    """
    True minimal preprocessing - agent discovers everything.

    Does ONLY:
    1. Numeric NaN → 0 (agent can detect absence via > 0)
    2. Categorical NaN → "None" string
    3. Convert categoricals to pd.Categorical dtype

    Does NOT:
    - Ordinal encoding (agent discovers from data_description)
    - Grouped imputation (agent discovers)
    - Feature engineering (agent discovers)
    """
```

---

## New Strategies Added to Gemini

### 1. presence_detection
- `HasGarage`: `(df['GarageArea'] > 0).astype(int)`
- `HasPool`: `(df['PoolQC'] != 'None').astype(int)`
- `HasBasement`: `(df['TotalBsmtSF'] > 0).astype(int)`
- `Has2ndFloor`: `(df['2ndFlrSF'] > 0).astype(int)`
- `HasFireplace`: `(df['Fireplaces'] > 0).astype(int)`

### 2. ordinal_from_description
- `ExterQual_Ord`: Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
- `KitchenQual_Ord`: Same quality scale
- `BsmtQual_Ord`: Same quality scale
- `BsmtExposure_Ord`: Gd=4, Av=3, Mn=2, No=1, None=0
- `GarageFinish_Ord`: Fin=3, RFn=2, Unf=1, None=0

### 3. grouped_features
- `LotFrontage_NeighMed`: Neighborhood median lot frontage
- `GrLivArea_NeighMed`: Neighborhood median living area
- `OverallQual_NeighMean`: Neighborhood mean quality
- `LotArea_NeighMed`: Neighborhood median lot size

---

## Prompt Enhancements

Added to both `_build_prompt()` and `_build_reflection_prompt()`:

```
## IMPORTANT: Understanding Categorical Values
- Quality features: Ex=Excellent(5), Gd=Good(4), TA=Typical(3), Fa=Fair(2), Po=Poor(1)
- Missing values filled with 'None' = often means ABSENCE

## Numeric columns with 0 = absent
GarageArea=0, PoolArea=0, TotalBsmtSF=0 mean the feature is ABSENT.

## WARNING: Target Leakage
Do NOT use SalePrice in feature engineering!
BAD: df.groupby('Neighborhood')['SalePrice'].transform('median')
GOOD: df.groupby('Neighborhood')['OverallQual'].transform('mean')
```

---

## What Agent Can Now Discover

| Discovery | Example Code | Impact |
|-----------|--------------|--------|
| **Binary presence** | `df['HasGarage'] = (df['GarageArea'] > 0)` | Capture absence vs presence |
| **Ordinal encoding** | `df['ExterQual'].map({'Ex':5,'Gd':4,...})` | Quality ordering |
| **Grouped stats** | `df.groupby('Neighborhood')['X'].transform('median')` | Location effects |
| **TotalSF** | `df['TotalSF'] = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Size aggregation |
| **HouseAge** | `df['YrSold'] - df['YearBuilt']` | Temporal features |

---

## Expected Baseline RMSLE

With true minimal preprocessing:
- No ordinal encoding → model can't use quality ordering efficiently
- No binary presence → model treats 0 same as small values
- No grouped features → neighborhood effects not captured

Expected baseline RMSLE: **~0.14-0.16** (much higher than previous ~0.10)

This gives agent HUGE room for improvement!

---

## Success Metrics

- [ ] Baseline RMSLE is significantly higher (~0.14+)
- [ ] Agent discovers ordinal encodings from data_description
- [ ] Agent creates binary presence features (HasGarage, HasPool)
- [ ] Agent creates grouped features (neighborhood-based)
- [ ] Cumulative improvement reduces RMSLE toward ~0.10-0.11

---

# How the Agent Works: Complete Flow

## Overview

The Feature Engineering Agent is an autonomous system that iteratively discovers and creates new features to improve a LightGBM model's prediction of house prices. It uses Gemini AI to generate feature code, executes it safely, evaluates improvement, and learns from SHAP feedback.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ITERATION LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD DATA                                                   │
│     └── train.csv + data_description.txt                        │
│                                                                 │
│  2. MINIMAL PREPROCESSING                                       │
│     └── Numeric NaN→0, Categorical NaN→"None", pd.Categorical   │
│                                                                 │
│  3. BASELINE EVALUATION                                         │
│     └── 5-fold CV with LightGBM → Baseline RMSLE                │
│                                                                 │
│  4. FOR EACH ITERATION:                                         │
│     ┌──────────────────────────────────────────────────────┐    │
│     │ a. GEMINI GENERATES FEATURE CODE                     │    │
│     │    - Receives: data_description, columns, strategies │    │
│     │    - With feedback: also gets SHAP importance        │    │
│     │    - Avoids: previously tried codes                  │    │
│     │                                                      │    │
│     │ b. CODE EXECUTOR RUNS CODE SAFELY                    │    │
│     │    - Sandboxed: only pd, np available                │    │
│     │    - Validates: row count, no all-NaN columns        │    │
│     │                                                      │    │
│     │ c. EVALUATOR MEASURES IMPROVEMENT                    │    │
│     │    - 5-fold CV on new features                       │    │
│     │    - Compare: new RMSLE vs current best              │    │
│     │                                                      │    │
│     │ d. MEMORY RECORDS RESULT                             │    │
│     │    - Success: add to accumulated features            │    │
│     │    - Failure: record error, don't use again          │    │
│     │                                                      │    │
│     │ e. SHAP FEEDBACK (if enabled)                        │    │
│     │    - Compute feature importance                      │    │
│     │    - Guide next iteration toward high-impact areas   │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                 │
│  5. OUTPUT                                                      │
│     └── Memory file + visualizations + best model               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. MinimalPreprocessor (`src/baseline/minimal_preprocessor.py`)

**Purpose**: Prepare data with absolute minimum transformations so agent has maximum discovery opportunity.

| What it does | Why |
|--------------|-----|
| Numeric NaN → 0 | Agent can detect absence via `> 0` check |
| Categorical NaN → "None" | Agent can detect absence via `!= 'None'` |
| Convert to `pd.Categorical` | Preserves string values for agent to understand |

**What it does NOT do** (agent discovers these):
- Ordinal encoding (Ex→5, Gd→4, etc.)
- Binary presence features (HasGarage, HasPool)
- Grouped/neighborhood features
- Any feature engineering

---

### 2. GeminiClient (`src/agent/gemini_client.py`)

**Purpose**: Generate feature engineering code using Gemini AI.

#### Generation Modes

| Mode | Method | Description |
|------|--------|-------------|
| **Standard** | `generate_feature()` | Random strategy, no feedback |
| **With Feedback** | `generate_feature_with_feedback()` | Uses SHAP importance to guide generation |
| **Batch** | `generate_feature_batch()` | Multiple features in one call |

#### Prompt Structure

The prompt sent to Gemini includes:

```
1. ROLE: "Expert data scientist specializing in Ames Housing"

2. DATA CONTEXT:
   - data_description.txt (truncated to 2500 chars)
   - Available column names

3. FEW-SHOT EXAMPLES:
   - 3-5 examples from the selected strategy
   - Each with: name, code, rationale

4. CATEGORICAL GUIDANCE:
   - Quality mappings: Ex=5, Gd=4, TA=3, Fa=2, Po=1
   - Absence patterns: 0 or "None" means absent
   - Target leakage warning

5. REQUIREMENTS:
   - Return only executable Python code
   - Use 'df' as dataframe variable
   - Handle NaN with .fillna()
   - Start with # Feature: <name>

6. AVOID LIST:
   - Previously tried codes (last 15)
   - Baseline features that already exist

7. STRATEGY HINT:
   - Specific guidance for selected strategy
```

---

### 3. All Available Strategies (14 total)

The agent randomly selects from these strategies, or uses SHAP feedback to pick the most relevant:

| Strategy | Purpose | Example |
|----------|---------|---------|
| **foundational** | Essential aggregate features | `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
| **quality_interactions** | Quality × other features | `OverallQual * ExterQual` |
| **quality_tiers** | Bin quality into levels | `pd.cut(OverallQual, bins=[0,4,6,8,10])` |
| **size_efficiency** | Space per unit metrics | `GrLivArea / BedroomAbvGr` |
| **interaction** | Multiply related features | `OverallQual * GrLivArea` |
| **ratio** | Divide for proportions | `BsmtFinSF1 / TotalBsmtSF` |
| **aggregation** | Sum related features | `OpenPorchSF + EnclosedPorch + ...` |
| **binary** | 0/1 presence indicators | `(PoolArea > 0).astype(int)` |
| **polynomial** | Squared/log transforms | `OverallQual ** 2`, `np.log1p(LotArea)` |
| **temporal** | Time-based calculations | `YrSold - YearBuilt` |
| **presence_detection** | Detect absence from NaN patterns | `(GarageArea > 0)`, `(PoolQC != 'None')` |
| **ordinal_from_description** | Quality → ordinal using data_description | `.map({'Ex':5, 'Gd':4, 'TA':3, ...})` |
| **grouped_features** | Neighborhood-level statistics | `.groupby('Neighborhood')['X'].transform('median')` |

---

### 4. CodeExecutor (`src/agent/code_executor.py`)

**Purpose**: Safely execute LLM-generated code in a sandboxed environment.

#### Safety Features

| Feature | Implementation |
|---------|----------------|
| **Sandboxed globals** | Only `pd`, `np` available |
| **Restricted builtins** | Only `len`, `range`, `int`, `float`, `str`, `min`, `max`, etc. |
| **No imports** | Cannot import any modules |
| **No file access** | Cannot read/write files |
| **No network** | Cannot make HTTP requests |

#### Validation Checks

| Check | Error if... |
|-------|-------------|
| Row count | New df has different row count than original |
| All-NaN columns | New column is completely NaN |
| DataFrame type | Result is not a DataFrame |

---

### 5. FeatureEvaluator (`src/agent/evaluator.py`)

**Purpose**: Measure feature quality using K-Fold Cross-Validation.

#### Evaluation Process

```python
# For each fold (5 folds):
1. Split data into train/validation
2. Train LightGBM on train set
3. Predict on validation set
4. Calculate RMSLE

# Final score:
mean_rmsle = average of 5 fold scores
```

#### SHAP Feedback

After evaluation, computes SHAP values to identify most important features:

```python
# Get feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
importance = mean(|shap_values|)

# Return top features for next iteration's prompt
top_features = sorted(importance)[:10]
```

---

### 6. AgentMemory (`src/agent/memory.py`)

**Purpose**: Persistent storage of all attempts, successes, and failures.

#### Memory Structure

```json
{
  "baseline_rmsle": 0.145,
  "best_rmsle": 0.118,

  "successful_features": [
    {
      "code": "df['TotalSF'] = ...",
      "columns": ["TotalSF"],
      "rmsle": 0.130,
      "improvement": 0.015,
      "timestamp": "2024-..."
    }
  ],

  "failed_features": [
    {
      "code": "df['BadFeature'] = ...",
      "error": "KeyError: 'NonExistentColumn'",
      "timestamp": "2024-..."
    }
  ],

  "iteration_history": [
    {
      "iteration": 1,
      "rmsle": 0.130,
      "success": true,
      "feature_summary": "TotalSF",
      "feature_description": "Sum: TotalBsmtSF + 1stFlrSF + 2ndFlrSF"
    }
  ],

  "shap_history": [
    {
      "iteration": 1,
      "top_features": ["OverallQual", "GrLivArea", ...],
      "importance_scores": {"OverallQual": 0.45, ...}
    }
  ]
}
```

#### Key Methods

| Method | Purpose |
|--------|---------|
| `get_all_tried_codes()` | Avoid regenerating same features |
| `get_successful_codes()` | Apply previous successes to new iterations |
| `add_successful_feature()` | Record winning features |
| `add_failed_feature()` | Record failures to avoid repeating |

---

## Command Line Options

```bash
python -m src.agent.iterative_agent [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-n`, `--iterations` | Number of iterations/batches | 10 |
| `--clear` | Clear memory, start fresh | False |
| `-v`, `--visualize` | Generate plots after completion | False |
| `-f`, `--feedback` | Enable SHAP feedback loop | False |
| `-b`, `--batch` | Batch mode (multiple features per iteration) | False |
| `--batch-size` | Features per batch | 5 |

### Example Commands

```bash
# Basic: 10 single-feature iterations
python -m src.agent.iterative_agent

# With SHAP feedback (recommended)
python -m src.agent.iterative_agent -n 10 --feedback

# Batch mode: 3 batches × 5 features = 15 total
python -m src.agent.iterative_agent -n 3 --batch --clear

# Full featured run
python -m src.agent.iterative_agent -n 5 -v -f --batch --clear
```

---

## Feature Acceptance Criteria

A feature is **accepted** if:
1. Code executes without error
2. Creates at least one new column
3. New column is not all-NaN
4. 5-fold CV RMSLE is **lower** than current best

A feature is **rejected** if:
1. Syntax error in code
2. Runtime error (KeyError, TypeError, etc.)
3. No new columns created (duplicate)
4. RMSLE does not improve

---

## SHAP Feedback Loop

When `--feedback` is enabled:

```
┌─────────────────────────────────────────────────────────────┐
│                    SHAP FEEDBACK LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. EVALUATE features → train LightGBM                      │
│                                                             │
│  2. COMPUTE SHAP values on trained model                    │
│     └── TreeExplainer for fast computation                  │
│                                                             │
│  3. IDENTIFY top features by mean |SHAP|                    │
│     └── e.g., OverallQual: 0.45, GrLivArea: 0.32, ...       │
│                                                             │
│  4. GENERATE prompt with SHAP insights                      │
│     └── "Top feature is OverallQual. Consider:"             │
│         - Interacting it with other features                │
│         - Creating polynomial transforms                    │
│         - Combining with related features                   │
│                                                             │
│  5. SUGGEST strategy based on top features                  │
│     └── If top has "qual" → interaction/polynomial          │
│     └── If top has "sf/area" → ratio/aggregation            │
│     └── If top has "year" → temporal                        │
│                                                             │
│  6. IF feature rejected:                                    │
│     └── RECOMPUTE SHAP on accumulated features only         │
│     └── Prevents referencing non-existent columns           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/logs/agent_memory.json` | All attempts, successes, failures |
| `outputs/logs/progress_plot.png` | RMSLE over iterations (with `-v`) |
| `outputs/logs/progress_report.html` | HTML report (with `-v`) |
| `outputs/models/best_lgbm_params.json` | Tuned hyperparameters |
