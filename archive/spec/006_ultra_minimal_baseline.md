# Ultra-Minimal Baseline Plan

## Problem Statement

The baseline has **too much feature engineering already**, leaving the agent no room to improve:
- 5 engineered features (TotalSF, HouseAge, RemodAge, TotalBath, PorchArea)
- 15 ordinal-encoded quality features
- ~100+ one-hot encoded columns
- Outlier removal
- Skewness correction (Box-Cox)

The agent tries to improve an already-optimized model → no improvements possible.

## Solution: Ultra-Minimal Baseline

Create a minimal preprocessor that does **only**:
1. Impute missing values (median for numeric, mode for categorical)
2. Label encode all categoricals (simple A=0, B=1, C=2)

**NO feature engineering** → Agent can discover TotalSF, HouseAge, ordinal encodings, etc.

---

## Implementation Plan

### Step 1: Create MinimalPreprocessor

**File**: `src/baseline/minimal_preprocessor.py` (NEW)

```python
class MinimalPreprocessor:
    """Ultra-minimal preprocessing: impute + label encode only"""

    def __init__(self):
        self.numeric_medians = {}      # col -> median
        self.categorical_modes = {}    # col -> mode
        self.label_encoders = {}       # col -> {value: int}

    def fit(self, df):
        # Learn medians for numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            self.numeric_medians[col] = df[col].median()

        # Learn modes for categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            self.categorical_modes[col] = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            # Create label encoding map
            unique_vals = df[col].dropna().unique()
            self.label_encoders[col] = {v: i for i, v in enumerate(sorted(unique_vals))}

        return self

    def transform(self, df):
        df = df.copy()

        # Impute numeric with median
        for col, median in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # Impute categorical with mode, then label encode
        for col, mode in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)
                encoder = self.label_encoders.get(col, {})
                df[col] = df[col].map(lambda x: encoder.get(x, -1))

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
```

### Step 2: Update run.py to Use MinimalPreprocessor

**File**: `src/agent/run.py`

Changes:
1. Import MinimalPreprocessor instead of AmesPreprocessor
2. Remove advanced preprocessing calls (outlier removal, skewness)
3. Use minimal preprocessing only

```python
# BEFORE
from src.baseline.preprocessor import AmesPreprocessor
preprocessor = AmesPreprocessor()
train_processed = preprocessor.fit_transform(train_df)
train_processed = preprocess_for_agent(train_processed, ...)

# AFTER
from src.baseline.minimal_preprocessor import MinimalPreprocessor
preprocessor = MinimalPreprocessor()
train_processed = preprocessor.fit_transform(train_df)
# NO advanced preprocessing
```

### Step 3: Update iterative_agent.py Similarly

**File**: `src/agent/iterative_agent.py`

Same changes as run.py - use MinimalPreprocessor, remove advanced preprocessing.

### Step 4: Update Gemini Prompts

**File**: `src/agent/gemini_client.py`

Update prompts to encourage creating foundational features:

```python
FEATURE_STRATEGIES = {
    'area_combinations': [
        {'name': 'TotalSF', 'code': "df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF']"},
        {'name': 'TotalPorchSF', 'code': "df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']"},
    ],
    'age_features': [
        {'name': 'HouseAge', 'code': "df['HouseAge'] = df['YrSold'] - df['YearBuilt']"},
        {'name': 'RemodAge', 'code': "df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']"},
    ],
    'bathroom_features': [
        {'name': 'TotalBath', 'code': "df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'].fillna(0) + 0.5*df['BsmtHalfBath'].fillna(0)"},
    ],
    'ordinal_encoding': [
        {'name': 'OverallQual_Ordinal', 'code': "# Already numeric 1-10"},
        {'name': 'ExterQual_Ordinal', 'code': "df['ExterQual_Ordinal'] = df['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}).fillna(0)"},
    ],
    # ... more strategies
}
```

---

## Files to Modify

| File | Action |
|------|--------|
| `src/baseline/minimal_preprocessor.py` | NEW: Ultra-minimal preprocessor |
| `src/agent/run.py` | Use MinimalPreprocessor, remove advanced preprocessing |
| `src/agent/iterative_agent.py` | Use MinimalPreprocessor, remove advanced preprocessing |
| `src/agent/gemini_client.py` | Update strategies to include foundational features |

---

## Before vs After

| Aspect | Before (Heavy) | After (Minimal) |
|--------|----------------|-----------------|
| Engineered features | 5 (TotalSF, HouseAge, etc.) | 0 |
| Encoding | Ordinal + One-hot (~100+ cols) | Label encode only (~40 cols) |
| Outlier removal | Yes | No |
| Skewness correction | Yes | No |
| Agent opportunity | Very limited | Maximum |

---

## Expected Outcomes

With ultra-minimal baseline:
- Agent can discover TotalSF → significant improvement
- Agent can discover HouseAge → improvement
- Agent can discover ordinal encodings → improvement
- Agent can discover outlier handling → improvement
- Agent can discover log transforms → improvement

Each of these should yield measurable RMSLE improvements, giving the agent many opportunities to succeed.

---

## Success Metrics

- [ ] Baseline RMSLE is higher (more room to improve)
- [ ] Agent successfully creates TotalSF-like features
- [ ] Multiple successful iterations (vs 0 currently)
- [ ] Cumulative RMSLE improvement over minimal baseline
