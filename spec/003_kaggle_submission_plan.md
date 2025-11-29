# Kaggle Submission Plan: Converting Python Scripts to Notebook

## Problem Statement

The current baseline model (Phase A3) is implemented as modular Python scripts:
- `src/baseline/train.py` - Main training script
- `src/baseline/preprocessor.py` - Feature engineering pipeline
- `src/baseline/model.py` - Hyperparameter tuning
- `outputs/predictions/baseline_phase_a3.csv` - Generated predictions

**Kaggle Competition Requirements:**
1. Code must be submitted as a Jupyter notebook
2. Notebook runs in Kaggle's kernel environment
3. Paths must use `/kaggle/input/` and `/kaggle/working/`
4. No external file imports allowed (all code must be inline)
5. Must produce `submission.csv` in correct format (Id, SalePrice)

## Current Baseline Performance

- **CV RMSLE**: 0.10353 (±0.00682)
- **Features**: 210 (after preprocessing and engineering)
- **Model**: LightGBM with Optuna-tuned hyperparameters
- **Preprocessing**: Semantic imputation, ordinal encoding, one-hot encoding, engineered features

## Solution: Self-Contained Kaggle Notebook

### Notebook Structure

**File**: `notebooks/kaggle_submission_baseline.ipynb`

**Cell 1: Imports and Setup**
```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import json
```

**Cell 2: Helper Functions**
- `rmsle()` - Calculate RMSLE metric
- `rmse()` - Calculate RMSE metric

**Cell 3: AmesPreprocessor Class (Inline)**
- Copy entire class from `src/baseline/preprocessor.py`
- Includes:
  - `_create_engineered_features()` - TotalSF, HouseAge, RemodAge, TotalBath, PorchArea
  - `_handle_missing_values()` - Semantic imputation, grouped imputation
  - `_encode_quality_features()` - Ordinal encoding for quality features
  - `_encode_categorical_features()` - One-hot encoding

**Cell 4: Load Data**
```python
# Kaggle paths
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
```

**Cell 5: Preprocessing**
- Extract target and log-transform
- Initialize and fit preprocessor on training data
- Transform test data

**Cell 6: Model Parameters**
- Inline best hyperparameters from `outputs/models/best_lgbm_params.json`
```python
model_params = {
    'n_estimators': 800,
    'learning_rate': 0.0444,
    'max_depth': 13,
    'num_leaves': 69,
    'min_child_samples': 35,
    'subsample': 0.6174,
    'colsample_bytree': 0.6242,
    'reg_alpha': 0.0159,
    'reg_lambda': 6.57e-08,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'random_state': 42
}
```

**Cell 7: Cross-Validation**
- 5-fold CV
- Train models and calculate RMSLE per fold
- Display mean CV RMSLE and individual fold scores

**Cell 8: Train Final Model**
- Train on full training dataset
- Use log-transformed target

**Cell 9: Generate Predictions**
- Predict on test set (in log space)
- Transform back to original scale with `np.expm1()`
- Create submission DataFrame

**Cell 10: Save Submission**
```python
submission.to_csv('/kaggle/working/submission.csv', index=False)
```

**Cell 11: Display Summary**
- Show CV scores
- Preview submission file
- Display statistics

## Path Adaptations for Kaggle

| Local Path | Kaggle Path |
|------------|-------------|
| `data/train.csv` | `/kaggle/input/house-prices-advanced-regression-techniques/train.csv` |
| `data/test.csv` | `/kaggle/input/house-prices-advanced-regression-techniques/test.csv` |
| `outputs/predictions/submission.csv` | `/kaggle/working/submission.csv` |

**Note**: Kaggle automatically detects `submission.csv` in `/kaggle/working/` for competition submissions.

## Submission Methods

### Option A: Manual Upload via Kaggle UI (Recommended)

**Steps:**
1. Open Kaggle competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
2. Click "Code" tab → "New Notebook"
3. Copy-paste cells from local `notebooks/kaggle_submission_baseline.ipynb`
4. Verify paths point to `/kaggle/input/house-prices-advanced-regression-techniques/`
5. Click "Save Version" → "Save & Run All"
6. Wait for notebook to complete execution
7. Check output for CV RMSE (~0.10353 expected)
8. Click "Submit to Competition"
9. View leaderboard score

**Advantages:**
- Direct feedback on execution
- Easy to debug if errors occur
- Can iterate quickly
- No API setup required

### Option B: Kaggle API Submission (Automated)

**Prerequisites:**
```bash
pip install kaggle
```

**Setup API Credentials:**
1. Go to Kaggle Account Settings → "Create New API Token"
2. Download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json` (chmod 600)

**Commands:**
```bash
# Push notebook to Kaggle
kaggle kernels push -p notebooks/

# Check kernel status
kaggle kernels status <kernel-slug>

# Submit to competition
kaggle competitions submit -c house-prices-advanced-regression-techniques \
  -f submission.csv -m "Phase A3 baseline with tuned hyperparameters"
```

**Advantages:**
- Automation-friendly
- Can integrate into CI/CD
- Good for multiple submissions

## Testing Strategy

### Local Testing

**Before uploading to Kaggle:**

1. **Run notebook locally:**
   ```bash
   jupyter notebook notebooks/kaggle_submission_baseline.ipynb
   ```

2. **Verify outputs:**
   - CV RMSLE should match ~0.10353 (±0.00682)
   - Check submission.csv format:
     - Has columns: `Id`, `SalePrice`
     - 1459 rows (test set size)
     - No missing values
     - Prices are in original scale (not log)

3. **Compare with Phase A3:**
   ```bash
   diff outputs/predictions/baseline_phase_a3.csv /kaggle/working/submission.csv
   ```
   (Predictions should be nearly identical)

### Kaggle Testing

**After upload:**
1. Check for runtime errors
2. Verify execution time < 9 hours (Kaggle limit)
3. Confirm submission.csv is generated
4. Review competition leaderboard score
   - Private/Public score split
   - Compare with CV RMSLE

## Version Control

### Git Workflow

**Initial Commit:**
```bash
git add notebooks/kaggle_submission_baseline.ipynb
git add spec/kaggle_submission_plan.md
git commit -m "Add Kaggle submission notebook - Phase A3 baseline (CV RMSLE: 0.10353)"
git push origin main
```

**After Kaggle Submission:**
```bash
git tag -a v1.0-baseline-a3 -m "Phase A3 baseline: CV RMSLE 0.10353, LB score TBD"
git push origin v1.0-baseline-a3
```

### File Organization

**Keep Both Versions:**
```
kaggle-iaa-hackathon/
├── src/baseline/          # Modular code for development & agent
│   ├── train.py
│   ├── preprocessor.py
│   ├── model.py
│   └── tune_hyperparameters.py
├── notebooks/             # Kaggle submission notebooks
│   └── kaggle_submission_baseline.ipynb  # "Compiled" version
└── outputs/
    └── predictions/
        └── baseline_phase_a3.csv  # Reference
```

**Rationale:**
- **Modular code** (`src/`): Easier to develop, test, and use for agent
- **Notebook** (`notebooks/`): Required for Kaggle submission
- **Update workflow**: Improve modular code → Regenerate notebook → Submit to Kaggle

## Expected Results

### Local Validation
- **CV RMSLE**: 0.10353 (±0.00682)
- **CV RMSE (log)**: 0.10353 (±0.00682)
- **Features**: 210
- **Execution time**: ~2-3 minutes

### Kaggle Leaderboard
- **Expected Public LB**: ~0.12-0.13 (typical gap from CV)
- **Expected Private LB**: ~0.11-0.12
- **Rank**: Mid-tier baseline (top 50-60%)

**Note**: First submission establishes benchmark. Agent-generated features (Track B) should improve this.

## Troubleshooting

### Common Issues

**Issue 1: Path Not Found**
```
FileNotFoundError: /kaggle/input/house-prices-advanced-regression-techniques/train.csv
```
**Solution**: Verify competition data is added as input source in Kaggle notebook settings.

**Issue 2: Memory Error**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce `n_estimators` or use lighter preprocessing.

**Issue 3: Submission Format Error**
```
Submission file must have columns: Id, SalePrice
```
**Solution**: Verify column names are exact (case-sensitive), no extra columns.

**Issue 4: Different CV Score**
```
CV RMSLE: 0.11+ (expected 0.10353)
```
**Solution**: Check if preprocessing is identical to local version.

## Next Steps

### Immediate (After First Submission)
1. ✅ Submit Phase A3 baseline to Kaggle
2. ✅ Note leaderboard score
3. ✅ Compare with CV RMSLE
4. ✅ Commit notebook to git

### Short-term (Track B: Agent)
1. Use this baseline as evaluation target
2. Agent should generate features to improve 0.10353 RMSLE
3. Create new notebook versions with agent-generated features
4. Submit improved versions to Kaggle

### Long-term (Optimization)
1. Ensemble multiple models
2. Feature selection refinement
3. Additional feature engineering
4. Model blending

## References

- **Local baseline**: `src/baseline/train.py --use-tuned`
- **Tuned params**: `outputs/models/best_lgbm_params.json`
- **Preprocessor**: `src/baseline/preprocessor.py`
- **Phase A3 predictions**: `outputs/predictions/baseline_phase_a3.csv`
- **Kaggle competition**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- **Kaggle API docs**: https://github.com/Kaggle/kaggle-api
