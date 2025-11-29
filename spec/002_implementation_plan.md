# Implementation Plan: Autonomous Feature Engineering Agent

## Overview

This document outlines a progressive implementation strategy with two parallel tracks:

- **Track A: Baseline Model** - Traditional ML pipeline for Ames Housing price prediction
- **Track B: Feature Engineering Agent** - Autonomous agent powered by Gemini that generates features to improve the baseline

## Project Structure

```
kaggle-iaa-hackathon/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── src/
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   └── agent/
│       ├── __init__.py
│       ├── gemini_client.py
│       ├── code_executor.py
│       ├── evaluator.py
│       ├── memory.py
│       └── simple_agent.py
├── notebooks/
│   └── experiments/
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── logs/
├── spec/
│   ├── Autonomous Feature Engineering Design.md
│   └── implementation_plan.md
└── requirements.txt
```

---

## Track A: Baseline Model Implementation

### Phase A0: Minimal Baseline (2-3 hours)

**Goal**: Get a working end-to-end pipeline that produces a submission file.

**Steps**:
1. Load train.csv and test.csv
2. Minimal preprocessing:
   - Drop non-numeric columns
   - Fill missing values with median
   - Separate features (X) from target (SalePrice)
3. Train XGBoost regressor on all training data
4. Make predictions on test set
5. Calculate RMSE on training set (basic sanity check)
6. Save predictions to CSV

**Files to Create**:
- `src/baseline/train.py` - Main training script
- `requirements.txt` - Dependencies (pandas, xgboost, scikit-learn)

**Success Criteria**:
- Script runs without errors
- Produces predictions.csv with Id and SalePrice
- Training RMSE < 50000 (very rough baseline)

**Example Code Structure**:
```python
# src/baseline/train.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def main():
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Minimal preprocessing
    target = train['SalePrice']
    train_ids = train['Id']
    test_ids = test['Id']

    # Drop non-numeric and target
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [f for f in numeric_features if f not in ['Id', 'SalePrice']]

    X_train = train[numeric_features].fillna(train[numeric_features].median())
    X_test = test[numeric_features].fillna(train[numeric_features].median())

    # Train model
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, target)

    # Evaluate
    train_pred = model.predict(X_train)
    rmse = mean_squared_error(target, train_pred, squared=False)
    print(f"Training RMSE: {rmse:.2f}")

    # Predict and save
    test_pred = model.predict(X_test)
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
    submission.to_csv('outputs/predictions/baseline_phase0.csv', index=False)
    print("Predictions saved!")

if __name__ == '__main__':
    main()
```

---

### Phase A1: Standard Preprocessing (4-6 hours)

**Goal**: Improve baseline with proper preprocessing and feature engineering.

**Enhancements**:
1. **Missing Value Handling**:
   - Semantic imputation (e.g., NA in PoolQC = "No Pool" → 0)
   - Grouped imputation for LotFrontage by Neighborhood
2. **Categorical Encoding**:
   - One-hot encoding for low cardinality features
   - Ordinal encoding for quality features (Ex=5, Gd=4, TA=3, Fa=2, Po=1)
3. **Basic Feature Engineering**:
   - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
   - HouseAge = YrSold - YearBuilt
   - RemodAge = YrSold - YearRemodAdd
4. **Target Transformation**:
   - Apply log1p to SalePrice (competition uses RMSLE)

**Files to Create**:
- `src/baseline/preprocessor.py` - Preprocessing pipeline
- `src/baseline/data_loader.py` - Data loading utilities

**Success Criteria**:
- Training RMSE < 30000
- Clean separation of preprocessing logic
- Reusable preprocessing pipeline

**Example Preprocessor**:
```python
# src/baseline/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class AmesPreprocessor:
    def __init__(self):
        self.numeric_impute_values = {}
        self.label_encoders = {}

    def fit_transform(self, df, target_col='SalePrice'):
        df = df.copy()

        # Basic features
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

        # Quality features - ordinal encoding
        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                     'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

        for col in qual_cols:
            if col in df.columns:
                df[col] = df[col].fillna('NA').map(qual_map)

        # Numeric features - median imputation
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col not in ['Id', target_col]:
                median_val = df[col].median()
                self.numeric_impute_values[col] = median_val
                df[col] = df[col].fillna(median_val)

        return df

    def transform(self, df):
        # Apply same transformations to test set
        df = df.copy()

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                     'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

        for col in qual_cols:
            if col in df.columns:
                df[col] = df[col].fillna('NA').map(qual_map)

        for col, val in self.numeric_impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        return df
```

---

### Phase A2: Cross-Validation Strategy (2-3 hours)

**Goal**: Add robust validation to prevent overfitting.

**Enhancements**:
1. Implement K-Fold cross-validation (5 folds)
2. Calculate mean CV RMSE and standard deviation
3. Save validation results to track improvements
4. Log feature importance

**Files to Update**:
- `src/baseline/train.py` - Add CV loop

**Success Criteria**:
- CV RMSE with confidence interval
- Reproducible validation scores
- Feature importance logged

**Example CV Loop**:
```python
from sklearn.model_selection import KFold

def train_with_cv(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_fold, y_train_fold)

        val_pred = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, val_pred, squared=False)
        cv_scores.append(rmse)
        print(f"Fold {fold+1}: RMSE = {rmse:.2f}")

    print(f"\nMean CV RMSE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
    return np.mean(cv_scores)
```

---

### Phase A3: Hyperparameter Tuning (3-4 hours)

**Goal**: Optimize model performance.

**Enhancements**:
1. Grid search or random search for XGBoost parameters
2. Test alternative models (LightGBM, CatBoost)
3. Ensemble different models
4. Save best model configuration

**Files to Create**:
- `src/baseline/model.py` - Model definitions and tuning

**Success Criteria**:
- CV RMSE < 25000
- Best hyperparameters logged
- Model saved for reuse

---

## Track B: Feature Engineering Agent Implementation

### Phase B0: Manual Gemini Test (1-2 hours)

**Goal**: Verify that Gemini can generate useful feature engineering code.

**Steps**:
1. Set up Google Gemini API credentials
2. Create a simple Python script that:
   - Reads `data_description.txt`
   - Extracts column names and types from train.csv
   - Sends prompt to Gemini asking for ONE feature engineering idea
   - Prints the generated code
3. Manually copy-paste the code and test if it works
4. Check if the feature improves RMSE

**Files to Create**:
- `src/agent/gemini_client.py` - Wrapper for Gemini API
- `.env` - Store API key (add to .gitignore)

**Success Criteria**:
- Gemini generates valid Python code
- Code creates a meaningful feature
- Feature can be added to baseline

**Example Gemini Client**:
```python
# src/agent/gemini_client.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_feature(self, data_description, column_info, existing_features=None):
        prompt = self._build_prompt(data_description, column_info, existing_features)
        response = self.model.generate_content(prompt)
        return response.text

    def _build_prompt(self, data_description, column_info, existing_features):
        prompt = f"""You are an expert data scientist working on the Ames Housing dataset.

Data Description:
{data_description[:2000]}  # Truncate if too long

Available Columns:
{column_info}

Task: Generate ONE new feature that will help predict house prices.
Provide ONLY executable Python code that creates the feature.
Assume the dataframe is called 'df'.

Example format:
```python
# Feature: Total Square Footage
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
```

"""
        if existing_features:
            prompt += f"\nAlready tried features (don't repeat):\n{existing_features}\n"

        prompt += "\nGenerate the code:"
        return prompt
```

---

### Phase B1: Single-Shot Automation (4-5 hours)

**Goal**: Automate the entire loop: Gemini → code execution → evaluation.

**Enhancements**:
1. **Code Executor**: Safely execute generated code with try/catch
2. **Evaluator**: Compare baseline RMSE vs RMSE with new feature
3. **End-to-end script**: Run without manual intervention

**Files to Create**:
- `src/agent/code_executor.py` - Execute generated code safely
- `src/agent/evaluator.py` - Evaluate feature quality
- `src/agent/simple_agent.py` - Main agent script

**Success Criteria**:
- Agent generates 1 feature
- Feature is tested automatically
- RMSE improvement (or not) is logged
- Script completes without manual intervention

**Example Code Executor**:
```python
# src/agent/code_executor.py
import pandas as pd
import traceback

class CodeExecutor:
    def __init__(self):
        self.safe_globals = {
            'pd': pd,
            'np': __import__('numpy'),
        }

    def execute(self, code, df):
        """Execute generated code on dataframe"""
        try:
            # Create safe execution environment
            local_vars = {'df': df.copy()}
            exec(code, self.safe_globals, local_vars)
            return local_vars['df'], None
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
```

**Example Evaluator**:
```python
# src/agent/evaluator.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

class FeatureEvaluator:
    def __init__(self, baseline_rmse):
        self.baseline_rmse = baseline_rmse
        self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)

    def evaluate_feature(self, X, y, cv=3):
        """Evaluate features using cross-validation"""
        # Use negative MSE for cross_val_score
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_mean_squared_error'
        )
        rmse = np.sqrt(-scores.mean())
        return rmse

    def is_improvement(self, new_rmse):
        """Check if new features improved over baseline"""
        improvement = self.baseline_rmse - new_rmse
        improvement_pct = (improvement / self.baseline_rmse) * 100
        return improvement > 0, improvement, improvement_pct
```

**Example Simple Agent**:
```python
# src/agent/simple_agent.py
import pandas as pd
from src.baseline.preprocessor import AmesPreprocessor
from src.agent.gemini_client import GeminiClient
from src.agent.code_executor import CodeExecutor
from src.agent.evaluator import FeatureEvaluator

def main():
    # Load data
    train = pd.read_csv('data/train.csv')
    data_desc = open('data/data_description.txt').read()

    # Baseline preprocessing
    preprocessor = AmesPreprocessor()
    train_processed = preprocessor.fit_transform(train)

    # Get baseline RMSE
    X_baseline = train_processed.select_dtypes(include=['int64', 'float64']).drop(['Id', 'SalePrice'], axis=1)
    y = train_processed['SalePrice']

    evaluator = FeatureEvaluator(baseline_rmse=30000)  # Replace with actual baseline
    baseline_rmse = evaluator.evaluate_feature(X_baseline, y)
    print(f"Baseline RMSE: {baseline_rmse:.2f}")

    # Generate feature with Gemini
    gemini = GeminiClient()
    column_info = ', '.join(train.columns)
    generated_code = gemini.generate_feature(data_desc, column_info)
    print(f"\nGenerated Code:\n{generated_code}\n")

    # Execute code
    executor = CodeExecutor()
    new_df, error = executor.execute(generated_code, train_processed)

    if error:
        print(f"Error executing code: {error}")
        return

    # Evaluate new features
    X_new = new_df.select_dtypes(include=['int64', 'float64']).drop(['Id', 'SalePrice'], axis=1)
    new_rmse = evaluator.evaluate_feature(X_new, y)

    is_better, improvement, pct = evaluator.is_improvement(new_rmse)
    print(f"New RMSE: {new_rmse:.2f}")
    print(f"Improvement: {improvement:.2f} ({pct:.2f}%)")
    print(f"Better: {is_better}")

if __name__ == '__main__':
    main()
```

---

### Phase B2: Simple Iterative Loop (1-2 days)

**Goal**: Run multiple iterations and accumulate successful features.

**Enhancements**:
1. **Memory System**: Store successful features in list/JSON
2. **Iteration Loop**: Run 5-10 times
3. **Avoid Duplicates**: Pass "already tried" features to Gemini
4. **Best Feature Tracking**: Keep only features that improve RMSE

**Files to Create**:
- `src/agent/memory.py` - Simple memory system

**Success Criteria**:
- Agent runs 10 iterations automatically
- Successful features are accumulated
- Final RMSE better than baseline
- Memory persisted to JSON

**Example Memory System**:
```python
# src/agent/memory.py
import json
from datetime import datetime

class AgentMemory:
    def __init__(self, filepath='outputs/logs/agent_memory.json'):
        self.filepath = filepath
        self.memory = self._load()

    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'successful_features': [],
                'failed_features': [],
                'iteration_history': []
            }

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def add_successful_feature(self, code, rmse_improvement):
        self.memory['successful_features'].append({
            'code': code,
            'improvement': rmse_improvement,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def add_failed_feature(self, code, error):
        self.memory['failed_features'].append({
            'code': code,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def log_iteration(self, iteration, rmse, features_tried):
        self.memory['iteration_history'].append({
            'iteration': iteration,
            'rmse': rmse,
            'features_tried': features_tried,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def get_successful_codes(self):
        return [f['code'] for f in self.memory['successful_features']]
```

**Updated Agent Loop**:
```python
def iterative_loop(n_iterations=10):
    memory = AgentMemory()
    current_best_rmse = baseline_rmse
    accumulated_features = []

    for i in range(n_iterations):
        print(f"\n=== Iteration {i+1}/{n_iterations} ===")

        # Generate new feature
        existing = memory.get_successful_codes()
        generated_code = gemini.generate_feature(data_desc, column_info, existing)

        # Execute and evaluate
        new_df, error = executor.execute(generated_code, train_processed)

        if error:
            memory.add_failed_feature(generated_code, error)
            continue

        # Test improvement
        X_new = new_df.select_dtypes(include=['int64', 'float64']).drop(['Id', 'SalePrice'], axis=1)
        new_rmse = evaluator.evaluate_feature(X_new, y)

        if new_rmse < current_best_rmse:
            improvement = current_best_rmse - new_rmse
            memory.add_successful_feature(generated_code, improvement)
            accumulated_features.append(generated_code)
            current_best_rmse = new_rmse
            print(f"✓ Improvement! New RMSE: {new_rmse:.2f}")
        else:
            print(f"✗ No improvement. RMSE: {new_rmse:.2f}")

        memory.log_iteration(i+1, new_rmse, len(accumulated_features))

    print(f"\n=== Final Results ===")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Final RMSE: {current_best_rmse:.2f}")
    print(f"Total Improvement: {baseline_rmse - current_best_rmse:.2f}")
    print(f"Successful Features: {len(accumulated_features)}")
```

---

### Phase B3: Reflection with Feature Importance (2-3 days)

**Goal**: Add feedback loop where agent learns from feature importance.

**Enhancements**:
1. **Feature Importance Extraction**: Use SHAP or permutation importance
2. **Feedback to Gemini**: Tell Gemini which features helped/hurt and why
3. **Guided Generation**: Gemini uses insights to generate better features
4. **Error Analysis**: Analyze residuals and suggest targeted features

**Success Criteria**:
- Agent receives feature importance feedback
- Generates increasingly targeted features
- Demonstrates learning across iterations
- RMSE improvement accelerates over iterations

**Example Reflection Prompt**:
```python
def build_reflection_prompt(data_desc, column_info, feature_importance, top_features):
    prompt = f"""You are an expert data scientist working on Ames Housing prediction.

Previous Iteration Results:
- Top 5 Important Features: {top_features}
- Feature Importance Scores: {feature_importance}

Analysis:
The model is currently relying heavily on {top_features[0]}.
Consider creating interaction terms or transformations that combine
these important features in meaningful ways.

Data Description:
{data_desc[:2000]}

Task: Generate ONE new feature that:
1. Interacts with or transforms the currently important features
2. Has clear semantic meaning for house price prediction
3. Is different from previously tried features

Provide ONLY executable Python code.
"""
    return prompt
```

---

## Dependencies

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Environment Setup

### .env (Add to .gitignore!)
```
GEMINI_API_KEY=your_api_key_here
```

---

## Success Metrics

### Baseline Model
- Phase A0: RMSE < 50000
- Phase A1: RMSE < 30000
- Phase A2: CV RMSE with std dev < 2000
- Phase A3: CV RMSE < 25000

### Feature Engineering Agent
- Phase B0: Gemini generates valid code
- Phase B1: Single iteration completes successfully
- Phase B2: 10 iterations, at least 3 successful features
- Phase B3: Agent shows learning (later iterations more successful)

**Ultimate Goal**: Agent-generated features improve baseline RMSE by at least 10%

---

## Timeline Estimate

| Phase | Track A | Track B | Total |
|-------|---------|---------|-------|
| Phase 0 | 2-3h | 1-2h | 3-5h |
| Phase 1 | 4-6h | 4-5h | 8-11h |
| Phase 2 | 2-3h | 1-2d | 1-2d |
| Phase 3 | 3-4h | 2-3d | 2-3d |

**Total**: 3-5 days for complete implementation

---

## Next Steps

1. Set up environment and install dependencies
2. Implement Track A Phase 0 (minimal baseline)
3. Implement Track B Phase 0 (Gemini test)
4. Iterate through phases, testing each thoroughly
5. Document results and insights
6. Consider advanced features (OpenFE integration, evolutionary optimization) if basic agent works well

---

## Progress Update (November 2024)

### Completed Phases

#### Track A: Baseline Model ✅
- **Phase A0-A3**: Complete
- CV RMSLE: ~0.10353
- 210 features (5 engineered + ordinal + one-hot encoded)
- LightGBM with Optuna-tuned hyperparameters

#### Track B: Feature Engineering Agent
- **Phase B0**: Gemini client ✅ (`src/agent/gemini_client.py`)
- **Phase B1**: Code executor ✅ (`src/agent/code_executor.py`)
- **Phase B2**: Feature evaluator ✅ (`src/agent/evaluator.py`)
- **Phase B3**: Simple agent ✅ (`src/agent/simple_agent.py`)
- **Phase B4**: Memory system ✅ (`src/agent/memory.py`)
- **Phase B5**: Iterative agent ✅ (`src/agent/iterative_agent.py`)

### Current Files

```
src/agent/
├── __init__.py
├── gemini_client.py      # Gemini 2.5-flash API wrapper
├── code_executor.py      # Safe sandboxed code execution
├── evaluator.py          # CV-based feature evaluation
├── memory.py             # JSON persistence for iterations
├── simple_agent.py       # Single iteration agent
└── iterative_agent.py    # Multi-iteration agent with memory
```

### Next Phase: Visualization

See `spec/004_visualization_plan.md` for details on:
- Progress plot (iterations vs RMSLE)
- Feature summary table
- HTML report generation
