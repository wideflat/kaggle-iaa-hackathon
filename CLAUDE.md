# CLAUDE.md - Project Context for Claude Code

## Project Overview

Kaggle competition project for Ames Housing price prediction with an autonomous LLM-powered feature engineering agent.

## Project Structure

```
kaggle-iaa-hackathon/
├── data/                    # Dataset files (train.csv, test.csv, data_description.txt)
├── config/
│   └── default.yaml         # Agent configuration (iterations, model, logging)
├── src/
│   ├── baseline/            # Phase A: Traditional ML pipeline
│   │   ├── data_loader.py   # AmesDataLoader class
│   │   ├── preprocessor.py  # AmesPreprocessor with feature engineering
│   │   ├── minimal_preprocessor.py  # Ultra-minimal preprocessing for agent
│   │   ├── model.py         # Model definitions
│   │   ├── train.py         # Main training script
│   │   ├── tune_hyperparameters.py  # Optuna tuning
│   │   ├── outlier_remover.py      # Remove known Ames Housing outliers
│   │   ├── model_stacker.py        # 5-model ensemble (Ridge, Lasso, XGB, LGB, etc.)
│   │   ├── feature_selector.py     # Importance/correlation-based selection
│   │   ├── hyperparameter_tuner.py # Optuna tuning for stacking models
│   │   └── make_submission.py      # End-to-end submission pipeline
│   └── agent/               # Phase B: LLM-powered agent
│       ├── gemini_client.py # Gemini API wrapper with few-shot prompting
│       ├── code_executor.py # Safe code execution sandbox
│       ├── evaluator.py     # CV-based evaluation + SHAP importance
│       ├── memory.py        # JSON persistence for iterations
│       ├── config.py        # YAML config loader
│       ├── logger.py        # Colored console + file logging
│       ├── visualizer.py    # Progress plot, table, HTML report
│       ├── simple_agent.py  # Single iteration agent
│       ├── iterative_agent.py  # Multi-iteration agent
│       └── run.py           # Production CLI entry point
├── scripts/
│   └── visualize_progress.py  # Standalone visualization CLI
├── outputs/
│   ├── models/              # Saved model params (best_lgbm_params.json)
│   ├── predictions/         # Submission CSVs
│   └── logs/                # Agent memory, logs, reports
│       ├── agent_memory.json
│       ├── agent.log
│       ├── progress_plot.png
│       └── progress_report.html
├── notebooks/               # Kaggle submission notebooks
└── spec/                    # Design documents
```

## Current State

### Phase A (Complete)
- Baseline LightGBM model with CV RMSLE ~0.103
- Tuned hyperparameters via Optuna
- 5 engineered features: TotalSF, HouseAge, RemodAge, TotalBath, PorchArea
- Ordinal encoding for quality features, one-hot for categoricals
- Total: 212 features after preprocessing

### Phase B (Complete)
- **B0-B3**: Core agent (Gemini client, code executor, evaluator)
- **B4-B5**: Memory system + iteration loop
- **B6**: Enhanced prompting with 18+ few-shot examples across 6 strategies
- **B7**: SHAP-based feedback loop for guided feature generation
- **B8**: Production CLI with config files, logging, error recovery
- **B9**: Batch mode + 15 strategies (including skewness_correction, frequency_encoding)

### Phase C: Top Kaggle Improvements (Complete)
Based on analysis of Top 1% Kaggle solutions:

- **Outlier Removal**: Remove famous Ames outliers (GrLivArea > 4000 with low SalePrice)
- **Model Stacking**: 5-model ensemble (Ridge, Lasso, ElasticNet, XGBoost, LightGBM)
- **Feature Selection**: LightGBM importance-based filtering
- **Hyperparameter Tuning**: Optuna-based tuning for LightGBM and XGBoost

Expected RMSLE improvement: 0.02-0.04 reduction

---

## How the Iterative Agent Works

### Overview

The agent is an autonomous feature engineering system that:
1. Uses Gemini LLM to generate Python code for new features
2. Executes the code safely in a sandboxed environment
3. Evaluates the feature using 5-fold cross-validation
4. Keeps features that improve RMSLE, rejects those that don't
5. Uses SHAP analysis to guide the next iteration

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ITERATION LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SHAP       │───▶│   Gemini     │───▶│   Code       │      │
│  │   Analysis   │    │   LLM        │    │   Executor   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ▲                                       │               │
│         │                                       ▼               │
│  ┌──────────────┐                       ┌──────────────┐       │
│  │   LightGBM   │◀──────────────────────│   New        │       │
│  │   Evaluator  │                       │   Feature    │       │
│  └──────────────┘                       └──────────────┘       │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │   Compare    │───▶│   Memory     │                          │
│  │   RMSLE      │    │   (JSON)     │                          │
│  └──────────────┘    └──────────────┘                          │
│         │                                                       │
│         ▼                                                       │
│    KEPT (improved) ──▶ Add to pipeline                         │
│    REJECTED ─────────▶ Discard, try next                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Logic

#### 1. Initialization
```
Load train.csv (1465 rows, 81 columns)
    ↓
Preprocess with AmesPreprocessor
    - Create 5 baseline features (TotalSF, HouseAge, etc.)
    - Ordinal encode quality features
    - One-hot encode categoricals
    ↓
Result: 212 features
    ↓
Calculate baseline RMSLE using 5-fold CV (~0.103)
```

#### 2. SHAP Analysis (if --feedback enabled)
```
Train LightGBM on full dataset
    ↓
Compute SHAP values for 100 samples
    ↓
Extract top 5-10 most important features
    Example: OverallQual (0.095), TotalSF (0.087), GrLivArea (0.029)
    ↓
Generate suggestions: "Consider interactions with OverallQual"
```

#### 3. Feature Generation (Gemini)
```
Build prompt with:
    - Data description (column meanings)
    - Available columns
    - Few-shot examples for selected strategy
    - SHAP insights (top features, suggestions)
    - Previously tried features (to avoid duplicates)
    ↓
Send to Gemini 2.5-flash
    ↓
Extract Python code from response
    Example: df['QualSF'] = df['OverallQual'] * df['TotalSF']
```

#### 4. Code Execution (Sandboxed)
```
Create restricted environment:
    - Only allow: pandas, numpy, math operations
    - Block: file I/O, imports, exec, eval
    ↓
Execute generated code on DataFrame copy
    ↓
Check for new columns created
    ↓
If error or no new columns → Mark as FAILED
```

#### 5. Evaluation
```
Get features + new column(s)
    ↓
Run 5-fold CV with LightGBM
    - Train on 4 folds, validate on 1
    - Repeat 5 times, average RMSLE
    ↓
Compare new RMSLE vs current best
    ↓
If new_rmsle < best_rmsle:
    → KEPT: Add feature to pipeline, update best
Else:
    → REJECTED: Discard feature
```

#### 6. Memory Persistence
```
Save to agent_memory.json:
    - successful_features: [code, columns, improvement]
    - failed_features: [code, error]
    - iteration_history: [rmsle, success, description]
    - shap_history: [top_features, importance_scores]
    - baseline_rmsle, best_rmsle
```

#### 7. Next Iteration
```
If KEPT:
    Apply successful feature to accumulated DataFrame
    ↓
Repeat from Step 2 with updated model
```

### Feature Generation Strategies

The agent uses 6 strategies, randomly selected or based on SHAP insights:

| Strategy | Description | Example |
|----------|-------------|---------|
| `interaction` | Multiply related features | `OverallQual * TotalSF` |
| `ratio` | Divide for proportions | `BsmtFinSF1 / TotalBsmtSF` |
| `aggregation` | Sum related features | `OpenPorchSF + EnclosedPorch` |
| `binary` | 0/1 indicators | `(PoolArea > 0).astype(int)` |
| `polynomial` | Squared/log transforms | `OverallQual ** 2` |
| `temporal` | Time-based calculations | `YrSold - YearRemodAdd` |

### SHAP Feedback Loop

When `--feedback` is enabled:
1. After each evaluation, SHAP values are computed
2. Top important features are identified
3. Strategy is auto-selected based on feature types:
   - Quality features → `interaction` or `polynomial`
   - Area features → `ratio` or `aggregation`
   - Year features → `temporal`
4. Gemini receives SHAP summary in prompt
5. Generated features are guided to interact with top features

---

## Key Commands

```bash
# Production CLI (recommended)
./venv/bin/python -m src.agent.run -n 10 --clear --visualize

# Show config without running
./venv/bin/python -m src.agent.run --dry-run

# Run with custom options
./venv/bin/python -m src.agent.run \
    -n 20 \
    --feedback \
    --log-level DEBUG \
    --visualize

# Resume previous session (don't clear memory)
./venv/bin/python -m src.agent.run -n 10 --visualize

# Legacy iterative agent
./venv/bin/python -m src.agent.iterative_agent -n 10 -f -v --clear

# Visualize existing results
./venv/bin/python scripts/visualize_progress.py --all

# Run baseline training
./venv/bin/python src/baseline/train.py --use-tuned
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-n, --iterations` | Number of iterations to run |
| `--clear` | Clear memory and start fresh |
| `--feedback / --no-feedback` | Enable/disable SHAP feedback |
| `--visualize, -v` | Generate report after completion |
| `--config PATH` | Use custom YAML config |
| `--log-level` | DEBUG, INFO, WARNING, ERROR |
| `--dry-run` | Show config and exit |

## Output Files

| File | Description |
|------|-------------|
| `outputs/logs/agent_memory.json` | All iterations, features, SHAP history |
| `outputs/logs/agent.log` | Timestamped execution log |
| `outputs/logs/progress_plot.png` | RMSLE over iterations chart |
| `outputs/logs/progress_report.html` | Interactive HTML report |

## Environment

- Python 3.13 (venv/)
- Key deps: lightgbm, pandas, numpy, google-generativeai, shap, optuna, pyyaml
- API key: .env file with GEMINI_API_KEY

## Metrics

- Primary: RMSLE (Root Mean Squared Logarithmic Error)
- Target is log-transformed during training
- 5-fold CV for evaluation
- Lower RMSLE = better model

## Submission Workflow

```bash
# 1. Run agent to discover features
./venv/bin/python -m src.agent.iterative_agent -n 20 --feedback --batch --clear

# 2. (Optional) Tune hyperparameters for stacking models
./venv/bin/python -m src.baseline.hyperparameter_tuner

# 3. Generate final submission with model stacking
./venv/bin/python -m src.baseline.make_submission

# Options for make_submission:
#   --skip-features    Skip applying agent features
#   --skip-selection   Skip feature selection
#   --output PATH      Custom output path
```

### Model Stacking Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL STACKER                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Linear Models (30% weight, scaled features):               │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐                 │
│  │  Ridge  │  │  Lasso  │  │  ElasticNet │                 │
│  │  (10%)  │  │  (10%)  │  │    (10%)    │                 │
│  └────┬────┘  └────┬────┘  └──────┬──────┘                 │
│       │            │              │                         │
│       └────────────┼──────────────┘                         │
│                    │                                        │
│  Tree Models (70% weight, raw features):                    │
│  ┌───────────┐  ┌───────────┐                              │
│  │  XGBoost  │  │  LightGBM │                              │
│  │   (35%)   │  │   (35%)   │                              │
│  └─────┬─────┘  └─────┬─────┘                              │
│        │              │                                     │
│        └──────┬───────┘                                     │
│               │                                             │
│               ▼                                             │
│       ┌──────────────┐                                      │
│       │   Weighted   │                                      │
│       │    Blend     │                                      │
│       └──────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Design Docs

- `spec/001_Autonomous Feature Engineering Design.md` - Full architecture vision
- `spec/002_implementation_plan.md` - Implementation roadmap with progress
- `spec/003_kaggle_submission_plan.md` - Submission strategy
- `spec/004_visualization_plan.md` - Visualization design
- `spec/007_true_minimal_preprocessor.md` - Minimal preprocessing for agent
- `spec/008_top_kaggle_improvements.md` - Model stacking, outlier removal, feature selection
