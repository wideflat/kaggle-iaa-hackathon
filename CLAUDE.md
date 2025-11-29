# CLAUDE.md - Project Context for Claude Code

## Project Overview

Kaggle competition project for Ames Housing price prediction with an autonomous LLM-powered feature engineering agent.

## Project Structure

```
kaggle-iaa-hackathon/
├── data/                    # Dataset files (train.csv, test.csv, data_description.txt)
├── src/
│   ├── baseline/            # Phase A: Traditional ML pipeline
│   │   ├── data_loader.py   # AmesDataLoader class
│   │   ├── preprocessor.py  # AmesPreprocessor with feature engineering
│   │   ├── model.py         # Model definitions
│   │   ├── train.py         # Main training script
│   │   └── tune_hyperparameters.py  # Optuna tuning
│   └── agent/               # Phase B: LLM-powered agent
│       ├── gemini_client.py # Gemini API wrapper
│       ├── code_executor.py # Safe code execution sandbox
│       ├── evaluator.py     # CV-based feature evaluation
│       └── simple_agent.py  # Main agent orchestration
├── outputs/
│   ├── models/              # Saved model params (best_lgbm_params.json)
│   └── predictions/         # Submission CSVs
├── notebooks/               # Kaggle submission notebooks
└── spec/                    # Design documents
```

## Current State

### Phase A (Complete)
- Baseline LightGBM model with CV RMSLE ~0.103
- Tuned hyperparameters via Optuna
- Features: TotalSF, HouseAge, RemodAge, TotalBath, PorchArea
- Ordinal encoding for quality features, one-hot for categoricals

### Phase B (B0-B3 Complete)
- Gemini 2.5-flash integration for feature generation
- Safe code executor with sandboxed environment
- CV-based evaluator comparing against baseline
- Single-iteration agent working

### Phase B (B4-B8 Pending)
- B4: Memory system for tracking tried features
- B5: Iterative loop (10+ iterations)
- B6: Enhanced prompting with few-shot examples
- B7: SHAP-based feedback loop
- B8: Production CLI

## Key Commands

```bash
# Run baseline training
venv/bin/python src/baseline/train.py --use-tuned

# Run feature engineering agent (single iteration)
venv/bin/python -m src.agent.simple_agent

# Tune hyperparameters
venv/bin/python src/baseline/tune_hyperparameters.py
```

## Environment

- Python 3.13 (venv/)
- Key deps: lightgbm, pandas, numpy, google-generativeai, optuna
- API key: .env file with GEMINI_API_KEY

## Metrics

- Primary: RMSLE (Root Mean Squared Logarithmic Error)
- Target is log-transformed during training
- 5-fold CV for evaluation

## Design Docs

- `spec/001_Autonomous Feature Engineering Design.md` - Full architecture vision
- `spec/002_implementation_plan.md` - Implementation roadmap
- `spec/003_kaggle_submission_plan.md` - Submission strategy
