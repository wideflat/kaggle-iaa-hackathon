# Ames Housing Price Prediction with Autonomous Feature Engineering

> **"What if an AI could read the data description, understand feature relationships, and engineer new features—all while learning from its own mistakes?"**

A Kaggle competition project featuring an LLM-powered autonomous agent that discovers and engineers features for the Ames Housing dataset.

## Watch the Agent in Action

[![Watch the video](https://img.youtube.com/vi/2UByOVGtdY8/maxresdefault.jpg)](https://www.youtube.com/watch?v=2UByOVGtdY8)

---

## The Innovation

### The Problem

Feature engineering is often the most impactful yet tedious part of machine learning. Data scientists spend hours:
- Reading documentation to understand feature meanings
- Experimenting with transformations (log, square, ratios)
- Creating interaction features based on domain knowledge
- Testing each feature to see if it actually helps

**What if we could automate this entire process?**

### Our Solution: An Autonomous Feature Engineering Agent

We built an **LLM-powered agent** that:

| Innovation | Description |
|------------|-------------|
| **Reads Documentation** | Understands `data_description.txt` to learn what each column means |
| **Uses SHAP Feedback** | Analyzes which features matter most and targets improvements there |
| **Self-Evaluates** | Tests each feature with 5-fold CV, keeps only what improves RMSLE |
| **Runs in Parallel** | Producer-consumer pattern with batched Gemini API calls |
| **Learns from Failure** | Remembers what didn't work to avoid repeating mistakes |

---

## Key Features

- **Autonomous Feature Engineering**: LLM-powered agent generates Python code for new features
- **SHAP-based Feedback Loop**: Guides feature generation based on feature importance
- **Parallel Processing**: Multiple workers evaluate features concurrently
- **Three-Layer Model Stacking**: 7-model ensemble with meta-learners
- **Hyperparameter Tuning**: Optuna-based optimization
- **Real-time Dashboard**: Monitor agent progress in browser

## Project Structure

```
kaggle-iaa-hackathon/
├── data/                    # Dataset files
├── config/                  # Agent configuration
├── src/
│   ├── baseline/            # Traditional ML pipeline
│   │   ├── preprocessor.py  # Feature engineering
│   │   ├── model_stacker.py # Three-layer ensemble
│   │   └── make_submission.py
│   └── agent/               # LLM-powered agent
│       ├── gemini_client.py # Gemini API wrapper
│       ├── code_executor.py # Safe code execution
│       ├── evaluator.py     # CV evaluation + SHAP
│       ├── parallel_agent.py # Main agent entry
│       └── memory.py        # Iteration persistence
├── outputs/
│   ├── models/              # Saved model parameters
│   ├── predictions/         # Submission CSVs
│   └── logs/                # Agent logs and reports
└── notebooks/               # Kaggle submission notebooks
```

## Installation

```bash
# Clone the repository
git clone https://github.com/wideflat/kaggle-iaa-hackathon.git
cd kaggle-iaa-hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Run the Autonomous Agent

```bash
# Run parallel agent with dashboard
python -m src.agent.parallel_agent -n 20 -w 2 -b 5 --dashboard --clear

# With SHAP-based feedback
python -m src.agent.parallel_agent -n 20 -w 2 -b 5 --dashboard --feedback
```

### Generate Submission

```bash
# Run agent + hyperparameter tuning
python -m src.agent.parallel_agent -n 50 -w 2 -b 5 --dashboard --clear --tune

# Generate final submission with model stacking
python -m src.baseline.make_submission --skip-selection --blend-weight 0.5
```

## How the Agent Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS AGENT LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📄 Data Description    ──────▶  🧠 Gemini LLM                 │
│   📊 SHAP Insights       ──────▶  (Feature Generator)           │
│   ❌ Failed Attempts     ──────▶                                │
│                                        │                        │
│                                        ▼                        │
│                               ┌──────────────┐                  │
│                               │  Python Code │                  │
│                               │  Generator   │                  │
│                               └──────┬───────┘                  │
│                                      │                          │
│                                      ▼                          │
│   ┌──────────────┐           ┌──────────────┐                  │
│   │   LightGBM   │◀──────────│   Sandbox    │                  │
│   │   5-Fold CV  │           │   Executor   │                  │
│   └──────┬───────┘           └──────────────┘                  │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐           ┌──────────────┐                  │
│   │  RMSLE       │──────────▶│   Memory     │                  │
│   │  Improved?   │           │   (JSON)     │                  │
│   └──────────────┘           └──────────────┘                  │
│          │                                                      │
│    ✅ KEEP  /  ❌ REJECT                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **SHAP Analysis**: Identifies top important features
2. **Feature Generation**: Gemini generates Python code based on insights
3. **Safe Execution**: Code runs in sandboxed environment
4. **Evaluation**: 5-fold CV determines if feature improves RMSLE
5. **Memory**: Successful features are persisted for future iterations

## Feature Generation Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| `interaction` | Multiply related features | `OverallQual * TotalSF` |
| `ratio` | Divide for proportions | `BsmtFinSF1 / TotalBsmtSF` |
| `aggregation` | Sum related features | `OpenPorchSF + EnclosedPorch` |
| `binary` | 0/1 indicators | `(PoolArea > 0).astype(int)` |
| `polynomial` | Squared/log transforms | `OverallQual ** 2` |
| `temporal` | Time-based calculations | `YrSold - YearRemodAdd` |

## CLI Options

| Option | Description |
|--------|-------------|
| `-n, --iterations` | Total number of iterations (default: 10) |
| `-w, --workers` | Number of parallel workers (default: 2) |
| `-b, --batch-size` | Features per API batch (default: 5) |
| `--clear` | Clear memory and start fresh |
| `-d, --dashboard` | Open real-time dashboard in browser |
| `-f, --feedback` | Enable SHAP-based feedback |
| `--tune` | Run hyperparameter tuning after feature engineering |

## Three-Layer Model Stacking Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   THREE-LAYER MODEL STACKING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: Base Models (7 models)                                │
│  ┌────────┐ ┌────────┐ ┌───────────┐ ┌─────┐                   │
│  │ Ridge  │ │ Lasso  │ │ElasticNet │ │ SVR │                   │
│  │  10%   │ │  10%   │ │    10%    │ │  5% │                   │
│  └────┬───┘ └───┬────┘ └─────┬─────┘ └──┬──┘                   │
│       │         │            │          │                       │
│  ┌────┴───┐ ┌───┴────┐ ┌─────┴─────┐                           │
│  │  GBR   │ │XGBoost │ │ LightGBM  │                           │
│  │  20%   │ │  20%   │ │    25%    │                           │
│  └────┬───┘ └───┬────┘ └─────┬─────┘                           │
│       │         │            │                                  │
│       └─────────┼────────────┘                                  │
│                 │                                               │
│                 ▼                                               │
│  LAYER 2: Meta-Learners                                         │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │ Layer 2a: Weighted Avg  │  │ Layer 2b: StackingCV        │  │
│  │ (blend of 7 models)     │  │ (XGBoost meta-learner)      │  │
│  └───────────┬─────────────┘  └──────────────┬──────────────┘  │
│              │                               │                  │
│              └───────────┬───────────────────┘                  │
│                          │                                      │
│                          ▼                                      │
│  LAYER 3: Final Ensemble                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         50% Layer 2a  +  50% Layer 2b                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Base Models

| Model | Type | Weight | CV RMSLE |
|-------|------|--------|----------|
| Ridge | Linear | 10% | 0.09991 |
| Lasso | Linear | 10% | 0.09968 |
| ElasticNet | Linear | 10% | 0.09968 |
| SVR | Kernel | 5% | 0.15547 |
| GBR | Tree | 20% | 0.09359 |
| XGBoost | Tree | 20% | 0.09608 |
| LightGBM | Tree | 25% | 0.09806 |

### Performance Summary

| Stage | CV RMSE | vs Baseline |
|-------|---------|-------------|
| Best Single Model (GBR) | 0.09359 | -6.8% |
| Layer 2a (Weighted Avg) | 0.09206 | -8.3% |
| Layer 3 (Final Blend) | 0.05803 | -42.2% |

**Baseline RMSLE**: 0.10041 (single LightGBM without agent features)

## Agent Results

| Metric | Value |
|--------|-------|
| **Baseline RMSLE** | 0.10041 |
| **Best RMSLE** | 0.09848 |
| **Improvement** | 1.9% |
| **Features Tried** | 20 |
| **Features Kept** | 4 |

### Features Discovered by the Agent

| # | Feature | Strategy | RMSLE Improvement |
|---|---------|----------|-------------------|
| 1 | `OverallQual_GrLivArea_Product` | Interaction | +0.00118 |
| 2 | `TotalLivingSF` | Aggregation | +0.00045 |
| 3 | `GrLivArea_Cubic` | Polynomial | +0.00024 |
| 4 | `LivAreaRatioToLot` | Ratio | +0.00005 |

## Requirements

- Python 3.10+
- LightGBM, XGBoost, scikit-learn
- Google Generative AI (Gemini)
- SHAP, Optuna
- pandas, numpy

## Resources

- **GitHub**: https://github.com/wideflat/kaggle-iaa-hackathon
- **Video Demo**: https://youtu.be/2UByOVGtdY8

## License

MIT License
