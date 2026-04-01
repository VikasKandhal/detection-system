# FraudShield AI — Production Fraud Detection System

A production-grade fraud detection system built on the IEEE Fraud Detection dataset, featuring ensemble ML models, autoencoder anomaly detection, SHAP explainability, and a FastAPI + React dashboard.

## Features

- **Multi-Model Ensemble**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Autoencoder Anomaly Detection**: Semi-supervised learning on legitimate transactions
- **Advanced Feature Engineering**: Behavioral, velocity, risk-based, and aggregate features
- **SHAP Explainability**: Global feature importance and per-transaction explanations
- **FastAPI Backend**: REST API with `/predict`, `/predict/batch`, `/health`, `/model/info`
- **React Dashboard**: Real-time monitoring, transaction scoring, and analytics
- **Optimized for Precision**: Threshold tuning targeting ≥90% precision with strong recall

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Full pipeline (with hyperparameter tuning — ~1-2 hours)
python scripts/run_pipeline.py

# Quick mode (skip tuning — ~15-30 minutes)
python scripts/run_pipeline.py --quick

# Debug mode (small dataset)
python scripts/run_pipeline.py --quick --nrows 50000
```

### 3. Start API Server

```bash
python scripts/run_api.py
# API docs: http://localhost:8000/docs
```

### 4. Start React Frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

## Project Structure

```
├── src/                    # Core ML pipeline
│   ├── config.py           # Central configuration
│   ├── data_loader.py      # Data loading and merging
│   ├── eda.py              # Exploratory data analysis
│   ├── preprocessing.py    # Missing values, encoding, scaling
│   ├── feature_engineering.py  # Behavioral, velocity, risk features
│   ├── model_training.py   # Multi-model training with Optuna
│   ├── autoencoder.py      # Anomaly detection
│   ├── evaluation.py       # Metrics, plots, threshold optimization
│   ├── explainability.py   # SHAP values and explanations
│   └── utils.py            # Shared utilities
├── api/                    # FastAPI deployment
│   ├── main.py             # API endpoints
│   ├── schemas.py          # Pydantic models
│   └── model_service.py    # Model loading and inference
├── frontend/               # React dashboard
├── scripts/                # Entry-point scripts
├── models/                 # Trained model artifacts
├── reports/figures/        # Generated plots
├── docs/                   # Architecture documentation
├── Dockerfile
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score multiple transactions |
| GET | `/model/info` | Deployed model metadata |

## Scalability

See [docs/scalability.md](docs/scalability.md) for Kafka, Redis, and Kubernetes architecture.

## License

MIT
