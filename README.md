# CreditGuard V2

A production-grade credit risk and fraud detection API built with FastAPI, XGBoost, MLflow, and Docker. Features JWT authentication, async PostgreSQL, model retraining endpoints, and a full CI pipeline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + async SQLAlchemy |
| Database | PostgreSQL (Docker) + Alembic migrations |
| Auth | JWT (python-jose) + bcrypt |
| ML | XGBoost + SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |
| CI | GitHub Actions (lint → test → docker build) |
| Linting | Ruff |

---

## Models

### Credit Risk Model
- **Dataset:** Give Me Some Credit (Kaggle)
- **Algorithm:** XGBoost + SMOTE
- **AUC-ROC:** 0.8309
- **Threshold:** 0.5877 (Precision-Recall optimized)
- **MLflow Experiment:** `creditguard-credit-risk`
- **Artifacts:** `Model/credit_pipeline.pkl`, `Model/credit_threshold.pkl`

> Note: A `Medium` risk label with `prediction=0` is expected behavior — probability is above 0.3 but below the 0.58 threshold.

### Fraud Detection Model
- **Dataset:** ULB Creditcard (Kaggle)
- **Algorithm:** XGBoost + SMOTE
- **AUC-ROC:** 0.979
- **Threshold:** 0.99 (high precision mode)
- **MLflow Experiment:** `creditguard-fraud`
- **Artifacts:** `Model/fraud_model.pkl`, `Model/fraud_thresholds.pkl`

---

## Project Structure

```
CreditGuard/
├── Api/
│   ├── routes/
│   │   ├── auth.py          # JWT auth endpoints
│   │   ├── credit.py        # Credit risk endpoints
│   │   └── fraud.py         # Fraud detection endpoints
│   ├── db/
│   │   ├── models.py        # SQLAlchemy models
│   │   └── session.py       # Async DB session
│   ├── schemas/             # Pydantic request/response schemas
│   ├── config.py            # Pydantic settings (.env)
│   └── main.py              # FastAPI app
├── Model/
│   ├── train.py             # Credit model training
│   └── fraud_train.py       # Fraud model training
├── Data/                    # Training datasets (not committed)
├── alembic/                 # DB migrations
├── frontend/
│   └── index.html           # See note below
├── tests/
│   └── test_app.py          # CI test suite
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env                     # Not committed
```

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login, returns JWT token |
| GET | `/auth/me` | Get current user info |

### Credit Risk
| Method | Endpoint | Description |
|---|---|---|
| POST | `/credit/predict` | Single credit risk prediction |
| POST | `/credit/predict/batch` | Batch prediction from CSV |
| GET | `/credit/history` | Prediction history |
| GET | `/credit/model/metrics` | Current model metrics |
| GET | `/credit/model/runs` | MLflow run history |
| POST | `/credit/retrain` | Trigger background retraining |
| POST | `/credit/model/promote/{run_id}` | Promote MLflow run to production |

### Fraud Detection
| Method | Endpoint | Description |
|---|---|---|
| POST | `/fraud/predict` | Single transaction fraud check |
| GET | `/fraud/history` | Prediction history |
| GET | `/fraud/model/runs` | MLflow run history |
| POST | `/fraud/retrain` | Trigger background retraining |
| POST | `/fraud/model/promote/{run_id}` | Promote MLflow run to production |

---

## Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Training datasets in `Data/` directory

### Environment Variables

Create a `.env` file at the project root:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/creditguard
SECRET_KEY=your-secret-key-here
CREDIT_MODEL_PATH=Model/credit_pipeline.pkl
CREDIT_THRESHOLD_PATH=Model/credit_threshold.pkl
MLFLOW_TRACKING_URI=http://mlflow:5000
FRAUD_MODEL_PATH=Model/fraud_model.pkl
FRAUD_THRESHOLD_PATH=Model/fraud_thresholds.pkl
```

> **Note:** Use `@db:5432` inside Docker and `@localhost:5432` for local Alembic migrations.

### Running with Docker

```bash
# Start all services (API, PostgreSQL, MLflow)
docker-compose up --build

# Run database migrations
docker-compose exec api alembic upgrade head

# Train the credit model
docker-compose exec api python -m Model.train

# Train the fraud model
docker-compose exec api python -m Model.fraud_train
```

### Services

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Frontend | http://localhost:8000 |

---

## CI Pipeline

GitHub Actions runs on every push to the `main` branch:

```
push to main
    │
    ▼
 lint (ruff)          ~6s
    │
    ├──────────────────────┐
    ▼                      ▼
 test (pytest)       docker build
    ~58s                ~1m
```

- **lint** — Ruff checks the entire codebase, excluding `alembic/`
- **test** — Pytest runs against a SQLite test database with mocked env vars
- **docker-build** — Confirms the Dockerfile builds cleanly

---

## Model Retraining Workflow

1. Hit `POST /credit/retrain` or `POST /fraud/retrain` — starts training in a background task
2. Open MLflow UI at `http://localhost:5000` and copy the `run_id` from the new run
3. Hit `POST /credit/model/promote/{run_id}` — loads the model from MLflow, saves to disk, hot-swaps the global model in memory without restarting the API

---

## Frontend

> ⚠️ **The frontend (`frontend/index.html`) was not written by me.** It was generated by Claude (Anthropic) as part of the development process. The UI covers all API functionality — auth, credit risk prediction, fraud detection, prediction history, MLflow run inspection, and model retrain/promote controls.

The frontend is a single static HTML file served by FastAPI at the root route. No framework, no build step.

---

## Development Notes

- Models load lazily at startup — if `.pkl` files are missing the app starts with `model=None` instead of crashing
- Async throughout: `AsyncSession` for all DB operations
- Batch prediction endpoint accepts CSV uploads
- MLflow runs store: `auc_roc`, `best_threshold`, `precision_class_1`, `recall_class_1`, `n_estimators`, `learning_rate`, `max_depth`
- NaN values in MLflow run DataFrames are replaced with `None` before JSON serialization

---

## Author

**Om Singh** — 1st year AI-DS, SDSF DAVV, Indore  
GitHub: [@omsingh-19](https://github.com/omsingh-19)
