# CreditGuard V2

CreditGuard is a production-grade **credit risk scoring and fraud detection platform** built as a REST API. It takes financial data about a borrower or a transaction, runs it through trained XGBoost machine learning models, and returns a risk prediction — along with a probability score and a human-readable risk label (Low / Medium / High).

**Who is this for?** Fintech applications, lending platforms, or anyone who needs to integrate automated credit risk assessment or real-time fraud detection into their product without building the ML infrastructure from scratch.

**What problem does it solve?**
- A lender wants to know if a borrower is likely to default — send their financial profile to `/credit/predict`, get back a risk score and label in milliseconds.
- A payment system wants to flag suspicious transactions in real time — send the transaction features to `/fraud/predict`, get back a fraud probability and verdict.
- A data team wants to retrain models on fresh data and promote the best run to production — all without restarting the server.

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

Predicts the probability that a borrower will default on a loan within the next two years.

- **Dataset:** Give Me Some Credit (Kaggle) — 150,000 borrower records
- **Algorithm:** XGBoost + SMOTE (handles class imbalance)
- **AUC-ROC:** 0.8309
- **Threshold:** 0.5877 (Precision-Recall optimized — trades some recall for higher precision)
- **Output:** `prediction` (0/1), `probability` (0.0–1.0), `risk_label` (Low / Medium / High)
- **MLflow Experiment:** `creditguard-credit-risk`
- **Artifacts:** `Model/credit_pipeline.pkl`, `Model/credit_threshold.pkl`

**Input features:** `RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents`

> **Note:** A `Medium` risk label with `prediction=0` is expected behavior — it means the probability is above 0.3 (enough to flag as medium risk) but below the 0.58 classification threshold (not high enough to predict default).

### Fraud Detection Model

Classifies whether a credit card transaction is fraudulent.

- **Dataset:** ULB Creditcard (Kaggle) — 284,807 transactions, only 0.17% fraudulent
- **Algorithm:** XGBoost + SMOTE
- **AUC-ROC:** 0.979
- **Threshold:** 0.99 (high precision mode — minimizes false positives at the cost of missing some fraud)
- **Output:** `is_fraud` (true/false), `fraud_probability` (0.0–1.0)
- **MLflow Experiment:** `creditguard-fraud`
- **Artifacts:** `Model/fraud_model.pkl`, `Model/fraud_thresholds.pkl`

**Input features:** PCA-transformed components V1–V28 (anonymized by ULB for privacy), plus `Amount` and `Time`

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
├── Data/raw                 # Training datasets (not committed)
├── alembic/                 # DB migrations
├── frontend/
│   └── index.html           # Static frontend served at /
├── tests/
│   └── test_app.py          # CI test suite
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── notebooks/
│   └── credit_eda.ipynb     # Exploratory data analysis
├── mlartifacts/             # MLflow stored model artifacts
├── mlruns/                  # MLflow run metadata
├── .gitignore
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── ruff.toml
└── .env                     # Not committed — see Environment Variables
```

---

## API Endpoints

All prediction and history endpoints require a valid JWT token in the `Authorization: Bearer <token>` header. Register and login first to obtain a token.

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login — returns JWT token |
| GET | `/auth/me` | Get current user info |

### Credit Risk
| Method | Endpoint | Description |
|---|---|---|
| POST | `/credit/predict` | Single borrower credit risk prediction |
| POST | `/credit/predict/batch` | Batch prediction from uploaded CSV |
| GET | `/credit/history` | Prediction history for current user |
| GET | `/credit/model/metrics` | Current production model metrics |
| GET | `/credit/model/runs` | All MLflow run history |
| POST | `/credit/retrain` | Trigger background model retraining |
| POST | `/credit/model/promote/{run_id}` | Promote an MLflow run to production |

### Fraud Detection
| Method | Endpoint | Description |
|---|---|---|
| POST | `/fraud/predict` | Single transaction fraud check |
| GET | `/fraud/history` | Prediction history for current user |
| GET | `/fraud/model/runs` | All MLflow run history |
| POST | `/fraud/retrain` | Trigger background model retraining |
| POST | `/fraud/model/promote/{run_id}` | Promote an MLflow run to production |

---

## Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Training datasets placed in the `Data/` directory before running training commands:
  - `Data/raw/cs-training.csv` — Give Me Some Credit (Kaggle)
  - `Data/raw/creditcard.csv` — ULB Creditcard (Kaggle)

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

> **Note:** Use `@db:5432` inside Docker Compose and `@localhost:5432` when running Alembic migrations locally.

### Running with Docker

```bash
# Start all services (API, PostgreSQL, MLflow)
docker-compose up --build

# Run database migrations
docker-compose exec api alembic upgrade head

# Train the credit risk model
docker-compose exec api python -m Model.train

# Train the fraud detection model
docker-compose exec api python -m Model.fraud_train
```

### Services

| Service | URL | Description |
|---|---|---|
| API | http://localhost:8000 | Main FastAPI application |
| API Docs | http://localhost:8000/docs | Interactive Swagger UI |
| MLflow UI | http://localhost:5000 | Experiment tracking dashboard |
| Frontend | http://localhost:8000 | Static HTML frontend |

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
- **test** — Pytest runs against a SQLite test database with mocked environment variables; no real models or PostgreSQL required
- **docker-build** — Confirms the Dockerfile builds cleanly

---

## Model Retraining Workflow

CreditGuard supports a full retrain → evaluate → promote workflow without any server restarts.

1. Hit `POST /credit/retrain` or `POST /fraud/retrain` — starts training in a background task; a new MLflow run is created and logged automatically
2. Open the MLflow UI at `http://localhost:5000` — inspect metrics (AUC-ROC, precision, recall, threshold) across all runs and copy the `run_id` of the run you want to promote
3. Hit `POST /credit/model/promote/{run_id}` — the API loads the model from MLflow, saves it to disk, and hot-swaps the in-memory global model; live traffic immediately uses the new model

**MLflow tracks per run:** `auc_roc`, `best_threshold`, `precision_class_1`, `recall_class_1`, `n_estimators`, `learning_rate`, `max_depth`

---

## Frontend

The project includes a single-file static frontend (`frontend/index.html`) served by FastAPI at the root route. No framework, no build step — just open `http://localhost:8000` in a browser.

**Features:**
- JWT login and registration
- Credit risk prediction form
- Fraud detection form with **LOAD LEGIT TX** / **LOAD FRAUD TX** presets (real ULB PCA values for quick testing)
- Prediction history tables
- MLflow run inspection
- Retrain and promote controls

> ⚠️ The frontend was generated by Claude (Anthropic) and is not part of Om's own code. It is a development aid, not a portfolio artifact.

---

## Development Notes

- **Lazy model loading** — if `.pkl` files are missing at startup, the app starts with `model=None` instead of crashing; predictions return a 503 until a model is trained and promoted
- **Async throughout** — `AsyncSession` for all database operations; no blocking calls in the request path
- **Batch prediction** — `/credit/predict/batch` accepts CSV file uploads and returns predictions for every row
- **NaN handling** — MLflow run DataFrames replace `NaN` with `None` before JSON serialization (`.where(pd.notna(), None)`)
- **CI uses SQLite + fake env vars** — no real database or model files required to pass CI; `joblib.load` is wrapped in `try/except FileNotFoundError`

---

## Author

**Om Singh** — 1st year AI-DS, SDSF DAVV, Indore
GitHub: [@omsingh-19](https://github.com/omsingh-19)
