# CreditGuard — Credit Risk Intelligence Platform

An end-to-end Machine Learning API that predicts credit default risk for loan applicants. Built with FastAPI, XGBoost, and SQLAlchemy. Fully containerized with Docker.

---

## Live Demo

Start the server and visit `http://localhost:8000` to open the dashboard.

---

## Features

- Single applicant risk scoring with probability score and risk label (Low / Medium / High)
- Bulk CSV prediction via batch endpoint
- Prediction history stored in SQLite database
- Model performance metrics endpoint
- Fully Dockerized — runs with one command
- Interactive dashboard served directly from the API

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost Classifier |
| Preprocessing | Scikit-learn Pipeline + ColumnTransformer |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Backend | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy |
| Containerization | Docker + docker-compose |
| Frontend | HTML/CSS/JS (see note below) |

---

## Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.8309 |
| Decision Threshold | 0.5877 |
| Train Samples | 120,000 (after SMOTE) |
| Test Samples | 30,000 |

> **Note:** Medium `risk_label` with `prediction=0` is expected behavior. It means the probability is above 0.30 (Medium range) but below the decision threshold of 0.58, so the model does not predict default. This is intentional — risk label and prediction are two different things.

---

## Dataset

Download **Give Me Some Credit** from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data

Place `cs-training.csv` in `data/raw/`.

---

## Project Structure

```
CreditGuard/
│
├── data/raw/               # Dataset (not committed)
├── notebooks/
│   └── credit_eda.ipynb    # Exploratory data analysis
│
├── Model/                  # Saved model files (not committed)
│   ├── credit_pipeline.pkl
│   └── threshold.pkl
│
├── ml/
│   └── train.py            # Model training script
│
├── Api/
│   ├── main.py             # FastAPI app entry point
│   ├── routes/
│   │   └── credit.py       # API endpoints
│   ├── schemas/
│   │   └── credit.py       # Pydantic request/response models
│   └── db/
│       ├── models.py        # SQLAlchemy table definitions
│       └── session.py       # Database connection setup
│
├── frontend/
│   └── index.html          # Dashboard UI
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/omsingh-19/CreditGuard
cd CreditGuard
```

Train the model first (required — model files are not committed):
```bash
pip install -r requirements.txt
python Model/train.py
```

Then run with Docker:
```bash
docker-compose up --build
```

Visit `http://localhost:8000`

---

### Option 2 — Local

```bash
pip install -r requirements.txt
python Model/train.py
uvicorn Api.main:app --reload
```

Visit `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/credit/predict` | Single applicant prediction |
| POST | `/credit/predict/batch` | CSV bulk prediction |
| GET | `/credit/history` | Prediction history (`?limit=10`) |
| GET | `/credit/model/metrics` | Model performance stats |

### Sample Request

```bash
curl -X POST http://localhost:8000/credit/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "income": 5000,
    "debt_ratio": 0.35,
    "revolving_utilization": 0.5,
    "num_open_credit_lines": 4,
    "num_real_estate_loans": 1,
    "num_late_30_59": 0,
    "num_late_60_89": 0,
    "num_late_90": 0,
    "dependents": 2
  }'
```

### Sample Response

```json
{
  "prediction": 0,
  "risk_score": 0.2134,
  "risk_label": "Low",
  "threshold_used": 0.5877
}
```

---

## Retrain the Model

```bash
python Model/train.py
```

This will clean the data, train a new XGBoost pipeline with SMOTE, find the optimal threshold via precision-recall curve, and save the model to `Model/`.

---

## Frontend

> **Disclosure:** The frontend dashboard (`frontend/index.html`) was generated using Claude (claude.ai) based on the API structure and design requirements. All backend code, ML pipeline, and API logic was written manually.

---

## Author

**Om Singh Lodhi**

1st Year AI-DS Student — SDSF, DAVV, Indore


---

## License

Free to use for learning and educational purposes.
