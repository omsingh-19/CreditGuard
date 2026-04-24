from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from Api.config import settings
import joblib
from Api.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from Api.schemas.fraud import FraudInput , FraudResponse , FraudHistoryResponse
from Api.db.models import FraudPrediction ,User
import pandas as pd
from Api.routes.auth import get_current_user
import mlflow.sklearn

router = APIRouter(prefix="/fraud", tags=["fraud"])

try:
    model = joblib.load(settings.fraud_model_path)
    threshold = joblib.load(settings.fraud_threshold_path)
except FileNotFoundError:
    model = None
    threshold = None

def reload_fraud_model():
    """Reload the fraud model and threshold from disk into global scope"""

    global model, threshold
    model = joblib.load(settings.fraud_model_path)
    threshold = joblib.load(settings.fraud_threshold_path)

    return {"model_reloaded": True, "threshold": float(threshold)}


def get_fraud_prediction(input_data : FraudInput):

    data_dict = {
        "Time": input_data.time,
        "Amount": input_data.amount,
    }

    # Add V1-V28
    for i in range(1, 29):
        data_dict[f"V{i}"] = getattr(input_data, f"v{i}")

    df = pd.DataFrame([data_dict])

    # Ensure column order matches training (Time, Amount, V1-V28)
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = df[cols]

    fraud_probability = model.predict_proba(df)[0][1]
    prediction = int(fraud_probability >= threshold)

    if fraud_probability > 0.7:
        risk_label = "High"
    elif fraud_probability > 0.4:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return fraud_probability, prediction, risk_label

@router.post("/predict", response_model=FraudResponse)
async def predict_fraud(
    input_data: FraudInput,
    db: AsyncSession = Depends(get_db)
):
    fraud_probability, prediction, risk_label = get_fraud_prediction(input_data)

    record = FraudPrediction(
        amount=input_data.amount,
        time=input_data.time,
        fraud_probability=fraud_probability,
        risk_label=risk_label,
        prediction=prediction,
        threshold_used=threshold
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)

    return FraudResponse(
        prediction=prediction,
        fraud_probability=fraud_probability,
        risk_label=risk_label,
        threshold_used=threshold
    )


@router.get("/history", response_model=list[FraudHistoryResponse])
async def get_fraud_history(limit : int =5,
                            db: AsyncSession = Depends(get_db)):
    
    result = await db.execute(
        select(FraudPrediction)
        .order_by(FraudPrediction.created_at.desc())
        .limit(limit)
    )

    records = result.scalars().all()

    return records

@router.post("/retrain")
async def retrain_fraud_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger fraud model retraining in background"""
    
    def train_in_background():
        from Model.fraud_train import Train_Model
        run_id = Train_Model()
        print(f"✓ Fraud model retrained. MLflow run_id: {run_id}")
    
    background_tasks.add_task(train_in_background)
    
    return {
        "status": "training_started",
        "message": "Fraud model retraining started in background. Check MLflow for progress."
    }


@router.post("/model/promote/{run_id}")
async def promote_fraud_model(
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    """Promote a specific MLflow run to production"""
    
    try:
        # Load model from MLflow
        model_uri = f"runs:/{run_id}/fraud_pipeline"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Get metrics from the run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        threshold_value = run.data.metrics.get("best_threshold")
        auc_score = run.data.metrics.get("auc_roc")
        
        if threshold_value is None:
            raise HTTPException(status_code=400, detail="Run missing best_threshold metric")
        
        # Save to production paths
        joblib.dump(loaded_model, settings.fraud_model_path)
        joblib.dump(threshold_value, settings.fraud_threshold_path)
        
        # Reload global model
        reload_result = reload_fraud_model()
        
        return {
            "status": "promoted",
            "run_id": run_id,
            "auc_roc": auc_score,
            "threshold": threshold_value,
            "model_reloaded": reload_result["model_reloaded"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")
                            

@router.get("/model/runs")
async def get_fraud_runs(limit: int = 5):
    
    runs_df = mlflow.search_runs(experiment_names=["creditguard-fraud"])
    
    if runs_df.empty:
        return []
    
    columns = [
        "run_id",
        "start_time",
        "metrics.auc_roc",
        "metrics.best_threshold",
        "metrics.precision_class_1",
        "metrics.recall_class_1",
        "params.n_estimators",
        "params.learning_rate",
        "params.max_depth"
    ]
    
    runs_df["start_time"] = runs_df["start_time"].astype(str)
    runs_df = runs_df[columns]
    run_limited = runs_df.head(limit)
    
    # Replace NaN with None for JSON compatibility
    run_limited = run_limited.where(pd.notna(run_limited), None)
    
    run_dict = run_limited.to_dict(orient="records")
    
    return run_dict