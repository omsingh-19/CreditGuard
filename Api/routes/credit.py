import joblib
import pandas as pd
from fastapi import APIRouter , Depends , File , UploadFile , HTTPException,BackgroundTasks
from sqlalchemy.orm import Session
from Api.db.session import get_db
from Api.db.models import CreditPrediction
from Api.schemas.credit import CreditInput ,CreditResponse ,CreditHistoryResponse
from Api.config import settings
import mlflow
from Api.routes.auth import get_current_user
from Api.db.models import User


try:
    model = joblib.load(settings.credit_model_path)
    threshold = joblib.load(settings.credit_threshold_path)
except FileNotFoundError:
    model = None
    threshold = None

def reload_credit_model():

    """Reload the credit model and threshold from disk into global scope"""
    global model, threshold
    model = joblib.load(settings.credit_model_path)
    threshold = joblib.load(settings.credit_threshold_path)

    return {"model_reloaded": True, "threshold": float(threshold)}


def get_prediction(input_data : CreditInput)->CreditResponse:

    data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": input_data.revolving_utilization,
        "age": input_data.age,
        "NumberOfTime30-59DaysPastDueNotWorse": input_data.num_late_30_59,
        "DebtRatio": input_data.debt_ratio,
        "MonthlyIncome": input_data.income,
        "NumberOfOpenCreditLinesAndLoans": input_data.num_open_credit_lines,
        "NumberOfTimes90DaysLate": input_data.num_late_90,
        "NumberRealEstateLoansOrLines": input_data.num_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": input_data.num_late_60_89,
        "NumberOfDependents": input_data.dependents
    }])
    prob = model.predict_proba(data)[:,1][0]
    prediction = int(prob >= threshold)
    if prob>0.6:
        risk_label = "High"
    elif prob>0.3:
        risk_label = "Medium"
    else: 
        risk_label = "Low"

    return {
        "prediction" : prediction,
        "risk_label" : risk_label,
        "risk_score" : prob,
        "threshold_used" : threshold
    }

router = APIRouter(prefix="/credit",tags=["credit"])
@router.post("/predict",response_model=CreditResponse)
async def credit_prediction(
    input_data : CreditInput ,
    db : Session=Depends(get_db)
):
    
    prediction = get_prediction(input_data)
    data = CreditPrediction(
        age=input_data.age,
        income=input_data.income,
        debt_ratio=input_data.debt_ratio,
        revolving_utilization=input_data.revolving_utilization,
        num_open_credit_lines=input_data.num_open_credit_lines,
        num_real_estate_loans=input_data.num_real_estate_loans,
        num_late_30_59=input_data.num_late_30_59,
        num_late_60_89=input_data.num_late_60_89,
        num_late_90=input_data.num_late_90,
        dependents=input_data.dependents,
        risk_score=prediction["risk_score"],
        prediction=prediction["prediction"],
        risk_label=prediction["risk_label"],
        threshold_used=threshold,
    )
    
    db.add(data)
    await db.commit()
    await db.refresh(data)

    return prediction

@router.get("/history",response_model=list[CreditHistoryResponse])
def get_history(limit : int = 5, db : Session = Depends(get_db)):

    records = db.query(CreditPrediction).limit(limit).all()
    return records

@router.get("/model/metrics")
def get_model_stats():

    return {
        "auc_roc" : 0.8309,
        "model" : "XGBoost",
        "threshold": 0.5876834,
        "dataset" : "Give Me Some Credit",
        "train_samples" : 120000,
        "test_samples" : 30000
    }

@router.post("/predict/batch" , response_model=list)
def predict_batch(file : UploadFile = File(...)):
    
    df = pd.read_csv(file.file)
    COLUMNS = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    missing = set(COLUMNS) - set(df.columns)
    if missing :
        raise HTTPException(status_code=400 , detail=f"missing columns{missing}")
    
    df = df[COLUMNS]
    
    probs = model.predict_proba(df)[:, 1]
    results=[]
    for prob in probs :
        results.append({
            "risk_score":float(prob),
            "risk_label" : "High" if prob>0.6 else "Medium" if prob>0.3 else "Low",
            "prediction" : int(prob>=threshold)
        })

    return results

@router.get("/model/runs")
def get_model_runs(limit: int = 5):

    runs_df = mlflow.search_runs(experiment_names=["creditguard-credit-risk"])
    runs_df = runs_df.dropna(subset=["metrics.auc_roc"])

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

    # 🔥 FIX HERE
    runs_df = runs_df.fillna(0)

    run_limited = runs_df.head(limit)

    return run_limited.to_dict(orient="records")

@router.post("/retrain")
async def retrain_credit_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger credit model retraining in background"""
    
    def train_in_background():
        from Model.train import Train_Model
        run_id = Train_Model()
        print(f"✓ Credit model retrained. MLflow run_id: {run_id}")
    
    background_tasks.add_task(train_in_background)
    
    return {
        "status": "training_started",
        "message": "Credit model retraining started in background. Check /credit/model/runs for progress."
    }


@router.post("/model/promote/{run_id}")
async def promote_credit_model(
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    """Promote a specific MLflow run to production"""
    
    try:
        # Load model from MLflow
        model_uri = f"runs:/{run_id}/credit_pipeline"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Get metrics from the run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        threshold_value = run.data.metrics.get("best_threshold")
        auc_score = run.data.metrics.get("auc_roc")
        
        if threshold_value is None:
            raise HTTPException(status_code=400, detail="Run missing best_threshold metric")
        
        # Save to production paths
        joblib.dump(loaded_model, settings.credit_model_path)
        joblib.dump(threshold_value, settings.credit_threshold_path)
        
        # Reload global model
        reload_result = reload_credit_model()
        
        return {
            "status": "promoted",
            "run_id": run_id,
            "auc_roc": auc_score,
            "threshold": threshold_value,
            "model_reloaded": reload_result["model_reloaded"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")