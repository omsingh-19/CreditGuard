import joblib
import pandas as pd
from fastapi import APIRouter , Depends , File , UploadFile , HTTPException
from sqlalchemy.orm import Session
from Api.db.session import get_db
from Api.db.models import CreditPrediction
from Api.schema.credit import CreditInput ,CreditResponse ,CreditHistoryResponse
from Api.config import settings
import os


model = joblib.load(settings.model_path)
threshold = joblib.load(settings.threshold_path)

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
def credit_prediction(
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
    db.commit()
    db.refresh(data)

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