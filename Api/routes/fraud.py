from fastapi import APIRouter, Depends
from Api.config import settings
import joblib
from Api.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from Api.schemas.fraud import FraudInput , FraudResponse , FraudHistoryResponse
from Api.db.models import FraudPrediction
import pandas as pd

router = APIRouter(prefix="/fraud", tags=["fraud"])

model = joblib.load(settings.fraud_model_path)
threshold = joblib.load(settings.fraud_threshold_path)


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
                            