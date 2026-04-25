from pydantic import BaseModel
from datetime import datetime

class CreditInput(BaseModel):
    age : int
    income : float
    debt_ratio : float
    revolving_utilization : float
    num_open_credit_lines : int
    num_real_estate_loans : int
    num_late_30_59 : int
    num_late_60_89 : int
    num_late_90 : int
    dependents : int


class CreditResponse(BaseModel):
    prediction : int
    risk_score : float
    risk_label : str
    threshold_used: float

class CreditHistoryResponse(BaseModel):
    id: int
    age: int
    income: float
    risk_score: float
    risk_label: str
    prediction: int
    threshold_used: float
    created_at: datetime

    class Config:
        from_attributes = True