from sqlalchemy import Column , Integer, DateTime,Float ,String
from sqlalchemy.sql import func
from Api.db.session import Base

class CreditPrediction(Base):
    __tablename__ = "credit_predictions"

    id = Column(Integer , primary_key = True,index = True)
    age = Column(Integer)
    income = Column(Float)
    debt_ratio = Column(Float)
    revolving_utilization = Column(Float)
    num_open_credit_lines = Column(Integer)
    num_real_estate_loans = Column(Integer)
    num_late_30_59 = Column(Integer)
    num_late_60_89 = Column(Integer)
    num_late_90 = Column(Integer)
    dependents = Column(Integer)
    risk_score = Column(Float)
    risk_label = Column(String)
    threshold_used = Column(Float)
    prediction = Column(Integer)
    created_at = Column(DateTime, default=func.now())