from fastapi import FastAPI
from Api.routes.credit import router as credit_router
from Api.db.session import Base ,engine

Base.metadata.create_all(bind = engine)
app = FastAPI(
    title="CreditGuard API",
    description="Credit Risk Scoring API",
    version="1.0.0"
)
app.include_router(credit_router)

@app.get("/health")
def health():
    return {"status":"ok","version":"1.0.0"}