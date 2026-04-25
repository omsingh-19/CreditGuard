from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):

    model_config = ConfigDict(env_file=".env")

    database_url: str
    secret_key: str
    credit_model_path: str
    credit_threshold_path: str
    mlflow_tracking_uri: str
    fraud_model_path : str
    fraud_threshold_path : str

settings = Settings()