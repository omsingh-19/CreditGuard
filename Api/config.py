from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):

    model_config = ConfigDict(env_file=".env")

    database_url: str
    secret_key: str
    model_path: str
    threshold_path: str
    mlflow_tracking_uri: str

settings = Settings()