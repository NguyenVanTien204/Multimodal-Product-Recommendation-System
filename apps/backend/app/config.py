from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://mini_market:mini_market@localhost:5432/mini_market"
    jwt_secret: str = "development-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    qdrant_url: str = "http://localhost:6333"
    datn_recommender_url: str | None = None
    cors_origins: str = "http://localhost:3000"


settings = Settings()
