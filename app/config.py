from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    strapi_url: str = "http://localhost:1337"
    strapi_api_token: str = ""
    forecast_api_key: str = ""
    forecast_horizon_days: int = 120
    # Import lead time for Adatex: 35d manufacturing + 30d transit + 5d nationalization = 70d
    default_lead_time_days: int = 70
    default_service_level: float = 0.95
    cache_ttl_seconds: int = 3600
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
