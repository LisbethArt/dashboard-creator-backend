from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    Server-side secrets must never be exposed to the browser bundle.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"
    supabase_url: str
    supabase_service_role_key: str
    supabase_bucket: str = "uploads"
    cors_origins: str = ""


def get_settings() -> Settings:
    return Settings()
