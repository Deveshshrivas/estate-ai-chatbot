from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model: str = "openrouter/free"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    knowledge_similarity_threshold: float = 0.38
    strict_grounding: bool = True
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "estate_ai"
    app_url: str = "http://localhost:8000"
    app_name: str = "EstateAI"
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    max_history_messages: int = 20
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_graph_version: str = "v23.0"
    wacrm_api_url: str = ""
    wacrm_api_key: str = ""
    wacrm_webhook_secret: str = ""
    cron_secret: str = ""
    followup_delay_hours: int = 5
    followup_max_attempts: int = 3
    followup_template_name: str = ""
    followup_template_language: str = "en_US"
    public_form_delay_minutes: int = 3
    public_form_template_name: str = "property_enquiry_followup"
    public_form_template_language: str = "en_US"
    property_admin_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=(Path(__file__).parent.parent / ".env", Path(__file__).with_name(".env")),
        extra="ignore",
    )

    @property
    def origins(self) -> list[str]:
        return [value.strip() for value in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
