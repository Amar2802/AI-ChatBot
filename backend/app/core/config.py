import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "Data"
    DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "chatbot.db"
    FAQ_PATH: Path = Path(__file__).resolve().parent.parent.parent / "Data" / "faqs.json"
    PUBLIC_DIR: Path = Path(__file__).resolve().parent.parent.parent / "Public"

    # Models configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_GEN_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # API Keys & Flags
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    USE_LOCAL_LLM: Optional[bool] = None

    # Algorithm settings
    TOP_K: int = 3
    SIM_THRESHOLD: float = 0.65
    FAQ_CONTEXT_MIN: float = 0.40
    MAX_TURNS_MEMORY: int = 6
    MAX_NEW_TOKENS: int = 256

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:
        # Resolve USE_LOCAL_LLM logic dynamically if not set explicitly
        if self.USE_LOCAL_LLM is None:
            self.USE_LOCAL_LLM = (self.OPENAI_API_KEY is None) and (self.GEMINI_API_KEY is None)

settings = Settings()
