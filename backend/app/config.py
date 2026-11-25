import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_keys: str = ""
    llm_model: str = "gemini/gemini-2.0-flash"
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: str = "http://localhost:3000"
    enable_failure_simulation: bool = False
    default_system_prompt: str = "You are a helpful AI assistant."

    class Config:
        env_file = ("../.env", ".env")
        extra = "ignore"

    @property
    def api_keys_list(self) -> List[str]:
        if not self.llm_api_keys:
            return []
        return [k.strip() for k in self.llm_api_keys.split(",") if k.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.allowed_origins:
            return ["*"]
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        for default_origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            if default_origin not in origins:
                origins.append(default_origin)
        return origins

    @property
    def sqlite_path(self) -> str:
        url = self.database_url
        if url.startswith("sqlite:////"):
            return "/" + url[len("sqlite:////"):]
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
