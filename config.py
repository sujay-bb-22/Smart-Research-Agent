import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    environment: str = os.getenv("APP_ENV", "development")
    model: str = os.getenv("MODEL", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    default_k: int = int(os.getenv("DEFAULT_K", "3"))
    minimum_k: int = int(os.getenv("MINIMUM_K", "1"))
    maximum_k: int = int(os.getenv("MAXIMUM_K", "10"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    )
    storage_path: str = os.getenv("STORAGE_PATH", "data")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "db")
    database_path: str = os.getenv("DATABASE_PATH", os.path.join("data", "documents.json"))
    auth_required: bool = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


settings = AppConfig()
