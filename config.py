import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    environment: str = os.getenv("APP_ENV", "development")
    model: str = os.getenv("MODEL", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    google_embedding_model: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    embedding_dimension: int | None = (
        int(os.getenv("EMBEDDING_DIMENSION"))
        if os.getenv("EMBEDDING_DIMENSION")
        else None
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    default_k: int = int(os.getenv("DEFAULT_K", "3"))
    minimum_k: int = int(os.getenv("MINIMUM_K", "1"))
    maximum_k: int = int(os.getenv("MAXIMUM_K", "10"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    lexical_weight: float = float(os.getenv("LEXICAL_WEIGHT", "0.3"))
    rerank_candidate_multiplier: int = int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "3"))
    summary_group_size: int = int(os.getenv("SUMMARY_GROUP_SIZE", "12"))
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
    api_token: str = os.getenv("API_TOKEN") or os.getenv("AUTH_TOKEN") or "dev-token"
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "default-user")
    default_workspace_id: str = os.getenv("DEFAULT_WORKSPACE_ID", "default-workspace")
    object_storage_bucket: str | None = os.getenv("OBJECT_STORAGE_BUCKET")
    object_storage_endpoint: str | None = os.getenv("OBJECT_STORAGE_ENDPOINT")
    postgres_dsn: str | None = os.getenv("POSTGRES_DSN")
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    pgvector_dsn: str | None = os.getenv("PGVECTOR_DSN") or os.getenv("POSTGRES_DSN")


settings = AppConfig()
