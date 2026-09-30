from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    recommender_url: str = "http://recommender:8100"

    # OpenAI-compatible chat endpoint. Gemini:
    #   RAG_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    #   RAG_LLM_MODEL=gemini-3.5-flash-lite   RAG_LLM_API_KEY=<key>   RAG_LLM_REASONING_EFFORT=low
    # (Ollama: http://host:11434/v1, vLLM, OpenAI also work.) Leave llm_base_url empty for template mode.
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout_s: float = 30.0
    llm_reasoning_effort: str = ""  # Gemini: "low" keeps thinking tokens (and latency) small
    llm_max_tokens: int = 1024
    llm_max_retries: int = 2

    device: str = "auto"  # "cpu" | "cuda" | "auto" for the Jina CLIP v2 query encoder
    load_encoder: bool = True
    session_ttl_s: float = 3600.0
    max_image_mb: float = 8.0
    default_k: int = 5
    max_k: int = 12


settings = Settings()
