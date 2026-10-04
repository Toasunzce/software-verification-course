from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/docmind.db"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"

    categories: list[str] = [
        "technology", "finance", "medicine", "law",
        "science", "education", "sports", "business",
    ]