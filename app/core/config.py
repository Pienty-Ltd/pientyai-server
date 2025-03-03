import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    TOKEN_ALGORITHM = "HS256"
    TOKEN_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    REDIS_URL = os.getenv("REDIS_URL")
    DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

config = Config()