import os
from dotenv import load_dotenv
import logging

# Load .env file
load_dotenv(override=True)


class Config:
    # Environment
    API_PRODUCTION = os.getenv("API_PRODUCTION", "false").lower() == "true"

    # Configure logging based on environment
    LOG_LEVEL = logging.WARNING if API_PRODUCTION else logging.DEBUG

    # Authentication
    SECRET_KEY = os.getenv("SECRET_KEY")
    TOKEN_ALGORITHM = "HS256"
    TOKEN_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # Database and Redis
    REDIS_URL = os.getenv("REDIS_URL")
    DATABASE_URL = os.getenv("DATABASE_URL",
                             "").replace("postgresql://",
                                         "postgresql+asyncpg://")

    # Stripe Configuration
    STRIPE_TEST_PUBLIC_KEY = os.getenv("STRIPE_TEST_PUBLIC_KEY")
    STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
    STRIPE_LIVE_PUBLIC_KEY = os.getenv("STRIPE_LIVE_PUBLIC_KEY")
    STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET = os.getenv(
        "STRIPE_WEBHOOK_SECRET")  # Added webhook secret
    STRIPE_PRICE_ID = os.getenv(
        "STRIPE_PRICE_ID")  # Added subscription price ID


config = Config()

# Configure logging globally
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
