import os
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse

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

    # Parse and reconstruct DATABASE_URL to ensure proper format
    DATABASE_URL = os.getenv("DATABASE_URL")

    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Stripe Configuration
    STRIPE_TEST_PUBLIC_KEY = os.getenv("STRIPE_TEST_PUBLIC_KEY")
    STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
    STRIPE_LIVE_PUBLIC_KEY = os.getenv("STRIPE_LIVE_PUBLIC_KEY")
    STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")

    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_LIVE_WEBHOOK_SECRET")
    STRIPE_PRICE_ID = os.getenv("STRIPE_LIVE_PRICE_ID")

    STRIPE_TEST_WEBHOOK_SECRET = os.getenv("STRIPE_TEST_WEBHOOK_SECRET")
    STRIPE_TEST_PRICE_ID = os.getenv("STRIPE_TEST_PRICE_ID")


config = Config()

# Configure logging globally
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
