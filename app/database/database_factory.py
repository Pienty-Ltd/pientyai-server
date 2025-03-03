from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import logging

logger = logging.getLogger(__name__)

# Log the database URL for debugging (excluding sensitive info)
parsed_url = config.DATABASE_URL.split('@')[-1] if config.DATABASE_URL else "None"
logger.info(f"Connecting to database at: {parsed_url}")

DATABASE_URL = config.DATABASE_URL

engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()