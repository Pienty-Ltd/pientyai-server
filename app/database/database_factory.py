from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = config.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def create_tables():
    """Create all database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Successfully created database tables")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

async def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()