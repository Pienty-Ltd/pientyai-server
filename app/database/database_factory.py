from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = config.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True)
async_session_maker = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def create_tables():
    """Create all database tables"""
    try:
        async with engine.begin() as conn:
            # Import all models here to ensure they're registered with SQLAlchemy
            from app.database.models import (
                User, Organization, UserSubscription,
                PaymentHistory, PromoCode, PromoCodeUsage,
                UserRole, PaymentStatus, SubscriptionStatus,
                File, KnowledgeBase, FileStatus
            )
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Successfully created database tables")
            # Log created tables for verification
            tables = Base.metadata.tables.keys()
            logger.info(f"Created tables: {', '.join(tables)}")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

async def get_db():
    """Get database session"""
    db = async_session_maker()
    try:
        yield db
    finally:
        await db.close()