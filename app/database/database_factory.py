from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import logging
import os
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)

DATABASE_URL = config.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True)
async_session_maker = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def execute_sql_file(session: AsyncSession, file_path: str) -> None:
    """Execute a SQL file"""
    try:
        with open(file_path, 'r') as f:
            sql = f.read()
            # Execute the entire SQL file as a single transaction
            async with session.begin():
                await session.execute(text(sql))
                await session.commit()
            logger.info(f"Successfully executed SQL file: {file_path}")
    except Exception as e:
        logger.error(f"Error executing SQL file {file_path}: {str(e)}")
        raise

async def init_database_procedures() -> None:
    """Initialize database procedures from SQL files"""
    try:
        # Get the SQL directory path
        sql_dir = Path(__file__).parent / 'sql'
        if not sql_dir.exists():
            logger.warning(f"SQL directory not found: {sql_dir}")
            return

        # Get all .sql files and sort them
        sql_files = sorted(sql_dir.glob('*.sql'))
        if not sql_files:
            logger.warning("No SQL files found in the SQL directory")
            return

        async with async_session_maker() as session:
            for sql_file in sql_files:
                logger.info(f"Executing SQL file: {sql_file}")
                await execute_sql_file(session, str(sql_file))

        logger.info("Successfully initialized all database procedures")
    except Exception as e:
        logger.error(f"Error initializing database procedures: {str(e)}")
        raise

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

            # Initialize database procedures after creating tables
            await init_database_procedures()
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