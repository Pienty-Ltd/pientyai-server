from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import logging
import os
from pathlib import Path
from sqlalchemy import text
from app.database.utils import execute_sql_commands

logger = logging.getLogger(__name__)

DATABASE_URL = config.DATABASE_URL

# Create async engine with proper configuration
engine = create_async_engine(DATABASE_URL,
                             echo=True,
                             pool_pre_ping=True,
                             pool_size=20,
                             max_overflow=10)

# Configure async session maker
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Important for async operations
    autocommit=False,
    autoflush=False)

Base = declarative_base()


async def execute_sql_file(session: AsyncSession, file_path: str) -> None:
    """Execute a SQL file by splitting it into individual commands"""
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()

        # Execute commands using the utility function
        await execute_sql_commands(session, sql_content)
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

        # Execute init.sql first if it exists
        init_sql = sql_dir / 'init.sql'
        if init_sql.exists():
            logger.info("Executing database initialization SQL")
            async with async_session_maker() as session:
                await execute_sql_file(session, str(init_sql))
                logger.info("Successfully initialized database procedures")
        else:
            logger.warning("init.sql not found in the SQL directory")

    except Exception as e:
        logger.error(f"Error initializing database procedures: {str(e)}")
        raise


async def create_tables():
    """Create all database tables"""
    try:
        async with engine.begin() as conn:
            # Import all models here to ensure they're registered with SQLAlchemy
            from app.database.models import (User, Organization,
                                             UserSubscription, PaymentHistory,
                                             PromoCode, PromoCodeUsage,
                                             UserRole, PaymentStatus,
                                             SubscriptionStatus, File,
                                             KnowledgeBase, FileStatus)
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
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
