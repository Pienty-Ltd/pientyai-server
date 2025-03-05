from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import func, text
from sqlalchemy import and_, or_
from app.database.models.db_models import DashboardStats, User, Organization, KnowledgeBase, File
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DashboardStatsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_stats(self, user_id: int) -> Optional[DashboardStats]:
        """Get dashboard statistics for a specific user"""
        try:
            logger.info(f"Fetching stats for user_id: {user_id}")
            result = await self.db.execute(
                select(DashboardStats).filter(DashboardStats.user_id == user_id)
            )
            stats = result.scalar_one_or_none()
            if stats:
                logger.info(f"Found stats for user_id {user_id}: kb_count={stats.total_knowledge_base_count}, file_count={stats.total_file_count}")
            else:
                logger.warning(f"No stats found for user_id {user_id}")
            return stats
        except Exception as e:
            logger.error(f"Error fetching user stats: {str(e)}", exc_info=True)
            return None

    async def get_organization_stats(self, organization_id: int) -> Optional[DashboardStats]:
        """Get dashboard statistics for a specific organization"""
        try:
            logger.info(f"Fetching stats for organization_id: {organization_id}")
            result = await self.db.execute(
                select(DashboardStats).filter(DashboardStats.organization_id == organization_id)
            )
            stats = result.scalar_one_or_none()
            if stats:
                logger.info(f"Found stats for organization_id {organization_id}: kb_count={stats.total_knowledge_base_count}, file_count={stats.total_file_count}")
            else:
                logger.warning(f"No stats found for organization_id {organization_id}")
            return stats
        except Exception as e:
            logger.error(f"Error fetching organization stats: {str(e)}", exc_info=True)
            return None

    async def update_stats(self) -> None:
        """Update all dashboard statistics"""
        try:
            # Update user statistics
            await self.db.execute(text("""
                INSERT INTO dashboard_stats (user_id, total_knowledge_base_count, total_file_count, total_storage_used, last_activity_date)
                SELECT 
                    u.id as user_id,
                    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
                    COUNT(DISTINCT f.id) as total_file_count,
                    COALESCE(SUM(f.file_size), 0) as total_storage_used,
                    MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
                FROM users u
                LEFT JOIN files f ON f.user_id = u.id
                LEFT JOIN knowledge_base kb ON kb.file_id = f.id
                GROUP BY u.id
                ON CONFLICT (user_id) WHERE user_id IS NOT NULL
                DO UPDATE SET
                    total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
                    total_file_count = EXCLUDED.total_file_count,
                    total_storage_used = EXCLUDED.total_storage_used,
                    last_activity_date = EXCLUDED.last_activity_date,
                    last_updated = CURRENT_TIMESTAMP
            """))

            # Update organization statistics
            await self.db.execute(text("""
                INSERT INTO dashboard_stats (organization_id, total_knowledge_base_count, total_file_count, total_storage_used, last_activity_date)
                SELECT 
                    o.id as organization_id,
                    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
                    COUNT(DISTINCT f.id) as total_file_count,
                    COALESCE(SUM(f.file_size), 0) as total_storage_used,
                    MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
                FROM organizations o
                LEFT JOIN files f ON f.organization_id = o.id
                LEFT JOIN knowledge_base kb ON kb.file_id = f.id
                GROUP BY o.id
                ON CONFLICT (organization_id) WHERE organization_id IS NOT NULL
                DO UPDATE SET
                    total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
                    total_file_count = EXCLUDED.total_file_count,
                    total_storage_used = EXCLUDED.total_storage_used,
                    last_activity_date = EXCLUDED.last_activity_date,
                    last_updated = CURRENT_TIMESTAMP
            """))

            await self.db.commit()
            logger.info("Successfully updated dashboard statistics")
        except Exception as e:
            logger.error(f"Error updating dashboard statistics: {str(e)}")
            await self.db.rollback()
            raise