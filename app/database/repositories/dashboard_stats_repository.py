from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import func, text
from sqlalchemy import and_, or_
from app.database.models.db_models import DashboardStats, User, Organization, KnowledgeBase, File
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta
import asyncio
import traceback

logger = logging.getLogger(__name__)

class DashboardStatsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes cache timeout
        self._cache_lock = asyncio.Lock()

    async def _get_from_cache(self, cache_key: str) -> Optional[DashboardStats]:
        """Get stats from cache if available and not expired"""
        async with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self._cache_timeout):
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return data
                else:
                    logger.debug(f"Cache expired for key: {cache_key}")
                    del self._cache[cache_key]
            return None

    async def _set_cache(self, cache_key: str, data: DashboardStats) -> None:
        """Set stats in cache with current timestamp"""
        async with self._cache_lock:
            self._cache[cache_key] = (data, datetime.now())
            logger.debug(f"Cache set for key: {cache_key}")

    async def get_user_stats(self, user_id: int) -> Optional[DashboardStats]:
        """Get dashboard statistics for a specific user with caching"""
        try:
            logger.info(f"Fetching stats for user_id: {user_id}")
            cache_key = f"user_stats_{user_id}"

            # Try cache first
            cached_stats = await self._get_from_cache(cache_key)
            if cached_stats:
                logger.info(f"Returning cached stats for user_id: {user_id}")
                return cached_stats

            logger.debug("Cache miss, querying database...")

            # Try to get stats from materialized view
            try:
                logger.debug("Querying materialized view...")
                async with self.db.begin() as transaction:
                    result = await transaction.execute(text("""
                        SELECT * FROM mv_user_stats WHERE user_id = :user_id
                    """), {"user_id": user_id})
                    row = result.fetchone()
                    logger.debug(f"Materialized view query result: {row}")
            except Exception as e:
                logger.error(f"Error querying materialized view: {str(e)}\n{traceback.format_exc()}")
                return None

            if not row:
                logger.info(f"No stats found for user_id {user_id}")
                return None

            # Convert materialized view row to DashboardStats
            stats = DashboardStats(
                user_id=row.user_id,
                total_knowledge_base_count=row.total_knowledge_base_count,
                total_file_count=row.total_file_count,
                total_storage_used=row.total_storage_used,
                last_activity_date=row.last_activity_date
            )

            if stats:
                logger.info(f"Found stats for user_id {user_id}: kb_count={stats.total_knowledge_base_count}, file_count={stats.total_file_count}")
                await self._set_cache(cache_key, stats)
            else:
                logger.warning(f"No stats found for user_id {user_id}")

            return stats

        except Exception as e:
            logger.error(f"Error in get_user_stats: {str(e)}\n{traceback.format_exc()}")
            return None

    async def get_organization_stats(self, organization_id: int) -> Optional[DashboardStats]:
        """Get dashboard statistics for a specific organization with caching"""
        try:
            logger.info(f"Fetching stats for organization_id: {organization_id}")
            cache_key = f"org_stats_{organization_id}"

            # Try cache first
            cached_stats = await self._get_from_cache(cache_key)
            if cached_stats:
                logger.info(f"Returning cached stats for organization_id: {organization_id}")
                return cached_stats

            logger.debug("Cache miss, querying database...")

            # Try to get stats from materialized view first
            try:
                logger.debug("Querying materialized view...")
                async with self.db.begin() as transaction:
                    result = await transaction.execute(text("""
                        SELECT * FROM mv_organization_stats WHERE organization_id = :organization_id
                    """), {"organization_id": organization_id})
                    row = result.fetchone()
                    logger.debug(f"Materialized view query result: {row}")
            except Exception as e:
                logger.error(f"Error querying materialized view: {str(e)}\n{traceback.format_exc()}")
                row = None

            if not row:
                logger.info("No data in materialized view, updating stats...")
                try:
                    async with self.db.begin() as transaction:
                        # Update stats for this organization
                        await transaction.execute(text("""
                            SELECT update_organization_stats(:organization_id)
                        """), {"organization_id": organization_id})
                        # Get updated stats
                        result = await transaction.execute(
                            select(DashboardStats).filter(DashboardStats.organization_id == organization_id)
                        )
                        stats = result.scalar_one_or_none()
                        logger.debug(f"Direct query result: {stats}")
                except Exception as e:
                    logger.error(f"Error updating organization stats: {str(e)}\n{traceback.format_exc()}")
                    return None
            else:
                logger.debug("Converting materialized view row to DashboardStats")
                # Convert materialized view row to DashboardStats
                stats = DashboardStats(
                    organization_id=row.organization_id,
                    total_knowledge_base_count=row.total_knowledge_base_count,
                    total_file_count=row.total_file_count,
                    total_storage_used=row.total_storage_used,
                    last_activity_date=row.last_activity_date
                )

            if stats:
                logger.info(f"Found stats for organization_id {organization_id}: kb_count={stats.total_knowledge_base_count}, file_count={stats.total_file_count}")
                await self._set_cache(cache_key, stats)
            else:
                logger.warning(f"No stats found for organization_id {organization_id}")

            return stats

        except Exception as e:
            logger.error(f"Error in get_organization_stats: {str(e)}\n{traceback.format_exc()}")
            return None

    async def update_stats(self, batch_size: int = 1000) -> None:
        """Update all dashboard statistics using materialized views and batching"""
        try:
            logger.info("Starting dashboard stats update...")

            # First refresh materialized views
            try:
                async with self.db.begin() as transaction:
                    await transaction.execute(text("""
                        SELECT refresh_dashboard_stats_views()
                    """))
                logger.info("Successfully refreshed materialized views")
            except Exception as e:
                logger.error(f"Error refreshing materialized views: {str(e)}\n{traceback.format_exc()}")
                raise

            # Then update dashboard_stats table using the updated views
            try:
                async with self.db.begin() as transaction:
                    await transaction.execute(text("""
                        SELECT update_dashboard_stats(:batch_size)
                    """), {"batch_size": batch_size})
                logger.info(f"Successfully updated dashboard statistics with batch size {batch_size}")
            except Exception as e:
                logger.error(f"Error updating dashboard stats: {str(e)}\n{traceback.format_exc()}")
                raise

            # Clear cache after update
            async with self._cache_lock:
                self._cache.clear()
                logger.info("Cache cleared after statistics update")

        except Exception as e:
            logger.error(f"Error in update_stats: {str(e)}\n{traceback.format_exc()}")
            raise

    async def invalidate_cache(self, user_id: Optional[int] = None, organization_id: Optional[int] = None) -> None:
        """Invalidate cache for specific user or organization, or all if none specified"""
        async with self._cache_lock:
            if user_id:
                cache_key = f"user_stats_{user_id}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Cache invalidated for user_id {user_id}")
            elif organization_id:
                cache_key = f"org_stats_{organization_id}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Cache invalidated for organization_id {organization_id}")
            else:
                self._cache.clear()
                logger.debug("All cache invalidated")