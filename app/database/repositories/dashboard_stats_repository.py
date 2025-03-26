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

            # Try to get stats from materialized view first
            row = None
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
                # We'll try direct calculation instead of returning None
            
            # If materialized view doesn't exist or has no data, calculate stats directly
            if not row:
                logger.info(f"No stats in materialized view for user_id {user_id}, calculating directly...")
                try:
                    # Direct calculation of statistics using existing connection without begin()
                    # Calculate total knowledge base count for user's organizations
                    kb_result = await self.db.execute(text("""
                        SELECT COUNT(*) as kb_count FROM knowledge_base kb
                        JOIN organizations o ON kb.organization_id = o.id
                        JOIN user_organizations uo ON o.id = uo.organization_id
                        WHERE uo.user_id = :user_id
                    """), {"user_id": user_id})
                    kb_count = kb_result.scalar() or 0
                    
                    # Calculate total file count for user's organizations
                    file_result = await self.db.execute(text("""
                        SELECT COUNT(*) as file_count, COALESCE(SUM(file_size), 0) as total_size 
                        FROM files f
                        JOIN organizations o ON f.organization_id = o.id
                        JOIN user_organizations uo ON o.id = uo.organization_id
                        WHERE uo.user_id = :user_id
                    """), {"user_id": user_id})
                    file_row = file_result.fetchone()
                    file_count = file_row.file_count if file_row else 0
                    total_size = file_row.total_size if file_row else 0
                    
                    # Get last activity date
                    activity_result = await self.db.execute(text("""
                        SELECT MAX(f.created_at) as last_activity
                        FROM files f
                        JOIN organizations o ON f.organization_id = o.id
                        JOIN user_organizations uo ON o.id = uo.organization_id
                        WHERE uo.user_id = :user_id
                    """), {"user_id": user_id})
                    last_activity = activity_result.scalar()
                    
                    # Create stats object
                    stats = DashboardStats(
                        user_id=user_id,
                        total_knowledge_base_count=kb_count,
                        total_file_count=file_count,
                        total_storage_used=total_size,
                        last_activity_date=last_activity
                    )
                    
                    logger.info(f"Calculated stats directly: kb_count={kb_count}, file_count={file_count}")
                except Exception as e:
                    logger.error(f"Error calculating stats directly: {str(e)}\n{traceback.format_exc()}")
                    # Create default stats object with zeros
                    stats = DashboardStats(
                        user_id=user_id,
                        total_knowledge_base_count=0,
                        total_file_count=0,
                        total_storage_used=0,
                        last_activity_date=None
                    )
            else:
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
            # Return empty stats object instead of None
            return DashboardStats(
                user_id=user_id,
                total_knowledge_base_count=0,
                total_file_count=0,
                total_storage_used=0,
                last_activity_date=None
            )

    async def get_organization_stats(self, organization_id: int) -> DashboardStats:
        """Get dashboard statistics for a specific organization with caching"""
        try:
            logger.info(f"Fetching stats for organization_id: {organization_id}")
            cache_key = f"org_stats_{organization_id}"

            # Try cache first
            cached_stats = await self._get_from_cache(cache_key)
            if cached_stats:
                logger.info(f"Returning cached stats for organization_id: {organization_id}")
                return cached_stats

            # Create default stats in case we can't get any data
            default_stats = DashboardStats(
                organization_id=organization_id,
                total_knowledge_base_count=0,
                total_file_count=0,
                total_storage_used=0,
                last_activity_date=None
            )

            # Try direct calculation - simpler approach to avoid nested transactions
            try:
                # Calculate total knowledge base count
                kb_result = await self.db.execute(text("""
                    SELECT COUNT(*) as kb_count FROM knowledge_base 
                    WHERE organization_id = :organization_id
                """), {"organization_id": organization_id})
                kb_count = kb_result.scalar() or 0
                
                # Calculate total file count and storage
                file_result = await self.db.execute(text("""
                    SELECT COUNT(*) as file_count, COALESCE(SUM(file_size), 0) as total_size 
                    FROM files 
                    WHERE organization_id = :organization_id
                """), {"organization_id": organization_id})
                file_row = file_result.fetchone()
                file_count = file_row.file_count if file_row else 0
                total_size = file_row.total_size if file_row else 0
                
                # Get last activity date
                activity_result = await self.db.execute(text("""
                    SELECT MAX(created_at) as last_activity
                    FROM files 
                    WHERE organization_id = :organization_id
                """), {"organization_id": organization_id})
                last_activity = activity_result.scalar()
                
                # Create stats object
                stats = DashboardStats(
                    organization_id=organization_id,
                    total_knowledge_base_count=kb_count,
                    total_file_count=file_count,
                    total_storage_used=total_size,
                    last_activity_date=last_activity
                )
                
                logger.info(f"Calculated stats directly: kb_count={kb_count}, file_count={file_count}")
                
                # Cache the result
                await self._set_cache(cache_key, stats)
                return stats
                
            except Exception as e:
                logger.error(f"Error calculating organization stats: {str(e)}\n{traceback.format_exc()}")
                return default_stats
                
        except Exception as e:
            logger.error(f"Error in get_organization_stats: {str(e)}\n{traceback.format_exc()}")
            # Return default stats in case of any error
            return DashboardStats(
                organization_id=organization_id,
                total_knowledge_base_count=0,
                total_file_count=0,
                total_storage_used=0,
                last_activity_date=None
            )

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