from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func, update
from app.database.models.error_log import ErrorLog
from typing import Optional, Dict, List, Any, Tuple
import logging
import math
import datetime
import traceback

logger = logging.getLogger(__name__)

class ErrorLogRepository:
    """Repository for error logs operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_error_log(self, error_data: Dict[str, Any]) -> ErrorLog:
        """
        Create a new error log entry
        
        Args:
            error_data: Dictionary containing error information
            
        Returns:
            The created ErrorLog object
        """
        try:
            error_log = ErrorLog(**error_data)
            self.db.add(error_log)
            await self.db.commit()
            await self.db.refresh(error_log)
            logger.debug(f"Error log created: {error_log.fp}")
            return error_log
        except Exception as e:
            await self.db.rollback()
            # Don't raise here to prevent circular error logging
            logger.error(f"Failed to create error log: {str(e)}")
            return None

    async def get_error_log(self, fp: str) -> Optional[ErrorLog]:
        """
        Get error log by fingerprint
        
        Args:
            fp: Error log fingerprint
            
        Returns:
            ErrorLog if found, None otherwise
        """
        try:
            result = await self.db.execute(
                select(ErrorLog).where(ErrorLog.fp == fp)
            )
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error retrieving error log {fp}: {str(e)}")
            return None

    async def get_error_logs(self, 
                           page: int = 1, 
                           per_page: int = 20,
                           error_type_filter: Optional[str] = None,
                           component_filter: Optional[str] = None,
                           is_resolved_filter: Optional[int] = None,
                           start_date: Optional[datetime.datetime] = None,
                           end_date: Optional[datetime.datetime] = None,
                           user_fp_filter: Optional[str] = None) -> Tuple[List[ErrorLog], int, int]:
        """
        Get paginated error logs with optional filtering
        
        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            error_type_filter: Filter by error type
            component_filter: Filter by component
            is_resolved_filter: Filter by resolution status
            start_date: Filter by start date
            end_date: Filter by end date
            user_fp_filter: Filter by user fingerprint
            
        Returns:
            Tuple of (list of error logs, total count, total pages)
        """
        try:
            # Build the base query
            query = select(ErrorLog)
            count_query = select(func.count(ErrorLog.id))
            
            # Apply filters if provided
            if error_type_filter:
                query = query.where(ErrorLog.error_type == error_type_filter)
                count_query = count_query.where(ErrorLog.error_type == error_type_filter)
                
            if component_filter:
                query = query.where(ErrorLog.component == component_filter)
                count_query = count_query.where(ErrorLog.component == component_filter)
                
            if is_resolved_filter is not None:
                query = query.where(ErrorLog.is_resolved == is_resolved_filter)
                count_query = count_query.where(ErrorLog.is_resolved == is_resolved_filter)
                
            if start_date:
                query = query.where(ErrorLog.created_at >= start_date)
                count_query = count_query.where(ErrorLog.created_at >= start_date)
                
            if end_date:
                query = query.where(ErrorLog.created_at <= end_date)
                count_query = count_query.where(ErrorLog.created_at <= end_date)
                
            if user_fp_filter:
                query = query.where(ErrorLog.user_fp == user_fp_filter)
                count_query = count_query.where(ErrorLog.user_fp == user_fp_filter)
            
            # Order by creation date, newest first
            query = query.order_by(desc(ErrorLog.created_at))
            
            # Get total count
            count_result = await self.db.execute(count_query)
            total_count = count_result.scalar()
            
            # Calculate pagination
            total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
            offset = (page - 1) * per_page
            
            # Apply pagination
            query = query.offset(offset).limit(per_page)
            
            # Execute query
            result = await self.db.execute(query)
            error_logs = result.scalars().all()
            
            return error_logs, total_count, total_pages
            
        except Exception as e:
            logger.error(f"Error retrieving error logs: {str(e)}")
            return [], 0, 0

    async def update_error_log_status(self, 
                                     fp: str, 
                                     is_resolved: int,
                                     resolution_notes: Optional[str] = None) -> bool:
        """
        Update the resolution status of an error log
        
        Args:
            fp: Error log fingerprint
            is_resolved: Resolution status (0: unresolved, 1: resolved, 2: ignored)
            resolution_notes: Notes about the resolution
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                "is_resolved": is_resolved,
                "updated_at": func.now()
            }
            
            if is_resolved == 1:  # If marked as resolved
                update_data["resolved_at"] = func.now()
                
            if resolution_notes:
                update_data["resolution_notes"] = resolution_notes
                
            update_stmt = (
                update(ErrorLog)
                .where(ErrorLog.fp == fp)
                .values(**update_data)
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating error log status: {str(e)}")
            return False

    async def get_error_stats(self) -> Dict[str, Any]:
        """
        Get statistics about errors
        
        Returns:
            Dictionary with error statistics
        """
        try:
            # Total errors count
            total_query = select(func.count(ErrorLog.id))
            total_result = await self.db.execute(total_query)
            total_count = total_result.scalar() or 0
            
            # Unresolved errors count
            unresolved_query = select(func.count(ErrorLog.id)).where(ErrorLog.is_resolved == 0)
            unresolved_result = await self.db.execute(unresolved_query)
            unresolved_count = unresolved_result.scalar() or 0
            
            # Errors by type
            type_query = select(
                ErrorLog.error_type,
                func.count(ErrorLog.id).label("count")
            ).group_by(ErrorLog.error_type)
            type_result = await self.db.execute(type_query)
            type_stats = {row[0]: row[1] for row in type_result.all()}
            
            # Errors by component
            component_query = select(
                ErrorLog.component,
                func.count(ErrorLog.id).label("count")
            ).group_by(ErrorLog.component)
            component_result = await self.db.execute(component_query)
            component_stats = {row[0] or "unknown": row[1] for row in component_result.all()}
            
            # Recent errors trend (last 7 days)
            now = datetime.datetime.utcnow()
            week_ago = now - datetime.timedelta(days=7)
            trend_query = select(
                func.date_trunc('day', ErrorLog.created_at).label("day"),
                func.count(ErrorLog.id).label("count")
            ).where(ErrorLog.created_at >= week_ago).group_by("day").order_by("day")
            
            trend_result = await self.db.execute(trend_query)
            trend_data = {row[0].strftime("%Y-%m-%d"): row[1] for row in trend_result.all()}
            
            return {
                "total_count": total_count,
                "unresolved_count": unresolved_count,
                "resolved_percentage": round((total_count - unresolved_count) / total_count * 100, 2) if total_count > 0 else 0,
                "by_type": type_stats,
                "by_component": component_stats,
                "trend": trend_data
            }
            
        except Exception as e:
            logger.error(f"Error retrieving error statistics: {str(e)}")
            return {
                "total_count": 0,
                "unresolved_count": 0,
                "resolved_percentage": 0,
                "by_type": {},
                "by_component": {},
                "trend": {}
            }