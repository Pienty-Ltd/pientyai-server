from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func
from app.database.models.request_log import RequestLog
from app.database.models.db_models import User, UserRole
from typing import Optional, Dict, List, Any
import logging
import math
import json

logger = logging.getLogger(__name__)

class RequestLogRepository:
    """Repository for request log operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_request_log(self, log_data: Dict[str, Any]) -> RequestLog:
        """Create a new request log entry"""
        try:
            # Ensure JSON data is properly formatted
            if 'query_params' in log_data and log_data['query_params'] is not None:
                log_data['query_params'] = self._ensure_json_serializable(log_data['query_params'])
                
            if 'request_headers' in log_data and log_data['request_headers'] is not None:
                log_data['request_headers'] = self._ensure_json_serializable(log_data['request_headers'])
            
            # Create new request log
            request_log = RequestLog(**log_data)
            self.db.add(request_log)
            await self.db.commit()
            await self.db.refresh(request_log)
            return request_log
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating request log: {str(e)}")
            raise
    
    async def get_request_log_by_id(self, request_id: str) -> Optional[RequestLog]:
        """Get request log by request ID"""
        result = await self.db.execute(
            select(RequestLog).where(RequestLog.request_id == request_id)
        )
        return result.scalars().first()
    
    async def get_request_logs_by_user(
        self, 
        user_fp: str, 
        page: int = 1, 
        per_page: int = 20
    ) -> tuple[List[RequestLog], int, int]:
        """Get request logs for a specific user with pagination"""
        # Query for logs
        stmt = select(RequestLog).where(RequestLog.user_fp == user_fp) \
                                .order_by(desc(RequestLog.created_at)) \
                                .offset((page - 1) * per_page) \
                                .limit(per_page)
        result = await self.db.execute(stmt)
        logs = result.scalars().all()
        
        # Count total logs
        count_stmt = select(func.count()).select_from(RequestLog) \
                                        .where(RequestLog.user_fp == user_fp)
        count_result = await self.db.execute(count_stmt)
        total_count = count_result.scalar()
        
        # Calculate total pages
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        return logs, total_count, total_pages
    
    async def get_all_request_logs(
        self, 
        page: int = 1, 
        per_page: int = 20,
        path_filter: Optional[str] = None,
        status_filter: Optional[int] = None,
        user_fp_filter: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> tuple[List[RequestLog], int, int]:
        """
        Get all request logs with filtering and pagination
        This is primarily for admin access
        """
        # Base query
        query = select(RequestLog)
        
        # Apply filters
        if path_filter:
            query = query.where(RequestLog.path.contains(path_filter))
        
        if status_filter:
            query = query.where(RequestLog.response_status == status_filter)
            
        if user_fp_filter:
            query = query.where(RequestLog.user_fp == user_fp_filter)
            
        if start_date:
            query = query.where(RequestLog.created_at >= start_date)
            
        if end_date:
            query = query.where(RequestLog.created_at <= end_date)
            
        # Apply ordering
        query = query.order_by(desc(RequestLog.created_at))
        
        # Count total matching logs
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await self.db.execute(count_query)
        total_count = count_result.scalar()
        
        # Apply pagination
        query = query.offset((page - 1) * per_page).limit(per_page)
        
        # Execute query
        result = await self.db.execute(query)
        logs = result.scalars().all()
        
        # Calculate total pages
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        return logs, total_count, total_pages
    
    def _ensure_json_serializable(self, data: Any) -> Dict:
        """
        Ensure data is JSON serializable
        Returns a dictionary that can be safely stored in a JSON column
        """
        try:
            # Test serialization
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # If not serializable, convert to string representation
            return {"data_string": str(data)}