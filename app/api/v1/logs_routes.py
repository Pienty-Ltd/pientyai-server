from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel
import logging
import math

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user, admin_required
from app.schemas.base import BaseResponse
from app.database.models.db_models import User
from app.database.repositories.request_log_repository import RequestLogRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/logs",
    tags=["System Logs"],
    responses={404: {"description": "Not found"}}
)

class RequestLogResponse(BaseModel):
    fp: str
    request_id: str
    user_id: Optional[int] = None
    ip_address: Optional[str] = None
    method: str
    path: str
    response_status: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime

class RequestLogDetailResponse(RequestLogResponse):
    query_params: Optional[Dict] = None
    request_headers: Optional[Dict] = None
    request_body: Optional[str] = None
    response_body: Optional[str] = None

class PaginatedRequestLogResponse(BaseModel):
    logs: List[RequestLogResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int

@router.get("", response_model=BaseResponse[PaginatedRequestLogResponse],
           summary="List request logs",
           description="Get a paginated list of all request logs (Admin only)")
async def get_request_logs(
    request: Request,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page"),
    path_filter: Optional[str] = Query(None, description="Filter logs by path"),
    status_filter: Optional[int] = Query(None, description="Filter logs by status code"),
    user_id: Optional[int] = Query(None, description="Filter logs by user ID"),
    start_date: Optional[str] = Query(None, description="Filter logs from date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter logs to date (ISO format)"),
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of request logs with filtering options.
    This endpoint is only accessible by administrators.
    """
    repo = RequestLogRepository(db)
    logs, total, pages = await repo.get_all_request_logs(
        page=page, 
        per_page=per_page,
        path_filter=path_filter,
        status_filter=status_filter,
        user_id_filter=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    # Convert to response models
    log_responses = [
        RequestLogResponse(
            fp=log.fp,
            request_id=log.request_id,
            user_id=log.user_id,
            ip_address=log.ip_address,
            method=log.method,
            path=log.path,
            response_status=log.response_status,
            duration_ms=log.duration_ms,
            error=log.error,
            created_at=log.created_at
        ) for log in logs
    ]
    
    return BaseResponse.from_request(
        request=request,
        data=PaginatedRequestLogResponse(
            logs=log_responses,
            total_count=total,
            total_pages=pages,
            current_page=page,
            per_page=per_page
        ),
        message="Request logs retrieved successfully"
    )

@router.get("/{request_id}", response_model=BaseResponse[RequestLogDetailResponse],
           summary="Get request log details",
           description="Get detailed information about a specific request log (Admin only)")
async def get_request_log_detail(
    request: Request,
    request_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific request log, including
    headers, query parameters, and request/response bodies.
    This endpoint is only accessible by administrators.
    """
    repo = RequestLogRepository(db)
    log = await repo.get_request_log_by_id(request_id)
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Request log not found"}
        )
    
    return BaseResponse.from_request(
        request=request,
        data=RequestLogDetailResponse(
            fp=log.fp,
            request_id=log.request_id,
            user_id=log.user_id,
            ip_address=log.ip_address,
            method=log.method,
            path=log.path,
            query_params=log.query_params,
            request_headers=log.request_headers,
            request_body=log.request_body,
            response_status=log.response_status,
            response_body=log.response_body,
            duration_ms=log.duration_ms,
            error=log.error,
            created_at=log.created_at
        ),
        message="Request log details retrieved successfully"
    )

@router.get("/user/me", response_model=BaseResponse[PaginatedRequestLogResponse],
           summary="Get current user's request logs",
           description="Get request logs for the currently authenticated user")
async def get_my_request_logs(
    request: Request,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of request logs for the currently authenticated user.
    This allows users to see their own activity history.
    """
    repo = RequestLogRepository(db)
    logs, total, pages = await repo.get_request_logs_by_user(
        user_id=current_user.id,
        page=page,
        per_page=per_page
    )
    
    # Convert to response models
    log_responses = [
        RequestLogResponse(
            fp=log.fp,
            request_id=log.request_id,
            user_id=log.user_id,
            ip_address=log.ip_address,
            method=log.method,
            path=log.path,
            response_status=log.response_status,
            duration_ms=log.duration_ms,
            error=log.error,
            created_at=log.created_at
        ) for log in logs
    ]
    
    return BaseResponse.from_request(
        request=request,
        data=PaginatedRequestLogResponse(
            logs=log_responses,
            total_count=total,
            total_pages=pages,
            current_page=page,
            per_page=per_page
        ),
        message="Your request logs retrieved successfully"
    )