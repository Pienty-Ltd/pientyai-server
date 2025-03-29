from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks, Query, Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from sqlalchemy import select, join
from sqlalchemy.orm import joinedload

from app.api.v1.auth import get_current_user
from app.api.v1.middlewares.validation_middleware import validate_organization_id, validate_document_id
from app.core.services.document_analysis_service import DocumentAnalysisService
from app.core.services.document_service import DocumentService
from app.schemas.base import BaseResponse
from app.schemas.document_analysis import (
    DocumentAnalysisRequest, DocumentAnalysisResponse, AnalysisListItem,
    PaginatedAnalysisResponse, AnalysisDetailResponse, AnalysisStatusEnum
)
from app.database.models.db_models import User, DocumentAnalysis, File, AnalysisStatus
from app.database.database_factory import async_session_maker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/document-analysis", tags=["Document Analysis"])

@router.post("", response_model=BaseResponse[DocumentAnalysisResponse])
async def analyze_document(
    request: Request,
    analysis_request: DocumentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Queue a document for analysis against the organization's knowledge base.
    
    This endpoint:
    1. Validates the request and document access
    2. Creates an analysis record in the database 
    3. Queues the analysis process to run in the background
    4. Returns immediately with the analysis ID and status
    
    The analysis will continue processing in the background, and results
    can be retrieved later using the analysis listing and detail endpoints.
    """
    try:
        logger.info(f"Document analysis requested for doc {analysis_request.document_id} in org {analysis_request.organization_id}")
        
        # Validate organization and document IDs
        await validate_organization_id(analysis_request.organization_id)
        await validate_document_id(analysis_request.document_id)
        
        # Check if user has access to organization
        if not any(org.id == analysis_request.organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to analyze document {analysis_request.document_id} in org {analysis_request.organization_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )
        
        # Initialize service
        document_analysis_service = DocumentAnalysisService()
        
        # Create a record to track the analysis
        analysis_record = await document_analysis_service.create_analysis_record(
            document_id=analysis_request.document_id,
            organization_id=analysis_request.organization_id,
            user_id=current_user.id
        )
        
        # Get document details for the response
        document_service = DocumentService()
        document = await document_service.get_document_by_id(
            organization_id=analysis_request.organization_id, 
            document_id=analysis_request.document_id
        )
        
        # Add the analysis task to background_tasks
        async def run_analysis_task():
            try:
                logger.info(f"Background analysis task started for document {analysis_request.document_id}")
                await document_analysis_service.analyze_document_with_knowledge_base(
                    organization_id=analysis_request.organization_id,
                    document_id=analysis_request.document_id,
                    user_id=current_user.id,
                    max_relevant_chunks=analysis_request.max_relevant_chunks
                )
                logger.info(f"Background analysis task completed for document {analysis_request.document_id}")
            except Exception as e:
                logger.error(f"Error in background analysis task: {str(e)}", exc_info=True)
                # Update the analysis status to failed
                await document_analysis_service.update_analysis_status(
                    analysis_record.id, 
                    AnalysisStatus.FAILED
                )
        
        # Add the task to run in the background
        background_tasks.add_task(run_analysis_task)
        
        # Return the analysis record information immediately
        return BaseResponse.from_request(
            request=request,
            data=DocumentAnalysisResponse(
                id=analysis_record.id,
                fp=analysis_record.fp,
                document_id=analysis_record.document_id,
                organization_id=analysis_record.organization_id,
                analysis="Analysis in progress...",
                key_points=[],
                conflicts=[],
                recommendations=[],
                total_chunks_analyzed=0,
                processing_time_seconds=0.0,
                chunk_analyses=[],
                status=analysis_record.status.value,
                created_at=analysis_record.created_at,
                completed_at=None
            ),
            success=True,
            message=f"Document analysis queued successfully. Analysis ID: {analysis_record.id}. Check status using the detail endpoint."
        )
    
    except ValueError as e:
        logger.error(f"Value error in document analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error during document analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during document analysis"
        )

@router.get("", response_model=BaseResponse[PaginatedAnalysisResponse])
async def list_analyses(
    request: Request,
    organization_id: int = Query(..., description="Organization ID to filter by"),
    page: int = Query(1, description="Page number, starting at 1"),
    per_page: int = Query(10, description="Number of items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user)
):
    """
    List document analyses with pagination.
    
    This endpoint retrieves a paginated list of document analyses for a specific organization.
    Results can be filtered by status.
    """
    try:
        # Validate organization ID and check access
        await validate_organization_id(organization_id)
        
        if not any(org.id == organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to list analyses for org {organization_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )
            
        # Initialize service
        document_analysis_service = DocumentAnalysisService()
        
        # Get analyses with pagination
        analyses, total_count, total_pages = await document_analysis_service.get_analyses_for_organization(
            organization_id=organization_id,
            page=page,
            per_page=per_page
        )
        
        # Collect document IDs to fetch document details
        document_ids = [analysis.document_id for analysis in analyses]
        
        # Fetch document details for all analyses at once
        document_service = DocumentService()
        documents = {}
        if document_ids:
            async with async_session_maker() as session:
                query = select(File).where(File.id.in_(document_ids))
                result = await session.execute(query)
                file_records = result.scalars().all()
                
                for file in file_records:
                    documents[file.id] = {
                        "filename": file.filename,
                        "file_type": file.file_type
                    }
        
        # Create response data
        analyses_data = []
        for analysis in analyses:
            doc_info = documents.get(analysis.document_id, {})
            analyses_data.append(AnalysisListItem(
                id=analysis.id,
                fp=analysis.fp,
                document_id=analysis.document_id,
                organization_id=analysis.organization_id,
                status=analysis.status.value,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
                document_filename=doc_info.get("filename"),
                document_type=doc_info.get("file_type")
            ))
        
        return BaseResponse.from_request(
            request=request,
            data=PaginatedAnalysisResponse(
                analyses=analyses_data,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            ),
            success=True,
            message=f"Retrieved {len(analyses_data)} analyses (page {page} of {total_pages})"
        )
        
    except ValueError as e:
        logger.error(f"Value error in listing analyses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while listing analyses"
        )

@router.get("/{analysis_id}", response_model=BaseResponse[AnalysisDetailResponse])
async def get_analysis_detail(
    request: Request,
    analysis_id: int = Path(..., description="Analysis ID to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a document analysis.
    
    This endpoint retrieves detailed information about a specific document analysis, including:
    - Analysis summary
    - Key points, conflicts, and recommendations
    - Original document content
    - Suggested changes with highlighted differences
    """
    try:
        # Initialize service
        document_analysis_service = DocumentAnalysisService()
        
        # Get the analysis
        analysis = await document_analysis_service.get_analysis_by_id(analysis_id)
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
            
        # Check user access to the organization
        if not any(org.id == analysis.organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to access analysis {analysis_id} in org {analysis.organization_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )
        
        # Get document details
        document_service = DocumentService()
        document = await document_service.get_document_by_id(analysis.organization_id, analysis.document_id)
        
        # Prepare the response
        return BaseResponse.from_request(
            request=request,
            data=AnalysisDetailResponse(
                id=analysis.id,
                fp=analysis.fp,
                document_id=analysis.document_id,
                organization_id=analysis.organization_id,
                analysis=analysis.analysis,
                key_points=analysis.key_points or [],
                conflicts=analysis.conflicts or [],
                recommendations=analysis.recommendations or [],
                total_chunks_analyzed=analysis.total_chunks_analyzed or 0,
                processing_time_seconds=float(analysis.processing_time_seconds) if analysis.processing_time_seconds else 0.0,
                chunk_analyses=analysis.chunk_analyses or [],
                status=analysis.status.value,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
                original_content=analysis.original_content,
                suggested_changes=analysis.suggested_changes,
                document_filename=document.filename if document else None,
                document_type=document.file_type if document else None
            ),
            success=True,
            message=f"Retrieved details for analysis {analysis_id}"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving analysis details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving analysis details"
        )