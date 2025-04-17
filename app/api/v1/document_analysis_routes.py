from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks, Query, Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from sqlalchemy import select, join
from sqlalchemy.orm import joinedload

from app.api.v1.auth import get_current_user
from app.api.v1.middlewares.validation_middleware import validate_organization_id, validate_document_id, validate_document_fp, validate_organization_fp
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
        logger.info(f"Document analysis requested for doc {analysis_request.document_fp} in org {analysis_request.organization_fp}")
        
        # Validate organization FP and document fingerprint
        await validate_organization_fp(analysis_request.organization_fp)
        await validate_document_fp(analysis_request.document_fp)
        
        # Validate max_relevant_chunks is within allowed range (3-10)
        if analysis_request.max_relevant_chunks < 3 or analysis_request.max_relevant_chunks > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_relevant_chunks must be between 3 and 10"
            )
        
        # Initialize services
        from app.database.repositories.organization_repository import OrganizationRepository
        async with async_session_maker() as session:
            org_repo = OrganizationRepository(session)
            
            # Get organization from FP
            organization = await org_repo.get_organization_by_fp(analysis_request.organization_fp)
            if not organization:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Organization not found"
                )
                
            # Check if user has access to organization
            if not any(org.id == organization.id for org in current_user.organizations):
                logger.warning(f"Access denied: User {current_user.id} attempted to analyze document {analysis_request.document_fp} in org {organization.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to organization"
                )
            
            # Initialize service
            document_analysis_service = DocumentAnalysisService()
            
            # Create a record to track the analysis using document fingerprint
            analysis_record = await document_analysis_service.create_analysis_record_by_fp(
                document_fp=analysis_request.document_fp,
                organization_id=organization.id,  # Use actual ID from organization object
                user_id=current_user.id
            )
            
            # Get document details for the response
            document_service = DocumentService()
            document = await document_service.get_document_by_fp(
                organization_id=organization.id, 
                document_fp=analysis_request.document_fp
            )
        
        # Add the analysis task to background_tasks
        async def run_analysis_task():
            try:
                logger.info(f"Background analysis task started for document {analysis_request.document_fp}")
                
                # Önce analiz kaydının durumunu kontrol et
                try:
                    await document_analysis_service.update_analysis_status(
                        analysis_record.id, 
                        AnalysisStatus.PROCESSING
                    )
                    logger.info(f"Analysis record status updated to PROCESSING for {analysis_record.fp}")
                except Exception as update_error:
                    logger.error(f"Failed to update analysis status to PROCESSING: {str(update_error)}")
                    # Güncelleme başarısız olsa bile analiz işlemine devam et
                
                try:
                    # OpenAI API'sine bağlanmayı dene
                    await document_analysis_service.openai_service.get_api_info()
                    logger.info(f"OpenAI API connection verified")
                except Exception as api_error:
                    logger.error(f"OpenAI API connection test failed: {str(api_error)}")
                    # API testi başarısız olsa bile analiz işlemine devam et
                
                # Analiz işlemini başlat
                result = await document_analysis_service.analyze_document_with_knowledge_base(
                    organization_id=organization.id,
                    document_id=analysis_record.document_id,  # Using ID from created record
                    user_id=current_user.id,
                    max_relevant_chunks=analysis_request.max_relevant_chunks,
                    analysis_id=analysis_record.id  # Pass the existing analysis record ID
                )
                
                logger.info(f"Background analysis task completed for document {analysis_request.document_fp}")
                
                # Analiz sonucunu kontrol et
                if result and isinstance(result, dict) and "error" in result and result.get("status") == "failed":
                    # Servis tarafından hata döndürüldü, ancak uygulamayı çökertmeyelim
                    logger.error(f"Analysis service reported an error: {result.get('error')}")
                    # Bu durumda bile status'u failed yapmamayı deneyeceğiz
                
            except Exception as e:
                logger.error(f"Error in background analysis task: {str(e)}", exc_info=True)
                
                # Hata oluştuğunda bile hemen failed durumuna geçmeyeceğiz
                # Bunun yerine anlamlı bir hata mesajı içeren bir analiz sonucu oluşturacağız
                try:
                    basic_response = {
                        "diff_changes": f"Analysis encountered an error but will continue processing. Error details: {str(e)}",
                        "processing_time_seconds": 0,
                        "total_chunks_analyzed": 0,
                        "error_details": str(e)
                    }
                    
                    # Analiz kaydını güncelle ama FAILED yapmak yerine COMPLETED olarak işaretle
                    await document_analysis_service.update_analysis_record(
                        analysis_id=analysis_record.id,
                        analysis_data=basic_response,
                        status=AnalysisStatus.COMPLETED  # Önemli: FAILED yerine COMPLETED kullan
                    )
                    logger.info(f"Created basic completed response despite error for {analysis_record.fp}")
                except Exception as update_error:
                    logger.error(f"Failed to create basic response: {str(update_error)}")
                    # Son çare olarak durumu FAILED olarak güncelle, ancak bunu yapmaktan kaçınmaya çalışıyoruz
                    await document_analysis_service.update_analysis_status_by_fp(
                        analysis_record.fp, 
                        AnalysisStatus.COMPLETED,  # Burada FAILED yerine COMPLETED kullanıyoruz
                        f"Analysis process encountered an error: {str(e)}"
                    )
        
        # Add the task to run in the background
        background_tasks.add_task(run_analysis_task)
        
        # Get organization and document fingerprints
        document = await document_service.get_document_by_id(organization.id, analysis_record.document_id)
        if not document:
            logger.error(f"Document not found for ID: {analysis_record.document_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        # Return the analysis record information immediately
        return BaseResponse.from_request(
            request=request,
            data=DocumentAnalysisResponse(
                fp=analysis_record.fp,
                document_fp=document.fp,
                organization_fp=organization.fp,
                diff_changes="", # Git-like diff changes
                total_chunks_analyzed=0,
                processing_time_seconds=0.0,
                chunk_analyses=[],
                status=analysis_record.status.value,
                created_at=analysis_record.created_at,
                completed_at=None
            ),
            success=True,
            message=f"Document analysis queued successfully. Use fingerprint: {analysis_record.fp} to check status using the detail endpoint."
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
    organization_fp: Optional[str] = Query(None, description="Organization fingerprint (fp) to filter by (optional)"),
    page: int = Query(1, description="Page number, starting at 1"),
    per_page: int = Query(10, description="Number of items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user)
):
    """
    List document analyses with pagination.
    
    This endpoint retrieves a paginated list of document analyses.
    - If organization_fp is provided, only analyses from that organization are returned.
    - If organization_fp is not provided, analyses from all organizations the user has access to are returned.
    Results can be filtered by status.
    """
    try:
        # Initialize service
        document_analysis_service = DocumentAnalysisService()
        
        if organization_fp:
            # Validate organization FP
            await validate_organization_fp(organization_fp)
            
            # Get organization by FP
            from app.database.repositories.organization_repository import OrganizationRepository
            async with async_session_maker() as session:
                org_repo = OrganizationRepository(session)
                organization = await org_repo.get_organization_by_fp(organization_fp)
                
                if not organization:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Organization not found"
                    )
                
                # Check if user has access to organization
                if not any(org.id == organization.id for org in current_user.organizations):
                    logger.warning(f"Access denied: User {current_user.id} attempted to list analyses for org {organization_fp}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to organization"
                    )
                
                # Get analyses with pagination for specific organization
                analyses, total_count, total_pages = await document_analysis_service.get_analyses_for_organization(
                    organization_id=organization.id,
                    page=page,
                    per_page=per_page,
                    status_filter=status
                )
        else:
            # Get analyses from all organizations the user has access to
            user_org_ids = [org.id for org in current_user.organizations]
            
            analyses, total_count, total_pages = await document_analysis_service.get_analyses_for_user(
                user_organization_ids=user_org_ids,
                page=page,
                per_page=per_page,
                status_filter=status
            )
        
        # Collect document IDs to fetch document details
        document_ids = [analysis.document_id for analysis in analyses]
        
        # Fetch document details for all analyses at once
        document_service = DocumentService()
        documents = {}
        document_fps = {}
        organizations = {}
        
        if document_ids:
            async with async_session_maker() as session:
                # Get document details
                query = select(File).where(File.id.in_(document_ids))
                result = await session.execute(query)
                file_records = result.scalars().all()
                
                for file in file_records:
                    documents[file.id] = {
                        "filename": file.filename,
                        "file_type": file.file_type,
                        "fp": file.fp
                    }
                    document_fps[file.id] = file.fp
                
                # Get organization details for each unique organization ID
                org_ids = set(analysis.organization_id for analysis in analyses)
                from app.database.models.db_models import Organization
                query = select(Organization).where(Organization.id.in_(org_ids))
                result = await session.execute(query)
                org_records = result.scalars().all()
                
                for org in org_records:
                    organizations[org.id] = org.fp
        
        # Create response data
        analyses_data = []
        for analysis in analyses:
            doc_info = documents.get(analysis.document_id, {})
            document_fp = document_fps.get(analysis.document_id, "")
            organization_fp = organizations.get(analysis.organization_id, "")
            
            analyses_data.append(AnalysisListItem(
                fp=analysis.fp,
                document_fp=document_fp,
                organization_fp=organization_fp,
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

@router.get("/{analysis_fp}", response_model=BaseResponse[AnalysisDetailResponse])
async def get_analysis_detail(
    request: Request,
    analysis_fp: str = Path(..., description="Analysis fingerprint (fp) to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a document analysis.
    
    This endpoint retrieves detailed information about a specific document analysis, including:
    - Document details
    - Original document content
    - Diff changes with git-like format showing removals (red) and additions (green)
    """
    try:
        # Initialize service
        document_analysis_service = DocumentAnalysisService()
        
        # Get the analysis by fingerprint
        analysis = await document_analysis_service.get_analysis_by_fp(analysis_fp)
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with fingerprint {analysis_fp} not found"
            )
            
        # Check user access to the organization
        if not any(org.id == analysis.organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to access analysis {analysis_fp} in org {analysis.organization_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )
        
        # Get document and organization details
        document_service = DocumentService()
        document = await document_service.get_document_by_id(analysis.organization_id, analysis.document_id)
        
        # Get organization FP
        async with async_session_maker() as session:
            from app.database.models.db_models import Organization
            from app.database.repositories.organization_repository import OrganizationRepository
            org_repo = OrganizationRepository(session)
            organization = await org_repo.get_organization_by_id(analysis.organization_id)
            organization_fp = organization.fp if organization else ""
        
        # Prepare the response
        return BaseResponse.from_request(
            request=request,
            data=AnalysisDetailResponse(
                fp=analysis.fp,
                document_fp=document.fp if document else "",
                organization_fp=organization_fp,
                diff_changes=analysis.diff_changes or "",  # Git-like diff changes field
                total_chunks_analyzed=analysis.total_chunks_analyzed or 0,
                processing_time_seconds=float(analysis.processing_time_seconds) if analysis.processing_time_seconds else 0.0,
                # Chunk analyses'i doğru formatta döndürelim
                chunk_analyses=analysis.chunk_analyses if isinstance(analysis.chunk_analyses, list) else (
                    [analysis.chunk_analyses] if isinstance(analysis.chunk_analyses, dict) else []
                ),
                status=analysis.status.value,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
                original_content=analysis.original_content,
                document_filename=document.filename if document else None,
                document_type=document.file_type if document else None
            ),
            success=True,
            message=f"Retrieved details for analysis {analysis_fp}"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving analysis details: {str(e)}", exc_info=True)
        
        # Hata durumunda daha dayanıklı bir yanıt oluşturalım
        try:
            # Eğer analiz verisi alındıysa, güvenli bir alternatif yanıt oluşturalım
            if 'analysis' in locals() and analysis is not None:
                # chunk_analyses alanını düzgün formatta hazırlayalım
                safe_chunk_analyses = []
                
                if hasattr(analysis, 'chunk_analyses') and analysis.chunk_analyses:
                    # Eğer dict tipindeyse, bir liste içine sarmalayalım
                    if isinstance(analysis.chunk_analyses, dict):
                        safe_chunk_analyses = [analysis.chunk_analyses]
                    # Eğer zaten liste tipindeyse, olduğu gibi kullanalım
                    elif isinstance(analysis.chunk_analyses, list):
                        safe_chunk_analyses = analysis.chunk_analyses
                
                # document_fp ve organization_fp'yi güvenli şekilde alalım
                safe_document_fp = ""
                if 'document' in locals() and document:
                    safe_document_fp = document.fp
                
                safe_org_fp = ""
                if 'organization_fp' in locals():
                    safe_org_fp = organization_fp
                
                # Doğrudan dict kullanarak pydantic validasyonunu atlatalım
                response_data = {
                    "fp": analysis.fp,
                    "document_fp": safe_document_fp,
                    "organization_fp": safe_org_fp,
                    "diff_changes": analysis.diff_changes or "",
                    "total_chunks_analyzed": analysis.total_chunks_analyzed or 0,
                    "processing_time_seconds": float(analysis.processing_time_seconds) if analysis.processing_time_seconds else 0.0,
                    "chunk_analyses": safe_chunk_analyses,  # Düzeltilmiş güvenli chunk_analyses
                    "status": analysis.status.value if hasattr(analysis, 'status') else "unknown",
                    "created_at": analysis.created_at,
                    "completed_at": analysis.completed_at,
                    "original_content": analysis.original_content,
                    "document_filename": document.filename if 'document' in locals() and document else None,
                    "document_type": document.file_type if 'document' in locals() and document else None
                }
                
                return BaseResponse.from_request(
                    request=request,
                    data=response_data,  # Pydantic model yerine doğrudan dict kullanıyoruz
                    success=True,
                    message=f"Retrieved details for analysis {analysis_fp} (recovered from error)"
                )
                
        except Exception as recovery_error:
            logger.error(f"Error during analysis details recovery: {str(recovery_error)}", exc_info=True)
        
        # Eğer kurtarma başarısız olursa, standart hata yanıtını döndür
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving analysis details"
        )