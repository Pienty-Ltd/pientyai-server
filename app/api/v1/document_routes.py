from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query
from typing import List
import logging
from app.core.services.document_service import DocumentService
from app.database.models import User, Organization
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.schemas.document import DocumentResponse, KnowledgeBaseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=BaseResponse[DocumentResponse])
async def upload_document(
    file: UploadFile = File(...),
    organization_id: int = None,
    current_user: User = Depends(get_current_user)
):
    """Upload and process a document for knowledge base"""
    try:
        # Validate file type
        allowed_types = ["pdf", "docx", "doc"]
        file_type = file.filename.split(".")[-1].lower()
        if file_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )

        # Get organization_id from user if not provided
        if not organization_id:
            if not current_user.organizations:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User is not associated with any organization"
                )
            organization_id = current_user.organizations[0].id

        # Process document
        document_service = DocumentService()
        db_file, knowledge_base_entries = await document_service.process_document(
            file=file.file,
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization_id
        )

        return BaseResponse(
            success=True,
            message="Document uploaded and processed successfully",
            data=DocumentResponse(
                id=db_file.id,
                fp=db_file.fp,
                filename=db_file.filename,
                file_type=db_file.file_type,
                status=db_file.status,
                created_at=db_file.created_at,
                chunks_count=len(knowledge_base_entries)
            )
        )

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing document"
        )

@router.get("/{organization_id}", response_model=BaseResponse[List[DocumentResponse]])
async def get_organization_documents(
    organization_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get all documents for an organization"""
    try:
        # Check if user has access to organization
        if not any(org.id == organization_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        files = await document_service.get_organization_documents(
            organization_id=organization_id
        )

        return BaseResponse(
            success=True,
            data=[
                DocumentResponse(
                    id=file.id,
                    fp=file.fp,
                    filename=file.filename,
                    file_type=file.file_type,
                    status=file.status,
                    created_at=file.created_at,
                    chunks_count=len(file.knowledge_base)
                ) for file in files
            ]
        )

    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching documents"
        )

@router.get("/search/{organization_id}", response_model=BaseResponse[List[KnowledgeBaseResponse]])
async def search_documents(
    organization_id: int,
    query: str = Query(..., min_length=3),
    limit: int = Query(default=5, gt=0, le=20),
    current_user: User = Depends(get_current_user)
):
    """
    Search through organization's documents using semantic search
    """
    try:
        # Check if user has access to organization
        if not any(org.id == organization_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        results = await document_service.search_documents(
            organization_id=organization_id,
            query=query,
            limit=limit
        )

        return BaseResponse(
            success=True,
            data=[
                KnowledgeBaseResponse(
                    id=result.id,
                    fp=result.fp,
                    chunk_index=result.chunk_index,
                    content=result.content,
                    meta_info=result.meta_info,
                    created_at=result.created_at
                ) for result in results
            ]
        )

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error searching documents"
        )