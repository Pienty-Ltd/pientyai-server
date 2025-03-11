from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query, BackgroundTasks
from typing import List
import logging
from app.core.services.document_service import DocumentService
from app.database.models import User, Organization
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.schemas.document import DocumentResponse, KnowledgeBaseResponse, PaginatedDocumentResponse
from app.database.models.db_models import FileStatus
import math

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

async def process_document_background(
    file_content: bytes,
    filename: str,
    file_type: str,
    user_id: int,
    organization_id: int,
    db_file_id: int
):
    """Background task for processing document after upload"""
    try:
        document_service = DocumentService()
        await document_service.process_document_async(
            file_content=file_content,
            filename=filename,
            file_type=file_type,
            user_id=user_id,
            organization_id=organization_id,
            db_file_id=db_file_id
        )
    except Exception as e:
        logger.error(f"Background document processing failed: {str(e)}")
        # Update file status to failed in case of error
        document_service = DocumentService()
        await document_service.update_file_status(db_file_id, FileStatus.FAILED)

@router.get("", response_model=BaseResponse[PaginatedDocumentResponse])
async def list_user_documents(
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user)
):
    """List all documents accessible to the current user across their organizations"""
    try:
        document_service = DocumentService()
        documents, total_count = await document_service.get_user_accessible_documents(
            user_id=current_user.id,
            page=page,
            per_page=per_page
        )

        total_pages = math.ceil(total_count / per_page)

        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=[
                    DocumentResponse(
                        id=doc.id,
                        fp=doc.fp,
                        filename=doc.filename,
                        file_type=doc.file_type,
                        status=doc.status,
                        created_at=doc.created_at,
                        chunks_count=doc.chunk_count,
                        organization_id=doc.organization_id
                    ) for doc in documents
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    except Exception as e:
        logger.error(f"Error listing user documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/organization/{org_id}", response_model=BaseResponse[PaginatedDocumentResponse])
async def list_organization_documents(
    org_id: int,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user)
):
    """List all documents for a specific organization with pagination"""
    try:
        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        documents, total_count = await document_service.get_organization_documents_paginated(
            organization_id=org_id,
            page=page,
            per_page=per_page
        )

        total_pages = math.ceil(total_count / per_page)

        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=[
                    DocumentResponse(
                        id=doc.id,
                        fp=doc.fp,
                        filename=doc.filename,
                        file_type=doc.file_type,
                        status=doc.status,
                        created_at=doc.created_at,
                        chunks_count=doc.chunk_count,
                        organization_id=doc.organization_id
                    ) for doc in documents
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing organization documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/organization/{org_id}/{document_id}", response_model=BaseResponse[DocumentResponse])
async def get_document(
    org_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific document by ID"""
    try:
        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        document = await document_service.get_document_by_id(org_id, document_id)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return BaseResponse(
            success=True,
            data=DocumentResponse(
                id=document.id,
                fp=document.fp,
                filename=document.filename,
                file_type=document.file_type,
                status=document.status,
                created_at=document.created_at,
                chunks_count=document.chunk_count,
                organization_id=document.organization_id
            )
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/organization/{org_id}/{document_id}", response_model=BaseResponse)
async def delete_document(
    org_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a document and its associated knowledge base entries"""
    try:
        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        success = await document_service.delete_document(org_id, document_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return BaseResponse(
            success=True,
            message="Document and associated data deleted successfully"
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/upload", response_model=BaseResponse[DocumentResponse])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    organization_id: int = Query(..., description="Organization ID to upload the document to"),
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

        # Check if user has access to organization
        if not any(org.id == organization_id for org in current_user.organizations):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        # Read file content
        file_content = await file.read()

        # Create initial file record with PROCESSING status
        document_service = DocumentService()
        db_file = await document_service.create_file_record(
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization_id
        )

        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            file_content=file_content,
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization_id,
            db_file_id=db_file.id
        )

        return BaseResponse(
            success=True,
            message="Document upload started. Processing in background.",
            data=DocumentResponse(
                id=db_file.id,
                fp=db_file.fp,
                filename=db_file.filename,
                file_type=db_file.file_type,
                status=db_file.status,
                created_at=db_file.created_at,
                chunks_count=0,  # Will be updated during processing
                organization_id=organization_id
            )
        )

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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