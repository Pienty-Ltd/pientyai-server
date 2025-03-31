from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query, BackgroundTasks, Request
from typing import List, Optional
import logging
from datetime import datetime
import mimetypes
from app.core.services.document_service import DocumentService
from app.database.models import User, Organization
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.schemas.document import DocumentResponse, KnowledgeBaseResponse, PaginatedDocumentResponse
from app.database.models.db_models import FileStatus
from app.api.v1.middlewares.validation_middleware import (
    validate_pagination_parameters,
    validate_document_id,
    validate_document_fp,
    validate_organization_id,
    validate_organization_fp
)
import math

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Enhance ALLOWED_MIMETYPES with additional validation
ALLOWED_MIMETYPES = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/msword': 'doc'
}

# Mapping of file extensions to expected MIME types
EXTENSION_MIME_MAP = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword'
}

async def validate_file_upload(file: UploadFile) -> str:
    """
    Validate uploaded file type and content
    Returns the validated file extension
    """
    try:
        # Check file size (10MB limit)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds 10MB limit"
            )

        # Reset file position for later reading
        await file.seek(0)

        # Basic filename validation
        if not file.filename or '.' not in file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )

        # Extract and validate file extension
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in EXTENSION_MIME_MAP:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_MIMETYPES.values())}"
            )

        # Check content type
        content_type = file.content_type
        if content_type not in ALLOWED_MIMETYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type. Expected types: {', '.join(ALLOWED_MIMETYPES.keys())}"
            )

        # Validate file extension matches content type
        expected_ext = ALLOWED_MIMETYPES[content_type]
        if file_ext != expected_ext:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension '{file_ext}' doesn't match content type. Expected: {expected_ext}"
            )

        # Additional validation: ensure the detected MIME type matches the extension
        expected_mime = EXTENSION_MIME_MAP[file_ext]
        if content_type != expected_mime:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content type mismatch. Got: {content_type}, Expected: {expected_mime}"
            )

        return file_ext

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file validation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or corrupted file"
        )

async def process_document_background(
    file_content: bytes,
    filename: str,
    file_type: str,
    user_id: int,
    organization_id: int,
    db_file_id: int,
    is_knowledge_base: bool = True
):
    """Background task for processing document after upload"""
    try:
        logger.info(f"Starting background processing for document {filename} (ID: {db_file_id}, knowledge_base: {is_knowledge_base})")
        start_time = datetime.now()

        document_service = DocumentService()
        await document_service.process_document_async(
            file_content=file_content,
            filename=filename,
            file_type=file_type,
            user_id=user_id,
            organization_id=organization_id,
            db_file_id=db_file_id,
            is_knowledge_base=is_knowledge_base
        )

        processing_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed background processing for document {filename} (Duration: {processing_duration}s)")
    except Exception as e:
        logger.error(f"Background document processing failed: {str(e)}", exc_info=True)
        # Update file status to failed in case of error
        document_service = DocumentService()
        await document_service.update_file_status(db_file_id, FileStatus.FAILED)

@router.get("", response_model=BaseResponse[PaginatedDocumentResponse])
async def list_user_documents(
    request: Request,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    current_user: User = Depends(get_current_user)
):
    """List all documents accessible to the current user across their organizations"""
    try:
        # Validate pagination parameters
        page, per_page = await validate_pagination_parameters(request, page, per_page)

        logger.info(f"Fetching documents for user {current_user.id}, page {page}, per_page {per_page}")
        start_time = datetime.now()

        document_service = DocumentService()  # No S3 client initialization here
        documents, total_count = await document_service.get_user_accessible_documents(
            user_id=current_user.id,
            page=page,
            per_page=per_page
        )

        total_pages = math.ceil(total_count / per_page)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved {len(documents)} documents (Total: {total_count}, Duration: {duration}s)")

        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=[
                    DocumentResponse(
                        fp=doc.fp,
                        filename=doc.filename,
                        file_type=doc.file_type,
                        status=doc.status,
                        created_at=doc.created_at,
                        chunks_count=doc.chunk_count if hasattr(doc, 'chunk_count') else 0,
                        organization_id=doc.organization_id
                    ) for doc in documents
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    except ValueError as ve:
        logger.error(f"Validation error in list_user_documents: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error listing user documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching documents"
        )

@router.get("/organization/{org_id}", response_model=BaseResponse[PaginatedDocumentResponse])
async def list_organization_documents(
    request: Request,
    org_id: int,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    current_user: User = Depends(get_current_user)
):
    """List all documents for a specific organization with pagination"""
    try:
        # Validate parameters
        org_id = await validate_organization_id(org_id)
        page, per_page = await validate_pagination_parameters(request, page, per_page)

        logger.info(f"Fetching documents for organization {org_id}, page {page}, per_page {per_page}")
        start_time = datetime.now()

        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to access org {org_id}")
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
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved {len(documents)} documents for org {org_id} (Total: {total_count}, Duration: {duration}s)")

        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=[
                    DocumentResponse(
                        fp=doc.fp,
                        filename=doc.filename,
                        file_type=doc.file_type,
                        status=doc.status,
                        created_at=doc.created_at,
                        chunks_count=0,  # Set to 0 since we don't load knowledge_base
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
        logger.error(f"Error listing organization documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching organization documents"
        )

@router.get("/organization/{org_id}/{document_fp}", response_model=BaseResponse[DocumentResponse])
async def get_document(
    org_id: int,
    document_fp: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific document by fingerprint (fp)"""
    try:
        logger.info(f"Fetching document {document_fp} from organization {org_id}")
        start_time = datetime.now()

        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)

        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to access document {document_fp} in org {org_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        document = await document_service.get_document_by_fp(org_id, document_fp)

        if not document:
            logger.warning(f"Document not found: FP {document_fp} in org {org_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved document {document_fp} (Duration: {duration}s)")

        return BaseResponse(
            success=True,
            data=DocumentResponse(
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
        logger.error(f"Error fetching document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the document"
        )

@router.delete("/organization/{org_id}/{document_fp}", response_model=BaseResponse)
async def delete_document(
    org_id: int,
    document_fp: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a document and its associated knowledge base entries by fingerprint (fp)"""
    try:
        logger.info(f"Attempting to delete document {document_fp} from organization {org_id}")
        start_time = datetime.now()

        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)

        # Check if user has access to organization
        if not any(org.id == org_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to delete document {document_fp} in org {org_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        document_service = DocumentService()
        success = await document_service.delete_document_by_fp(org_id, document_fp)

        if not success:
            logger.warning(f"Document not found for deletion: FP {document_fp} in org {org_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Deleted document {document_fp} successfully (Duration: {duration}s)")

        return BaseResponse(
            success=True,
            message="Document and associated data deleted successfully"
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the document"
        )

@router.post("/upload", response_model=BaseResponse[DocumentResponse])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    organization_id: int = Query(..., description="Organization ID to upload the document to"),
    is_knowledge_base: bool = Query(True, description="Whether the document is for knowledge base (True) or for analysis (False)"),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document
    
    - Set is_knowledge_base=True for documents that should be part of the knowledge base for reference
    - Set is_knowledge_base=False for documents that will be analyzed against the knowledge base
    """
    try:
        logger.info(f"Processing upload request for file {file.filename} to organization {organization_id} (knowledge_base: {is_knowledge_base})")
        start_time = datetime.now()

        # Enhanced file validation
        try:
            file_type = await validate_file_upload(file)
        except HTTPException as e:
            logger.warning(f"File validation failed: {str(e.detail)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during file validation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format or corrupted file"
            )

        # Check if user has access to organization
        if not any(org.id == organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to upload to org {organization_id}")
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
            organization_id=organization_id,
            is_knowledge_base=is_knowledge_base
        )

        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            file_content=file_content,
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization_id,
            db_file_id=db_file.id,
            is_knowledge_base=is_knowledge_base
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Document upload initiated for {file.filename} (Duration: {duration}s)")

        return BaseResponse(
            success=True,
            message="Document upload started. Processing in background.",
            data=DocumentResponse(
                fp=db_file.fp,
                filename=db_file.filename,
                file_type=db_file.file_type,
                status=db_file.status,
                created_at=db_file.created_at,
                chunks_count=0,  # Will be updated during processing
                organization_id=organization_id,
                is_knowledge_base=is_knowledge_base
            )
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while uploading the document"
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
        logger.info(f"Searching documents in organization {organization_id} with query '{query}', limit {limit}")
        start_time = datetime.now()

        # Check if user has access to organization
        if not any(org.id == organization_id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to search in org {organization_id}")
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

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Search completed (Duration: {duration}s), {len(results)} results found.")

        return BaseResponse(
            success=True,
            data=[
                KnowledgeBaseResponse(
                    fp=result.fp,
                    chunk_index=result.chunk_index,
                    content=result.content,
                    meta_info=result.meta_info,
                    created_at=result.created_at
                ) for result in results
            ]
        )

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching documents"
        )