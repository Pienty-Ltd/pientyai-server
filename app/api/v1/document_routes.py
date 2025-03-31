from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query, BackgroundTasks, Request
from typing import List, Optional
import logging
from datetime import datetime
import mimetypes
import math
from app.core.services.document_service import DocumentService
from app.database.models import User, Organization
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.schemas.document import (
    DocumentResponse, 
    KnowledgeBaseResponse, 
    PaginatedDocumentResponse, 
    PaginatedKnowledgeBaseResponse,
    DocumentWithChunksResponse
)
from app.database.models.db_models import FileStatus
from app.api.v1.middlewares.validation_middleware import (
    validate_pagination_parameters,
    validate_document_fp,
    validate_organization_id,
    validate_organization_fp
)
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
    organization_fp: Optional[str] = Query(None, description="Optional organization fingerprint to filter documents"),
    current_user: User = Depends(get_current_user)
):
    """
    List documents accessible to the current user
    
    - If organization_fp is provided, returns documents only from that organization
    - Otherwise, returns documents from all organizations the user has access to
    """
    try:
        # Validate pagination parameters
        validated_page, validated_per_page = await validate_pagination_parameters(request, page, per_page)
        page = validated_page if validated_page is not None else page
        per_page = validated_per_page if validated_per_page is not None else per_page
        
        # Validate organization fingerprint if provided
        if organization_fp:
            organization_fp = await validate_organization_fp(organization_fp)
            logger.info(f"Fetching documents for user {current_user.id} in organization {organization_fp}, page {page}, per_page {per_page}")
        else:
            logger.info(f"Fetching documents for user {current_user.id} across all organizations, page {page}, per_page {per_page}")
        
        start_time = datetime.now()

        document_service = DocumentService()
        documents, total_count = await document_service.get_user_accessible_documents(
            user_id=current_user.id,
            page=page,
            per_page=per_page,
            organization_fp=organization_fp
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
                per_page=per_page,
                organization_fp=organization_fp  # Pass the organization_fp if it was provided
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

@router.get("/organization/{org_fp}", response_model=BaseResponse[PaginatedDocumentResponse])
async def list_organization_documents(
    request: Request,
    org_fp: str,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    current_user: User = Depends(get_current_user)
):
    """List all documents for a specific organization with pagination"""
    try:
        # Validate parameters
        org_fp = await validate_organization_fp(org_fp)
        validated_page, validated_per_page = await validate_pagination_parameters(request, page, per_page)
        page = validated_page if validated_page is not None else page
        per_page = validated_per_page if validated_per_page is not None else per_page

        logger.info(f"Fetching documents for organization {org_fp}, page {page}, per_page {per_page}")
        start_time = datetime.now()

        # Fetch organization by fingerprint
        document_service = DocumentService()
        organization = await document_service.get_organization_by_fp(org_fp)
        
        if not organization:
            logger.warning(f"Organization not found: FP {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Check if user has access to organization
        if not any(org.id == organization.id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to access org {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        documents, total_count = await document_service.get_organization_documents_paginated(
            organization_id=organization.id,
            page=page,
            per_page=per_page
        )

        total_pages = math.ceil(total_count / per_page)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved {len(documents)} documents for org {org_fp} (Total: {total_count}, Duration: {duration}s)")

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
                per_page=per_page,
                organization_fp=org_fp
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

@router.get("/document/{document_fp}/chunks", response_model=BaseResponse[PaginatedKnowledgeBaseResponse])
async def get_document_chunks(
    document_fp: str,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    current_user: User = Depends(get_current_user)
):
    """
    Get all chunks for a specific document with pagination
    Returns a list of chunks ordered by chunk_index
    """
    try:
        logger.info(f"Fetching chunks for document {document_fp}, page={page}, per_page={per_page}")
        start_time = datetime.now()
        
        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)
        
        # Get document service
        document_service = DocumentService()
        
        # First check if the user has access to this document
        document, organization = await document_service.get_document_by_fp_for_user(
            user_id=current_user.id, 
            document_fp=document_fp
        )
        
        if not document or not organization:
            logger.warning(f"Document not found or access denied: FP {document_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get chunks with pagination
        chunks, total_count = await document_service.get_document_chunks(
            document_id=document.id,
            page=page,
            per_page=per_page
        )
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved {len(chunks)} chunks for document {document_fp} (Total: {total_count}, Duration: {duration}s)")
        
        # Convert KnowledgeBase models to KnowledgeBaseResponse objects
        knowledge_base_responses = [
            KnowledgeBaseResponse(
                fp=chunk.fp,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                meta_info=chunk.meta_info,
                created_at=chunk.created_at,
                is_knowledge_base=chunk.is_knowledge_base
            ) for chunk in chunks
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedKnowledgeBaseResponse(
                chunks=knowledge_base_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching document chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching document chunks"
        )

@router.get("/document/{document_fp}/search", response_model=BaseResponse[PaginatedKnowledgeBaseResponse])
async def search_document_chunks(
    document_fp: str,
    query: str = Query(..., min_length=3, description="Search query"),
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    current_user: User = Depends(get_current_user)
):
    """
    Search within a specific document's chunks using semantic search
    Returns a list of chunks sorted by relevance to the search query
    """
    try:
        logger.info(f"Searching chunks in document {document_fp} with query '{query}', page={page}, per_page={per_page}")
        start_time = datetime.now()
        
        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)
        
        # Get document service
        document_service = DocumentService()
        
        # Search for chunks in the document with user access check
        chunks, total_count, document, organization = await document_service.search_chunks_in_document_by_fp(
            document_fp=document_fp,
            query=query,
            page=page,
            per_page=per_page
        )
        
        # Check if document exists and user has access to it
        if not document or not organization:
            logger.warning(f"Document not found: FP {document_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
            
        # Check if user has access to organization
        if not any(org.id == organization.id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to search chunks in document {document_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to document"
            )
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Found {len(chunks)} chunks in document {document_fp} matching query (Total: {total_count}, Duration: {duration}s)")
        
        # Convert KnowledgeBase models to KnowledgeBaseResponse objects
        knowledge_base_responses = [
            KnowledgeBaseResponse(
                fp=chunk.fp,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                meta_info=chunk.meta_info,
                created_at=chunk.created_at,
                is_knowledge_base=chunk.is_knowledge_base
            ) for chunk in chunks
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedKnowledgeBaseResponse(
                chunks=knowledge_base_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error searching document chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching document chunks"
        )

@router.get("/document/{document_fp}", response_model=BaseResponse[DocumentWithChunksResponse])
async def get_document(
    document_fp: str,
    include_chunks: bool = Query(True, description="Whether to include document chunks in the response"),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific document by fingerprint (fp)
    
    - include_chunks: Set to true to include the document chunks in the response
    """
    try:
        logger.info(f"Fetching document {document_fp}, include_chunks={include_chunks}")
        start_time = datetime.now()

        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)

        # Get document service and find document with a single query
        document_service = DocumentService()
        document, organization = await document_service.get_document_by_fp_for_user(
            user_id=current_user.id,
            document_fp=document_fp
        )
                
        if not document or not organization:
            logger.warning(f"Document not found: FP {document_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # We already have the organization from the query
        org_fp = organization.fp
        
        # Create response with basic document info
        response = DocumentWithChunksResponse(
            fp=document.fp,
            filename=document.filename,
            file_type=document.file_type,
            status=document.status,
            created_at=document.created_at,
            chunks_count=document.chunk_count,
            organization_id=document.organization_id,
            is_knowledge_base=document.is_knowledge_base,
            organization_fp=org_fp,
            chunks=[]  # Default empty list of chunks
        )
        
        # Add chunks if requested and available
        if include_chunks and document.knowledge_base:
            response.chunks = [
                KnowledgeBaseResponse(
                    fp=chunk.fp,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    meta_info=chunk.meta_info,
                    created_at=chunk.created_at,
                    is_knowledge_base=chunk.is_knowledge_base
                ) for chunk in document.knowledge_base
            ]
            
            # Sort chunks by index
            response.chunks.sort(key=lambda x: x.chunk_index)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Retrieved document {document_fp} with {len(response.chunks)} chunks (Duration: {duration}s)")

        return BaseResponse(
            success=True,
            data=response
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the document"
        )

@router.delete("/document/{document_fp}", response_model=BaseResponse)
async def delete_document(
    document_fp: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a document and its associated knowledge base entries by fingerprint (fp)"""
    try:
        logger.info(f"Attempting to delete document {document_fp}")
        start_time = datetime.now()

        # Validate document fingerprint
        document_fp = await validate_document_fp(document_fp)
        
        # Get document service and delete document with a single query
        document_service = DocumentService()
        
        # Delete the document across all organizations the user has access to
        success = await document_service.delete_document_by_fp_for_user(
            user_id=current_user.id,
            document_fp=document_fp
        )
                
        if not success:
            logger.warning(f"Document not found for deletion: FP {document_fp}")
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
    organization_fp: str = Query(..., description="Organization fingerprint to upload the document to"),
    is_knowledge_base: bool = Query(True, description="Whether the document is for knowledge base (True) or for analysis (False)"),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document
    
    - Set is_knowledge_base=True for documents that should be part of the knowledge base for reference
    - Set is_knowledge_base=False for documents that will be analyzed against the knowledge base
    """
    try:
        logger.info(f"Processing upload request for file {file.filename} to organization {organization_fp} (knowledge_base: {is_knowledge_base})")
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
            
        # Validate organization fingerprint
        organization_fp = await validate_organization_fp(organization_fp)
        
        # Fetch organization by fingerprint
        document_service = DocumentService()
        organization = await document_service.get_organization_by_fp(organization_fp)
        
        if not organization:
            logger.warning(f"Organization not found: FP {organization_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check if user has access to organization
        if not any(org.id == organization.id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to upload to org {organization_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        # Read file content
        file_content = await file.read()

        # Create initial file record with PROCESSING status
        db_file = await document_service.create_file_record(
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization.id,
            is_knowledge_base=is_knowledge_base
        )

        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            file_content=file_content,
            filename=file.filename,
            file_type=file_type,
            user_id=current_user.id,
            organization_id=organization.id,
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
                organization_id=organization.id,
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

@router.get("/search/{org_fp}", response_model=BaseResponse[PaginatedDocumentResponse])
async def search_documents(
    org_fp: str,
    query: str = Query(..., min_length=3),
    page: int = Query(default=1, gt=0),
    per_page: int = Query(default=20, gt=0, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search through organization's documents using semantic search
    Returns a list of documents (files) sorted by relevance to the search query
    """
    try:
        logger.info(f"Searching documents in organization {org_fp} with query '{query}', page {page}, per_page {per_page}")
        start_time = datetime.now()

        # Validate organization fingerprint
        org_fp = await validate_organization_fp(org_fp)
        
        # Fetch organization by fingerprint
        document_service = DocumentService()
        organization = await document_service.get_organization_by_fp(org_fp)
        
        if not organization:
            logger.warning(f"Organization not found: FP {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check if user has access to organization
        if not any(org.id == organization.id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to search in org {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        # Use modified search method that returns documents instead of chunks
        documents, total_count = await document_service.search_documents_for_organization(
            organization_id=organization.id,
            query=query,
            page=page,
            per_page=per_page
        )

        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Organization document search completed (Duration: {duration}s), {len(documents)} results found, {total_count} total.")

        # Convert File models to DocumentResponse objects
        document_responses = [
            DocumentResponse(
                fp=doc.fp,
                filename=doc.filename,
                file_type=doc.file_type,
                status=doc.status,
                created_at=doc.created_at,
                chunks_count=doc.chunk_count if hasattr(doc, 'chunk_count') else 0,
                organization_id=doc.organization_id,
                is_knowledge_base=doc.is_knowledge_base,
                organization_fp=org_fp
            ) for doc in documents
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=document_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page,
                organization_fp=org_fp
            )
        )

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching documents"
        )
        
@router.get("/search", response_model=BaseResponse[PaginatedDocumentResponse])
async def search_user_documents(
    query: str = Query(..., min_length=3),
    page: int = Query(default=1, gt=0),
    per_page: int = Query(default=20, gt=0, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search across all documents the user has access to across all organizations
    Returns a list of documents (files) sorted by relevance to the search query
    """
    try:
        logger.info(f"Searching all user accessible documents with query '{query}', page {page}, per_page {per_page}")
        start_time = datetime.now()
        
        document_service = DocumentService()
        
        results, total_count = await document_service.search_documents_for_user(
            user_id=current_user.id,
            query=query,
            page=page,
            per_page=per_page
        )
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"User document search completed (Duration: {duration}s), {len(results)} results found, {total_count} total.")
        
        # Convert File models to DocumentResponse objects
        document_responses = [
            DocumentResponse(
                fp=file.fp,
                filename=file.filename,
                file_type=file.file_type,
                status=file.status,
                created_at=file.created_at,
                chunks_count=file.chunk_count,
                organization_id=file.organization_id,
                is_knowledge_base=file.is_knowledge_base
            ) for file in results
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedDocumentResponse(
                documents=document_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    
    except Exception as e:
        logger.error(f"Error searching user documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching documents"
        )
        
@router.get("/search/chunks/{org_fp}", response_model=BaseResponse[PaginatedKnowledgeBaseResponse])
async def search_organization_document_chunks(
    org_fp: str,
    query: str = Query(..., min_length=3),
    page: int = Query(default=1, gt=0),
    per_page: int = Query(default=20, gt=0, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search through document chunks in a specific organization
    Returns a list of chunks sorted by relevance to the search query
    """
    try:
        logger.info(f"Searching document chunks in organization {org_fp} with query '{query}', page {page}, per_page {per_page}")
        start_time = datetime.now()

        # Validate organization fingerprint
        org_fp = await validate_organization_fp(org_fp)
        
        # Fetch organization by fingerprint
        document_service = DocumentService()
        organization = await document_service.get_organization_by_fp(org_fp)
        
        if not organization:
            logger.warning(f"Organization not found: FP {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check if user has access to organization
        if not any(org.id == organization.id for org in current_user.organizations):
            logger.warning(f"Access denied: User {current_user.id} attempted to search chunks in org {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization"
            )

        # Use the new organization-specific chunk search
        results, total_count = await document_service.search_chunks_for_organization(
            organization_id=organization.id,
            query=query,
            page=page,
            per_page=per_page
        )
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Organization chunk search completed (Duration: {duration}s), {len(results)} chunks found, {total_count} total.")
        
        # Convert KnowledgeBase models to KnowledgeBaseResponse objects
        knowledge_base_responses = [
            KnowledgeBaseResponse(
                fp=chunk.fp,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                meta_info=chunk.meta_info,
                created_at=chunk.created_at,
                is_knowledge_base=chunk.is_knowledge_base
            ) for chunk in results
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedKnowledgeBaseResponse(
                chunks=knowledge_base_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    
    except Exception as e:
        logger.error(f"Error searching document chunks in organization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching document chunks"
        )
        
@router.get("/search/chunks", response_model=BaseResponse[PaginatedKnowledgeBaseResponse])
async def search_user_document_chunks(
    query: str = Query(..., min_length=3),
    page: int = Query(default=1, gt=0),
    per_page: int = Query(default=20, gt=0, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search through all document chunks the user has access to across all organizations
    Returns a list of chunks sorted by relevance to the search query
    """
    try:
        logger.info(f"Searching document chunks with query '{query}', page {page}, per_page {per_page}")
        start_time = datetime.now()
        
        document_service = DocumentService()
        
        # Use the renamed method for chunk search
        results, total_count = await document_service.search_chunks_for_user(
            user_id=current_user.id,
            query=query,
            page=page,
            per_page=per_page
        )
        
        # Calculate pagination info
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chunk search completed (Duration: {duration}s), {len(results)} chunks found, {total_count} total.")
        
        # Convert KnowledgeBase models to KnowledgeBaseResponse objects
        knowledge_base_responses = [
            KnowledgeBaseResponse(
                fp=chunk.fp,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                meta_info=chunk.meta_info,
                created_at=chunk.created_at,
                is_knowledge_base=chunk.is_knowledge_base
            ) for chunk in results
        ]
        
        return BaseResponse(
            success=True,
            data=PaginatedKnowledgeBaseResponse(
                chunks=knowledge_base_responses,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    
    except Exception as e:
        logger.error(f"Error searching document chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching document chunks"
        )