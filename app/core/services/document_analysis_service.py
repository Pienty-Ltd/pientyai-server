import os
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
import math
from datetime import datetime
from sqlalchemy import select, func, desc, and_, or_

from app.core.services.openai_service import OpenAIService
from app.core.services.document_service import DocumentService
from app.database.models.db_models import (
    KnowledgeBase, File, DocumentAnalysis, AnalysisStatus, User
)
from app.database.database_factory import async_session_maker

logger = logging.getLogger(__name__)

class DocumentAnalysisService:
    """Service for document analysis against organization's knowledge base"""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.document_service = DocumentService()
        
    async def analyze_document_with_knowledge_base(
        self,
        organization_id: int,
        document_id: int,
        user_id: int,
        max_relevant_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze a document against the organization's knowledge base
        
        Args:
            organization_id: ID of the organization
            document_id: ID of the document to analyze
            user_id: ID of the user requesting the analysis
            max_relevant_chunks: Maximum number of relevant chunks to retrieve from knowledge base
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            logger.info(f"Starting document analysis for doc {document_id} in organization {organization_id}")
            
            # Create a record to track the analysis
            analysis_record = await self.create_analysis_record(
                document_id=document_id,
                organization_id=organization_id,
                user_id=user_id
            )
            
            start_time = datetime.now()
            
            # Fetch the document to be analyzed
            document = await self.document_service.get_document_by_id(organization_id, document_id)
            if not document:
                error_msg = f"Document not found: {document_id}"
                await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
                raise ValueError(error_msg)
                
            if document.status != "completed":
                error_msg = f"Document processing is not complete. Current status: {document.status}"
                await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
                raise ValueError(error_msg)
            
            # Update the status to processing
            await self.update_analysis_status(analysis_record.id, AnalysisStatus.PROCESSING)
            
            # Get the original document content - we'll join all chunks to represent the original content
            document_chunks = await self.get_document_chunks(document_id)
            
            if not document_chunks:
                error_msg = f"No chunks found for document {document_id}"
                await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
                raise ValueError(error_msg)
                
            # Store the original content (concatenated chunks)
            original_content = "\n\n".join([chunk.content for chunk in document_chunks])
            await self.update_original_content(analysis_record.id, original_content)
            
            # Set up tracking variables for analysis
            analysis_results = []
            total_chunks = len(document_chunks)
            
            logger.info(f"Processing {total_chunks} chunks for document {document_id}")
            
            # For each chunk in the document, perform semantic search and analyze
            for chunk_index, document_chunk in enumerate(document_chunks):
                chunk_start_time = datetime.now()
                logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks}")
                
                # Get the content of the current chunk
                chunk_content = document_chunk.content
                
                # Find the most relevant chunks from the knowledge base for this document chunk
                relevant_kb_chunks = await self.find_relevant_knowledge_base_chunks(
                    organization_id=organization_id,
                    query_text=chunk_content,
                    current_document_id=document_id,  # Exclude the current document from search
                    limit=max_relevant_chunks
                )
                
                # Extract just the content from the KB chunks for analysis
                kb_content_list = [kb_chunk.content for kb_chunk in relevant_kb_chunks]
                
                # Get the analysis for this chunk with context from relevant KB chunks
                chunk_analysis = await self.openai_service.analyze_document(
                    document_chunk=chunk_content,
                    knowledge_base_chunks=kb_content_list
                )
                
                # Check if the analysis contains an error
                if "error" in chunk_analysis:
                    error_msg = f"Chunk {chunk_index + 1} analysis failed: {chunk_analysis.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    # If this is the first chunk and it failed, we might want to fail the whole analysis
                    if chunk_index == 0:
                        await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
                        raise ValueError(error_msg)
                
                # Add metadata about the chunk to the analysis result
                chunk_analysis["chunk_index"] = chunk_index
                chunk_analysis["chunk_content"] = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
                chunk_analysis["processing_time_seconds"] = (datetime.now() - chunk_start_time).total_seconds()
                
                # Add to the results list
                analysis_results.append(chunk_analysis)
                
                # Small delay between chunks to avoid rate limits
                if chunk_index < total_chunks - 1:
                    await asyncio.sleep(0.5)
            
            # Create the final combined analysis
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Combine the analyses into a single result
            combined_analysis = self._combine_chunk_analyses(analysis_results)
            combined_analysis["document_id"] = document_id
            combined_analysis["organization_id"] = organization_id
            combined_analysis["total_chunks_analyzed"] = total_chunks
            combined_analysis["processing_time_seconds"] = total_duration
            
            # Generate suggested changes based on the analysis
            suggested_changes = await self._generate_suggested_changes(
                original_content=original_content,
                analysis_results=analysis_results
            )
            
            # Update the record with the results
            await self.update_analysis_record(
                analysis_id=analysis_record.id,
                analysis_data=combined_analysis,
                suggested_changes=suggested_changes,
                status=AnalysisStatus.COMPLETED
            )
            
            logger.info(f"Completed document analysis for doc {document_id} (Duration: {total_duration}s)")
            
            # Get the updated record to return
            updated_record = await self.get_analysis_by_id(analysis_record.id)
            
            # Get document and organization FPs
            document = None
            organization = None
            try:
                async with async_session_maker() as session:
                    # Get document FP
                    from app.database.models.db_models import File
                    query = select(File).where(File.id == updated_record.document_id)
                    result = await session.execute(query)
                    document = result.scalar_one_or_none()
                    
                    # Get organization FP
                    from app.database.models.db_models import Organization
                    query = select(Organization).where(Organization.id == updated_record.organization_id)
                    result = await session.execute(query)
                    organization = result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting document or organization details: {e}", exc_info=True)
            
            return {
                "id": updated_record.id,
                "fp": updated_record.fp,
                "document_fp": document.fp if document else "",
                "organization_fp": organization.fp if organization else "",
                "analysis": updated_record.analysis,
                "key_points": updated_record.key_points,
                "conflicts": updated_record.conflicts,
                "recommendations": updated_record.recommendations,
                "total_chunks_analyzed": updated_record.total_chunks_analyzed,
                "processing_time_seconds": float(updated_record.processing_time_seconds) if updated_record.processing_time_seconds else 0.0,
                "chunk_analyses": updated_record.chunk_analyses,
                "status": updated_record.status.value,
                "created_at": updated_record.created_at,
                "completed_at": updated_record.completed_at
            }
            
        except Exception as e:
            error_msg = f"Error in document analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # If there's a record, update it to failed status with the error message
            if 'analysis_record' in locals() and analysis_record:
                await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
            raise
            
    async def get_document_chunks(self, document_id: int) -> List[KnowledgeBase]:
        """Get all chunks for a specific document from knowledge base"""
        try:
            async with async_session_maker() as session:
                stmt = select(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id
                ).order_by(KnowledgeBase.chunk_index)
                
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
            
    async def find_relevant_knowledge_base_chunks(
        self,
        organization_id: int,
        query_text: str,
        current_document_id: Optional[int] = None,
        limit: int = 5
    ) -> List[KnowledgeBase]:
        """
        Find the most relevant knowledge base chunks for a query text
        Only searches through actual knowledge base chunks (is_knowledge_base=True)
        
        Also includes adjacent chunks (2 before and 2 after) for each relevant chunk to provide more context.
        This helps avoid truncated information from chunk boundaries.
        
        Args:
            organization_id: ID of the organization
            query_text: Text to search for (usually a chunk from the document to analyze)
            current_document_id: Optional ID of the current document to exclude from results
            limit: Maximum number of chunks to return (between 3 and 10)
            
        Returns:
            List of relevant knowledge base chunks including adjacent chunks for better context
        """
        # Ensure limit is within allowed range
        if limit < 3:
            limit = 3
        elif limit > 10:
            limit = 10
        try:
            # Generate embedding for the query text
            try:
                query_embedding = await self.openai_service.create_embeddings([query_text])
                if not query_embedding or len(query_embedding) == 0:
                    logger.error("Failed to generate embedding for query text")
                    return []
            except Exception as e:
                error_details = str(e)
                if hasattr(e, '__cause__') and e.__cause__ is not None:
                    error_details = f"{error_details} | Cause: {str(e.__cause__)}"
                
                logger.error(f"Error generating embedding for query text: {error_details}")
                # Return empty list so document analysis can continue without vector search
                return []
                
            async with async_session_maker() as session:
                # Step 1: First get the most relevant chunks based on vector similarity
                base_query = select(KnowledgeBase).where(
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_knowledge_base == True  # Only include knowledge base chunks
                )
                
                # If a current document ID is provided, exclude it from the results
                if current_document_id is not None:
                    base_query = base_query.where(
                        KnowledgeBase.file_id != current_document_id
                    )
                
                # Order by vector similarity (L2 distance) and get top matches
                # We'll get fewer initial chunks since we'll be adding adjacent ones later
                initial_limit = min(limit // 2, 3)  # Get fewer initial chunks to leave room for adjacent ones
                if initial_limit < 1:
                    initial_limit = 1
                
                try:
                    # Vektör sorgusu yaparken hata yakalama ekleyelim
                    if query_embedding and len(query_embedding) > 0 and len(query_embedding[0]) > 0:
                        similarity_query = base_query.order_by(
                            func.l2_distance(KnowledgeBase.embedding, query_embedding[0])
                        ).limit(initial_limit)
                    else:
                        # Eğer embedding oluşturulamadıysa veya geçersizse,
                        # Basit bir sorgu ile en son eklenen knowledge base parçalarını getirelim
                        logger.warning("Invalid embedding for query, falling back to recent chunks")
                        similarity_query = base_query.order_by(
                            KnowledgeBase.created_at.desc()
                        ).limit(initial_limit)
                except Exception as e:
                    # L2 distance fonksiyonu çalışmazsa en son parçaları getirelim
                    logger.error(f"Vector search failed: {str(e)}, falling back to recent chunks")
                    similarity_query = base_query.order_by(
                        KnowledgeBase.created_at.desc()
                    ).limit(initial_limit)
                
                result = await session.execute(similarity_query)
                top_chunks = result.scalars().all()
                
                if not top_chunks:
                    return []
                
                # Step 2: Collect file_ids and chunk_indexes from top results
                # We'll use a dictionary to group by file_id for efficient lookup
                final_chunks = []
                seen_chunk_keys = set()  # To track chunks we've already included
                chunks_to_fetch = []
                
                # Group top chunks by file_id and track their chunk_indexes
                file_chunks = {}
                for chunk in top_chunks:
                    if chunk.file_id not in file_chunks:
                        file_chunks[chunk.file_id] = []
                    file_chunks[chunk.file_id].append(chunk.chunk_index)
                    # Add the original chunks to our final list
                    final_chunks.append(chunk)
                    seen_chunk_keys.add(f"{chunk.file_id}_{chunk.chunk_index}")
                
                # Step 3: For each file, get adjacent chunks (2 before and 2 after each chunk)
                for file_id, chunk_indexes in file_chunks.items():
                    # For each chunk index, calculate the adjacent indexes we want
                    for chunk_index in chunk_indexes:
                        # Get up to 2 chunks before and 2 chunks after
                        for adj_index in range(chunk_index - 2, chunk_index + 3):
                            # Skip the original chunk (we already added it)
                            if adj_index == chunk_index:
                                continue
                            
                            # Skip if we've already seen this chunk
                            chunk_key = f"{file_id}_{adj_index}"
                            if chunk_key in seen_chunk_keys:
                                continue
                            
                            # Only fetch valid indexes (non-negative)
                            if adj_index >= 0:
                                chunks_to_fetch.append((file_id, adj_index))
                                seen_chunk_keys.add(chunk_key)
                
                # Step 4: Fetch all adjacent chunks in a single efficient query if there are any to fetch
                if chunks_to_fetch:
                    # Build conditions for OR query to get all adjacent chunks at once
                    conditions = []
                    for file_id, adj_index in chunks_to_fetch:
                        conditions.append(
                            and_(
                                KnowledgeBase.file_id == file_id,
                                KnowledgeBase.chunk_index == adj_index
                            )
                        )
                    
                    # Only execute the query if we have conditions
                    if conditions:
                        adjacent_query = select(KnowledgeBase).where(
                            or_(*conditions),
                            KnowledgeBase.organization_id == organization_id,
                            KnowledgeBase.is_knowledge_base == True
                        )
                        
                        result = await session.execute(adjacent_query)
                        adjacent_chunks = result.scalars().all()
                        
                        # Add all adjacent chunks to our final list
                        final_chunks.extend(adjacent_chunks)
                
                # Step 5: Sort the final chunks for better readability
                # Sort first by file_id, then by chunk_index to maintain proper document order
                final_chunks.sort(key=lambda x: (x.file_id, x.chunk_index))
                
                # Ensure we don't exceed the original limit
                if len(final_chunks) > limit * 2:  # Allow slightly more for context
                    logger.info(f"Trimming chunk count from {len(final_chunks)} to {limit * 2}")
                    final_chunks = final_chunks[:limit * 2]
                
                return final_chunks
                
        except Exception as e:
            error_msg = f"Error finding relevant knowledge base chunks: {str(e)}"
            error_details = str(e)
            
            # Try to extract more detailed error information for PostgreSQL/pgvector errors
            if hasattr(e, '__cause__') and e.__cause__ is not None:
                error_details = f"{error_details} | Cause: {str(e.__cause__)}"
                if hasattr(e.__cause__, '__cause__') and e.__cause__.__cause__ is not None:
                    error_details = f"{error_details} | Root cause: {str(e.__cause__.__cause__)}"
            
            # PostgreSQL error codes
            if 'asyncpg.exceptions.DataError' in error_details:
                logger.error(f"PostgreSQL vector operation failed: {error_details}")
            elif 'asyncpg.exceptions' in error_details:
                logger.error(f"PostgreSQL error: {error_details}")
                
            logger.error(error_msg)
            logger.error(f"Detailed error info: {error_details}")
            
            # Return empty list regardless of error to avoid blocking the document analysis process
            return []
            
    # CRUD Operations for DocumentAnalysis
    
    async def create_analysis_record(
        self,
        document_id: int,
        organization_id: int,
        user_id: int
    ) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document ID"""
        try:
            async with async_session_maker() as session:
                # Check if there's already an active analysis for this document
                stmt = select(DocumentAnalysis).where(
                    (DocumentAnalysis.document_id == document_id) &
                    (DocumentAnalysis.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]))
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    logger.info(f"Analysis already in progress for document {document_id}, returning existing record")
                    return existing
                
                # Create new analysis record
                analysis = DocumentAnalysis(
                    document_id=document_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    total_chunks_analyzed=0
                )
                
                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)
                
                logger.info(f"Created analysis record {analysis.id} for document {document_id}")
                return analysis
                
        except Exception as e:
            logger.error(f"Error creating analysis record: {str(e)}")
            raise
            
    async def create_analysis_record_by_fp(
        self,
        document_fp: str,
        organization_id: int,
        user_id: int
    ) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document fingerprint (fp)"""
        try:
            # First get the document ID from the fingerprint
            document_service = DocumentService()
            document = await document_service.get_document_by_fp(organization_id, document_fp)
            
            if not document:
                raise ValueError(f"Document with fingerprint {document_fp} not found")
                
            document_id = document.id
            
            # Check if there's already an active analysis for this document
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    (DocumentAnalysis.document_id == document_id) &
                    (DocumentAnalysis.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]))
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    logger.info(f"Analysis already in progress for document {document_fp} (ID: {document_id}), returning existing record")
                    return existing
                
                # Create new analysis record
                analysis = DocumentAnalysis(
                    document_id=document_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    total_chunks_analyzed=0
                )
                
                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)
                
                logger.info(f"Created analysis record {analysis.id} for document {document_fp} (ID: {document_id})")
                return analysis
                
        except Exception as e:
            logger.error(f"Error creating analysis record by FP: {str(e)}")
            raise
            
    async def get_analysis_by_id(self, analysis_id: int) -> Optional[DocumentAnalysis]:
        """Get analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Error getting analysis by ID: {str(e)}")
            raise
            
    async def get_analysis_by_fp(self, analysis_fp: str) -> Optional[DocumentAnalysis]:
        """Get analysis record by fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Error getting analysis by fingerprint: {str(e)}")
            raise
    
    async def get_analysis_by_document_id(self, document_id: int) -> List[DocumentAnalysis]:
        """Get all analysis records for a document, ordered by creation date"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.document_id == document_id
                ).order_by(desc(DocumentAnalysis.created_at))
                
                result = await session.execute(stmt)
                return result.scalars().all()
                
        except Exception as e:
            logger.error(f"Error getting analyses for document: {str(e)}")
            raise
    
    async def get_analyses_for_organization(
        self,
        organization_id: int,
        page: int = 1,
        per_page: int = 10,
        status_filter: Optional[str] = None
    ) -> Tuple[List[DocumentAnalysis], int, int]:
        """Get analyses for a specific organization with pagination"""
        try:
            async with async_session_maker() as session:
                # Start with base condition on organization ID
                conditions = [DocumentAnalysis.organization_id == organization_id]
                
                # Add status filter if provided
                if status_filter:
                    try:
                        status_enum = AnalysisStatus[status_filter.upper()]
                        conditions.append(DocumentAnalysis.status == status_enum)
                    except KeyError:
                        logger.warning(f"Invalid status filter: {status_filter}")
                
                # Get total count
                count_stmt = select(func.count()).select_from(DocumentAnalysis).where(
                    *conditions
                )
                result = await session.execute(count_stmt)
                total_count = result.scalar_one()
                
                # Calculate pagination
                total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page
                
                # Get paginated data
                stmt = select(DocumentAnalysis).where(
                    *conditions
                ).order_by(
                    desc(DocumentAnalysis.created_at)
                ).offset(offset).limit(per_page)
                
                result = await session.execute(stmt)
                analyses = result.scalars().all()
                
                return analyses, total_count, total_pages
                
        except Exception as e:
            logger.error(f"Error getting analyses for organization: {str(e)}")
            raise
                
    async def get_analyses_for_user(
        self,
        user_organization_ids: List[int],
        page: int = 1, 
        per_page: int = 10,
        status_filter: Optional[str] = None
    ) -> Tuple[List[DocumentAnalysis], int, int]:
        """Get analyses across all organizations a user has access to"""
        try:
            if not user_organization_ids:
                return [], 0, 1
                
            async with async_session_maker() as session:
                # Base condition: analyses from any of the user's organizations
                conditions = [DocumentAnalysis.organization_id.in_(user_organization_ids)]
                
                # Add status filter if provided
                if status_filter:
                    try:
                        status_enum = AnalysisStatus[status_filter.upper()]
                        conditions.append(DocumentAnalysis.status == status_enum)
                    except KeyError:
                        logger.warning(f"Invalid status filter: {status_filter}")
                
                # Get total count
                count_stmt = select(func.count()).select_from(DocumentAnalysis).where(
                    *conditions
                )
                result = await session.execute(count_stmt)
                total_count = result.scalar_one()
                
                # Calculate pagination
                total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page
                
                # Get paginated data
                stmt = select(DocumentAnalysis).where(
                    *conditions
                ).order_by(
                    desc(DocumentAnalysis.created_at)
                ).offset(offset).limit(per_page)
                
                result = await session.execute(stmt)
                analyses = result.scalars().all()
                
                return analyses, total_count, total_pages
                
        except Exception as e:
            logger.error(f"Error getting analyses for organization: {str(e)}")
            raise
    
# Method removed to fix duplicate function definition
# The get_analysis_by_fp method is already defined at line 375
    
    async def update_analysis_status(
        self,
        analysis_id: int,
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found: {analysis_id}")
                    return
                
                analysis.status = status
                
                # If status is FAILED and error_message is provided, save it
                if status == AnalysisStatus.FAILED and error_message:
                    analysis.error_message = error_message
                
                # If completed, set the completed_at timestamp
                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()
                    
                await session.commit()
                logger.info(f"Updated analysis {analysis_id} status to {status.value}")
                
        except Exception as e:
            logger.error(f"Error updating analysis status: {str(e)}")
            raise
    
    async def update_analysis_status_by_fp(
        self,
        analysis_fp: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of an analysis record by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found with fp: {analysis_fp}")
                    return
                
                analysis.status = status
                
                # If status is FAILED and error_message is provided, save it
                if status == AnalysisStatus.FAILED and error_message:
                    analysis.error_message = error_message
                
                # If completed, set the completed_at timestamp
                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()
                    
                await session.commit()
                logger.info(f"Updated analysis with fp {analysis_fp} status to {status.value}")
                
        except Exception as e:
            logger.error(f"Error updating analysis status: {str(e)}")
            raise
    
    async def update_original_content(
        self,
        analysis_id: int,
        content: str
    ) -> None:
        """Update the original content of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found: {analysis_id}")
                    return
                
                analysis.original_content = content
                await session.commit()
                logger.info(f"Updated original content for analysis {analysis_id}")
                
        except Exception as e:
            logger.error(f"Error updating original content: {str(e)}")
            raise
            
    async def update_original_content_by_fp(
        self,
        analysis_fp: str,
        content: str
    ) -> None:
        """Update the original content of an analysis record by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found with fp: {analysis_fp}")
                    return
                
                analysis.original_content = content
                await session.commit()
                logger.info(f"Updated original content for analysis with fp {analysis_fp}")
                
        except Exception as e:
            logger.error(f"Error updating original content: {str(e)}")
            raise
    
    async def update_analysis_record(
        self,
        analysis_id: int,
        analysis_data: Dict[str, Any],
        suggested_changes: Dict[str, Any],
        status: AnalysisStatus = AnalysisStatus.COMPLETED
    ) -> None:
        """Update an analysis record with analysis results by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found: {analysis_id}")
                    return
                
                # Update the analysis record with the results
                analysis.analysis = analysis_data.get("analysis") or ""  # Ensure analysis is never None
                analysis.key_points = analysis_data.get("key_points", [])
                analysis.conflicts = analysis_data.get("conflicts", [])
                analysis.recommendations = analysis_data.get("recommendations", [])
                analysis.chunk_analyses = analysis_data.get("chunk_analyses", [])
                analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                analysis.processing_time_seconds = analysis_data.get("processing_time_seconds", 0)
                analysis.suggested_changes = suggested_changes
                analysis.status = status
                
                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()
                
                await session.commit()
                logger.info(f"Updated analysis record {analysis_id} with results")
                
        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise

    async def update_analysis_record_by_fp(
        self,
        analysis_fp: str,
        analysis_data: Dict[str, Any],
        suggested_changes: Dict[str, Any],
        status: AnalysisStatus = AnalysisStatus.COMPLETED
    ) -> None:
        """Update an analysis record with analysis results by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    logger.error(f"Analysis record not found with fp: {analysis_fp}")
                    return
                
                # Update the analysis record with the results
                analysis.analysis = analysis_data.get("analysis") or ""  # Ensure analysis is never None
                analysis.key_points = analysis_data.get("key_points", [])
                analysis.conflicts = analysis_data.get("conflicts", [])
                analysis.recommendations = analysis_data.get("recommendations", [])
                analysis.chunk_analyses = analysis_data.get("chunk_analyses", [])
                analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                analysis.processing_time_seconds = analysis_data.get("processing_time_seconds", 0)
                analysis.suggested_changes = suggested_changes
                analysis.status = status
                
                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()
                
                await session.commit()
                logger.info(f"Updated analysis record with fp {analysis_fp} with results")
                
        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise
    
    async def _generate_suggested_changes(
        self,
        original_content: str,
        analysis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate suggested changes for the document based on analysis results.
        This method creates a structured representation of suggested changes
        that can be used to show a "diff" between original and suggested content.
        """
        try:
            # Extract all recommendations from the analysis results
            all_recommendations = []
            for chunk in analysis_results:
                if "recommendations" in chunk and isinstance(chunk["recommendations"], list):
                    for rec in chunk["recommendations"]:
                        if isinstance(rec, str) and rec not in all_recommendations:
                            all_recommendations.append(rec)
            
            # Structure the suggestions
            suggestions = {
                "recommendations": all_recommendations,
                "sections": []
            }
            
            # Go through each chunk and find parts that need changes
            for chunk in analysis_results:
                chunk_index = chunk.get("chunk_index", 0)
                chunk_content = chunk.get("chunk_content", "")
                conflicts = chunk.get("conflicts", [])
                
                if conflicts:
                    # For each conflict, add a suggested change section
                    for conflict in conflicts:
                        suggestions["sections"].append({
                            "chunk_index": chunk_index,
                            "original_text": chunk_content,
                            "conflict": conflict,
                            "suggested_improvement": "This section has conflicts that should be addressed."
                        })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggested changes: {str(e)}")
            return {"recommendations": [], "sections": []}
    
    def _combine_chunk_analyses(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple chunk analyses into a single comprehensive analysis
        
        Args:
            chunk_analyses: List of analysis results for individual chunks
            
        Returns:
            Combined analysis as a dictionary
        """
        if not chunk_analyses:
            return {
                "analysis": "No chunks available for analysis",
                "key_points": [],
                "conflicts": [],
                "recommendations": []
            }
            
        # Initialize the combined result
        combined = {
            "analysis": "",
            "key_points": [],
            "conflicts": [],
            "recommendations": [],
            "chunk_analyses": []  # Store individual chunk analyses for reference
        }
        
        # Collect all unique items from each analysis
        all_key_points = set()
        all_conflicts = set()
        all_recommendations = set()
        
        # Process each chunk analysis
        for chunk in chunk_analyses:
            # Store a reference to the individual chunk analysis
            chunk_summary = {
                "chunk_index": chunk.get("chunk_index"),
                "analysis": chunk.get("analysis", ""),
            }
            combined["chunk_analyses"].append(chunk_summary)
            
            # Collect key points (convert to tuple for set handling)
            if "key_points" in chunk and isinstance(chunk["key_points"], list):
                for point in chunk["key_points"]:
                    if isinstance(point, str):
                        all_key_points.add(point)
            
            # Collect conflicts
            if "conflicts" in chunk and isinstance(chunk["conflicts"], list):
                for conflict in chunk["conflicts"]:
                    if isinstance(conflict, str):
                        all_conflicts.add(conflict)
            
            # Collect recommendations
            if "recommendations" in chunk and isinstance(chunk["recommendations"], list):
                for rec in chunk["recommendations"]:
                    if isinstance(rec, str):
                        all_recommendations.add(rec)
        
        # Build a comprehensive analysis summary
        analysis_parts = []
        for chunk in chunk_analyses:
            if "analysis" in chunk and chunk["analysis"]:
                analysis_parts.append(chunk["analysis"])
        
        # Join the analysis parts with logical transitions
        if analysis_parts:
            combined["analysis"] = "\n\n".join(analysis_parts)
        else:
            combined["analysis"] = "No analysis available for the document."
        
        # Convert sets back to lists
        combined["key_points"] = list(all_key_points)
        combined["conflicts"] = list(all_conflicts)
        combined["recommendations"] = list(all_recommendations)
        
        return combined