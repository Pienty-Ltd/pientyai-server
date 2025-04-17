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
from app.database.models.db_models import (KnowledgeBase, File,
                                           DocumentAnalysis, AnalysisStatus,
                                           User)
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
            max_relevant_chunks: int = 5) -> Dict[str, Any]:
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
            logger.info(
                f"Starting document analysis for doc {document_id} in organization {organization_id}"
            )

            # Create a record to track the analysis
            analysis_record = await self.create_analysis_record(
                document_id, organization_id, user_id)

            if not analysis_record:
                raise Exception(
                    f"Failed to create analysis record for document {document_id}"
                )

            # Update status to processing
            await self.update_analysis_status(analysis_record.id,
                                             AnalysisStatus.PROCESSING)

            # Get document chunks
            document_chunks = await self.get_document_chunks(document_id)

            if not document_chunks:
                await self.update_analysis_status(
                    analysis_record.id, AnalysisStatus.FAILED,
                    "Document has no content to analyze.")
                return {"error": "Document has no content to analyze."}

            # Get the original document text to store in the analysis record
            original_content = "\n\n".join(
                chunk.content for chunk in document_chunks if chunk.content)

            # Update original content in the analysis record
            await self.update_original_content(analysis_record.id,
                                              original_content)

            # Process each chunk against knowledge base
            chunk_analyses = []
            total_processing_time = 0.0

            for i, chunk in enumerate(document_chunks):
                if not chunk.content or not chunk.content.strip():
                    continue

                logger.debug(
                    f"Processing chunk {i+1}/{len(document_chunks)} (document {document_id})"
                )

                # Find related knowledge base chunks
                relevant_chunks = await self.find_relevant_knowledge_base_chunks(
                    organization_id,
                    chunk.content,
                    current_document_id=document_id,
                    limit=max_relevant_chunks)

                # Skip if no relevant chunks found - nothing to compare against
                if not relevant_chunks:
                    logger.warning(
                        f"No relevant knowledge base chunks found for chunk {i+1}"
                    )
                    continue

                # Analyze this chunk against knowledge base
                start_time = datetime.now()
                chunk_result = await self.openai_service.analyze_document_chunk_with_git_diff(
                    chunk.content, relevant_chunks)
                end_time = datetime.now()
                processing_time = (end_time -
                                  start_time).total_seconds()
                total_processing_time += processing_time

                # Add chunk index and processing time to the chunk results
                chunk_result["chunk_index"] = i
                chunk_result["processing_time_seconds"] = processing_time
                chunk_result["original_text"] = chunk.content

                # Process and store results
                chunk_analyses.append(chunk_result)

            # Calculate the git-like diff changes
            diff_changes = ""
            for chunk in chunk_analyses:
                if "diff_changes" in chunk and chunk["diff_changes"]:
                    diff_changes += chunk["diff_changes"] + "\n\n"

            diff_changes = diff_changes.strip()

            # Create the final analysis result
            analysis_result = {
                "document_id": document_id,
                "organization_id": organization_id,
                "diff_changes": diff_changes,
                "total_chunks_analyzed": len(chunk_analyses),
                "processing_time_seconds": total_processing_time,
                "chunk_analyses": chunk_analyses,
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }

            # Update the analysis record with the results
            await self.update_analysis_record(
                analysis_record.id, 
                analysis_result,
                AnalysisStatus.COMPLETED
            )

            logger.info(
                f"Completed document analysis for doc {document_id} in organization {organization_id}"
            )

            return analysis_result

        except Exception as e:
            logger.error(
                f"Error analyzing document {document_id}: {str(e)}")
            # If we created an analysis record, update its status to failed
            if 'analysis_record' in locals() and analysis_record:
                await self.update_analysis_status(
                    analysis_record.id, AnalysisStatus.FAILED, str(e))
            raise

    async def get_document_chunks(self,
                                  document_id: int) -> List[KnowledgeBase]:
        """Get all chunks for a specific document from knowledge base"""
        try:
            async with async_session_maker() as session:
                query = select(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id).order_by(
                        KnowledgeBase.chunk_index)

                result = await session.execute(query)
                chunks = result.scalars().all()
                return list(chunks)
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []

    async def find_relevant_knowledge_base_chunks(
            self,
            organization_id: int,
            query_text: str,
            current_document_id: Optional[int] = None,
            limit: int = 5,
            query_embedding: Optional[List[float]] = None
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
            query_embedding: Optional pre-generated embedding vector for the query text.
                            If provided, skips embedding generation to improve performance.
            
        Returns:
            List of relevant knowledge base chunks including adjacent chunks for better context
        """
        try:
            # Sanitize limit
            if limit < 3:
                limit = 3
            elif limit > 10:
                limit = 10

            # Get the embedding for the query text
            if query_embedding is None:
                query_embedding = await self.openai_service.get_embedding(
                    query_text)

            if not query_embedding:
                logger.warning("Failed to generate embedding for query text")
                return []

            async with async_session_maker() as session:
                # We'll use a raw SQL query for vector similarity search
                # This allows us to efficiently filter by organization and exclude current document

                # Start with a filter for knowledge base chunks only in this organization
                filter_conditions = [
                    "kb.organization_id = :org_id",
                    "kb.is_knowledge_base = TRUE",  # Only use knowledge base chunks
                ]

                # Exclude the current document if provided
                if current_document_id is not None:
                    filter_conditions.append("kb.file_id != :doc_id")

                # Combine all filter conditions
                filter_sql = " AND ".join(filter_conditions)

                # Build the SQL query for vector similarity search
                sql_query = f"""
                WITH similarity_results AS (
                    SELECT kb.*, 
                           kb.embedding <=> :query_embedding AS distance,
                           row_number() OVER (PARTITION BY kb.file_id ORDER BY kb.embedding <=> :query_embedding) as doc_rank
                    FROM knowledge_base kb
                    WHERE {filter_sql}
                    ORDER BY kb.embedding <=> :query_embedding
                    LIMIT :limit
                )
                SELECT * FROM similarity_results
                ORDER BY distance;
                """

                # Prepare parameters
                params = {
                    "query_embedding": query_embedding,
                    "org_id": organization_id,
                    "limit": limit
                }

                # Add document ID parameter if needed
                if current_document_id is not None:
                    params["doc_id"] = current_document_id

                # Execute the query
                result = await session.execute(sql_query, params)
                relevant_chunks = result.fetchall()

                # Convert raw results to KnowledgeBase objects
                relevant_kb_chunks = []

                # If we found any relevant chunks, get their adjacent chunks for context
                if relevant_chunks:
                    # Extract document IDs and chunk indices from relevant chunks
                    document_ids = set()
                    chunk_indices_by_doc = {}

                    for row in relevant_chunks:
                        doc_id = row.document_id
                        chunk_idx = row.chunk_index
                        document_ids.add(doc_id)

                        if doc_id not in chunk_indices_by_doc:
                            chunk_indices_by_doc[doc_id] = set()
                        chunk_indices_by_doc[doc_id].add(chunk_idx)

                        # Also include adjacent chunks (2 before and 2 after)
                        for adj_idx in range(chunk_idx - 2, chunk_idx + 3):
                            if adj_idx >= 0:  # Ensure non-negative index
                                chunk_indices_by_doc[doc_id].add(adj_idx)

                    # Fetch all the chunks in a single query
                    # This is more efficient than making multiple queries
                    adjacent_query = f"""
                    SELECT * FROM knowledge_base 
                    WHERE file_id = ANY(:doc_ids) AND is_knowledge_base = TRUE
                    """
                    result = await session.execute(
                        adjacent_query, {"doc_ids": list(document_ids)})
                    all_chunks = result.fetchall()

                    # Filter to get only the chunks we want (primary + adjacent)
                    for chunk in all_chunks:
                        doc_id = chunk.file_id
                        if doc_id in chunk_indices_by_doc and chunk.chunk_index in chunk_indices_by_doc[
                                doc_id]:
                            # Create KnowledgeBase object
                            kb_chunk = KnowledgeBase(
                                id=chunk.id,
                                organization_id=chunk.organization_id,
                                file_id=chunk.file_id,
                                chunk_index=chunk.chunk_index,
                                content=chunk.content,
                                is_knowledge_base=chunk.is_knowledge_base,
                                embedding=chunk.embedding
                                if hasattr(chunk, 'embedding') else None,
                                meta_info=chunk.meta_info
                                if hasattr(chunk, 'meta_info') else None,
                                created_at=chunk.created_at,
                                updated_at=chunk.updated_at,
                            )
                            relevant_kb_chunks.append(kb_chunk)

                # Sort the chunks by file ID and chunk index to preserve the original structure
                relevant_kb_chunks.sort(
                    key=lambda x: (x.file_id, x.chunk_index))

                return relevant_kb_chunks

        except Exception as e:
            logger.error(
                f"Error finding relevant knowledge base chunks: {str(e)}")
            return []

    async def create_analysis_record(self, document_id: int,
                                     organization_id: int,
                                     user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document ID"""
        try:
            async with async_session_maker() as session:
                # Get the document details
                stmt = select(File).where(File.id == document_id)
                result = await session.execute(stmt)
                file = result.scalar_one_or_none()

                if not file:
                    raise Exception(
                        f"Document with ID {document_id} not found")

                # Create new analysis record
                analysis = DocumentAnalysis(
                    document_id=document_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    created_at=datetime.now(),
                )

                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)

                return analysis

        except Exception as e:
            logger.error(f"Error creating analysis record: {str(e)}")
            raise

    async def create_analysis_record_by_fp(self, document_fp: str,
                                           organization_id: int,
                                           user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                # Get the document details
                stmt = select(File).where(File.fp == document_fp)
                result = await session.execute(stmt)
                file = result.scalar_one_or_none()

                if not file:
                    raise Exception(
                        f"Document with fp {document_fp} not found")

                # Create new analysis record
                analysis = DocumentAnalysis(
                    document_id=file.id,
                    organization_id=organization_id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    created_at=datetime.now(),
                )

                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)

                return analysis

        except Exception as e:
            logger.error(f"Error creating analysis record by fp: {str(e)}")
            raise

    async def get_analysis_by_id(
            self, analysis_id: int) -> Optional[DocumentAnalysis]:
        """Get analysis record by ID"""
        async with async_session_maker() as session:
            stmt = select(DocumentAnalysis).where(
                DocumentAnalysis.id == analysis_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_analysis_by_fp(
            self, analysis_fp: str) -> Optional[DocumentAnalysis]:
        """Get analysis record by fingerprint (fp)"""
        async with async_session_maker() as session:
            stmt = select(DocumentAnalysis).where(
                DocumentAnalysis.fp == analysis_fp)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_analysis_by_document_id(
            self, document_id: int) -> List[DocumentAnalysis]:
        """Get all analysis records for a document, ordered by creation date"""
        async with async_session_maker() as session:
            stmt = select(DocumentAnalysis).where(
                DocumentAnalysis.document_id == document_id).order_by(
                    desc(DocumentAnalysis.created_at))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_analyses_for_organization(
        self,
        organization_id: int,
        page: int = 1,
        per_page: int = 10,
        status_filter: Optional[str] = None
    ) -> Tuple[List[DocumentAnalysis], int, int]:
        """Get analyses for a specific organization with pagination"""
        async with async_session_maker() as session:
            # Base query filters
            filters = [DocumentAnalysis.organization_id == organization_id]
            
            # Add status filter if provided
            if status_filter:
                filters.append(DocumentAnalysis.status == status_filter)
            
            # Count total matching analyses
            count_stmt = select(func.count()).select_from(DocumentAnalysis).where(
                *filters)
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()
            
            # Calculate total pages
            total_pages = math.ceil(total_count / per_page)
            
            # Get the paginated results
            stmt = select(DocumentAnalysis).where(*filters).order_by(
                desc(DocumentAnalysis.created_at)).offset((page - 1) * per_page).limit(per_page)
            result = await session.execute(stmt)
            analyses = list(result.scalars().all())
            
            return analyses, total_count, total_pages

    async def get_analyses_for_user(
        self,
        user_organization_ids: List[int],
        page: int = 1,
        per_page: int = 10,
        status_filter: Optional[str] = None
    ) -> Tuple[List[DocumentAnalysis], int, int]:
        """Get analyses across all organizations a user has access to"""
        async with async_session_maker() as session:
            # Base query filter - analyses from organizations the user has access to
            filters = [DocumentAnalysis.organization_id.in_(user_organization_ids)]
            
            # Add status filter if provided
            if status_filter:
                filters.append(DocumentAnalysis.status == status_filter)
            
            # Count total matching analyses
            count_stmt = select(func.count()).select_from(DocumentAnalysis).where(
                *filters)
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()
            
            # Calculate total pages
            total_pages = math.ceil(total_count / per_page)
            
            # Get the paginated results
            stmt = select(DocumentAnalysis).where(*filters).order_by(
                desc(DocumentAnalysis.created_at)).offset((page - 1) * per_page).limit(per_page)
            result = await session.execute(stmt)
            analyses = list(result.scalars().all())
            
            return analyses, total_count, total_pages

    async def update_analysis_status(
            self,
            analysis_id: int,
            status: AnalysisStatus,
            error_message: Optional[str] = None) -> None:
        """Update the status of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                analysis = await session.get(DocumentAnalysis, analysis_id)
                
                if analysis:
                    analysis.status = status
                    
                    # Set completed_at timestamp if status is completed or failed
                    if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                        analysis.completed_at = datetime.now()
                    
                    # Store error message if provided
                    if error_message and status == AnalysisStatus.FAILED:
                        # Store error message directly in error_message column
                        analysis.error_message = error_message
                        
                    await session.commit()
                    logger.info(
                        f"Updated analysis {analysis_id} status to {status.value}")
                else:
                    logger.warning(
                        f"Analysis record with ID {analysis_id} not found for status update"
                    )

        except Exception as e:
            logger.error(f"Error updating analysis status: {str(e)}")
            raise

    async def update_analysis_status_by_fp(
            self,
            analysis_fp: str,
            status: AnalysisStatus,
            error_message: Optional[str] = None) -> None:
        """Update the status of an analysis record by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    analysis.status = status
                    
                    # Set completed_at timestamp if status is completed or failed
                    if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                        analysis.completed_at = datetime.now()
                    
                    # Store error message if provided
                    if error_message and status == AnalysisStatus.FAILED:
                        # Store error message directly in error_message column
                        analysis.error_message = error_message
                        
                    await session.commit()
                    logger.info(
                        f"Updated analysis {analysis_fp} status to {status.value}")
                else:
                    logger.warning(
                        f"Analysis record with fingerprint {analysis_fp} not found for status update"
                    )

        except Exception as e:
            logger.error(f"Error updating analysis status by fp: {str(e)}")
            raise

    async def update_original_content(self, analysis_id: int,
                                      content: str) -> None:
        """Update the original content of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                analysis = await session.get(DocumentAnalysis, analysis_id)
                
                if analysis:
                    analysis.original_content = content
                    await session.commit()
                    logger.debug(
                        f"Updated original content for analysis {analysis_id}")
                else:
                    logger.warning(
                        f"Analysis record with ID {analysis_id} not found for content update"
                    )

        except Exception as e:
            logger.error(f"Error updating original content: {str(e)}")
            raise

    async def update_original_content_by_fp(self, analysis_fp: str,
                                            content: str) -> None:
        """Update the original content of an analysis record by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    analysis.original_content = content
                    await session.commit()
                    logger.debug(
                        f"Updated original content for analysis {analysis_fp}")
                else:
                    logger.warning(
                        f"Analysis record with fp {analysis_fp} not found for content update"
                    )

        except Exception as e:
            logger.error(f"Error updating original content by fp: {str(e)}")
            raise

    async def update_analysis_record(
            self,
            analysis_id: int,
            analysis_data: Dict[str, Any],
            status: AnalysisStatus = AnalysisStatus.COMPLETED) -> None:
        """Update an analysis record with analysis results by ID"""
        try:
            async with async_session_maker() as session:
                analysis = await session.get(DocumentAnalysis, analysis_id)
                
                if analysis:
                    # Update status
                    analysis.status = status
                    analysis.completed_at = datetime.now()
                    
                    # Update diff_changes
                    if "diff_changes" in analysis_data:
                        analysis.diff_changes = analysis_data["diff_changes"]
                    
                    # Update fields with analysis data directly
                    if "total_chunks_analyzed" in analysis_data:
                        analysis.total_chunks_analyzed = analysis_data["total_chunks_analyzed"]
                    
                    if "processing_time_seconds" in analysis_data:
                        analysis.processing_time_seconds = analysis_data["processing_time_seconds"]
                        
                    if "chunk_analyses" in analysis_data:
                        analysis.chunk_analyses = analysis_data["chunk_analyses"]
                    
                    await session.commit()
                    logger.info(f"Updated analysis record with ID {analysis_id}")
                else:
                    logger.warning(
                        f"Analysis record with ID {analysis_id} not found for update"
                    )

        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise

    async def update_analysis_record_by_fp(
            self,
            analysis_fp: str,
            analysis_data: Dict[str, Any],
            status: AnalysisStatus = AnalysisStatus.COMPLETED) -> None:
        """Update an analysis record with analysis results by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    # Update status
                    analysis.status = status
                    analysis.completed_at = datetime.now()
                    
                    # Update diff_changes
                    if "diff_changes" in analysis_data:
                        analysis.diff_changes = analysis_data["diff_changes"]
                    
                    # Update fields with analysis data directly
                    if "total_chunks_analyzed" in analysis_data:
                        analysis.total_chunks_analyzed = analysis_data["total_chunks_analyzed"]
                    
                    if "processing_time_seconds" in analysis_data:
                        analysis.processing_time_seconds = analysis_data["processing_time_seconds"]
                        
                    if "chunk_analyses" in analysis_data:
                        analysis.chunk_analyses = analysis_data["chunk_analyses"]
                    
                    await session.commit()
                    logger.info(
                        f"Updated analysis record with fp {analysis_fp} with git-like diff changes"
                    )
                else:
                    logger.warning(
                        f"Analysis record with fp {analysis_fp} not found for update"
                    )

        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise