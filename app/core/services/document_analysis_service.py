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
                document_id=document_id,
                organization_id=organization_id,
                user_id=user_id)

            start_time = datetime.now()

            # Fetch the document to be analyzed
            document = await self.document_service.get_document_by_id(
                organization_id, document_id)
            if not document:
                error_msg = f"Document not found: {document_id}"
                await self.update_analysis_status(analysis_record.id,
                                                  AnalysisStatus.FAILED,
                                                  error_msg)
                raise ValueError(error_msg)

            if document.status != "completed":
                error_msg = f"Document processing is not complete. Current status: {document.status}"
                await self.update_analysis_status(analysis_record.id,
                                                  AnalysisStatus.FAILED,
                                                  error_msg)
                raise ValueError(error_msg)

            # Update the status to processing
            await self.update_analysis_status(analysis_record.id,
                                              AnalysisStatus.PROCESSING)

            # Get the original document content - we'll join all chunks to represent the original content
            document_chunks = await self.get_document_chunks(document_id)

            if not document_chunks:
                error_msg = f"No chunks found for document {document_id}"
                await self.update_analysis_status(analysis_record.id,
                                                  AnalysisStatus.FAILED,
                                                  error_msg)
                raise ValueError(error_msg)

            # Store the original content (concatenated chunks)
            original_content = "\n\n".join(
                [chunk.content for chunk in document_chunks])
            await self.update_original_content(analysis_record.id,
                                               original_content)

            # Set up tracking variables for analysis
            total_chunks = len(document_chunks)

            logger.info(
                f"Processing {total_chunks} chunks for document {document_id}")

            # Consolidated approach: Gather knowledge base chunks for all document chunks,
            # then perform a single analysis
            logger.info(
                f"Running consolidated document analysis process with comprehensive knowledge base collection"
            )

            # Collect relevant knowledge base chunks for EACH document chunk to ensure extensive coverage
            logger.info(
                f"Finding relevant KB chunks for all {total_chunks} chunks in the document"
            )

            full_document_content = original_content

            # Set to track unique knowledge base chunks across all chunk searches
            all_relevant_kb_chunks = set()  # Using a set to avoid duplicates

            # Process each document chunk to find relevant KB chunks
            for chunk_idx, doc_chunk in enumerate(document_chunks):
                chunk_content = doc_chunk.content
                logger.info(
                    f"Finding relevant KB chunks for chunk {chunk_idx+1}/{total_chunks}"
                )

                # Find the most relevant chunks from the knowledge base for this specific chunk
                # Use the document chunk's existing embedding if available
                doc_embedding = None
                if hasattr(doc_chunk,
                           'embedding') and doc_chunk.embedding is not None:
                    # Use the embedding as is
                    doc_embedding = doc_chunk.embedding

                chunk_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                    organization_id=organization_id,
                    query_text=chunk_content,
                    current_document_id=document_id,  # Exclude the current document from search
                    limit=max_relevant_chunks,  # Get context specific to this chunk
                    query_embedding=doc_embedding)

                # Add all relevant chunks to our global set, tracking by file_id and chunk_index
                # Also add sufficient context by including adjacent chunks when available
                for kb_chunk in chunk_relevant_kb:
                    # Add the current chunk
                    chunk_key = f"{kb_chunk.file_id}_{kb_chunk.chunk_index}"
                    all_relevant_kb_chunks.add((chunk_key, kb_chunk))

                    # Try to get and add all surrounding chunks too for better context
                    try:
                        # Get adjacent chunks for better context (2 before and 2 after)
                        # This helps with cases where information is split across chunks
                        async with async_session_maker() as session:
                            # Get up to 2 chunks before
                            before_query = select(KnowledgeBase).where(
                                KnowledgeBase.file_id == kb_chunk.file_id,
                                KnowledgeBase.is_knowledge_base == True,
                                KnowledgeBase.chunk_index
                                < kb_chunk.chunk_index).order_by(
                                    desc(KnowledgeBase.chunk_index)).limit(2)

                            before_result = await session.execute(before_query)
                            before_chunks = before_result.scalars().all()

                            # Get up to 2 chunks after
                            after_query = select(KnowledgeBase).where(
                                KnowledgeBase.file_id == kb_chunk.file_id,
                                KnowledgeBase.is_knowledge_base == True,
                                KnowledgeBase.chunk_index
                                > kb_chunk.chunk_index).order_by(
                                    KnowledgeBase.chunk_index).limit(2)

                            after_result = await session.execute(after_query)
                            after_chunks = after_result.scalars().all()

                            # Add all adjacent chunks to our set
                            for adj_chunk in before_chunks + after_chunks:
                                adj_key = f"{adj_chunk.file_id}_{adj_chunk.chunk_index}"
                                all_relevant_kb_chunks.add(
                                    (adj_key, adj_chunk))
                    except Exception as e:
                        logger.warning(
                            f"Error getting adjacent chunks: {str(e)}")
                        # Continue with the process even if getting adjacent chunks fails

            logger.info(
                f"Collected {len(all_relevant_kb_chunks)} unique KB chunks across all document chunks"
            )

            # Also search using the full document text to catch overall context
            logger.info(
                f"Finding additional KB chunks using the full document content"
            )

            # Prepare embedding for the full document if any document chunks have embeddings
            full_doc_embedding = None
            # Simple approach: Use the first document chunk's embedding if available
            if document_chunks and hasattr(
                    document_chunks[0],
                    'embedding') and document_chunks[0].embedding is not None:
                # Use the embedding as is
                full_doc_embedding = document_chunks[0].embedding
                logger.info(
                    f"Using existing embedding from document chunks for full document analysis"
                )

            full_doc_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                organization_id=organization_id,
                query_text=full_document_content,
                current_document_id=document_id,
                limit=max_relevant_chunks * 2,  # Get more chunks for the full document
                query_embedding=full_doc_embedding)

            # Add these to our set of unique chunks as well along with their adjacent context
            for kb_chunk in full_doc_relevant_kb:
                # Add the main chunk
                chunk_key = f"{kb_chunk.file_id}_{kb_chunk.chunk_index}"
                all_relevant_kb_chunks.add((chunk_key, kb_chunk))

                # Try to get and add all surrounding chunks too for better context
                try:
                    # Get adjacent chunks for better context (2 before and 2 after)
                    # This helps with cases where information is split across chunks
                    async with async_session_maker() as session:
                        # Get up to 2 chunks before
                        before_query = select(KnowledgeBase).where(
                            KnowledgeBase.file_id == kb_chunk.file_id,
                            KnowledgeBase.is_knowledge_base == True,
                            KnowledgeBase.chunk_index
                            < kb_chunk.chunk_index).order_by(
                                desc(KnowledgeBase.chunk_index)).limit(2)

                        before_result = await session.execute(before_query)
                        before_chunks = before_result.scalars().all()

                        # Get up to 2 chunks after
                        after_query = select(KnowledgeBase).where(
                            KnowledgeBase.file_id == kb_chunk.file_id,
                            KnowledgeBase.is_knowledge_base == True,
                            KnowledgeBase.chunk_index
                            > kb_chunk.chunk_index).order_by(
                                KnowledgeBase.chunk_index).limit(2)

                        after_result = await session.execute(after_query)
                        after_chunks = after_result.scalars().all()

                        # Add all adjacent chunks to our set
                        for adj_chunk in before_chunks + after_chunks:
                            adj_key = f"{adj_chunk.file_id}_{adj_chunk.chunk_index}"
                            all_relevant_kb_chunks.add((adj_key, adj_chunk))
                except Exception as e:
                    logger.warning(f"Error getting adjacent chunks: {str(e)}")
                    # Continue with the process even if getting adjacent chunks fails

            logger.info(
                f"Total unique KB chunks after full document search: {len(all_relevant_kb_chunks)}"
            )

            # Extract the actual chunk objects from the set of tuples
            document_relevant_kb = [
                chunk_tuple[1] for chunk_tuple in all_relevant_kb_chunks
            ]

            # Convert the chunks to a list of KB chunk info dictionaries with rich metadata
            # Include ALL chunks regardless of similarity score, but sort by similarity for better results
            consolidated_kb_chunks = []

            # Create a list of chunks with their similarity scores
            chunks_with_scores = []
            for kb_chunk in document_relevant_kb:
                # Get file name for better context
                file_name = "Unknown"
                try:
                    async with async_session_maker() as session:
                        file_query = select(File).where(
                            File.id == kb_chunk.file_id)
                        file_result = await session.execute(file_query)
                        file = file_result.scalar_one_or_none()
                        if file:
                            file_name = file.filename
                except Exception as e:
                    logger.error(f"Error retrieving filename: {str(e)}")

                # Get similarity score (default to 0 if not set)
                similarity_score = getattr(kb_chunk, 'similarity_score', 0)

                # Log the similarity score for debugging
                logger.debug(
                    f"KB Chunk {kb_chunk.file_id}_{kb_chunk.chunk_index} has similarity: {similarity_score}"
                )

                # Create a rich metadata object and add to the list
                chunk_info = {
                    "document_name": file_name,
                    "document_id": kb_chunk.file_id,
                    "chunk_index": kb_chunk.chunk_index,
                    "similarity_score": similarity_score,
                    "content": kb_chunk.content,
                    "meta_info":
                    kb_chunk.meta_info if kb_chunk.meta_info else {}
                }
                chunks_with_scores.append((chunk_info, similarity_score))
                logger.debug(
                    f"Added KB chunk with similarity score: {similarity_score}"
                )

            # Sort chunks by similarity score (highest first)
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Extract just the chunk info objects, preserving order
            consolidated_kb_chunks = [
                chunk_data[0] for chunk_data in chunks_with_scores
            ]

            logger.info(
                f"Prepared {len(consolidated_kb_chunks)} KB chunks for analysis"
            )

            # Now perform a single consolidated analysis with all document content and KB chunks
            chunk_start_time = datetime.now()
            
            # Perform a single analysis with all document content and knowledge base chunks
            logger.info(f"Performing consolidated analysis with all document content and KB chunks")
            consolidated_analysis = await self.openai_service.analyze_document(
                document_chunk=full_document_content,
                knowledge_base_chunks=consolidated_kb_chunks)

            # Add metadata to the analysis
            consolidated_analysis["is_consolidated_analysis"] = True
            consolidated_analysis["processing_time_seconds"] = (
                datetime.now() - chunk_start_time).total_seconds()
            consolidated_analysis["total_kb_chunks"] = len(consolidated_kb_chunks)
            consolidated_analysis["total_document_chunks"] = total_chunks

            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds()

            # Collect the final response
            analysis_response = {
                "analysis": consolidated_analysis,
                "document": {
                    "id": document_id,
                    "name": document.filename if hasattr(document, 'filename') else "Unknown",
                    "total_chunks": total_chunks
                },
                "meta": {
                    "total_processing_time_seconds": total_processing_time,
                    "total_kb_chunks_analyzed": len(consolidated_kb_chunks),
                    "analysis_fp": analysis_record.fp,
                    "created_at": analysis_record.created_at.isoformat() if hasattr(analysis_record, 'created_at') else None,
                }
            }

            # Update the analysis record
            await self.update_analysis_record(
                analysis_id=analysis_record.id,
                analysis_data=analysis_response,
                status=AnalysisStatus.COMPLETED)

            logger.info(
                f"Document analysis completed in {total_processing_time:.2f} seconds"
            )

            return analysis_response

        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}", exc_info=True)
            
            # Try to update the analysis record if it was created
            if 'analysis_record' in locals():
                try:
                    await self.update_analysis_status(
                        analysis_record.id, AnalysisStatus.FAILED, str(e))
                except Exception as update_error:
                    logger.error(
                        f"Error updating analysis status: {str(update_error)}")

            # Re-raise the exception
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
                return chunks
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise

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
            # Ensure limit is within reasonable bounds
            limit = max(3, min(10, limit))

            # Get or create embedding for the query text
            if not query_embedding:
                query_embedding = await self.openai_service.get_embedding(
                    query_text)

            # Use vector similarity search to find the most relevant chunks
            # Only search in knowledge base chunks (is_knowledge_base=True)
            async with async_session_maker() as session:
                # Build the query
                conditions = [
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_knowledge_base == True
                ]

                # Exclude the current document if provided
                if current_document_id:
                    conditions.append(KnowledgeBase.file_id != current_document_id)

                # For PostgreSQL with pgvector, use the <=> operator for cosine distance
                # 'embedding::vector <=> :query_embedding::vector AS similarity_score'
                # More relevant chunks have lower cosine distance
                from sqlalchemy import text
                embedding_string = json.dumps(query_embedding)

                sql_query = text(f"""
                    WITH similarity_results AS (
                        SELECT
                            *,
                            embedding::vector <=> :query_embedding::vector AS similarity_score
                        FROM knowledge_base
                        WHERE organization_id = :organization_id
                          AND is_knowledge_base = true
                          {f"AND file_id != :current_document_id" if current_document_id else ""}
                        ORDER BY similarity_score ASC
                        LIMIT :limit
                    )
                    SELECT * FROM similarity_results
                """)

                result = await session.execute(
                    sql_query,
                    {
                        "query_embedding": embedding_string,
                        "organization_id": organization_id,
                        "current_document_id": current_document_id,
                        "limit": limit
                    })

                # Convert the result to KnowledgeBase objects
                relevant_chunks = []
                for row in result:
                    chunk = KnowledgeBase(
                        id=row.id,
                        fp=row.fp,
                        organization_id=row.organization_id,
                        file_id=row.file_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        embedding=row.embedding,
                        meta_info=row.meta_info,
                        is_knowledge_base=row.is_knowledge_base,
                        created_at=row.created_at,
                    )
                    # Add the similarity score as an attribute
                    setattr(chunk, 'similarity_score', row.similarity_score)
                    relevant_chunks.append(chunk)

            return relevant_chunks
        except Exception as e:
            logger.error(
                f"Error finding relevant knowledge base chunks: {str(e)}",
                exc_info=True)
            # Return empty list in case of error to allow the process to continue
            return []

    async def create_analysis_record(self, document_id: int,
                                     organization_id: int,
                                     user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document ID"""
        try:
            async with async_session_maker() as session:
                # Get the document
                document_query = select(File).where(File.id == document_id)
                document_result = await session.execute(document_query)
                document = document_result.scalar_one_or_none()

                if not document:
                    raise ValueError(f"Document not found: {document_id}")

                # Create a new analysis record
                import uuid
                new_analysis = DocumentAnalysis(
                    fp=f"analysis_{uuid.uuid4().hex[:20]}",
                    organization_id=organization_id,
                    document_id=document_id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    original_content=None,  # Will be updated later
                    # Store document fingerprint in results temporarily
                    results={"document_fp": document.fp}
                )

                session.add(new_analysis)
                await session.commit()
                await session.refresh(new_analysis)

                return new_analysis
        except Exception as e:
            logger.error(f"Error creating analysis record: {str(e)}")
            raise

    async def create_analysis_record_by_fp(self, document_fp: str,
                                           organization_id: int,
                                           user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                # Get the document
                document_query = select(File).where(File.fp == document_fp)
                document_result = await session.execute(document_query)
                document = document_result.scalar_one_or_none()

                if not document:
                    raise ValueError(f"Document not found: {document_fp}")

                # Create a new analysis record
                import uuid
                new_analysis = DocumentAnalysis(
                    fp=f"analysis_{uuid.uuid4().hex[:20]}",
                    organization_id=organization_id,
                    document_id=document.id,
                    user_id=user_id,
                    status=AnalysisStatus.PENDING,
                    original_content=None,  # Will be updated later
                    # Store document fingerprint in results temporarily
                    results={"document_fp": document_fp}
                )

                session.add(new_analysis)
                await session.commit()
                await session.refresh(new_analysis)

                return new_analysis
        except Exception as e:
            logger.error(f"Error creating analysis record: {str(e)}")
            raise

    async def get_analysis_by_id(
            self, analysis_id: int) -> Optional[DocumentAnalysis]:
        """Get analysis record by ID"""
        try:
            async with async_session_maker() as session:
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting analysis by ID: {str(e)}")
            return None

    async def get_analysis_by_fp(
            self, analysis_fp: str) -> Optional[DocumentAnalysis]:
        """Get analysis record by fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting analysis by FP: {str(e)}")
            return None

    async def get_analysis_by_document_id(
            self, document_id: int) -> List[DocumentAnalysis]:
        """Get all analysis records for a document, ordered by creation date"""
        try:
            async with async_session_maker() as session:
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.document_id == document_id).order_by(
                        desc(DocumentAnalysis.created_at))
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting analysis by document ID: {str(e)}")
            return []

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
                # Build conditions for the query
                conditions = [DocumentAnalysis.organization_id == organization_id]
                
                # Add status filter if provided
                if status_filter:
                    conditions.append(DocumentAnalysis.status == status_filter)
                
                # Count total records
                count_query = select(func.count()).select_from(DocumentAnalysis).where(
                    and_(*conditions))
                count_result = await session.execute(count_query)
                total_count = count_result.scalar_one()
                
                # Calculate pagination
                total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page
                
                # Get paginated records
                query = select(DocumentAnalysis).where(
                    and_(*conditions)).order_by(
                        desc(DocumentAnalysis.created_at)).offset(offset).limit(per_page)
                result = await session.execute(query)
                analyses = result.scalars().all()
                
                return analyses, total_count, total_pages
        except Exception as e:
            logger.error(f"Error getting analyses for organization: {str(e)}")
            return [], 0, 1

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
                # Build conditions for the query
                conditions = [DocumentAnalysis.organization_id.in_(user_organization_ids)]
                
                # Add status filter if provided
                if status_filter:
                    conditions.append(DocumentAnalysis.status == status_filter)
                
                # Count total records
                count_query = select(func.count()).select_from(DocumentAnalysis).where(
                    and_(*conditions))
                count_result = await session.execute(count_query)
                total_count = count_result.scalar_one()
                
                # Calculate pagination
                total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page
                
                # Get paginated records
                query = select(DocumentAnalysis).where(
                    and_(*conditions)).order_by(
                        desc(DocumentAnalysis.created_at)).offset(offset).limit(per_page)
                result = await session.execute(query)
                analyses = result.scalars().all()
                
                return analyses, total_count, total_pages
        except Exception as e:
            logger.error(f"Error getting analyses for user: {str(e)}")
            return [], 0, 1

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
                    analysis.error_message = error_message
                    analysis.updated_at = datetime.now()
                    await session.commit()
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
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(query)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    analysis.status = status
                    analysis.error_message = error_message
                    analysis.updated_at = datetime.now()
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating analysis status by FP: {str(e)}")
            raise

    async def update_original_content(self, analysis_id: int,
                                      content: str) -> None:
        """Update the original content of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                analysis = await session.get(DocumentAnalysis, analysis_id)
                if analysis:
                    analysis.original_content = content
                    analysis.updated_at = datetime.now()
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating original content: {str(e)}")
            raise

    async def update_original_content_by_fp(self, analysis_fp: str,
                                            content: str) -> None:
        """Update the original content of an analysis record by fingerprint"""
        try:
            async with async_session_maker() as session:
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(query)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    analysis.original_content = content
                    analysis.updated_at = datetime.now()
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating original content by FP: {str(e)}")
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
                    analysis.results = analysis_data
                    analysis.status = status
                    analysis.updated_at = datetime.now()
                    await session.commit()
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
                query = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(query)
                analysis = result.scalar_one_or_none()
                
                if analysis:
                    analysis.results = analysis_data
                    analysis.status = status
                    analysis.updated_at = datetime.now()
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating analysis record by FP: {str(e)}")
            raise