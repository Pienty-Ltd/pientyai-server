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
            
            # Process document chunk by chunk and also analyze the whole document
            logger.info(f"Processing document in two phases: 1) Chunk-by-chunk analysis 2) Full document analysis")
            
            # Create a structure to store all analyses
            all_analyses = []
            chunk_kb_mapping = {}  # Store mapping between document chunks and their relevant KB chunks
            
            # SINGLE PHASE: Complete document analysis in a unified approach
            logger.info(f"Running unified document analysis process")
            
            # First collect relevant knowledge base chunks for the full document
            logger.info(f"Finding relevant KB chunks for the entire document")
            
            # Find the most relevant chunks from the knowledge base for the full document
            full_document_content = original_content
            
            # Find most relevant knowledge base chunks for the entire document
            document_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                organization_id=organization_id,
                query_text=full_document_content,
                current_document_id=document_id,  # Exclude the current document from search
                limit=max_relevant_chunks * 2  # Get more chunks for a full document
            )
            
            # Convert the chunks to a list of KB chunk info dictionaries with rich metadata
            consolidated_kb_chunks = []
            from app.database.models.db_models import File
            
            for kb_chunk in document_relevant_kb:
                # Get file name for better context
                file_name = "Unknown"
                try:
                    async with async_session_maker() as session:
                        file_query = select(File).where(File.id == kb_chunk.file_id)
                        file_result = await session.execute(file_query)
                        file = file_result.scalar_one_or_none()
                        if file:
                            file_name = file.filename
                except Exception as e:
                    logger.error(f"Error retrieving filename: {str(e)}")
                
                # Add rich metadata to provide better context
                chunk_info = {
                    "document_name": file_name,
                    "document_id": kb_chunk.file_id,
                    "chunk_index": kb_chunk.chunk_index,
                    "similarity_score": getattr(kb_chunk, 'similarity_score', 0),
                    "content": kb_chunk.content,
                    "meta_info": kb_chunk.meta_info if kb_chunk.meta_info else {}
                }
                consolidated_kb_chunks.append(chunk_info)
            
            logger.info(f"Found {len(consolidated_kb_chunks)} relevant knowledge base chunks for analysis")
            
            # Perform a single comprehensive analysis with the full document against all relevant KB chunks
            chunk_start_time = datetime.now()
            
            # Send only ONE request to OpenAI for the entire document analysis
            logger.info(f"Analyzing full document with {len(consolidated_kb_chunks)} relevant KB chunks in a single request")
            
            # Get the full document analysis with all relevant knowledge base chunks
            full_doc_analysis = await self.openai_service.analyze_document(
                document_chunk=full_document_content,
                knowledge_base_chunks=consolidated_kb_chunks
            )
            
            # Add metadata to the full document analysis
            full_doc_analysis["is_full_document_analysis"] = True
            full_doc_analysis["processing_time_seconds"] = (datetime.now() - chunk_start_time).total_seconds()
            
            # Include both individual chunk analyses and the full document analysis
            all_analyses.append(full_doc_analysis)
            
            # Use the full document analysis as our main chunk_analysis
            chunk_analysis = full_doc_analysis
            
            # Check if the analysis contains an error
            if "error" in chunk_analysis:
                error_msg = f"Document analysis failed: {chunk_analysis.get('error', 'Unknown error')}"
                logger.error(error_msg)
                await self.update_analysis_status(analysis_record.id, AnalysisStatus.FAILED, error_msg)
                raise ValueError(error_msg)
            
            # Add metadata about the full document to the analysis result
            chunk_analysis["processing_time_seconds"] = (datetime.now() - chunk_start_time).total_seconds()
            
            # Create a single analysis result
            analysis_results = [chunk_analysis]
            
            # Create the final combined analysis
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Combine the analyses into a single result
            combined_analysis = self._combine_chunk_analyses(analysis_results)
            combined_analysis["document_id"] = document_id
            combined_analysis["organization_id"] = organization_id
            combined_analysis["total_chunks_analyzed"] = total_chunks
            combined_analysis["processing_time_seconds"] = total_duration
            
            # Generate detailed suggested changes based on the analysis
            # Extract more information from the combined analysis to enhance the suggested changes
            conflicts = combined_analysis.get("conflicts", [])
            recommendations = combined_analysis.get("recommendations", [])
            
            # Add specific details to the analysis_results to generate better suggested changes
            enhanced_results = []
            for result in analysis_results:
                # Copy the original result
                enhanced_result = result.copy()
                
                # Add the original document content for context
                enhanced_result["original_content"] = original_content
                
                # Add overall conflicts and recommendations for better context
                if "conflicts" not in enhanced_result or not enhanced_result["conflicts"]:
                    enhanced_result["conflicts"] = conflicts
                
                if "recommendations" not in enhanced_result or not enhanced_result["recommendations"]:
                    enhanced_result["recommendations"] = recommendations
                
                enhanced_results.append(enhanced_result)
            
            # Generate more detailed suggested changes with the enhanced context
            suggested_changes = await self._generate_suggested_changes(
                original_content=original_content,
                analysis_results=enhanced_results
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
                    # Native pgvector similarity search with cosine distance
                    if query_embedding and len(query_embedding) > 0 and len(query_embedding[0]) > 0:
                        # Using PostgreSQL's native vector operations with direct SQL execution for better performance
                        logger.debug(f"Running pgvector search with embedding length: {len(query_embedding[0])}")
                        
                        # Create base query conditions
                        conditions = [
                            "organization_id = :org_id",
                            "is_knowledge_base = TRUE"
                        ]
                        
                        # Add condition to exclude current document if specified
                        if current_document_id is not None:
                            conditions.append("file_id != :doc_id")
                            
                        # Join all conditions
                        where_clause = " AND ".join(conditions)
                        
                        # Build complete SQL query using native pgvector operator
                        # 1 - (embedding <=> :query_vector) gives us similarity score between 0-1
                        sql_query = f"""
                            SELECT 
                                kb.*,
                                1 - (embedding <=> :query_vector) AS similarity_score
                            FROM 
                                knowledge_base kb
                            WHERE 
                                {where_clause}
                            ORDER BY 
                                similarity_score DESC
                            LIMIT :limit
                        """
                        
                        # Parameters for the query
                        params = {
                            "org_id": organization_id,
                            "query_vector": query_embedding[0],
                            "limit": initial_limit
                        }
                        
                        if current_document_id is not None:
                            params["doc_id"] = current_document_id
                        
                        # Execute raw SQL query
                        from sqlalchemy import text
                        async with async_session_maker() as session:
                            result = await session.execute(text(sql_query), params)
                            rows = result.fetchall()
                            
                            # Convert row results to KnowledgeBase objects with similarity scores
                            top_chunks = []
                            for row in rows:
                                # Get KnowledgeBase object from row mapping
                                chunk = row[0]  # The first column contains the entire KnowledgeBase object
                                similarity = float(row[1])  # The last column is our similarity score
                                
                                # Attach similarity score to the object
                                setattr(chunk, 'similarity_score', similarity)
                                top_chunks.append(chunk)
                                logger.debug(f"Retrieved chunk {chunk.chunk_index} with similarity {similarity:.4f}")
                            
                            # Now get adjacent chunks (one before and one after each top chunk)
                            # This helps avoid truncated information due to chunk boundaries
                            all_chunks = list(top_chunks)  # Make a copy of top chunks
                            
                            # Group chunks by file_id for efficient retrieval of adjacent chunks
                            file_chunks = {}
                            for chunk in top_chunks:
                                if chunk.file_id not in file_chunks:
                                    file_chunks[chunk.file_id] = []
                                file_chunks[chunk.file_id].append(chunk.chunk_index)
                            
                            # For each file, get adjacent chunks
                            for file_id, chunk_indexes in file_chunks.items():
                                # Get ALL adjacent chunks in a single query for efficiency
                                adjacent_indexes = set()
                                for chunk_idx in chunk_indexes:
                                    # Get one chunk before and one after
                                    if chunk_idx > 0:  # Ensure we don't go below 0
                                        adjacent_indexes.add(chunk_idx - 1)
                                    adjacent_indexes.add(chunk_idx + 1)  # Next chunk
                                
                                # Remove indexes we already have
                                adjacent_indexes = adjacent_indexes - set(chunk_indexes)
                                
                                if adjacent_indexes:
                                    # Build a query to get all adjacent chunks at once
                                    adjacent_conditions = []
                                    for adj_idx in adjacent_indexes:
                                        adjacent_conditions.append(
                                            f"(file_id = {file_id} AND chunk_index = {adj_idx})"
                                        )
                                        
                                    adjacent_query = f"""
                                        SELECT kb.*
                                        FROM knowledge_base kb
                                        WHERE organization_id = :org_id AND ({" OR ".join(adjacent_conditions)})
                                    """
                                    
                                    adj_result = await session.execute(text(adjacent_query), {"org_id": organization_id})
                                    adjacent_chunks = adj_result.scalars().all()
                                    
                                    # Add adjacent chunks to our results
                                    all_chunks.extend(adjacent_chunks)
                            
                            # Sort by file_id and chunk_index to maintain document order
                            all_chunks.sort(key=lambda x: (x.file_id, x.chunk_index))
                            
                            return all_chunks
                    else:
                        # If embedding couldn't be created or is invalid,
                        # Fall back to getting the most recently added knowledge base chunks
                        logger.warning("Invalid embedding for query, falling back to recent chunks")
                        similarity_query = base_query.order_by(
                            KnowledgeBase.created_at.desc()
                        ).limit(initial_limit)
                        
                        # Execute the ORM query
                        result = await session.execute(similarity_query)
                        top_chunks = result.scalars().all()
                except Exception as e:
                    # If vector search fails, get the most recent chunks
                    logger.error(f"Vector search failed: {str(e)}, falling back to recent chunks")
                    similarity_query = base_query.order_by(
                        KnowledgeBase.created_at.desc()
                    ).limit(initial_limit)
                    
                    # Execute the ORM query
                    result = await session.execute(similarity_query)
                    top_chunks = result.scalars().all()
                
                # We've optimized chunk retrieval in the main vector search block above
                
                if not top_chunks:
                    return []
                
                # Return the chunks from fallback methods (these use the ORM query which doesn't hit the optimized path)
                # In the future, these could be optimized as well if needed
                return top_chunks
                
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
        Generate suggested changes for the document based on the AI's analysis results.
        This method uses the AI's recommendations and conflicts directly instead of 
        trying to parse them with regex patterns.
        
        The suggestions come directly from the AI model output, preserving the 
        detailed analysis and making the changes more accurate.
        """
        try:
            # Initialize the suggestions structure
            suggestions = {
                "recommendations": [],
                "sections": [],
                "policy_conflicts": [],
                "legal_compliance_issues": []
            }
            
            # Extract all original AI recommendations directly without filtering
            for chunk in analysis_results:
                if "recommendations" in chunk and isinstance(chunk["recommendations"], list):
                    # Add all recommendations directly from AI
                    for rec in chunk["recommendations"]:
                        if isinstance(rec, str) and rec not in suggestions["recommendations"]:
                            suggestions["recommendations"].append(rec)
                
                # Extract conflicts directly from AI output
                if "conflicts" in chunk and isinstance(chunk["conflicts"], list):
                    for conflict in chunk["conflicts"]:
                        if not isinstance(conflict, str):
                            continue
                            
                        # Try to identify section reference if present in the conflict description
                        import re
                        section_ref = "Unspecified section"
                        section_pattern = r'([Ss][Ee][Cc][Tt][Ii][Oo][Nn]|[Mm][Aa][Dd][Dd][Ee]|[Bb][Öö][Ll][Üü][Mm])\s+(\d+(\.\d+)*)'
                        section_match = re.search(section_pattern, conflict)
                        
                        if section_match:
                            section_type = section_match.group(1)  # "Section", "Madde", etc.
                            section_num = section_match.group(2)   # "4.2", "3", etc.
                            section_ref = f"{section_type} {section_num}"
                        
                        # Determine if this is related to policy or legal requirements
                        # This is just classification, not changing the actual content
                        is_policy_conflict = any(term in conflict.lower() for term in [
                            "politika", "policy", "şirket", "company", "standart",
                            "standard", "prosedür", "procedure", "kural", "rule"
                        ])
                        
                        is_legal_issue = any(term in conflict.lower() for term in [
                            "kanun", "yasa", "law", "regulation", "mevzuat", 
                            "legal", "yasal", "anayasa", "constitution", "tüzük"
                        ])
                        
                        # Look for a corresponding recommendation for this conflict
                        matching_recommendation = None
                        for rec in suggestions["recommendations"]:
                            # If the recommendation mentions similar terms as the conflict,
                            # it's likely related
                            if any(term in rec.lower() and term in conflict.lower() 
                                  for term in ["faiz", "interest", "ödeme", "payment", 
                                              "gün", "day", "mahkeme", "court", "ceza", 
                                              "penalty", "süre", "period", "vade", "term"]):
                                matching_recommendation = rec
                                break
                        
                        # Add to sections with direct conflict and recommendation
                        change_section = {
                            "chunk_index": chunk.get("chunk_index", 0),
                            "section": section_ref,
                            "original_text": chunk.get("chunk_content", ""),
                            "conflict": conflict,
                            # Use the AI's recommendation directly if we found a match
                            "suggested_improvement": matching_recommendation or conflict,
                            "is_policy_conflict": is_policy_conflict,
                            "is_legal_issue": is_legal_issue
                        }
                        
                        suggestions["sections"].append(change_section)
                        
                        # Add to appropriate categorized lists
                        if is_policy_conflict and conflict not in suggestions["policy_conflicts"]:
                            suggestions["policy_conflicts"].append(conflict)
                        if is_legal_issue and conflict not in suggestions["legal_compliance_issues"]:
                            suggestions["legal_compliance_issues"].append(conflict)
            
            # If no sections were found but we have recommendations, create sections from them
            if not suggestions["sections"] and suggestions["recommendations"]:
                for i, rec in enumerate(suggestions["recommendations"]):
                    # Classify recommendation
                    is_policy = any(term in rec.lower() for term in [
                        "politika", "policy", "şirket", "company", "standart"
                    ])
                    is_legal = any(term in rec.lower() for term in [
                        "kanun", "yasa", "law", "regulation", "mevzuat"
                    ])
                    
                    # Create a section from the recommendation itself
                    suggestions["sections"].append({
                        "chunk_index": 0,
                        "section": f"Recommendation {i+1}",
                        "original_text": "",  # No specific text identified
                        "conflict": rec,      # Use recommendation as conflict too
                        "suggested_improvement": rec,
                        "is_policy_conflict": is_policy,
                        "is_legal_issue": is_legal
                    })
                    
                    # Add to appropriate categorized lists if not already there
                    if is_policy and rec not in suggestions["policy_conflicts"]:
                        suggestions["policy_conflicts"].append(rec)
                    if is_legal and rec not in suggestions["legal_compliance_issues"]:
                        suggestions["legal_compliance_issues"].append(rec)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggested changes: {str(e)}")
            return {"recommendations": [], "sections": [], "policy_conflicts": [], "legal_compliance_issues": []}
    
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
        
        # Keep track of chunks we've already processed to avoid duplicates
        processed_chunk_indices = set()
        
        # Process each chunk analysis
        for chunk in chunk_analyses:
            chunk_index = chunk.get("chunk_index")
            
            # Skip if this chunk index has already been processed (avoid duplicates)
            if chunk_index is not None and chunk_index in processed_chunk_indices:
                continue
            
            # Mark this chunk as processed
            if chunk_index is not None:
                processed_chunk_indices.add(chunk_index)
            
            # Store a reference to the individual chunk analysis
            chunk_summary = {
                "chunk_index": chunk_index,
                "original_text": chunk.get("original_text", ""),
                "analysis": chunk.get("analysis", ""),
                "conflict": chunk.get("conflict", ""),
                "suggested_improvement": chunk.get("suggested_improvement", "")
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
            # Skip duplicates when building analysis parts too
            chunk_index = chunk.get("chunk_index")
            if chunk_index is not None and chunk_index in processed_chunk_indices:
                processed_chunk_indices.remove(chunk_index)  # Remove so we process each index exactly once
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