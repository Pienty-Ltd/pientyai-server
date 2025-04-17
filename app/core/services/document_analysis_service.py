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
            analysis_results = []
            total_chunks = len(document_chunks)

            logger.info(
                f"Processing {total_chunks} chunks for document {document_id}")

            # Process document chunk by chunk and also analyze the whole document
            logger.info(
                f"Processing document in two phases: 1) Chunk-by-chunk analysis 2) Full document analysis"
            )

            # Create a structure to store all analyses
            all_analyses = []
            chunk_kb_mapping = {
            }  # Store mapping between document chunks and their relevant KB chunks

            # SINGLE PHASE: Complete document analysis in a unified approach
            logger.info(
                f"Running unified document analysis process with comprehensive knowledge base collection"
            )

            # First collect relevant knowledge base chunks for EACH document chunk to ensure extensive coverage
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
                # Ensure we flatten the embedding array if it's a nested list
                doc_embedding = None
                if hasattr(doc_chunk,
                           'embedding') and doc_chunk.embedding is not None:
                    # Simply use the embedding as is - pgvector expects a flat list of floats
                    # which is already stored correctly in the database
                    doc_embedding = doc_chunk.embedding

                chunk_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                    organization_id=organization_id,
                    query_text=chunk_content,
                    current_document_id=
                    document_id,  # Exclude the current document from search
                    limit=
                    max_relevant_chunks,  # Get context specific to this chunk
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
                # Simply use the embedding as is - pgvector expects a flat list of floats
                # which is already stored correctly in the database
                full_doc_embedding = document_chunks[0].embedding
                logger.info(
                    f"Using existing embedding from document chunks for full document analysis"
                )

            full_doc_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                organization_id=organization_id,
                query_text=full_document_content,
                current_document_id=document_id,
                limit=max_relevant_chunks *
                2,  # Get more chunks for the full document
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
            from app.database.models.db_models import File

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

            # Take the top 2 chunks with highest similarity scores
            top_limit = min(2, len(chunks_with_scores))
            top_chunks = [
                chunk_data[0] for chunk_data in chunks_with_scores[:top_limit]
            ]

            # Also include adjacent chunks (one before and one after) for each top chunk
            # First, group by document_id
            doc_chunks = {}
            for chunk in top_chunks:
                doc_id = chunk["document_id"]
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk["chunk_index"])

            # Identify adjacent chunk indexes
            adjacent_chunks = []
            for doc_id, chunk_indexes in doc_chunks.items():
                for idx in chunk_indexes:
                    # For each top chunk, find chunks with index +1 and -1
                    for adj_idx in [idx - 1, idx + 1]:
                        # Find this adjacent chunk in our original chunk_with_scores list
                        for chunk_info, _ in chunks_with_scores:
                            if chunk_info[
                                    "document_id"] == doc_id and chunk_info[
                                        "chunk_index"] == adj_idx:
                                adjacent_chunks.append(chunk_info)
                                break

            # Combine top chunks and adjacent chunks
            consolidated_kb_chunks = top_chunks + adjacent_chunks

            logger.info(
                f"Selected {len(top_chunks)} top chunks and {len(adjacent_chunks)} adjacent chunks for analysis"
            )

            # Perform a single comprehensive analysis with the full document against all relevant KB chunks
            chunk_start_time = datetime.now()

            # Send only ONE request to OpenAI for the entire document analysis
            logger.info(
                f"Analyzing full document with {len(consolidated_kb_chunks)} relevant KB chunks in a single request"
            )

            # Get the full document analysis with all relevant knowledge base chunks
            full_doc_analysis = await self.openai_service.analyze_document(
                document_chunk=full_document_content,
                knowledge_base_chunks=consolidated_kb_chunks)

            # Add metadata to the full document analysis
            full_doc_analysis["is_full_document_analysis"] = True
            full_doc_analysis["processing_time_seconds"] = (
                datetime.now() - chunk_start_time).total_seconds()

            # Include both individual chunk analyses and the full document analysis
            all_analyses.append(full_doc_analysis)

            # Use the full document analysis as our main chunk_analysis
            chunk_analysis = full_doc_analysis

            # Check if the analysis contains an error
            if "error" in chunk_analysis:
                error_msg = f"Document analysis failed: {chunk_analysis.get('error', 'Unknown error')}"
                logger.error(error_msg)
                await self.update_analysis_status(analysis_record.id,
                                                  AnalysisStatus.FAILED,
                                                  error_msg)
                raise ValueError(error_msg)

            # Add metadata about the full document to the analysis result
            chunk_analysis["processing_time_seconds"] = (
                datetime.now() - chunk_start_time).total_seconds()

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

            # Artık conflict ve recommendation alanları yerine sadece diff_changes formatı kullanıyoruz
            # Aşağıdaki satırları kaldırıyoruz ve doğrudan diff_changes'e odaklanıyoruz

            # Add specific details to the analysis_results to generate better suggested changes
            enhanced_results = []
            for result in analysis_results:
                # Copy the original result
                enhanced_result = result.copy()

                # Add the original document content for context
                enhanced_result["original_content"] = original_content

                # Artık conflicts ve recommendations alanları kullanılmıyor, 
                # sadece diff_changes formatına odaklanıyoruz

                enhanced_results.append(enhanced_result)

            # Generate more detailed suggested changes with the enhanced context
            suggested_changes = await self._generate_suggested_changes(
                original_content=original_content,
                analysis_results=enhanced_results)

            # Update the record with the results
            await self.update_analysis_record(
                analysis_id=analysis_record.id,
                analysis_data=combined_analysis,
                suggested_changes=suggested_changes,
                status=AnalysisStatus.COMPLETED)

            logger.info(
                f"Completed document analysis for doc {document_id} (Duration: {total_duration}s)"
            )

            # Get the updated record to return
            updated_record = await self.get_analysis_by_id(analysis_record.id)

            # Get document and organization FPs
            document = None
            organization = None
            try:
                async with async_session_maker() as session:
                    # Get document FP
                    from app.database.models.db_models import File
                    query = select(File).where(
                        File.id == updated_record.document_id)
                    result = await session.execute(query)
                    document = result.scalar_one_or_none()

                    # Get organization FP
                    from app.database.models.db_models import Organization
                    query = select(Organization).where(
                        Organization.id == updated_record.organization_id)
                    result = await session.execute(query)
                    organization = result.scalar_one_or_none()
            except Exception as e:
                logger.error(
                    f"Error getting document or organization details: {e}",
                    exc_info=True)

            return {
                "id": updated_record.id,
                "fp": updated_record.fp,
                "document_fp": document.fp if document else "",
                "organization_fp": organization.fp if organization else "",
                "diff_changes": updated_record.diff_changes,
                "total_chunks_analyzed": updated_record.total_chunks_analyzed,
                "processing_time_seconds": float(updated_record.processing_time_seconds)
                if updated_record.processing_time_seconds else 0.0,
                "chunk_analyses":
                updated_record.chunk_analyses,
                "status":
                updated_record.status.value,
                "created_at":
                updated_record.created_at,
                "completed_at":
                updated_record.completed_at
            }

        except Exception as e:
            error_msg = f"Error in document analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # If there's a record, update it to failed status with the error message
            if 'analysis_record' in locals() and analysis_record:
                await self.update_analysis_status(analysis_record.id,
                                                  AnalysisStatus.FAILED,
                                                  error_msg)
            raise

    async def get_document_chunks(self,
                                  document_id: int) -> List[KnowledgeBase]:
        """Get all chunks for a specific document from knowledge base"""
        try:
            async with async_session_maker() as session:
                stmt = select(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id).order_by(
                        KnowledgeBase.chunk_index)

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
        # Ensure limit is within allowed range
        if limit < 3:
            limit = 3
        elif limit > 10:
            limit = 10
        try:
            # Use provided embedding if available, otherwise generate a new one
            if query_embedding is None:
                # Generate embedding for the query text
                try:
                    query_embedding_result = await self.openai_service.create_embeddings(
                        [query_text])
                    if not query_embedding_result or len(
                            query_embedding_result) == 0:
                        logger.error(
                            "Failed to generate embedding for query text")
                        return []

                    # Handle case where embedding is a nested list [[values...]]
                    if isinstance(query_embedding_result[0], list) and len(
                            query_embedding_result[0]) > 0 and isinstance(
                                query_embedding_result[0][0], list):
                        query_embedding = query_embedding_result[0][
                            0]  # Extract from nested list
                        logger.debug(
                            f"Extracted embedding from nested list structure")
                    else:
                        query_embedding = query_embedding_result[0]

                except Exception as e:
                    error_details = str(e)
                    if hasattr(e, '__cause__') and e.__cause__ is not None:
                        error_details = f"{error_details} | Cause: {str(e.__cause__)}"

                    logger.error(
                        f"Error generating embedding for query text: {error_details}"
                    )
                    # Return empty list so document analysis can continue without vector search
                    return []
            else:
                # Use the provided embedding, but ensure it's in the right format
                # Handle all possible embedding formats and normalize them to a flat list
                if isinstance(query_embedding, list):
                    # Handle nested list case where embedding is wrapped in outer lists
                    if len(query_embedding) > 0 and isinstance(
                            query_embedding[0], list):
                        # This is a nested list like [[0.1, 0.2, ...]], extract the inner list
                        query_embedding = query_embedding[0]
                        logger.debug(
                            f"Extracted embedding from nested list structure")

                    # Now the embedding should be a flat list of floats
                    # Ensure all values are numeric to avoid pgvector errors
                    try:
                        # Make sure all values are floats (this will raise an error if any value can't be converted)
                        query_embedding = [
                            float(val) for val in query_embedding
                        ]
                        logger.debug(
                            f"Verified embedding has correct numeric format")
                    except Exception as e:
                        logger.error(
                            f"Error converting embedding values to float: {str(e)}"
                        )
                        # If we can't convert, create a new empty list (will fall back to recent chunks)
                        query_embedding = []

                    if query_embedding:
                        logger.debug(
                            f"Using provided embedding of length {len(query_embedding)}"
                        )
                else:
                    logger.warning(
                        f"Provided embedding is not a list, type: {type(query_embedding)}"
                    )
                    # Create an empty list to trigger fallback mechanism
                    query_embedding = []

            async with async_session_maker() as session:
                # Step 1: First get the most relevant chunks based on vector similarity
                base_query = select(KnowledgeBase).where(
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_knowledge_base ==
                    True  # Only include knowledge base chunks
                )

                # If a current document ID is provided, exclude it from the results
                if current_document_id is not None:
                    base_query = base_query.where(
                        KnowledgeBase.file_id != current_document_id)

                # Order by vector similarity (L2 distance) and get top matches
                # We'll get fewer initial chunks since we'll be adding adjacent ones later
                initial_limit = min(
                    limit // 2, 3
                )  # Get fewer initial chunks to leave room for adjacent ones
                if initial_limit < 1:
                    initial_limit = 1

                try:
                    # Native pgvector similarity search with cosine distance
                    if query_embedding and len(query_embedding) > 0:
                        # Using PostgreSQL's native vector operations with direct SQL execution for better performance
                        logger.debug(
                            f"Running pgvector search with embedding length: {len(query_embedding)}"
                        )

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
                        # 1 - (embedding <=> :query_vector::vector) gives us similarity score between 0-1
                        # Use the <=> operator with explicit type cast to vector to ensure proper parsing
                        sql_query = f"""
                            SELECT 
                                kb.*,
                                1 - (embedding <=> :query_vector::vector) AS similarity_score
                            FROM 
                                knowledge_base kb
                            WHERE 
                                {where_clause}
                                AND embedding IS NOT NULL
                            ORDER BY 
                                similarity_score DESC
                            LIMIT :limit
                        """

                        # Parameters for the query
                        # Format the embedding as a string representation of the array
                        # PostgreSQL pgvector extension expects embeddings in string array format: '[val1,val2,...]'
                        # Let's ensure the embedding is properly formatted for pgvector
                        try:
                            # Format embedding using the helper method
                            embedding_string = self.document_service.format_embedding_for_pgvector(
                                query_embedding)
                            if embedding_string and len(embedding_string) > 30:
                                logger.debug(
                                    f"Formatted embedding for pgvector: {embedding_string[:30]}..."
                                )
                            else:
                                logger.warning(
                                    f"Possibly malformed embedding string: {embedding_string}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error formatting embedding: {str(e)}")
                            # Create a fallback empty array string that will be valid for PostgreSQL
                            embedding_string = "[]"
                            logger.warning(
                                "Using empty embedding due to formatting error"
                            )

                        params = {
                            "org_id": organization_id,
                            "query_vector": embedding_string,
                            "limit": initial_limit
                        }

                        if current_document_id is not None:
                            params["doc_id"] = current_document_id

                        # Execute raw SQL query
                        from sqlalchemy import text
                        async with async_session_maker() as session:
                            result = await session.execute(
                                text(sql_query), params)
                            rows = result.fetchall()

                            # Convert row results to KnowledgeBase objects with similarity scores
                            chunks_with_similarity = []
                            for row in rows:
                                # Get KnowledgeBase object from row mapping
                                chunk = row[
                                    0]  # The first column contains the entire KnowledgeBase object
                                similarity = float(
                                    row[1]
                                )  # The last column is our similarity score

                                # Attach similarity score to the object and collect all chunks
                                # We'll select the top 2 chunks later regardless of similarity score
                                setattr(chunk, 'similarity_score', similarity)
                                chunks_with_similarity.append(
                                    (chunk, similarity))
                                logger.debug(
                                    f"Retrieved chunk {chunk.chunk_index} with similarity {similarity:.4f}"
                                )

                            # Sort chunks by similarity score (highest first)
                            chunks_with_similarity.sort(key=lambda x: x[1],
                                                        reverse=True)

                            # Get the top 2 chunks with highest similarity score (or all if less than 2)
                            top_limit = min(2, len(chunks_with_similarity))
                            top_chunks = [
                                chunk_data[0] for chunk_data in
                                chunks_with_similarity[:top_limit]
                            ]

                            if top_chunks:
                                logger.info(
                                    f"Selected top {len(top_chunks)} chunks, highest score: {chunks_with_similarity[0][1]:.4f}"
                                )

                            # Now get adjacent chunks (one before and one after each top chunk)
                            # This helps avoid truncated information due to chunk boundaries
                            all_chunks = list(
                                top_chunks)  # Make a copy of top chunks

                            # Group chunks by file_id for efficient retrieval of adjacent chunks
                            file_chunks = {}
                            for chunk in top_chunks:
                                if chunk.file_id not in file_chunks:
                                    file_chunks[chunk.file_id] = []
                                file_chunks[chunk.file_id].append(
                                    chunk.chunk_index)

                            # For each file, get adjacent chunks
                            for file_id, chunk_indexes in file_chunks.items():
                                # Get ALL adjacent chunks in a single query for efficiency
                                adjacent_indexes = set()
                                for chunk_idx in chunk_indexes:
                                    # Get one chunk before and one after
                                    if chunk_idx > 0:  # Ensure we don't go below 0
                                        adjacent_indexes.add(chunk_idx - 1)
                                    adjacent_indexes.add(chunk_idx +
                                                         1)  # Next chunk

                                # Remove indexes we already have
                                adjacent_indexes = adjacent_indexes - set(
                                    chunk_indexes)

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

                                    adj_result = await session.execute(
                                        text(adjacent_query),
                                        {"org_id": organization_id})
                                    adjacent_chunks = adj_result.scalars().all(
                                    )

                                    # Add adjacent chunks to our results
                                    all_chunks.extend(adjacent_chunks)

                            # Sort by file_id and chunk_index to maintain document order
                            all_chunks.sort(
                                key=lambda x: (x.file_id, x.chunk_index))

                            return all_chunks
                    else:
                        # If embedding couldn't be created or is invalid,
                        # Fall back to getting the most recently added knowledge base chunks
                        logger.warning(
                            "Invalid embedding for query, falling back to recent chunks"
                        )
                        similarity_query = base_query.order_by(
                            KnowledgeBase.created_at.desc()).limit(
                                initial_limit)

                        # Execute the ORM query
                        result = await session.execute(similarity_query)
                        top_chunks = result.scalars().all()
                except Exception as e:
                    # If vector search fails, log detailed error information and fall back
                    error_details = str(e)

                    # Try to extract more detailed error information for PostgreSQL/pgvector errors
                    if hasattr(e, '__cause__') and e.__cause__ is not None:
                        error_details = f"{error_details} | Cause: {str(e.__cause__)}"
                        if hasattr(e.__cause__, '__cause__'
                                   ) and e.__cause__.__cause__ is not None:
                            error_details = f"{error_details} | Root cause: {str(e.__cause__.__cause__)}"

                    # Check for common PostgreSQL error types
                    if 'asyncpg.exceptions.DataError' in error_details:
                        logger.error(
                            f"PostgreSQL vector operation failed: {error_details}"
                        )
                        logger.error(
                            "This is likely due to an issue with the embedding format or vector comparison"
                        )
                    elif 'asyncpg.exceptions.UndefinedFunctionError' in error_details and '<=> operator' in error_details:
                        logger.error(
                            f"The pgvector extension might not be properly installed: {error_details}"
                        )
                    elif 'asyncpg.exceptions' in error_details:
                        logger.error(
                            f"PostgreSQL error in vector search: {error_details}"
                        )

                    logger.error(
                        f"Vector search failed: {error_details}, falling back to recent chunks"
                    )

                    # Fall back to a simpler query that doesn't use vector operations
                    similarity_query = base_query.order_by(
                        KnowledgeBase.created_at.desc()).limit(initial_limit)

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
                if hasattr(e.__cause__,
                           '__cause__') and e.__cause__.__cause__ is not None:
                    error_details = f"{error_details} | Root cause: {str(e.__cause__.__cause__)}"

            # PostgreSQL error codes
            if 'asyncpg.exceptions.DataError' in error_details:
                logger.error(
                    f"PostgreSQL vector operation failed: {error_details}")
            elif 'asyncpg.exceptions' in error_details:
                logger.error(f"PostgreSQL error: {error_details}")

            logger.error(error_msg)
            logger.error(f"Detailed error info: {error_details}")

            # Return empty list regardless of error to avoid blocking the document analysis process
            return []

    # CRUD Operations for DocumentAnalysis

    async def create_analysis_record(self, document_id: int,
                                     organization_id: int,
                                     user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document ID"""
        try:
            async with async_session_maker() as session:
                # Check if there's already an active analysis for this document
                stmt = select(DocumentAnalysis).where(
                    (DocumentAnalysis.document_id == document_id)
                    & (DocumentAnalysis.status.in_(
                        [AnalysisStatus.PENDING, AnalysisStatus.PROCESSING])))
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    logger.info(
                        f"Analysis already in progress for document {document_id}, returning existing record"
                    )
                    return existing

                # Create new analysis record
                analysis = DocumentAnalysis(document_id=document_id,
                                            organization_id=organization_id,
                                            user_id=user_id,
                                            status=AnalysisStatus.PENDING,
                                            total_chunks_analyzed=0)

                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)

                logger.info(
                    f"Created analysis record {analysis.id} for document {document_id}"
                )
                return analysis

        except Exception as e:
            logger.error(f"Error creating analysis record: {str(e)}")
            raise

    async def create_analysis_record_by_fp(self, document_fp: str,
                                           organization_id: int,
                                           user_id: int) -> DocumentAnalysis:
        """Create a new document analysis record in pending state using document fingerprint (fp)"""
        try:
            # First get the document ID from the fingerprint
            document_service = DocumentService()
            document = await document_service.get_document_by_fp(
                organization_id, document_fp)

            if not document:
                raise ValueError(
                    f"Document with fingerprint {document_fp} not found")

            document_id = document.id

            # Check if there's already an active analysis for this document
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    (DocumentAnalysis.document_id == document_id)
                    & (DocumentAnalysis.status.in_(
                        [AnalysisStatus.PENDING, AnalysisStatus.PROCESSING])))
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    logger.info(
                        f"Analysis already in progress for document {document_fp} (ID: {document_id}), returning existing record"
                    )
                    return existing

                # Create new analysis record
                analysis = DocumentAnalysis(document_id=document_id,
                                            organization_id=organization_id,
                                            user_id=user_id,
                                            status=AnalysisStatus.PENDING,
                                            total_chunks_analyzed=0)

                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)

                logger.info(
                    f"Created analysis record {analysis.id} for document {document_fp} (ID: {document_id})"
                )
                return analysis

        except Exception as e:
            logger.error(f"Error creating analysis record by FP: {str(e)}")
            raise

    async def get_analysis_by_id(
            self, analysis_id: int) -> Optional[DocumentAnalysis]:
        """Get analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Error getting analysis by ID: {str(e)}")
            raise

    async def get_analysis_by_fp(
            self, analysis_fp: str) -> Optional[DocumentAnalysis]:
        """Get analysis record by fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Error getting analysis by fingerprint: {str(e)}")
            raise

    async def get_analysis_by_document_id(
            self, document_id: int) -> List[DocumentAnalysis]:
        """Get all analysis records for a document, ordered by creation date"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.document_id == document_id).order_by(
                        desc(DocumentAnalysis.created_at))

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
                conditions = [
                    DocumentAnalysis.organization_id == organization_id
                ]

                # Add status filter if provided
                if status_filter:
                    try:
                        status_enum = AnalysisStatus[status_filter.upper()]
                        conditions.append(
                            DocumentAnalysis.status == status_enum)
                    except KeyError:
                        logger.warning(
                            f"Invalid status filter: {status_filter}")

                # Get total count
                count_stmt = select(
                    func.count()).select_from(DocumentAnalysis).where(
                        *conditions)
                result = await session.execute(count_stmt)
                total_count = result.scalar_one()

                # Calculate pagination
                total_pages = math.ceil(total_count /
                                        per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page

                # Get paginated data
                stmt = select(DocumentAnalysis).where(*conditions).order_by(
                    desc(DocumentAnalysis.created_at)).offset(offset).limit(
                        per_page)

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
                conditions = [
                    DocumentAnalysis.organization_id.in_(user_organization_ids)
                ]

                # Add status filter if provided
                if status_filter:
                    try:
                        status_enum = AnalysisStatus[status_filter.upper()]
                        conditions.append(
                            DocumentAnalysis.status == status_enum)
                    except KeyError:
                        logger.warning(
                            f"Invalid status filter: {status_filter}")

                # Get total count
                count_stmt = select(
                    func.count()).select_from(DocumentAnalysis).where(
                        *conditions)
                result = await session.execute(count_stmt)
                total_count = result.scalar_one()

                # Calculate pagination
                total_pages = math.ceil(total_count /
                                        per_page) if total_count > 0 else 1
                offset = (page - 1) * per_page

                # Get paginated data
                stmt = select(DocumentAnalysis).where(*conditions).order_by(
                    desc(DocumentAnalysis.created_at)).offset(offset).limit(
                        per_page)

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
            error_message: Optional[str] = None) -> None:
        """Update the status of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id)
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
                logger.info(
                    f"Updated analysis {analysis_id} status to {status.value}")

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

                if not analysis:
                    logger.error(
                        f"Analysis record not found with fp: {analysis_fp}")
                    return

                analysis.status = status

                # If status is FAILED and error_message is provided, save it
                if status == AnalysisStatus.FAILED and error_message:
                    analysis.error_message = error_message

                # If completed, set the completed_at timestamp
                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()

                await session.commit()
                logger.info(
                    f"Updated analysis with fp {analysis_fp} status to {status.value}"
                )

        except Exception as e:
            logger.error(f"Error updating analysis status: {str(e)}")
            raise

    async def update_original_content(self, analysis_id: int,
                                      content: str) -> None:
        """Update the original content of an analysis record by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()

                if not analysis:
                    logger.error(f"Analysis record not found: {analysis_id}")
                    return

                analysis.original_content = content
                await session.commit()
                logger.info(
                    f"Updated original content for analysis {analysis_id}")

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

                if not analysis:
                    logger.error(
                        f"Analysis record not found with fp: {analysis_fp}")
                    return

                analysis.original_content = content
                await session.commit()
                logger.info(
                    f"Updated original content for analysis with fp {analysis_fp}"
                )

        except Exception as e:
            logger.error(f"Error updating original content: {str(e)}")
            raise

    async def update_analysis_record(
            self,
            analysis_id: int,
            analysis_data: Dict[str, Any],
            suggested_changes: Dict[str, Any],
            status: AnalysisStatus = AnalysisStatus.COMPLETED) -> None:
        """Update an analysis record with analysis results by ID"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()

                if not analysis:
                    logger.error(f"Analysis record not found: {analysis_id}")
                    return

                # Update the analysis record with only diff changes
                analysis.diff_changes = analysis_data.get("diff_changes", "")
                
                analysis.chunk_analyses = analysis_data.get("chunk_analyses", [])
                analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                analysis.processing_time_seconds = analysis_data.get("processing_time_seconds", 0)
                analysis.suggested_changes = suggested_changes
                analysis.status = status

                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()

                await session.commit()
                logger.info(
                    f"Updated analysis record {analysis_id} with git-like diff changes")

        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise

    async def update_analysis_record_by_fp(
            self,
            analysis_fp: str,
            analysis_data: Dict[str, Any],
            suggested_changes: Dict[str, Any],
            status: AnalysisStatus = AnalysisStatus.COMPLETED) -> None:
        """Update an analysis record with analysis results by fingerprint"""
        try:
            async with async_session_maker() as session:
                stmt = select(DocumentAnalysis).where(
                    DocumentAnalysis.fp == analysis_fp)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()

                if not analysis:
                    logger.error(
                        f"Analysis record not found with fp: {analysis_fp}")
                    return

                # Update the analysis record with only diff changes
                analysis.diff_changes = analysis_data.get("diff_changes", "")
                
                analysis.chunk_analyses = analysis_data.get("chunk_analyses", [])
                analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                analysis.processing_time_seconds = analysis_data.get("processing_time_seconds", 0)
                analysis.suggested_changes = suggested_changes
                analysis.status = status

                if status == AnalysisStatus.COMPLETED:
                    analysis.completed_at = datetime.now()

                await session.commit()
                logger.info(
                    f"Updated analysis record with fp {analysis_fp} with git-like diff changes"
                )

        except Exception as e:
            logger.error(f"Error updating analysis record: {str(e)}")
            raise

    async def _generate_suggested_changes(
            self, original_content: str,
            analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
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

            # Function to check if a recommendation is too similar to existing ones
            def is_too_similar(new_rec, existing_recs, similarity_threshold=0.70):
                # Extract key terms from the recommendation
                # For Turkish text, focusing on key terms like "değiştirilmelidir", "çıkarılmalıdır" etc.
                import re
                
                # Simple text normalization - lowercase and remove punctuation
                def normalize_text(text):
                    text = text.lower()
                    text = re.sub(r'[^\w\s]', ' ', text)
                    return text
                
                # Get significant parts of the recommendation (important for detecting duplicates)
                def get_key_parts(text):
                    # Extract quoted parts which typically contain the current and replacement text
                    quoted_parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
                    # Flatten the tuple list from re.findall
                    quoted_parts = [part[0] if part[0] else part[1] for part in quoted_parts if part[0] or part[1]]
                    
                    # Extract madde/section references
                    section_refs = re.findall(r'([Mm][Aa][Dd][Dd][Ee]|[Ss][Ee][Cc][Tt][Ii][Oo][Nn])\s+(\d+(\.\d+)*)', text)
                    section_refs = [f"{match[0]} {match[1]}" for match in section_refs]
                    
                    return quoted_parts, section_refs
                
                # Normalize new recommendation
                norm_new = normalize_text(new_rec)
                new_quoted, new_sections = get_key_parts(new_rec)
                
                for rec in existing_recs:
                    # Normalize existing recommendation
                    norm_existing = normalize_text(rec)
                    existing_quoted, existing_sections = get_key_parts(rec)
                    
                    # Check for section reference overlap
                    if new_sections and existing_sections:
                        common_sections = set(new_sections).intersection(set(existing_sections))
                        if common_sections:
                            # If they refer to the same section and have similar content, likely duplicate
                            if new_quoted and existing_quoted:
                                # Check for similarity in quoted content
                                for new_q in new_quoted:
                                    for ex_q in existing_quoted:
                                        if len(new_q) > 10 and len(ex_q) > 10:  # Only check substantive quotes
                                            # If quoted content is similar, it's likely a duplicate
                                            if new_q in ex_q or ex_q in new_q:
                                                return True
                    
                    # For cases where section may not be properly extracted, do a more general similarity check
                    if len(norm_new) > 20 and len(norm_existing) > 20:  # Only check substantive recommendations
                        # Check if they have very similar wording
                        if norm_new in norm_existing or norm_existing in norm_new:
                            return True
                
                return False  # Not similar enough to consider a duplicate

            # Extract all original AI recommendations directly without filtering
            for chunk in analysis_results:
                if "recommendations" in chunk and isinstance(
                        chunk["recommendations"], list):
                    # Add all recommendations directly from AI, but check for duplicates
                    for rec in chunk["recommendations"]:
                        if not isinstance(rec, str):
                            continue
                            
                        # Check if it's too similar to existing recommendations
                        if not is_too_similar(rec, suggestions["recommendations"]):
                            suggestions["recommendations"].append(rec)
                        else:
                            logger.debug(f"Skipping similar recommendation: {rec[:50]}...")

                # Extract conflicts directly from AI output
                if "conflicts" in chunk and isinstance(chunk["conflicts"],
                                                       list):
                    for conflict in chunk["conflicts"]:
                        if not isinstance(conflict, str):
                            continue

                        # Try to identify section reference if present in the conflict description
                        import re
                        section_ref = "Unspecified section"
                        section_pattern = r'([Ss][Ee][Cc][Tt][Ii][Oo][Nn]|[Mm][Aa][Dd][Dd][Ee]|[Bb][Öö][Ll][Üü][Mm])\s+(\d+(\.\d+)*)'
                        section_match = re.search(section_pattern, conflict)

                        if section_match:
                            section_type = section_match.group(
                                1)  # "Section", "Madde", etc.
                            section_num = section_match.group(
                                2)  # "4.2", "3", etc.
                            section_ref = f"{section_type} {section_num}"

                        # Determine if this is related to policy or legal requirements
                        # This is just classification, not changing the actual content
                        is_policy_conflict = any(
                            term in conflict.lower() for term in [
                                "politika", "policy", "şirket", "company",
                                "standart", "standard", "prosedür",
                                "procedure", "kural", "rule"
                            ])

                        is_legal_issue = any(
                            term in conflict.lower() for term in [
                                "kanun", "yasa", "law", "regulation",
                                "mevzuat", "legal", "yasal", "anayasa",
                                "constitution", "tüzük"
                            ])

                        # Find a corresponding recommendation - improved matching logic
                        matching_recommendation = None
                        
                        # Extract key identifiers from this conflict (section refs, specific terms)
                        conflict_sections = re.findall(r'([Mm][Aa][Dd][Dd][Ee]|[Ss][Ee][Cc][Tt][Ii][Oo][Nn])\s+(\d+(\.\d+)*)', conflict)
                        conflict_sections = [f"{match[0]} {match[1]}" for match in conflict_sections]
                        
                        # Extract quoted text from conflict if available
                        conflict_quotes = re.findall(r"'([^']*)'|\"([^\"]*)\"", conflict)
                        conflict_quotes = [q[0] if q[0] else q[1] for q in conflict_quotes if q[0] or q[1]]
                        
                        # Check recommendations for matches based on section references and quoted text
                        best_match_score = 0
                        for rec in suggestions["recommendations"]:
                            match_score = 0
                            
                            # Check for section reference match
                            rec_sections = re.findall(r'([Mm][Aa][Dd][Dd][Ee]|[Ss][Ee][Cc][Tt][Ii][Oo][Nn])\s+(\d+(\.\d+)*)', rec)
                            rec_sections = [f"{match[0]} {match[1]}" for match in rec_sections]
                            
                            # If both have section references and they match, strong indicator
                            if conflict_sections and rec_sections:
                                for cs in conflict_sections:
                                    if any(cs.lower() == rs.lower() for rs in rec_sections):
                                        match_score += 3  # Strong section match
                            
                            # Look for matching quoted text
                            rec_quotes = re.findall(r"'([^']*)'|\"([^\"]*)\"", rec)
                            rec_quotes = [q[0] if q[0] else q[1] for q in rec_quotes if q[0] or q[1]]
                            
                            if conflict_quotes and rec_quotes:
                                for cq in conflict_quotes:
                                    for rq in rec_quotes:
                                        if cq in rq or rq in cq:
                                            match_score += 2  # Matching quoted text
                            
                            # Check for shared key terms
                            key_terms = ["faiz", "interest", "ödeme", "payment", 
                                         "gün", "day", "mahkeme", "court", "ceza",
                                         "penalty", "süre", "period", "vade", "term"]
                                         
                            for term in key_terms:
                                if term in rec.lower() and term in conflict.lower():
                                    match_score += 1  # Matching key term
                            
                            # If this is the best match so far and it's above our threshold, save it
                            if match_score > best_match_score and match_score >= 2:
                                best_match_score = match_score
                                matching_recommendation = rec
                        
                        # Add to sections with direct conflict and recommendation
                        change_section = {
                            "chunk_index": chunk.get("chunk_index", 0),
                            "section": section_ref,
                            "original_text": chunk.get("chunk_content", ""),
                            "conflict": conflict,
                            # Use the AI's recommendation directly if we found a match
                            "suggested_improvement": matching_recommendation
                            or conflict,
                            "is_policy_conflict": is_policy_conflict,
                            "is_legal_issue": is_legal_issue
                        }

                        # Check if a very similar section already exists
                        existing_similar_section = False
                        for existing_section in suggestions["sections"]:
                            if existing_section["section"] == section_ref and existing_section["conflict"] == conflict:
                                existing_similar_section = True
                                break
                        
                        if not existing_similar_section:
                            suggestions["sections"].append(change_section)

                            # Add to appropriate categorized lists (avoiding duplicates)
                            if is_policy_conflict and not is_too_similar(conflict, suggestions["policy_conflicts"]):
                                suggestions["policy_conflicts"].append(conflict)
                            if is_legal_issue and not is_too_similar(conflict, suggestions["legal_compliance_issues"]):
                                suggestions["legal_compliance_issues"].append(conflict)

            # If no sections were found but we have recommendations, create sections from them
            if not suggestions["sections"] and suggestions["recommendations"]:
                for i, rec in enumerate(suggestions["recommendations"]):
                    # Classify recommendation
                    is_policy = any(term in rec.lower() for term in [
                        "politika", "policy", "şirket", "company", "standart"
                    ])
                    is_legal = any(
                        term in rec.lower() for term in
                        ["kanun", "yasa", "law", "regulation", "mevzuat"])

                    # Try to extract section reference from recommendation
                    import re
                    section_ref = f"Recommendation {i+1}"
                    section_pattern = r'([Ss][Ee][Cc][Tt][Ii][Oo][Nn]|[Mm][Aa][Dd][Dd][Ee]|[Bb][Öö][Ll][Üü][Mm])\s+(\d+(\.\d+)*)'
                    section_match = re.search(section_pattern, rec)

                    if section_match:
                        section_type = section_match.group(1)  # "Section", "Madde", etc.
                        section_num = section_match.group(2)  # "4.2", "3", etc.
                        section_ref = f"{section_type} {section_num}"

                    # Create a section from the recommendation itself
                    suggestions["sections"].append({
                        "chunk_index": 0,
                        "section": section_ref,
                        "original_text": "",  # No specific text identified
                        "conflict": rec,  # Use recommendation as conflict too
                        "suggested_improvement": rec,
                        "is_policy_conflict": is_policy,
                        "is_legal_issue": is_legal
                    })

                    # Add to appropriate categorized lists if not already there
                    if is_policy and not is_too_similar(rec, suggestions["policy_conflicts"]):
                        suggestions["policy_conflicts"].append(rec)
                    if is_legal and not is_too_similar(rec, suggestions["legal_compliance_issues"]):
                        suggestions["legal_compliance_issues"].append(rec)

            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggested changes: {str(e)}")
            return {
                "recommendations": [],
                "sections": [],
                "policy_conflicts": [],
                "legal_compliance_issues": []
            }

    def _combine_chunk_analyses(
            self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "chunk_analyses":
            []  # Store individual chunk analyses for reference
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
            if "recommendations" in chunk and isinstance(
                    chunk["recommendations"], list):
                for rec in chunk["recommendations"]:
                    if isinstance(rec, str):
                        all_recommendations.add(rec)

        # Build a comprehensive analysis summary
        analysis_parts = []
        for chunk in chunk_analyses:
            # Skip duplicates when building analysis parts too
            chunk_index = chunk.get("chunk_index")
            if chunk_index is not None and chunk_index in processed_chunk_indices:
                processed_chunk_indices.remove(
                    chunk_index
                )  # Remove so we process each index exactly once
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
