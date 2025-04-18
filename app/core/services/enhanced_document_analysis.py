"""
Enhanced Document Analysis Service Module for retrieving relevant knowledge base chunks.
This implements the improved algorithm for finding relevant chunks including neighbors and deduplication.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any
import json
from sqlalchemy import select, text
from app.database.database_factory import async_session_maker
from app.database.models.db_models import KnowledgeBase, File
from app.core.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

class EnhancedKnowledgeBaseRetrieval:
    """
    Enhanced service for retrieving knowledge base chunks based on similarity
    with proper adjacent chunk inclusion and deduplication.
    """
    
    def __init__(self, openai_service: Optional[OpenAIService] = None):
        """Initialize the service with dependencies"""
        self.openai_service = openai_service or OpenAIService()
    
    async def find_most_similar_kb_chunk_for_query(
        self,
        organization_id: int,
        query_text: str,
        current_document_id: Optional[int] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Optional[KnowledgeBase]:
        """
        Find the single most similar knowledge base chunk for a query text.
        
        Args:
            organization_id: ID of the organization
            query_text: Text to search for (document chunk content)
            current_document_id: Optional ID of the current document to exclude from results
            query_embedding: Optional pre-generated embedding vector for the query text
            
        Returns:
            The most similar knowledge base chunk or None if not found
        """
        try:
            # Get or create embedding for the query text
            if not query_embedding:
                query_embedding = await self.openai_service.get_embedding(query_text)
                
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query text")
                return None
                
            # Use vector similarity search to find the most similar chunk
            embedding_string = json.dumps(query_embedding)
            
            async with async_session_maker() as session:
                # Build conditions
                conditions = [
                    "organization_id = :org_id",
                    "is_knowledge_base = TRUE"
                ]
                
                if current_document_id:
                    conditions.append("file_id != :doc_id")
                    
                where_clause = " AND ".join(conditions)
                
                # SQL query to find the single most similar KB chunk
                sql_query = f"""
                SELECT 
                    *,
                    embedding::vector <=> :query_embedding::vector AS similarity_score
                FROM knowledge_base
                WHERE {where_clause}
                ORDER BY similarity_score ASC
                LIMIT 1
                """
                
                params = {
                    "query_embedding": embedding_string,
                    "org_id": organization_id
                }
                
                if current_document_id:
                    params["doc_id"] = current_document_id
                    
                result = await session.execute(text(sql_query), params)
                most_similar_chunk = result.first()
                
                if most_similar_chunk:
                    # Create a KnowledgeBase object from the result
                    kb_chunk = KnowledgeBase(
                        id=most_similar_chunk.id,
                        fp=most_similar_chunk.fp,
                        organization_id=most_similar_chunk.organization_id,
                        file_id=most_similar_chunk.file_id,
                        chunk_index=most_similar_chunk.chunk_index,
                        content=most_similar_chunk.content,
                        embedding=most_similar_chunk.embedding,
                        meta_info=most_similar_chunk.meta_info,
                        is_knowledge_base=most_similar_chunk.is_knowledge_base,
                        created_at=most_similar_chunk.created_at,
                    )
                    # Add similarity score as attribute
                    setattr(kb_chunk, 'similarity_score', most_similar_chunk.similarity_score)
                    return kb_chunk
                    
                return None
                
        except Exception as e:
            logger.error(f"Error finding most similar KB chunk: {str(e)}")
            return None
    
    async def get_adjacent_kb_chunks(
        self,
        document_id: int,
        chunk_index: int
    ) -> List[KnowledgeBase]:
        """
        Get adjacent chunks (one before and one after) for a given chunk.
        
        Args:
            document_id: ID of the document
            chunk_index: Index of the central chunk
            
        Returns:
            List of adjacent chunks (may include central chunk if requested)
        """
        try:
            adjacent_indices = []
            
            # Add previous chunk if not at the beginning
            if chunk_index > 0:
                adjacent_indices.append(chunk_index - 1)
                
            # Add next chunk
            adjacent_indices.append(chunk_index + 1)
            
            if not adjacent_indices:
                return []
                
            async with async_session_maker() as session:
                query = select(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id,
                    KnowledgeBase.chunk_index.in_(adjacent_indices),
                    KnowledgeBase.is_knowledge_base == True
                )
                
                result = await session.execute(query)
                adjacent_chunks = result.scalars().all()
                return list(adjacent_chunks)
                
        except Exception as e:
            logger.error(f"Error getting adjacent KB chunks: {str(e)}")
            return []
    
    async def find_relevant_chunks_with_neighbors(
        self,
        organization_id: int,
        document_chunks: List[Any],
        current_document_id: Optional[int] = None
    ) -> List[KnowledgeBase]:
        """
        Find the most relevant knowledge base chunks for each document chunk
        with their adjacent chunks, following these steps:
        1. For each document chunk, find the most similar KB chunk
        2. Include adjacent chunks (one before and one after when available)
        3. Remove duplicates based on fingerprint
        
        Args:
            organization_id: ID of the organization
            document_chunks: List of document chunks to analyze
            current_document_id: Optional ID of the current document to exclude from results
            
        Returns:
            List of deduplicated relevant knowledge base chunks with their adjacent chunks
        """
        try:
            # Use a set to track unique fingerprints and avoid duplicates
            unique_chunk_fps = set()
            all_relevant_chunks = []
            
            # Process each document chunk
            for chunk_idx, doc_chunk in enumerate(document_chunks):
                logger.info(f"Finding KB chunks for document chunk {chunk_idx+1}/{len(document_chunks)}")
                
                # Skip empty chunks
                if not doc_chunk.content or not doc_chunk.content.strip():
                    continue
                
                # Get embedding if available
                chunk_embedding = None
                if hasattr(doc_chunk, 'embedding') and doc_chunk.embedding is not None:
                    chunk_embedding = doc_chunk.embedding
                
                # Find most similar KB chunk for this document chunk
                most_similar_chunk = await self.find_most_similar_kb_chunk_for_query(
                    organization_id=organization_id,
                    query_text=doc_chunk.content,
                    current_document_id=current_document_id,
                    query_embedding=chunk_embedding
                )
                
                if most_similar_chunk:
                    # Process the most similar chunk
                    if most_similar_chunk.fp not in unique_chunk_fps:
                        unique_chunk_fps.add(most_similar_chunk.fp)
                        all_relevant_chunks.append(most_similar_chunk)
                    
                    # Get adjacent chunks
                    adjacent_chunks = await self.get_adjacent_kb_chunks(
                        document_id=most_similar_chunk.file_id,
                        chunk_index=most_similar_chunk.chunk_index
                    )
                    
                    # Add adjacent chunks if not already included
                    for adj_chunk in adjacent_chunks:
                        if adj_chunk.fp not in unique_chunk_fps:
                            unique_chunk_fps.add(adj_chunk.fp)
                            all_relevant_chunks.append(adj_chunk)
            
            # Sort chunks by document_id and chunk_index to maintain readability
            all_relevant_chunks.sort(key=lambda x: (x.file_id, x.chunk_index))
            
            logger.info(f"Retrieved {len(all_relevant_chunks)} unique KB chunks after deduplication")
            return all_relevant_chunks
            
        except Exception as e:
            logger.error(f"Error in retrieving relevant chunks with neighbors: {str(e)}")
            return []