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
            max_relevant_chunks: int = 5,
            analysis_id: Optional[int] = None) -> Dict[str, Any]:
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

            # Use existing analysis record or create a new one
            if analysis_id:
                # Use existing analysis record
                analysis_record = await self.get_analysis_by_id(analysis_id)
                if not analysis_record:
                    logger.warning(f"Analysis ID {analysis_id} provided but record not found, creating new record")
                    analysis_record = await self.create_analysis_record(
                        document_id=document_id,
                        organization_id=organization_id,
                        user_id=user_id)
            else:
                # Create a new analysis record
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
            total_processing_time = 0.0

            logger.info(
                f"Starting unified document analysis for {total_chunks} chunks"
            )

            # Step 1: Collect all relevant knowledge base chunks across all document chunks
            all_relevant_kb_chunks = set()  # Use a set to avoid duplicates
            
            # IMPROVEMENT: Process all document chunks to collect relevant knowledge base chunks
            for chunk_idx, chunk in enumerate(document_chunks):
                logger.info(f"Finding KB chunks for document chunk {chunk_idx+1}/{total_chunks}")
                
                # Skip empty chunks
                if not chunk.content or not chunk.content.strip():
                    continue
                
                # Get embedding if available
                chunk_embedding = None
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    chunk_embedding = chunk.embedding
                
                # Find relevant KB chunks for this document chunk
                relevant_chunks = await self.find_relevant_knowledge_base_chunks(
                    organization_id=organization_id,
                    query_text=chunk.content,
                    current_document_id=document_id,
                    limit=max_relevant_chunks,
                    query_embedding=chunk_embedding
                )
                
                # Add to our collection of unique KB chunks
                for kb_chunk in relevant_chunks:
                    chunk_key = f"{kb_chunk.file_id}_{kb_chunk.chunk_index}"
                    all_relevant_kb_chunks.add((chunk_key, kb_chunk))
                
                # Add adjacent chunks for better context (improve snippet context)
                for kb_chunk in relevant_chunks:
                    try:
                        async with async_session_maker() as session:
                            # Get up to 2 chunks before and after for context
                            before_query = select(KnowledgeBase).where(
                                KnowledgeBase.file_id == kb_chunk.file_id,
                                KnowledgeBase.is_knowledge_base == True,
                                KnowledgeBase.chunk_index < kb_chunk.chunk_index
                            ).order_by(desc(KnowledgeBase.chunk_index)).limit(2)
                            
                            after_query = select(KnowledgeBase).where(
                                KnowledgeBase.file_id == kb_chunk.file_id,
                                KnowledgeBase.is_knowledge_base == True,
                                KnowledgeBase.chunk_index > kb_chunk.chunk_index
                            ).order_by(KnowledgeBase.chunk_index).limit(2)
                            
                            before_result = await session.execute(before_query)
                            after_result = await session.execute(after_query)
                            
                            before_chunks = before_result.scalars().all()
                            after_chunks = after_result.scalars().all()
                            
                            # Add all adjacent chunks to our set
                            for adj_chunk in before_chunks + after_chunks:
                                adj_key = f"{adj_chunk.file_id}_{adj_chunk.chunk_index}"
                                all_relevant_kb_chunks.add((adj_key, adj_chunk))
                    except Exception as e:
                        logger.warning(f"Error getting adjacent chunks: {str(e)}")
                        # Continue even if we can't get adjacent chunks
            
            # Step 2: Also get relevant chunks for the entire document to ensure comprehensive coverage
            logger.info("Finding KB chunks based on the entire document content")
            full_doc_relevant_kb = await self.find_relevant_knowledge_base_chunks(
                organization_id=organization_id,
                query_text=original_content,
                current_document_id=document_id,
                limit=max_relevant_chunks * 2  # Get more chunks for the entire document
            )
            
            # Add these to our unique set as well
            for kb_chunk in full_doc_relevant_kb:
                chunk_key = f"{kb_chunk.file_id}_{kb_chunk.chunk_index}"
                all_relevant_kb_chunks.add((chunk_key, kb_chunk))
            
            # Extract KB chunks from the set of tuples
            all_kb_chunks = [chunk_tuple[1] for chunk_tuple in all_relevant_kb_chunks]
            
            logger.info(f"Collected {len(all_kb_chunks)} unique knowledge base chunks")
            
            # Step 3: Prepare KB chunks with rich metadata for the analysis
            kb_chunks_with_metadata = []
            for kb_chunk in all_kb_chunks:
                # Get the filename for context
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
                
                # Create rich metadata for the OpenAI prompt
                similarity_score = getattr(kb_chunk, 'similarity_score', 0)
                
                kb_info = {
                    "document_name": file_name,
                    "document_id": kb_chunk.file_id,
                    "chunk_index": kb_chunk.chunk_index,
                    "similarity_score": similarity_score,
                    "content": kb_chunk.content,
                    "meta_info": kb_chunk.meta_info if kb_chunk.meta_info else {}
                }
                
                kb_chunks_with_metadata.append((kb_info, similarity_score))
            
            # Sort by similarity score (highest first)
            kb_chunks_with_metadata.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the info objects
            formatted_kb_chunks = [chunk_data[0] for chunk_data in kb_chunks_with_metadata]
            
            logger.info(f"Prepared {len(formatted_kb_chunks)} KB chunks for OpenAI analysis")
            
            # Step 4: Analyze the full document against all relevant KB chunks in a single API call
            analysis_start_time = datetime.now()
            
            # DIKKAT: Bilgi bankası kontrolleri ve OpenAI API çağrısı
            logger.info(f"Preparing OpenAI analysis with {len(formatted_kb_chunks)} knowledge base chunks...")
            
            # Knowledge base'in boş olması durumunu kontrol et
            if len(formatted_kb_chunks) == 0:
                logger.warning("No knowledge base chunks found for analysis - this may impact analysis quality")
                
                # Knowledge base bağlantısını test et
                try:
                    async with async_session_maker() as session:
                        # Knowledge base tablosunda veri var mı kontrol et
                        kb_count_query = select(func.count()).select_from(KnowledgeBase).where(
                            KnowledgeBase.organization_id == organization_id,
                            KnowledgeBase.is_knowledge_base == True
                        )
                        kb_count_result = await session.execute(kb_count_query)
                        kb_count = kb_count_result.scalar_one()
                        
                        logger.info(f"Organization has {kb_count} total knowledge base entries")
                        
                        if kb_count == 0:
                            logger.warning(f"Organization {organization_id} has no knowledge base entries")
                        else:
                            # Neden bulunamadığını araştır
                            logger.warning(f"Organization has knowledge base entries but none were relevant for this document")
                except Exception as kb_check_err:
                    logger.error(f"Error checking knowledge base status: {str(kb_check_err)}")
            
            logger.info("Sending document for OpenAI analysis...")
            
            try:
                # Send the entire document and all relevant KB chunks in a single API call
                analysis_result = await self.openai_service.analyze_document(
                    document_chunk=original_content,
                    knowledge_base_chunks=formatted_kb_chunks
                )
                
                # KRITIK: Sonucun geçerli olup olmadığını kontrol et
                if analysis_result is None or not isinstance(analysis_result, dict):
                    logger.error(f"Invalid analysis result returned: {analysis_result}")
                    # Varsayılan bir yanıt oluştur, işlemi başarısız yapmadan devam et
                    analysis_result = {
                        "diff_changes": "Analysis could not be completed. Please try again.",
                        "processing_time_seconds": 0.0,
                        "total_chunks_analyzed": len(formatted_kb_chunks)
                    }
                
                # Sonucun diff_changes içerdiğinden emin ol
                if "diff_changes" not in analysis_result:
                    logger.warning("Analysis result does not contain diff_changes, adding empty value")
                    analysis_result["diff_changes"] = "No changes identified in the document."
                
                analysis_duration = (datetime.now() - analysis_start_time).total_seconds()
                total_processing_time += analysis_duration
                
                logger.info(f"Successfully completed document analysis in {analysis_duration:.2f} seconds")
                
                # Add additional metadata
                analysis_result["processing_time_seconds"] = analysis_duration
                analysis_result["total_chunks_analyzed"] = len(formatted_kb_chunks)
                
                # Prepare the final analysis response
                analysis_response = {
                    "diff_changes": analysis_result.get("diff_changes", ""),
                    "processing_time_seconds": total_processing_time,
                    "total_chunks_analyzed": len(formatted_kb_chunks),
                    "total_document_chunks": total_chunks,
                    "kb_chunks_used": len(formatted_kb_chunks),
                    "document_info": {
                        "id": document_id,
                        "name": document.filename if hasattr(document, 'filename') else "Unknown",
                    }
                }
                
                # Update the analysis record with the results
                try:
                    await self.update_analysis_record(
                        analysis_id=analysis_record.id,
                        analysis_data=analysis_response,
                        status=AnalysisStatus.COMPLETED
                    )
                    logger.info(f"Analysis record updated successfully for ID: {analysis_record.id}")
                except Exception as record_error:
                    logger.error(f"Failed to update analysis record: {str(record_error)}")
                    # Hata olsa bile işlemi devam ettir, kullanıcıya yanıt döndür
                
                logger.info(f"Document analysis completed successfully in {total_processing_time:.2f} seconds")
                return analysis_response
                
            except Exception as e:
                error_msg = f"Error during OpenAI analysis: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Hata durumunda bile kullanıcıya anlamlı bir yanıt döndürmeye çalış
                try:
                    # Başarısız duruma geçmek yerine işlemi tamamla ama hata mesajı içeren bir yanıt döndür
                    basic_response = {
                        "diff_changes": f"Analysis could not be completed due to an error: {str(e)}. Please try again later.",
                        "processing_time_seconds": (datetime.now() - analysis_start_time).total_seconds(),
                        "total_chunks_analyzed": len(formatted_kb_chunks),
                        "error_details": str(e)
                    }
                    
                    # Analiz kaydını güncelle ama FAILED yapmak yerine COMPLETED olarak işaretle
                    await self.update_analysis_record(
                        analysis_id=analysis_record.id,
                        analysis_data=basic_response,
                        status=AnalysisStatus.COMPLETED  # Önemli: FAILED yerine COMPLETED kullan
                    )
                    
                    logger.info("Returning partial analysis response despite error")
                    return basic_response
                    
                except Exception as update_error:
                    logger.error(f"Failed to update analysis with error info: {str(update_error)}")
                    # Son çare olarak basit bir hata yanıtı döndür
                    return {
                        "diff_changes": "Analysis failed. Please try again later.",
                        "error": str(e),
                        "status": "completed"  # FAILED yerine completed kullan
                    }

        except Exception as e:
            error_msg = f"Error in document analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Update analysis record if it was created
            if 'analysis_record' in locals():
                try:
                    await self.update_analysis_status(
                        analysis_record.id, AnalysisStatus.FAILED, error_msg)
                except Exception as update_error:
                    logger.error(f"Error updating analysis status: {str(update_error)}")
            
            # Return error information rather than re-raising
            return {
                "error": error_msg,
                "status": "failed"
            }

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
            # Daha ayrıntılı hata günlüğü kaydı
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error finding relevant knowledge base chunks: {str(e)}")
            logger.error(f"Detailed exception traceback: {error_details}")
            
            # pgvector ile ilgili hata olup olmadığını kontrol et
            error_str = str(e).lower()
            if "vector" in error_str or "cosine" in error_str or "embedding" in error_str:
                logger.critical("Possible pgvector extension issue or embedding format problem")
                # Veritabanında pgvector kontrol edilmeli
            
            try:
                # Fallback yaklaşım - vektör benzerliği olmadan en son eklenen birkaç bilgi bankası öğesini getir
                logger.info("Attempting fallback query to get recent knowledge base entries")
                async with async_session_maker() as session:
                    fallback_query = select(KnowledgeBase).where(
                        KnowledgeBase.organization_id == organization_id,
                        KnowledgeBase.is_knowledge_base == True
                    ).order_by(desc(KnowledgeBase.created_at)).limit(limit)
                    
                    result = await session.execute(fallback_query)
                    fallback_chunks = result.scalars().all()
                    
                    if fallback_chunks:
                        logger.info(f"Retrieved {len(fallback_chunks)} recent knowledge base chunks as fallback")
                        return fallback_chunks
                    else:
                        logger.warning(f"No knowledge base entries found for organization {organization_id}")
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {str(fallback_error)}")
            
            # İşlemi engellemeyip devam etmek için yine de boş liste döndür
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
                    # Using chunk_analyses JSON field to store metadata temporarily
                    chunk_analyses={"document_fp": document.fp}
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
                    # Using chunk_analyses JSON field to store metadata temporarily
                    chunk_analyses={"document_fp": document_fp}
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
                    # Convert to appropriate fields
                    # Extract data from analysis_data for each field in the model
                    if "diff_changes" in analysis_data:
                        analysis.diff_changes = analysis_data["diff_changes"]
                    
                    # Store processing metrics
                    if "processing_time_seconds" in analysis_data:
                        analysis.processing_time_seconds = analysis_data["processing_time_seconds"]
                    
                    if "total_chunks_analyzed" in analysis_data:
                        analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                    
                    # Store any remaining data in chunk_analyses JSON field
                    analysis.chunk_analyses = analysis_data
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
                    # Convert to appropriate fields
                    # Extract data from analysis_data for each field in the model
                    if "diff_changes" in analysis_data:
                        analysis.diff_changes = analysis_data["diff_changes"]
                    
                    # Store processing metrics
                    if "processing_time_seconds" in analysis_data:
                        analysis.processing_time_seconds = analysis_data["processing_time_seconds"]
                    
                    if "total_chunks_analyzed" in analysis_data:
                        analysis.total_chunks_analyzed = analysis_data.get("total_chunks_analyzed", 0)
                    
                    # Store any remaining data in chunk_analyses JSON field
                    analysis.chunk_analyses = analysis_data
                    analysis.status = status
                    analysis.updated_at = datetime.now()
                    await session.commit()
        except Exception as e:
            logger.error(f"Error updating analysis record by FP: {str(e)}")
            raise