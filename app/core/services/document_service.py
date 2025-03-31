import os
import json
from typing import List, Optional, Dict, Any, BinaryIO, Tuple
import boto3
from botocore.exceptions import ClientError
import logging
from docx import Document
from PyPDF2 import PdfReader
import asyncio
from app.core.config import config
from app.database.models.db_models import File, KnowledgeBase, FileStatus, User, Organization
from app.core.services.openai_service import OpenAIService
from sqlalchemy import select, desc, delete, func
from sqlalchemy.orm import joinedload
from sqlalchemy import delete, select, func, text
import math
from datetime import datetime

from app.database.database_factory import async_session_maker

logger = logging.getLogger(__name__)

class DocumentService:
    CHUNK_SIZE = 1000  # Characters per chunk
    BATCH_SIZE = 20    # Number of chunks to process in one batch

    def __init__(self):
        self._s3_client = None
        self.bucket_name = config.AWS_BUCKET_NAME
        self.openai_service = OpenAIService()

    @property
    def s3_client(self):
        """Lazy initialization of S3 client only when needed"""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                region_name=config.AWS_REGION
            )
        return self._s3_client

    async def create_file_record(
        self,
        filename: str,
        file_type: str,
        user_id: int,
        organization_id: int,
        is_knowledge_base: bool = True
    ) -> File:
        """Create initial file record with PROCESSING status"""
        try:
            # Make sure chunk_count is explicitly set to 0 initially
            db_file = File(
                filename=filename,
                file_type=file_type.lower(),
                status=FileStatus.PROCESSING,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=f"documents/{organization_id}/{filename}",
                chunk_count=0,  # Explicitly set initial chunk count
                is_knowledge_base=is_knowledge_base  # Set whether this file is for knowledge base or analysis
            )

            async with async_session_maker() as session:
                session.add(db_file)
                await session.commit()
                await session.refresh(db_file)
                logger.info(f"Created new file record with id: {db_file.id}, initial chunk_count: 0")
                return db_file

        except Exception as e:
            logger.error(f"Error creating file record: {str(e)}")
            raise

    async def update_file_status(
        self,
        file_id: int,
        status: FileStatus
    ) -> None:
        """Update file status"""
        try:
            async with async_session_maker() as session:
                stmt = select(File).where(File.id == file_id)
                result = await session.execute(stmt)
                db_file = result.scalar_one_or_none()

                if db_file:
                    db_file.status = status
                    await session.commit()
                    logger.info(f"Updated file {file_id} status to {status}")
                else:
                    logger.error(f"File {file_id} not found")

        except Exception as e:
            logger.error(f"Error updating file status: {str(e)}")
            raise

    async def process_document_async(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        user_id: int,
        organization_id: int,
        db_file_id: int,
        is_knowledge_base: bool = True
    ) -> None:
        """
        Process uploaded document asynchronously:
        - Upload to S3
        - Extract text
        - Generate embeddings
        - Save to knowledge base
        
        Parameters:
            is_knowledge_base: Whether document is for knowledge base (True) or for analysis (False)
        """
        try:
            logger.info(f"Starting document processing for file: {filename} (ID: {db_file_id})")

            # Upload to S3
            try:
                from io import BytesIO
                file_obj = BytesIO(file_content)
                s3_key = f"documents/{organization_id}/{filename}"

                start_time = datetime.now()
                self.s3_client.upload_fileobj(
                    file_obj,
                    self.bucket_name,
                    s3_key
                )
                upload_duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"File uploaded to S3: {s3_key} (Duration: {upload_duration}s)")
            except Exception as e:
                logger.error(f"Error uploading file to S3: {str(e)}, File: {filename}")
                await self.update_file_status(db_file_id, FileStatus.FAILED)
                raise

            # Extract text content and split into chunks
            start_time = datetime.now()
            text_chunks = await self._extract_text_chunks(BytesIO(file_content), file_type)
            extraction_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Extracted {len(text_chunks)} chunks from document (Duration: {extraction_duration}s)")

            # Process chunks in batches
            chunk_batches = [text_chunks[i:i + self.BATCH_SIZE] 
                           for i in range(0, len(text_chunks), self.BATCH_SIZE)]

            total_chunks = len(text_chunks)
            processing_start = datetime.now()
            logger.info(f"Starting document processing with {total_chunks} total chunks")

            async with async_session_maker() as session:
                for batch_idx, chunk_batch in enumerate(chunk_batches):
                    try:
                        batch_start = datetime.now()

                        # Generate embeddings and create knowledge base entries
                        try:
                            embeddings = await self.openai_service.create_embeddings(chunk_batch)

                            # Create knowledge base entries for the batch
                            batch_entries = []
                            start_idx = batch_idx * self.BATCH_SIZE

                            for idx, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                                kb_entry = KnowledgeBase(
                                    chunk_index=start_idx + idx,
                                    content=chunk,
                                    embedding=embedding,
                                    meta_info=json.dumps({
                                        "filename": filename,
                                        "chunk_number": start_idx + idx + 1,
                                        "total_chunks": total_chunks
                                    }),
                                    file_id=db_file_id,
                                    organization_id=organization_id,
                                    is_knowledge_base=is_knowledge_base
                                )
                                batch_entries.append(kb_entry)

                            # Bulk insert the batch
                            session.add_all(batch_entries)
                            await session.commit()

                            batch_duration = (datetime.now() - batch_start).total_seconds()
                            logger.info(f"Processed and saved batch {batch_idx + 1}/{len(chunk_batches)} (Duration: {batch_duration}s)")

                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                            continue

                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                        continue

                # Update chunk count using SQL COUNT before setting status to completed
                try:
                    from sqlalchemy import func, select
                    count_stmt = select(func.count()).select_from(KnowledgeBase).where(KnowledgeBase.file_id == db_file_id)
                    result = await session.execute(count_stmt)
                    actual_chunk_count = result.scalar()

                    logger.info(f"Counted {actual_chunk_count} chunks in knowledge base for file {db_file_id}")

                    # Update File record with the actual chunk count
                    update_stmt = select(File).where(File.id == db_file_id)
                    result = await session.execute(update_stmt)
                    db_file = result.scalar_one_or_none()

                    if db_file:
                        db_file.chunk_count = actual_chunk_count
                        await session.commit()
                        logger.info(f"Successfully updated chunk count for file {db_file_id} to {actual_chunk_count}")
                except Exception as e:
                    logger.error(f"Error updating chunk count: {str(e)}")
                    raise

            # Update file status to completed only after chunk count is updated
            await self.update_file_status(db_file_id, FileStatus.COMPLETED)
            logger.info(f"Completed processing document {filename} (ID: {db_file_id})")

        except Exception as e:
            logger.error(f"Error in async document processing: {str(e)}", exc_info=True)
            await self.update_file_status(db_file_id, FileStatus.FAILED)
            raise

    async def get_organization_documents(
        self,
        organization_id: int
    ) -> List[File]:
        """Get all documents for an organization"""
        try:
            async with async_session_maker() as session:
                stmt = select(File).where(
                    File.organization_id == organization_id
                ).order_by(desc(File.created_at))

                result = await session.execute(stmt)
                files = result.scalars().all()

                logger.info(f"Retrieved {len(files)} documents for organization {organization_id}")
                return files

        except Exception as e:
            logger.error(f"Error fetching organization documents: {str(e)}")
            raise

    async def _extract_text_chunks(
        self,
        file: BinaryIO,
        file_type: str,
        min_chunk_size: int = 100  # Minimum characters per chunk
    ) -> List[str]:
        """
        Extract text from document and split into optimally sized chunks
        """
        text = ""
        file_type = file_type.lower()

        try:
            if file_type == "pdf":
                reader = PdfReader(file)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
                    else:
                        logger.warning(f"Empty text extracted from PDF page")

            elif file_type in ["docx", "doc"]:
                doc = Document(file)
                for para in doc.paragraphs:
                    if para.text:
                        text += para.text + "\n"

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if not text.strip():
                raise ValueError("No text content extracted from document")

            # Implement smart chunking
            chunks = []
            current_chunk = []
            current_length = 0

            # Split text into sentences first
            sentences = text.replace("\n", " ").split(". ")

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_length = len(sentence)

                # If a single sentence exceeds chunk size, split it into smaller parts
                if sentence_length > self.CHUNK_SIZE:
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0

                    for word in words:
                        word_length = len(word) + 1  # +1 for space
                        if temp_length + word_length > self.CHUNK_SIZE:
                            if temp_chunk:  # Only add non-empty chunks
                                chunks.append(" ".join(temp_chunk))
                            temp_chunk = [word]
                            temp_length = word_length
                        else:
                            temp_chunk.append(word)
                            temp_length += word_length

                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))

                # For normal sentences, try to keep them together in chunks
                elif current_length + sentence_length + 1 <= self.CHUNK_SIZE:
                    current_chunk.append(sentence)
                    current_length += sentence_length + 1
                else:
                    if current_chunk and current_length >= min_chunk_size:
                        chunks.append(". ".join(current_chunk) + ".")
                    current_chunk = [sentence]
                    current_length = sentence_length

            # Add the last chunk if it meets minimum size
            if current_chunk and current_length >= min_chunk_size:
                chunks.append(". ".join(current_chunk) + ".")

            logger.info(f"Successfully extracted {len(chunks)} text chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error extracting text from document: {str(e)}")
            raise

    async def search_chunks_for_organization(
        self,
        organization_id: int,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int]:
        """
        Search through chunks in a specific organization using semantic search
        Returns a tuple of (chunks, total_count)
        This uses vector similarity search and pagination
        """
        try:
            logger.info(f"Starting semantic chunk search for org {organization_id}. Query: '{query}'")
            start_time = datetime.now()

            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")

            embedding_duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Generated query embedding (Duration: {embedding_duration}s)")
            
            # Calculate pagination
            offset = (page - 1) * per_page
            
            async with async_session_maker() as session:
                # Use raw SQL with window functions for better performance
                # This performs pagination and counting in a single query
                search_start = datetime.now()
                
                query_sql = text("""
                    WITH ranked_results AS (
                        SELECT 
                            kb.*,
                            COUNT(*) OVER() as total_count,
                            ROW_NUMBER() OVER(ORDER BY l2_distance(kb.embedding, :query_embedding::vector)) as row_num
                        FROM 
                            knowledge_base kb
                        WHERE
                            kb.organization_id = :org_id
                            AND kb.is_knowledge_base = TRUE
                    )
                    SELECT * FROM ranked_results
                    WHERE row_num > :offset AND row_num <= :offset_end
                    ORDER BY row_num
                """)
                
                result = await session.execute(
                    query_sql, 
                    {
                        "query_embedding": query_embedding[0],
                        "org_id": organization_id,
                        "offset": offset,
                        "offset_end": offset + per_page
                    }
                )
                
                rows = result.fetchall()
                
                # Extract total count from first row if results exist
                total_count = rows[0].total_count if rows else 0
                
                # Convert rows to KnowledgeBase objects
                chunks = [
                    KnowledgeBase(
                        id=row.id,
                        fp=row.fp,
                        file_id=row.file_id,
                        organization_id=row.organization_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        embedding=row.embedding,
                        meta_info=row.meta_info,
                        is_knowledge_base=row.is_knowledge_base,
                        created_at=row.created_at
                    ) for row in rows
                ]

                search_duration = (datetime.now() - search_start).total_seconds()
                logger.info(f"Found {len(chunks)} relevant chunks for org {organization_id} (Total: {total_count}, Search duration: {search_duration}s)")

                return chunks, total_count
        except Exception as e:
            logger.error(f"Error searching chunks for organization: {str(e)}", exc_info=True)
            raise
    
    async def search_documents(
        self,
        organization_id: int,
        query: str,
        limit: int = 5
    ) -> List[KnowledgeBase]:
        """
        [DEPRECATED - Use search_documents_for_organization or search_chunks_for_organization instead]
        Search through documents using semantic search with embeddings
        Only searches through knowledge base documents (is_knowledge_base=True)
        Returns knowledge base chunks
        """
        try:
            logger.info(f"Starting semantic search for org {organization_id}. Query: '{query}'")
            start_time = datetime.now()

            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")

            embedding_duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Generated query embedding (Duration: {embedding_duration}s)")

            async with async_session_maker() as session:
                # Using pgvector's L2 distance to find similar chunks
                # Only search in knowledge base chunks (is_knowledge_base=True)
                search_start = datetime.now()
                stmt = select(KnowledgeBase).where(
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_knowledge_base == True
                ).order_by(
                    func.l2_distance(KnowledgeBase.embedding, query_embedding[0])
                ).limit(limit)

                result = await session.execute(stmt)
                chunks = result.scalars().all()

                search_duration = (datetime.now() - search_start).total_seconds()
                logger.info(f"Found {len(chunks)} relevant chunks for query (Search duration: {search_duration}s)")

                # Log similarity scores for debugging
                if chunks:
                    for chunk in chunks:
                        similarity = 1 - func.l2_distance(chunk.embedding, query_embedding[0])
                        logger.debug(f"Chunk {chunk.id} similarity score: {similarity}")

                return chunks
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}", exc_info=True)
            raise
                
    async def search_chunks_for_user(
        self,
        user_id: int,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int]:
        """
        Search through all document chunks the user has access to across all organizations
        Returns a tuple of (chunks, total_count)
        This uses a single optimized SQL query with window functions for pagination
        """
        try:
            logger.info(f"Starting semantic chunk search across all orgs for user {user_id}. Query: '{query}'")
            start_time = datetime.now()

            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")

            embedding_duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Generated query embedding (Duration: {embedding_duration}s)")

            # Calculate pagination
            offset = (page - 1) * per_page
            
            async with async_session_maker() as session:
                # Use raw SQL with window functions for better performance
                # This performs pagination and counting in a single query
                search_start = datetime.now()
                
                query_sql = text("""
                    WITH ranked_results AS (
                        SELECT 
                            kb.*,
                            COUNT(*) OVER() as total_count,
                            ROW_NUMBER() OVER(ORDER BY l2_distance(kb.embedding, :query_embedding::vector)) as row_num
                        FROM 
                            knowledge_base kb
                        JOIN
                            organizations org ON kb.organization_id = org.id
                        JOIN
                            user_organizations uo ON org.id = uo.organization_id
                        WHERE
                            uo.user_id = :user_id
                            AND kb.is_knowledge_base = TRUE
                    )
                    SELECT * FROM ranked_results
                    WHERE row_num > :offset AND row_num <= :offset_end
                    ORDER BY row_num
                """)
                
                result = await session.execute(
                    query_sql, 
                    {
                        "query_embedding": query_embedding[0],
                        "user_id": user_id,
                        "offset": offset,
                        "offset_end": offset + per_page
                    }
                )
                
                rows = result.fetchall()
                
                # Extract total count from first row if results exist
                total_count = rows[0].total_count if rows else 0
                
                # Convert rows to KnowledgeBase objects
                chunks = [
                    KnowledgeBase(
                        id=row.id,
                        fp=row.fp,
                        file_id=row.file_id,
                        organization_id=row.organization_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        embedding=row.embedding,
                        meta_info=row.meta_info,
                        is_knowledge_base=row.is_knowledge_base,
                        created_at=row.created_at
                    ) for row in rows
                ]

                search_duration = (datetime.now() - search_start).total_seconds()
                logger.info(f"Found {len(chunks)} relevant chunks for user {user_id} (Total: {total_count}, Search duration: {search_duration}s)")

                return chunks, total_count
        except Exception as e:
            logger.error(f"Error searching chunks for user: {str(e)}", exc_info=True)
            raise
            
    async def search_documents_for_organization(
        self,
        organization_id: int,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[File], int]:
        """
        Search for documents (files) within a specific organization
        Returns a tuple of (files, total_count)
        This uses vector similarity search and pagination
        """
        try:
            # Generate embedding for query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")
            
            embedding = query_embedding[0]
            
            # Calculate offset
            offset = (page - 1) * per_page
            
            async with async_session_maker() as session:
                # First get the total count with a separate query for better performance
                count_statement = text("""
                    SELECT COUNT(DISTINCT f.id) as total_count
                    FROM files f
                    JOIN knowledge_base kb ON f.id = kb.file_id
                    WHERE f.organization_id = :org_id
                    AND f.status = 'COMPLETED'
                    AND kb.embedding IS NOT NULL
                    AND (
                        f.filename ILIKE :query_text
                        OR kb.content ILIKE :query_text
                        OR kb.embedding <=> :embedding < 0.2
                    )
                """)
                
                count_result = await session.execute(
                    count_statement,
                    {
                        "org_id": organization_id,
                        "query_text": f"%{query}%", 
                        "embedding": embedding
                    }
                )
                total_count = count_result.scalar() or 0
                
                # Now get the actual results with pagination
                # Note: We're using DISTINCT ON to get one row per file, ordered by the best match
                statement = text("""
                    WITH ranked_files AS (
                        SELECT DISTINCT ON (f.id) 
                            f.id,
                            f.fp,
                            f.filename,
                            f.file_type,
                            f.status,
                            f.created_at,
                            f.organization_id,
                            f.is_knowledge_base,
                            f.s3_key,
                            f.user_id,
                            (
                                SELECT COUNT(*) 
                                FROM knowledge_base kb_count 
                                WHERE kb_count.file_id = f.id
                            ) as chunk_count,
                            MIN(kb.embedding <=> :embedding) as similarity_score
                        FROM files f
                        JOIN knowledge_base kb ON f.id = kb.file_id
                        WHERE f.organization_id = :org_id
                        AND f.status = 'COMPLETED'
                        AND kb.embedding IS NOT NULL
                        AND (
                            f.filename ILIKE :query_text
                            OR kb.content ILIKE :query_text
                            OR kb.embedding <=> :embedding < 0.2
                        )
                        GROUP BY f.id, f.fp, f.filename, f.file_type, f.status, 
                            f.created_at, f.organization_id, f.is_knowledge_base, f.s3_key, f.user_id
                    )
                    SELECT * FROM ranked_files
                    ORDER BY similarity_score
                    LIMIT :limit OFFSET :offset
                """)
                
                result = await session.execute(
                    statement,
                    {
                        "org_id": organization_id,
                        "query_text": f"%{query}%", 
                        "embedding": embedding,
                        "limit": per_page,
                        "offset": offset
                    }
                )
                
                # Convert the result to File objects
                files = []
                for row in result:
                    file = File(
                        id=row.id,
                        fp=row.fp,
                        filename=row.filename,
                        file_type=row.file_type,
                        status=row.status,
                        created_at=row.created_at,
                        organization_id=row.organization_id,
                        is_knowledge_base=row.is_knowledge_base,
                        s3_key=row.s3_key,
                        user_id=row.user_id
                    )
                    # Add chunk_count as an attribute
                    setattr(file, 'chunk_count', row.chunk_count)
                    files.append(file)
                
                return files, total_count
                
        except Exception as e:
            logger.error(f"Error searching documents in organization {organization_id}: {str(e)}", exc_info=True)
            raise
            
    async def search_documents_for_user(
        self,
        user_id: int,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[File], int]:
        """
        Search for documents (files) the user has access to across all organizations
        Returns a tuple of (files, total_count)
        This uses a single optimized SQL query with vector similarity search and window functions for pagination
        """
        try:
            logger.info(f"Starting document search across all orgs for user {user_id}. Query: '{query}'")
            start_time = datetime.now()

            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")

            embedding_duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Generated query embedding (Duration: {embedding_duration}s)")

            # Calculate pagination
            offset = (page - 1) * per_page
            
            async with async_session_maker() as session:
                # Use raw SQL with window functions for better performance
                # This performs pagination and counting in a single query
                search_start = datetime.now()
                
                # First find most relevant chunks, then get their parent files with aggregation
                query_sql = text("""
                    WITH relevant_chunks AS (
                        SELECT 
                            kb.file_id,
                            MIN(l2_distance(kb.embedding, :query_embedding::vector)) as min_distance
                        FROM 
                            knowledge_base kb
                        JOIN
                            organizations org ON kb.organization_id = org.id
                        JOIN
                            user_organizations uo ON org.id = uo.organization_id
                        WHERE
                            uo.user_id = :user_id
                            AND kb.is_knowledge_base = TRUE
                        GROUP BY kb.file_id
                    ),
                    ranked_files AS (
                        SELECT 
                            f.*,
                            org.fp as organization_fp,
                            rc.min_distance,
                            COUNT(*) OVER() as total_count,
                            ROW_NUMBER() OVER(ORDER BY rc.min_distance) as row_num
                        FROM 
                            files f
                        JOIN
                            relevant_chunks rc ON f.id = rc.file_id
                        JOIN
                            organizations org ON f.organization_id = org.id
                        WHERE
                            f.is_knowledge_base = TRUE
                    )
                    SELECT * FROM ranked_files
                    WHERE row_num > :offset AND row_num <= :offset_end
                    ORDER BY row_num
                """)
                
                result = await session.execute(
                    query_sql, 
                    {
                        "query_embedding": query_embedding[0],
                        "user_id": user_id,
                        "offset": offset,
                        "offset_end": offset + per_page
                    }
                )
                
                rows = result.fetchall()
                
                # Extract total count from first row if results exist
                total_count = rows[0].total_count if rows else 0
                
                # Convert rows to File objects
                files = [
                    File(
                        id=row.id,
                        fp=row.fp,
                        filename=row.filename,
                        file_type=row.file_type,
                        s3_key=row.s3_key,
                        status=row.status,
                        file_size=row.file_size,
                        chunk_count=row.chunk_count,
                        is_knowledge_base=row.is_knowledge_base,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        user_id=row.user_id,
                        organization_id=row.organization_id,
                        organization_fp=row.organization_fp
                    ) for row in rows
                ]

                search_duration = (datetime.now() - search_start).total_seconds()
                logger.info(f"Found {len(files)} relevant documents for user {user_id} (Total: {total_count}, Search duration: {search_duration}s)")

                return files, total_count
        except Exception as e:
            logger.error(f"Error searching documents for user: {str(e)}", exc_info=True)
            raise

    async def get_user_accessible_documents(
        self,
        user_id: int,
        page: int = 1,
        per_page: int = 20,
        organization_fp: Optional[str] = None
    ) -> Tuple[List[File], int]:
        """
        Get all documents accessible to a user across their organizations.
        
        Args:
            user_id: ID of the user
            page: Page number for pagination
            per_page: Items per page
            organization_fp: Optional organization fingerprint to filter by organization
            
        Returns:
            Tuple of (documents, total_count)
        """
        try:
            async with async_session_maker() as session:
                # Get user's organizations
                user_stmt = select(User).where(User.id == user_id).options(
                    joinedload(User.organizations)
                )
                result = await session.execute(user_stmt)
                user = result.unique().scalar_one_or_none()

                if not user:
                    raise ValueError("User not found")

                org_ids = [org.id for org in user.organizations]
                
                # If organization fingerprint is provided, filter by that organization
                if organization_fp:
                    logger.info(f"Filtering documents by organization fingerprint: {organization_fp}")
                    
                    # Get organization by fingerprint
                    org_stmt = select(Organization).where(
                        Organization.fp == organization_fp,
                        Organization.id.in_(org_ids)  # Ensure user has access to this organization
                    )
                    result = await session.execute(org_stmt)
                    organization = result.scalar_one_or_none()
                    
                    if not organization:
                        logger.warning(f"Organization with fingerprint {organization_fp} not found or user has no access")
                        return [], 0
                    
                    # Use only this organization_id for filtering
                    org_ids = [organization.id]
                    logger.debug(f"Filtered to organization ID: {organization.id}")

                # Count total documents
                count_stmt = select(func.count(File.id)).where(
                    File.organization_id.in_(org_ids)
                )
                result = await session.execute(count_stmt)
                total_count = result.scalar()

                # Get paginated documents without knowledge base
                offset = (page - 1) * per_page
                stmt = select(File).where(
                    File.organization_id.in_(org_ids)
                ).order_by(
                    File.created_at.desc()
                ).offset(offset).limit(per_page)

                result = await session.execute(stmt)
                documents = result.scalars().all()

                return documents, total_count

        except Exception as e:
            logger.error(f"Error fetching user accessible documents: {str(e)}")
            raise

    async def get_organization_documents_paginated(
        self,
        organization_id: int,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[File], int]:
        """Get paginated documents for an organization"""
        try:
            async with async_session_maker() as session:
                # Count total documents
                count_stmt = select(func.count(File.id)).where(
                    File.organization_id == organization_id
                )
                result = await session.execute(count_stmt)
                total_count = result.scalar()

                # Get paginated documents without knowledge base
                offset = (page - 1) * per_page
                stmt = select(File).where(
                    File.organization_id == organization_id
                ).order_by(
                    File.created_at.desc()
                ).offset(offset).limit(per_page)

                result = await session.execute(stmt)
                documents = result.scalars().all()

                return documents, total_count

        except Exception as e:
            logger.error(f"Error fetching organization documents: {str(e)}")
            raise

    async def get_document_by_id(
        self,
        organization_id: int,
        document_id: int
    ) -> Optional[File]:
        """Get a single document by ID within an organization"""
        try:
            async with async_session_maker() as session:
                stmt = select(File).where(
                    File.id == document_id,
                    File.organization_id == organization_id
                ).options(
                    joinedload(File.knowledge_base)
                )
                result = await session.execute(stmt)
                document = result.unique().scalar_one_or_none()
                return document

        except Exception as e:
            logger.error(f"Error fetching document by ID: {str(e)}")
            raise
    
    async def get_organization_by_fp(
        self,
        organization_fp: str
    ) -> Optional[Organization]:
        """Get organization by fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                stmt = select(Organization).where(Organization.fp == organization_fp)
                result = await session.execute(stmt)
                organization = result.scalar_one_or_none()
                
                if not organization:
                    logger.warning(f"Organization with fingerprint {organization_fp} not found")
                return organization
                
        except Exception as e:
            logger.error(f"Error fetching organization by FP: {str(e)}")
            raise
            
    async def get_document_by_fp(
        self,
        organization_id: int,
        document_fp: str
    ) -> Optional[File]:
        """Get a single document by fingerprint (fp) within an organization"""
        try:
            async with async_session_maker() as session:
                stmt = select(File).where(
                    File.fp == document_fp,
                    File.organization_id == organization_id
                ).options(
                    joinedload(File.knowledge_base)
                )
                result = await session.execute(stmt)
                document = result.unique().scalar_one_or_none()
                return document

        except Exception as e:
            logger.error(f"Error fetching document by FP: {str(e)}")
            raise
            
    async def get_document_chunks(
        self,
        document_id: int,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int]:
        """
        Get paginated chunks for a specific document by ID
        Returns a tuple of (chunks, total_count)
        """
        try:
            async with async_session_maker() as session:
                # Count total chunks
                count_stmt = select(func.count(KnowledgeBase.id)).where(
                    KnowledgeBase.file_id == document_id
                )
                result = await session.execute(count_stmt)
                total_count = result.scalar() or 0
                
                # Calculate pagination
                offset = (page - 1) * per_page
                
                # Get paginated chunks
                stmt = select(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id
                ).order_by(
                    KnowledgeBase.chunk_index
                ).offset(offset).limit(per_page)
                
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                
                return chunks, total_count
                
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
            
    async def get_document_chunks_by_fp(
        self,
        document_fp: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int, Optional[File]]:
        """
        Get paginated chunks for a specific document by fingerprint (fp)
        Returns a tuple of (chunks, total_count, document)
        """
        try:
            async with async_session_maker() as session:
                # First get the document
                doc_stmt = select(File).where(
                    File.fp == document_fp
                )
                result = await session.execute(doc_stmt)
                document = result.scalar_one_or_none()
                
                if not document:
                    return [], 0, None
                
                # Now get chunks with pagination
                chunks, total_count = await self.get_document_chunks(
                    document_id=document.id,
                    page=page,
                    per_page=per_page
                )
                
                return chunks, total_count, document
                
        except Exception as e:
            logger.error(f"Error getting document chunks by FP: {str(e)}")
            raise
    
    async def search_chunks_in_document(
        self,
        document_id: int,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int]:
        """
        Search through chunks in a specific document using semantic search
        Returns a tuple of (chunks, total_count)
        """
        try:
            logger.info(f"Starting semantic chunk search in document {document_id}. Query: '{query}'")
            start_time = datetime.now()

            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate embedding for search query")
                raise ValueError("Failed to generate embedding for search query")

            embedding_duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Generated query embedding (Duration: {embedding_duration}s)")
            
            # Calculate pagination
            offset = (page - 1) * per_page
            
            async with async_session_maker() as session:
                # Use raw SQL with window functions for better performance
                # This performs pagination and counting in a single query
                search_start = datetime.now()
                
                query_sql = text("""
                    WITH ranked_results AS (
                        SELECT 
                            kb.*,
                            COUNT(*) OVER() as total_count,
                            ROW_NUMBER() OVER(ORDER BY l2_distance(kb.embedding, :query_embedding::vector)) as row_num
                        FROM 
                            knowledge_base kb
                        WHERE
                            kb.file_id = :document_id
                    )
                    SELECT * FROM ranked_results
                    WHERE row_num > :offset AND row_num <= :offset_end
                    ORDER BY row_num
                """)
                
                result = await session.execute(
                    query_sql, 
                    {
                        "query_embedding": query_embedding[0],
                        "document_id": document_id,
                        "offset": offset,
                        "offset_end": offset + per_page
                    }
                )
                
                rows = result.fetchall()
                
                # Extract total count from first row if results exist
                total_count = rows[0].total_count if rows else 0
                
                # Convert rows to KnowledgeBase objects
                chunks = [
                    KnowledgeBase(
                        id=row.id,
                        fp=row.fp,
                        file_id=row.file_id,
                        organization_id=row.organization_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        embedding=row.embedding,
                        meta_info=row.meta_info,
                        is_knowledge_base=row.is_knowledge_base,
                        created_at=row.created_at
                    ) for row in rows
                ]

                search_duration = (datetime.now() - search_start).total_seconds()
                logger.info(f"Found {len(chunks)} relevant chunks for document {document_id} (Total: {total_count}, Search duration: {search_duration}s)")

                return chunks, total_count
        except Exception as e:
            logger.error(f"Error searching chunks in document: {str(e)}", exc_info=True)
            raise
            
    async def search_chunks_in_document_by_fp(
        self,
        document_fp: str,
        query: str,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[KnowledgeBase], int, Optional[File], Optional[Organization]]:
        """
        Search through chunks in a specific document using semantic search, accessing by fingerprint
        Returns a tuple of (chunks, total_count, document, organization)
        """
        try:
            # First get the document and organization
            async with async_session_maker() as session:
                # Build query to find document and its organization
                stmt = select(File, Organization).join(
                    Organization, File.organization_id == Organization.id
                ).where(
                    File.fp == document_fp
                )
                
                result = await session.execute(stmt)
                row = result.first()
                
                if not row:
                    return [], 0, None, None
                    
                document, organization = row
                
                # Now search chunks
                chunks, total_count = await self.search_chunks_in_document(
                    document_id=document.id,
                    query=query,
                    page=page,
                    per_page=per_page
                )
                
                return chunks, total_count, document, organization
                
        except Exception as e:
            logger.error(f"Error searching chunks in document by FP: {str(e)}", exc_info=True)
            raise
    
    async def get_document_by_fp_for_user(
        self,
        user_id: int,
        document_fp: str
    ) -> Tuple[Optional[File], Optional[Organization]]:
        """
        Get a document by fingerprint across all organizations the user has access to
        Returns both the document and the organization it belongs to
        """
        try:
            async with async_session_maker() as session:
                # Build a more efficient query that joins with user_organizations
                # to find the document in any organization the user has access to
                stmt = select(File, Organization).join(
                    Organization, File.organization_id == Organization.id
                ).join(
                    "user_organizations", # Join through the user_organizations association table
                    Organization.id == Organization.user_organizations.c.organization_id
                ).where(
                    File.fp == document_fp,
                    Organization.user_organizations.c.user_id == user_id
                ).options(
                    joinedload(File.knowledge_base)
                )
                
                result = await session.execute(stmt)
                row = result.first()
                
                if row:
                    return row[0], row[1]  # Return document and organization
                return None, None
                
        except Exception as e:
            logger.error(f"Error fetching document by FP for user: {str(e)}")
            raise

    async def delete_document(
        self,
        organization_id: int,
        document_id: int
    ) -> bool:
        """Delete a document and its associated knowledge base entries by ID"""
        try:
            async with async_session_maker() as session:
                # First verify the document exists and belongs to the organization
                document = await self.get_document_by_id(organization_id, document_id)
                if not document:
                    return False

                # Delete knowledge base entries
                kb_stmt = delete(KnowledgeBase).where(
                    KnowledgeBase.file_id == document_id
                )
                await session.execute(kb_stmt)

                # Delete the document record
                file_stmt = delete(File).where(
                    File.id == document_id,
                    File.organization_id == organization_id
                )
                await session.execute(file_stmt)

                # Delete from S3
                try:
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name,
                        Key=document.s3_key
                    )
                except Exception as e:
                    logger.warning(f"Error deleting file from S3: {str(e)}")

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
            
    async def delete_document_by_fp(
        self,
        organization_id: int,
        document_fp: str
    ) -> bool:
        """Delete a document and its associated knowledge base entries by fingerprint (fp)"""
        try:
            async with async_session_maker() as session:
                # First verify the document exists and belongs to the organization
                document = await self.get_document_by_fp(organization_id, document_fp)
                if not document:
                    return False

                # Delete knowledge base entries
                kb_stmt = delete(KnowledgeBase).where(
                    KnowledgeBase.file_id == document.id
                )
                await session.execute(kb_stmt)

                # Delete the document record
                file_stmt = delete(File).where(
                    File.fp == document_fp,
                    File.organization_id == organization_id
                )
                await session.execute(file_stmt)

                # Delete from S3
                try:
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name,
                        Key=document.s3_key
                    )
                except Exception as e:
                    logger.warning(f"Error deleting file from S3: {str(e)}")

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error deleting document by FP: {str(e)}")
            raise
            
    async def delete_document_by_fp_for_user(
        self,
        user_id: int,
        document_fp: str
    ) -> bool:
        """
        Delete a document and its associated knowledge base entries, 
        checking across all organizations the user has access to
        """
        try:
            # First find the document and its organization
            document, organization = await self.get_document_by_fp_for_user(user_id, document_fp)
            
            if not document or not organization:
                logger.warning(f"Document {document_fp} not found for user {user_id}")
                return False
                
            # Now we can proceed with deletion
            async with async_session_maker() as session:
                # Delete knowledge base entries
                kb_stmt = delete(KnowledgeBase).where(
                    KnowledgeBase.file_id == document.id
                )
                await session.execute(kb_stmt)

                # Delete the document record
                file_stmt = delete(File).where(
                    File.fp == document_fp,
                    File.organization_id == organization.id
                )
                await session.execute(file_stmt)

                # Delete from S3
                try:
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name,
                        Key=document.s3_key
                    )
                except Exception as e:
                    logger.warning(f"Error deleting file from S3: {str(e)}")

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error deleting document by FP for user: {str(e)}")
            raise

    async def get_file_from_s3(self, s3_key: str) -> Optional[bytes]:
        """
        Retrieve file from S3
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error retrieving file from S3: {str(e)}")
            return None