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
from app.database.models.db_models import File, KnowledgeBase, FileStatus
from app.core.services.openai_service import OpenAIService
from sqlalchemy import select, desc
from sqlalchemy.sql.expression import func
from app.database.database_factory import async_session_maker

logger = logging.getLogger(__name__)

class DocumentService:
    CHUNK_SIZE = 1000  # Characters per chunk
    BATCH_SIZE = 20    # Number of chunks to process in one batch

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        self.bucket_name = config.AWS_BUCKET_NAME
        self.openai_service = OpenAIService()

    async def process_document(
        self,
        file: BinaryIO,
        filename: str,
        file_type: str,
        user_id: int,
        organization_id: int
    ) -> Tuple[File, List[KnowledgeBase]]:
        """
        Process uploaded document: upload to S3, extract text, create embeddings
        Implements chunking and batch processing for better performance
        """
        # Initialize file record
        db_file = File(
            filename=filename,
            file_type=file_type.lower(),
            status=FileStatus.PROCESSING,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=f"documents/{organization_id}/{filename}"
        )

        knowledge_base_entries = []

        try:
            # Upload to S3
            try:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    db_file.s3_key
                )
                logger.info(f"File uploaded to S3: {db_file.s3_key}")
            except ClientError as e:
                logger.error(f"Error uploading file to S3: {str(e)}")
                db_file.status = FileStatus.FAILED
                raise

            # Extract text content and split into chunks
            text_chunks = await self._extract_text_chunks(file, file_type)
            logger.info(f"Extracted {len(text_chunks)} chunks from document")

            # Process chunks in batches
            chunk_batches = [text_chunks[i:i + self.BATCH_SIZE] 
                           for i in range(0, len(text_chunks), self.BATCH_SIZE)]

            async with async_session_maker() as session:
                for batch_idx, chunk_batch in enumerate(chunk_batches):
                    try:
                        # Generate embeddings for the batch
                        embeddings = await self.openai_service.create_embeddings(chunk_batch)
                        logger.info(f"Generated embeddings for batch {batch_idx + 1}/{len(chunk_batches)}")

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
                                    "total_chunks": len(text_chunks)
                                }),
                                file_id=db_file.id,
                                organization_id=organization_id
                            )
                            batch_entries.append(kb_entry)
                            knowledge_base_entries.append(kb_entry)

                        # Bulk insert the batch
                        session.add_all(batch_entries)
                        await session.commit()
                        logger.info(f"Saved batch {batch_idx + 1} to database")

                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                        # Continue with next batch despite errors
                        continue

            db_file.status = FileStatus.COMPLETED
            return db_file, knowledge_base_entries

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            if db_file:
                db_file.status = FileStatus.FAILED
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

    async def search_documents(
        self,
        organization_id: int,
        query: str,
        limit: int = 5
    ) -> List[KnowledgeBase]:
        """
        Search through documents using semantic search with embeddings
        """
        try:
            # Generate embedding for search query
            query_embedding = await self.openai_service.create_embeddings([query])
            if not query_embedding or len(query_embedding) == 0:
                raise ValueError("Failed to generate embedding for search query")

            async with async_session_maker() as session:
                # Using pgvector's L2 distance to find similar chunks
                stmt = select(KnowledgeBase).where(
                    KnowledgeBase.organization_id == organization_id
                ).order_by(
                    func.l2_distance(KnowledgeBase.embedding, query_embedding[0])
                ).limit(limit)

                result = await session.execute(stmt)
                chunks = result.scalars().all()

                logger.info(f"Found {len(chunks)} relevant chunks for query: {query}")
                return chunks

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    async def get_organization_documents(
        self,
        organization_id: int
    ) -> List[File]:
        """
        Get all documents for an organization
        """
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