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

logger = logging.getLogger(__name__)

class DocumentService:
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

            # Extract text content
            text_chunks = await self._extract_text_chunks(file, file_type)

            # Generate embeddings
            embeddings = await self.openai_service.create_embeddings(text_chunks)

            # Create knowledge base entries
            for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                kb_entry = KnowledgeBase(
                    chunk_index=idx,
                    content=chunk,
                    embedding=embedding,
                    metadata=json.dumps({
                        "filename": filename,
                        "chunk_number": idx + 1,
                        "total_chunks": len(text_chunks)
                    }),
                    file_id=db_file.id,
                    organization_id=organization_id
                )
                knowledge_base_entries.append(kb_entry)

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
        chunk_size: int = 1000
    ) -> List[str]:
        """
        Extract text from document and split into chunks
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

            # Split text into chunks
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > chunk_size:
                    if current_chunk:  # Only add non-empty chunks
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            logger.info(f"Successfully extracted {len(chunks)} text chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error extracting text from document: {str(e)}")
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