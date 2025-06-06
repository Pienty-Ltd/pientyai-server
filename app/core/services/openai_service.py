import os
import asyncio
import time
import tempfile
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI
from app.core.config import config
import logging

logger = logging.getLogger(__name__)


class OpenAIService:
    BATCH_SIZE = 20  # Maximum number of texts to process in one API call for regular embedding creation
    BATCH_API_MAX_SIZE = 2000  # Maximum number of texts to process in one batch API call
    MAX_RETRIES = 3  # Maximum number of retries for failed API calls
    RETRY_DELAY = 1  # Delay between retries in seconds
    BATCH_CHECK_INTERVAL = 30  # Time in seconds to wait between batch status checks
    BATCH_MAX_WAIT_TIME = 24 * 60 * 60  # Maximum time (24 hours) to wait for a batch to complete

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    async def get_api_info(self) -> Dict[str, str]:
        """
        Test the OpenAI API connection and return API information.
        This is useful for verifying if the API key is valid.
        
        Returns:
            Dict with API status information
        """
        try:
            # Just a simple call to get models to verify connection
            models = self.client.models.list()
            return {
                "status": "connected",
                "models_available": str(len(models.data)),
                "first_model": models.data[0].id if models.data else "none"
            }
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {str(e)}")
            # Don't raise, just return error info
            return {
                "status": "error",
                "error": str(e)
            }
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text string
        
        Args:
            text: The text to create an embedding for
            
        Returns:
            The embedding vector as a list of floats
        """
        # Metin boş ise veya None ise varsayılan bir vektör döndür
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding generation")
            return [0.0] * 1536  # Ada-002 has 1536 dimensions
            
        try:
            retry_count = 0
            
            # API anahtarını kontrol et
            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured")
                logger.warning("Using zero vector due to missing API key")
                return [0.0] * 1536
                
            # Metni temizle ve hazırla
            # Çok uzun ise kısalt (OpenAI limitlerine uygun)
            cleaned_text = text.strip()
            if len(cleaned_text) > 8000:
                logger.warning(f"Text too long ({len(cleaned_text)} chars), truncating to 8000 chars")
                cleaned_text = cleaned_text[:8000]
            
            while retry_count < self.MAX_RETRIES:
                try:
                    logger.debug(f"Creating embedding for text: {cleaned_text[:50]}...")
                    
                    # Run the synchronous API call in a thread pool to avoid blocking
                    response = await asyncio.to_thread(
                        self.client.embeddings.create,
                        model="text-embedding-ada-002",
                        input=[cleaned_text])
                    
                    # Extract the embedding vector from the response
                    embedding = response.data[0].embedding
                    
                    # Vektörün boyutunu kontrol et
                    if len(embedding) != 1536:
                        logger.warning(f"Unexpected embedding dimensions: {len(embedding)}, expected 1536")
                    
                    logger.debug(f"Successfully created embedding vector with {len(embedding)} dimensions")
                    return embedding
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Error creating embedding (attempt {retry_count}/{self.MAX_RETRIES}): {str(e)}"
                    )
                    
                    if retry_count < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY * (2 ** (retry_count - 1)))
                    else:
                        logger.error(f"Failed to create embedding after {self.MAX_RETRIES} attempts")
                        raise
            
            # Bu noktaya gelmemesi gerekir, ama yine de bir return değeri ekleyelim
            logger.error("Unexpected code path in get_embedding")
            return [0.0] * 1536
            
        except Exception as e:
            logger.error(f"Error in get_embedding: {str(e)}")
            # Return a zero vector as a last resort
            return [0.0] * 1536  # Ada-002 has 1536 dimensions
            
    async def create_embeddings(self,
                                text_chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks using OpenAI's API
        Implements batching and retry logic for better reliability
        """
        all_embeddings = []

        try:
            # Process chunks in batches
            for i in range(0, len(text_chunks), self.BATCH_SIZE):
                batch = text_chunks[i:i + self.BATCH_SIZE]
                retry_count = 0

                while retry_count < self.MAX_RETRIES:
                    try:
                        logger.debug(
                            f"Attempting to create embeddings for batch of size {len(batch)}"
                        )

                        # Run the synchronous API call in a thread pool to avoid blocking
                        response = await asyncio.to_thread(
                            self.client.embeddings.create,
                            model="text-embedding-ada-002",
                            input=batch)

                        logger.debug(f"Response type: {type(response)}")
                        logger.debug(f"Response content: {response}")

                        # Extract embeddings from response data
                        batch_embeddings = [
                            item.embedding for item in response.data
                        ]
                        logger.debug(
                            f"Successfully extracted {len(batch_embeddings)} embeddings"
                        )
                        logger.debug(
                            f"Sample embedding length: {len(batch_embeddings[0]) if batch_embeddings else 0}"
                        )

                        all_embeddings.extend(batch_embeddings)
                        logger.info(
                            f"Successfully generated embeddings for batch {i//self.BATCH_SIZE + 1}"
                        )
                        break  # Success, exit retry loop

                    except Exception as e:
                        error_details = str(e)
                        # Özellikle OpenAI API hataları için ayrıntılı hata bilgilerini yakala
                        if hasattr(e, 'response') and hasattr(e.response, 'json'):
                            try:
                                error_json = e.response.json()
                                error_details = f"Error code: {e.response.status_code} - {error_json}"
                                logger.error(f"OpenAI API error details: {error_json}")
                            except:
                                pass
                                
                        retry_count += 1
                        if retry_count == self.MAX_RETRIES:
                            logger.error(
                                f"Failed to generate embeddings after {self.MAX_RETRIES} retries: {error_details}"
                            )
                            raise Exception(f"Error creating embeddings: {error_details}")

                        logger.warning(
                            f"Retry {retry_count}/{self.MAX_RETRIES} for batch {i//self.BATCH_SIZE + 1}: {error_details}"
                        )
                        await asyncio.sleep(self.RETRY_DELAY * retry_count
                                            )  # Exponential backoff

                # Add a small delay between batches to avoid rate limits
                if i + self.BATCH_SIZE < len(text_chunks):
                    await asyncio.sleep(0.5)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
            
    async def create_batch_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks using OpenAI's API
        This method handles large volumes of text chunks in a batch request.
        
        For very large datasets (>10,000 items), consider using create_async_batch_embeddings() 
        which uses OpenAI's true asynchronous Batch API.
        
        Args:
            text_chunks: List of text chunks to convert to embeddings
            
        Returns:
            List of embedding vectors, one for each input text chunk
        """
        if not text_chunks:
            logger.warning("Empty text chunks provided to create_batch_embeddings")
            return []
            
        # Ensure we don't exceed the maximum batch size
        if len(text_chunks) > self.BATCH_API_MAX_SIZE:
            logger.warning(f"Input size ({len(text_chunks)}) exceeds max batch size ({self.BATCH_API_MAX_SIZE}). "
                          f"Will process in multiple batch requests.")
            
            # Process in multiple batch API calls if needed
            all_embeddings = []
            for i in range(0, len(text_chunks), self.BATCH_API_MAX_SIZE):
                batch = text_chunks[i:i + self.BATCH_API_MAX_SIZE]
                batch_embeddings = await self.create_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        
        try:
            logger.info(f"Creating embeddings for {len(text_chunks)} text chunks using batch processing")
            retry_count = 0
            
            while retry_count < self.MAX_RETRIES:
                try:
                    # Submit the request - run in a thread pool to prevent blocking
                    logger.debug(f"Submitting embedding request with {len(text_chunks)} inputs")
                    
                    response = await asyncio.to_thread(
                        self.client.embeddings.create,
                        model="text-embedding-ada-002",
                        input=text_chunks  # OpenAI client handles batching internally
                    )
                    
                    # Extract embeddings from response
                    all_embeddings = [item.embedding for item in response.data]
                    
                    logger.info(f"Successfully received {len(all_embeddings)} embeddings from batch request")
                    logger.debug(f"Sample embedding length: {len(all_embeddings[0]) if all_embeddings else 0}")
                    
                    return all_embeddings
                    
                except Exception as e:
                    error_details = str(e)
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            error_json = e.response.json()
                            error_details = f"Error code: {e.response.status_code} - {error_json}"
                            logger.error(f"OpenAI API error details: {error_json}")
                        except:
                            pass
                    
                    retry_count += 1
                    if retry_count == self.MAX_RETRIES:
                        logger.error(f"Failed to generate embeddings after {self.MAX_RETRIES} retries: {error_details}")
                        raise Exception(f"Error creating embeddings: {error_details}")
                    
                    logger.warning(f"Retry {retry_count}/{self.MAX_RETRIES} for embeddings: {error_details}")
                    await asyncio.sleep(self.RETRY_DELAY * retry_count)  # Exponential backoff
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    async def create_async_batch_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks using OpenAI's true asynchronous Batch API
        This method follows the two-step process for Batch API:
        1. Create a batch job with the input file
        2. Wait for the job to complete and retrieve the results
        
        This is optimal for very large datasets (>10,000 items) where immediate response is not required.
        The batch will be processed asynchronously and will complete within 24 hours (often much faster).
        
        Args:
            text_chunks: List of text chunks to convert to embeddings
            
        Returns:
            List of embedding vectors, one for each input text chunk
        """
        if not text_chunks:
            logger.warning("Empty text chunks provided to create_async_batch_embeddings")
            return []
            
        # If the batch is small enough, use the synchronous method instead
        if len(text_chunks) <= self.BATCH_SIZE * 5:  # Use batch API only for significant workloads
            logger.info(f"Input size ({len(text_chunks)}) is small, using standard batch method instead of async batch API")
            return await self.create_batch_embeddings(text_chunks)
        
        try:
            # Step 1: Prepare input file for batch processing
            logger.info(f"Preparing async batch embedding request for {len(text_chunks)} chunks")
            
            # Create a temp file to hold the batch input data
            batch_id = str(uuid.uuid4())
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as batch_file:
                batch_file_path = batch_file.name
                
                # Create a JSONL file with each line containing a request for embeddings
                for i, chunk in enumerate(text_chunks):
                    # Create a proper batch API request entry for each chunk
                    entry = {
                        "custom_id": f"chunk_{i}",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "text-embedding-ada-002",
                            "input": chunk
                        }
                    }
                    batch_file.write(json.dumps(entry) + '\n')
                
            logger.info(f"Created batch input file at {batch_file_path} with {len(text_chunks)} entries")
            
            # Step 2: Upload the file for batch processing
            try:
                logger.info("Uploading batch input file to OpenAI")
                batch_file_obj = await asyncio.to_thread(
                    self.client.files.create,
                    file=open(batch_file_path, "rb"),
                    purpose="batch"
                )
                batch_file_id = batch_file_obj.id
                logger.info(f"Successfully uploaded batch file with ID: {batch_file_id}")
                
                # Step 3: Create the batch job
                logger.info("Creating batch embedding job")
                batch_job = await asyncio.to_thread(
                    self.client.batches.create,
                    input_file_id=batch_file_id,
                    endpoint="/v1/embeddings",
                    completion_window="24h"
                )
                
                batch_id = batch_job.id
                logger.info(f"Created batch job with ID: {batch_id}, current status: {batch_job.status}")
                
                # Step 4: Poll for batch completion
                start_time = time.time()
                while time.time() - start_time < self.BATCH_MAX_WAIT_TIME:
                    batch_status = await asyncio.to_thread(
                        self.client.batches.retrieve,
                        batch_id
                    )
                    
                    current_status = batch_status.status
                    logger.info(f"Batch job {batch_id} status: {current_status}")
                    
                    if current_status == "completed":
                        # Job completed, retrieve results
                        logger.info(f"Batch job {batch_id} completed, retrieving results")
                        output_file_id = batch_status.output_file_id
                        
                        if not output_file_id:
                            raise Exception(f"Batch job completed but no output file ID was provided")
                        
                        # Download the results file
                        file_response = await asyncio.to_thread(
                            self.client.files.content,
                            output_file_id
                        )
                        result_content = file_response.text
                        
                        # Parse the results
                        embeddings_by_chunk_id = {}
                        for line in result_content.splitlines():
                            result = json.loads(line)
                            custom_id = result.get("custom_id")
                            response_data = result.get("response", {}).get("body", {})
                            
                            # Extract chunk index from custom_id
                            chunk_idx = int(custom_id.split("_")[1]) if custom_id.startswith("chunk_") else -1
                            
                            # Extract embedding from response
                            if response_data and "data" in response_data:
                                embedding = response_data["data"][0]["embedding"]
                                embeddings_by_chunk_id[chunk_idx] = embedding
                        
                        # Reconstruct ordered list of embeddings
                        all_embeddings = []
                        for i in range(len(text_chunks)):
                            if i in embeddings_by_chunk_id:
                                all_embeddings.append(embeddings_by_chunk_id[i])
                            else:
                                logger.warning(f"Missing embedding for chunk {i}, will create it individually")
                                # Create missing embedding individually
                                individual_embedding = await self.create_embeddings([text_chunks[i]])
                                all_embeddings.append(individual_embedding[0])
                        
                        logger.info(f"Successfully processed batch embeddings for {len(all_embeddings)} chunks")
                        return all_embeddings
                    
                    elif current_status in ["failed", "expired", "cancelled"]:
                        error_file_id = batch_status.error_file_id
                        error_msg = f"Batch job {batch_id} failed with status {current_status}"
                        
                        if error_file_id:
                            # Download error file to get details
                            error_response = await asyncio.to_thread(
                                self.client.files.content,
                                error_file_id
                            )
                            error_content = error_response.text
                            error_msg += f", errors: {error_content}"
                        
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    
                    # Job still in progress, wait before checking again
                    await asyncio.sleep(self.BATCH_CHECK_INTERVAL)
                
                # Time limit exceeded
                raise Exception(f"Batch job {batch_id} timed out after {self.BATCH_MAX_WAIT_TIME} seconds")
                
            finally:
                # Clean up the temp file
                try:
                    os.unlink(batch_file_path)
                    logger.debug(f"Removed temporary batch file: {batch_file_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up temporary batch file: {str(cleanup_err)}")
        
        except Exception as e:
            logger.error(f"Error in async batch embeddings process: {str(e)}")
            logger.info("Falling back to standard batch embedding method")
            
            # Fallback to standard batch processing
            return await self.create_batch_embeddings(text_chunks)
    
    async def analyze_document_chunk_with_git_diff(self, 
                                            document_chunk: str,
                                            knowledge_base_chunks: List[Any]) -> Dict[str, Any]:
        """
        Analyze a single document chunk against knowledge base chunks to produce git-like diff changes
        
        Args:
            document_chunk: The text content of the chunk to analyze
            knowledge_base_chunks: List of relevant knowledge base chunks to use as context
            
        Returns:
            Dictionary containing the diff_changes for this chunk
        """
        try:
            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured")
                return {
                    "diff_changes": "Error: OpenAI API key is not configured"
                }
                
            logger.info("Analyzing document chunk with GPT-4.1")
            
            # Format KB chunks for the prompt, extract only content and metadata
            formatted_kb_chunks = []
            for chunk in knowledge_base_chunks:
                # Convert KB object to dict if needed
                if not isinstance(chunk, dict):
                    chunk_dict = {
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "file_id": chunk.file_id
                    }
                else:
                    chunk_dict = chunk
                formatted_kb_chunks.append(chunk_dict)
            
            # Call the main analyze_document method with the chunk
            analysis_result = await self.analyze_document(
                document_chunk=document_chunk,
                knowledge_base_chunks=formatted_kb_chunks
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analyze_document_chunk_with_git_diff: {str(e)}")
            return {
                "diff_changes": f"Error analyzing chunk: {str(e)}",
                "processing_time_seconds": 0,
                "total_chunks_analyzed": 0
            }
    
    async def analyze_document(self,
                               document_chunk: str,
                               knowledge_base_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a complete document against relevant knowledge base chunks using OpenAI API
        
        Args:
            document_chunk: The full text of the document to analyze
            knowledge_base_chunks: List of dictionaries containing relevant knowledge base chunks with metadata
            
        Returns:
            Dictionary containing the comprehensive analysis result with git-like diff changes
        """
        try:
            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured")
                return {
                    "diff_changes": "Error: OpenAI API key is not configured"
                }

            logger.info("Starting document analysis with OpenAI API using GPT-4.1")

            # Construct a prompt for creating a fully corrected document
            system_prompt = """You are an expert legal document analysis AI specializing in analyzing and correcting legal documents. Your task is to analyze the provided document and create a fully corrected version based on the knowledge base information.

            YOUR CORE RESPONSIBILITY:
            Create a completely corrected version of the provided document. Return the entire document with all necessary corrections already applied AND a detailed list of changes made.

            ANALYSIS METHODOLOGY:
            - Compare each section of the document against relevant knowledge base entries
            - Correct any sections that don't comply with policies or standards in the knowledge base
            - Use the exact same formatting, language, and style as the original document
            - Preserve section numbers, formatting, and document structure
            - DO NOT use diff format or show changes with markers - just return the fully corrected document
            - TRACK all modifications in a separate 'changes' array

            REQUIRED RESPONSE FORMAT (JSON):
            {
              "diff_changes": "The complete corrected document with all changes already applied (not a diff)",
              "changes": [
                {"original": "Original text segment that was changed", "corrected": "New corrected version of this segment", "reason": "Brief explanation of why this change was made"}
              ],
              "processing_time_seconds": 0,
              "total_chunks_analyzed": 0
            }

            CRITICAL INSTRUCTIONS:
            - MATCH DOCUMENT LANGUAGE - Detect the language of the input document and respond in EXACTLY the same language
            - For Turkish documents, respond in Turkish. For English documents, respond in English, etc.
            - FORMAT CONSISTENTLY - Keep all section references in the same format as the original document
            - DO NOT mark changes with + or - symbols - just provide the final document as it should be
            - You MUST include a 'changes' array with detailed information about each modification - this field is REQUIRED
            - All responses MUST contain the 'changes' array even if no changes were made
            - Make corrections based on policies and information found in the knowledge base
            - Return the ENTIRE document, not just sections that were changed
            
            REMEMBER: 
            1. Return the completely corrected document in the 'diff_changes' field
            2. You MUST provide the 'changes' array with each change including original text, corrected text, and reason
            3. The 'changes' array is a required field and must always be included in your response
            """

            # Format the knowledge base chunks as a JSON array for better structure
            kb_chunks_with_note = {
                "total_kb_chunks": len(knowledge_base_chunks),
                "kb_chunks": knowledge_base_chunks
            }
            kb_chunks_json = json.dumps(kb_chunks_with_note, ensure_ascii=False, indent=2)
            
            user_message = f"""
            ## KNOWLEDGE BASE CONTEXT (REFERENCE INFORMATION - JSON OBJECT):
            ```json
            {kb_chunks_json}
            ```
            
            ## DOCUMENT TO ANALYZE:
            {document_chunk}
            
            DETAILED INSTRUCTIONS:
            1. Review all {len(knowledge_base_chunks)} knowledge base chunks in the data
            2. Compare the document with the knowledge base entries to identify necessary changes
            3. Create a fully corrected version of the document based on the knowledge base content
            4. Maintain the original document's format and structure
            5. Return the ENTIRE document with corrections already applied
            6. TRACK all changes in a separate 'changes' array
            
            OUTPUT REQUIREMENTS:
            - Return the COMPLETE document with all corrections already applied in "diff_changes" field
            - Do NOT show annotations, marks, or indicate what has changed in the main document
            - INCLUDE a separate 'changes' array with detailed information about each modification
            - For each change, include: original text, corrected text, and reason for change
            - Make all changes in the EXACT same format and language as the original document
            - Ensure ALL necessary corrections are applied based on the knowledge base information
            
            Remember: Respond in EXACTLY the same language as the document being analyzed, with the same terminology and formatting. Return both the fully corrected document AND the detailed change list.
            """

            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    # Run the synchronous API call in a thread pool to avoid blocking
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="gpt-4.1",  # Using GPT-4.1 model instead of o3-mini
                        messages=[{
                            "role": "system",
                            "content": system_prompt
                        }, {
                            "role": "user",
                            "content": user_message
                        }],
                        response_format={"type": "json_object"}  # Ensure JSON response
                    )

                    logger.debug(f"OpenAI API response received: {response}")

                    # Extract and parse the response content
                    response_content = response.choices[0].message.content

                    try:
                        analysis_result = json.loads(response_content)
                        
                        # GPT-4.1 prompt yanıtını zorlamamıza rağmen istenilen formatta dönmeyebilir
                        # Bu durumda gerekli alanları ekleyelim ve uyumlu hale getirelim
                        
                        # Eğer eski formatta yanıt döndüyse, yeni formata dönüştür
                        if "diff_changes" not in analysis_result:
                            # Sadece diff_changes formatını kullan, diğer tüm alanları yoksay
                            return {
                                "diff_changes": analysis_result.get("diff", ""),
                                "changes": [],  # Boş changes array ekle
                                "processing_time_seconds": response.usage.total_tokens / 1000,  # Yaklaşık işlem süresi
                                "total_chunks_analyzed": len(knowledge_base_chunks)
                            }
                        
                        # "changes" alanı yoksa ekle - bu kritik bir gereksinim
                        if "changes" not in analysis_result:
                            # Bir hata log'u ekleyelim
                            logger.warning("OpenAI response format is missing 'changes' array. Attempting to create it.")
                            
                            try:
                                # Değişiklikleri algılamak için dokümanı analiz etmeye çalışalım
                                # NOT: Burada basitleştirilmiş bir yaklaşım kullanıyoruz, gerçek değişiklikleri belirlemek 
                                # için daha gelişmiş bir algoritma gerekebilir
                                
                                # Basit değişiklik listesi oluştur
                                analysis_result["changes"] = [
                                    {
                                        "original": "Değişiklikler otomatik olarak algılanamadı.",
                                        "corrected": "Değişiklikler OpenAI'dan alınamadı, ancak düzeltilmiş doküman mevcut.",
                                        "reason": "OpenAI API 'changes' dizisini döndürmedi."
                                    }
                                ]
                                logger.info("Created a placeholder changes array since OpenAI didn't provide one")
                            except Exception as change_err:
                                # Değişiklik oluşturma hatası durumunda boş dizi kullan
                                analysis_result["changes"] = []
                                logger.error(f"Error creating changes array: {str(change_err)}")
                        
                        # Yapılandırılmış formatta olduğundan emin olalım
                        if not isinstance(analysis_result["changes"], list):
                            logger.warning("OpenAI returned 'changes' but it's not an array. Converting.")
                            # Dizi değilse dizi haline getir
                            original_changes = analysis_result["changes"]
                            analysis_result["changes"] = [
                                {
                                    "original": "Format hatası",
                                    "corrected": "Düzeltilmiş içerik mevcut",
                                    "reason": f"OpenAI API yanıtı düzgün yapılandırılmamış: {str(original_changes)[:100]}"
                                }
                            ]
                        
                        return analysis_result
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON response: {str(e)}")
                        # If JSON parsing fails, return the raw content in a structured format
                        # Hata durumunda sadece diff_changes alanını döndür
                        return {
                            "diff_changes": "Error: " + str(e) + "\n\nRaw response: " + response_content,
                            "processing_time_seconds": response.usage.total_tokens / 1000,
                            "total_chunks_analyzed": len(knowledge_base_chunks)
                        }

                except Exception as e:
                    error_details = str(e)
                    # Özellikle OpenAI API hataları için ayrıntılı hata bilgilerini yakala
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            error_json = e.response.json()
                            error_details = f"Error code: {e.response.status_code} - {error_json}"
                            logger.error(f"OpenAI API error details: {error_json}")
                        except:
                            pass
                        
                    retry_count += 1
                    if retry_count == self.MAX_RETRIES:
                        logger.error(
                            f"Failed to get analysis after {self.MAX_RETRIES} retries: {error_details}"
                        )
                        # Özel bir hata objesi yarat ve yükselt
                        error = Exception(f"Error in document analysis: {error_details}")
                        raise error

                    logger.warning(
                        f"Retry {retry_count}/{self.MAX_RETRIES} for document analysis: {error_details}"
                    )
                    await asyncio.sleep(self.RETRY_DELAY * retry_count
                                        )  # Exponential backoff

        except Exception as e:
            error_msg = f"Error in document analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return a structured error response with only diff_changes format
            return {
                "diff_changes": f"Analysis failed: {error_msg}",
                "processing_time_seconds": 0.0,
                "total_chunks_analyzed": len(knowledge_base_chunks) if 'knowledge_base_chunks' in locals() else 0
            }
