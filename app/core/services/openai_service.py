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
    
    async def analyze_document(self,
                               document_chunk: str,
                               knowledge_base_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a complete document against relevant knowledge base chunks using OpenAI API
        
        Args:
            document_chunk: The full text of the document to analyze
            knowledge_base_chunks: List of dictionaries containing relevant knowledge base chunks with metadata
            
        Returns:
            Dictionary containing the comprehensive analysis result
        """
        try:
            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured")
                return {
                    "analysis":
                    "Error: OpenAI API key is not configured",
                    "key_points": [],
                    "conflicts": [],
                    "recommendations":
                    ["Configure OpenAI API key to enable document analysis"]
                }

            logger.info("Starting document analysis with OpenAI API")

            # Construct an extremely detailed and precise prompt for legal document analysis
            system_prompt = """You are an expert legal document analysis AI specializing in commercial contract analysis, trade law compliance, and corporate policy enforcement. Your primary task is to perform a meticulous and comprehensive analysis of the provided document, using the knowledge base information as authoritative reference material.

            YOUR CORE RESPONSIBILITIES:
            1. LEGAL COMPLIANCE VERIFICATION: Meticulously check if the document complies with relevant commercial laws, trade regulations, and legal requirements of the applicable jurisdiction. Cite specific laws when relevant.
            2. POLICY CONTRADICTION DETECTION: Identify EACH AND EVERY TERM that contradicts company policies found in the knowledge base. 
            3. INTEREST PROTECTION: Find ALL specific instances where contract terms deviate from company interests (e.g., payment terms, liability limitations, interest rates, penalty clauses, etc.)
            4. RISK ANALYSIS: Detect any clause that creates potential legal, financial, or operational risks for the company.
            5. REQUIRED CHANGES: For each identified conflict or issue, provide SPECIFIC CORRECTIVE TEXT that MUST be implemented to comply with company policy.

            ANALYSIS METHODOLOGY:
            - Compare each section of the document against relevant knowledge base entries
            - Do NOT just suggest changes - state definitive corrections that NEED to be made
            - For each issue, state the current problematic term and then provide the EXACT required replacement text
            - Use direct, declarative language such as "MUST BE CHANGED TO" rather than "consider changing"
            - Reference specific sections, page numbers, or paragraph numbers when possible

            REQUIRED RESPONSE FORMAT (JSON):
            {
              "analysis": "Comprehensive analysis focusing on compliance and policy alignment, written in precise language appropriate for legal professionals",
              "key_points": [
                "Detailed key point 1 with exact section reference",
                "Detailed key point 2 focusing on material terms"
              ],
              "conflicts": [
                "SECTION X.Y: Current interest rate of Z% DOES NOT COMPLY with company policy. MUST BE CHANGED to A%.",
                "CLAUSE X.Y.Z: Current delivery terms of N days VIOLATES standard policy. MUST BE CHANGED to M days."
              ],
              "recommendations": [
                "REPLACE Section X.Y text from 'current problematic text' TO 'exact corrective text' to comply with company policy",
                "CHANGE Clause X.Y.Z from 'current problematic text' TO 'exact corrective text' as required by policy"
              ]
            }

            CRITICAL INSTRUCTIONS:
            - BE EXTREMELY DECISIVE - All identified issues REQUIRE correction, not suggestions
            - PROVIDE EXACT REPLACEMENT TEXT - Don't just identify problems, provide precise corrective text
            - USE IMPERATIVE LANGUAGE - Use terms like "MUST", "REQUIRED", "CHANGE TO", rather than "should consider"
            - CITE REFERENCES - When identifying conflicts, cite the specific company policy or legal requirement
            - MATCH DOCUMENT LANGUAGE - Detect the language of the input document and respond in EXACTLY the same language
            - For Turkish documents, respond in Turkish. For English documents, respond in English, etc.
            - FORMAT CONSISTENTLY - Keep all section references in consistent format (e.g., "Section 4.2" or "Madde 4.2")
            
            REMEMBER: Your analysis will be used to make immediate corrections to legal documents. 
            You must provide precise, definitive changes that MUST be implemented, not mere suggestions.
            """

            # Format the knowledge base chunks as a JSON array for better structure
            # Add a note about chunk count to the KB chunks to help with processing
            kb_chunks_with_note = {
                "total_kb_chunks": len(knowledge_base_chunks),
                "kb_chunks": knowledge_base_chunks
            }
            kb_chunks_json = json.dumps(kb_chunks_with_note, ensure_ascii=False, indent=2)
            
            user_message = f"""
            ## KNOWLEDGE BASE CONTEXT (KURUMSAL POLİTİKA VE MEVZUAT BİLGİLERİ - JSON OBJECT):
            ```json
            {kb_chunks_json}
            ```
            
            ## DOCUMENT TO ANALYZE (ANALİZ EDİLECEK BELGE):
            {document_chunk}
            
            DETAILED ANALYSIS INSTRUCTIONS:
            1. Review all {len(knowledge_base_chunks)} knowledge base chunks in the data
            2. Perform a thorough comparison between the document and ALL knowledge base entries
            3. Identify EACH NUMBER, PERCENTAGE, DATE, TIMEFRAME or SPECIFIC TERM that does not match company policy
            4. For each issue found, specify EXACT section references (e.g., "Section 3.2.1" or "Madde 5.4")
            5. Whenever you identify a problem:
               - Quote the EXACT problematic text
               - State the EXACT replacement text that MUST be used instead
               - Explain why the change is mandatory (which policy it violates)
            6. For each identified issue, use language like "MUST BE CHANGED TO", not suggestions
            
            CRITICAL REQUIREMENTS:
            - DO NOT present analysis as recommendations or suggestions 
            - Present all findings as MANDATORY corrections
            - For each issue, provide both the problematic text AND the required replacement text
            - Use decisive language such as "MUST", "REQUIRED", "CHANGE TO"
            
            CRITICAL FOCUS AREAS:
            - Payment terms (ödeme koşulları)
            - Interest rates (faiz oranları)
            - Late payment penalties (gecikme cezaları)
            - Contract duration (sözleşme süresi)
            - Notice periods (bildirim süreleri)
            - Jurisdiction clauses (yargı yeri maddeleri)
            - Liability limitations (sorumluluk sınırlamaları)
            - Termination conditions (fesih koşulları)
            - Warranty periods (garanti süreleri)
            - Delivery timeframes (teslimat süreleri)
            
            Find EVERY SINGLE INSTANCE where the document differs from the company's standard terms.
            All identified discrepancies MUST be changed to comply with company policy.
            
            Remember: Respond in EXACTLY the same language as the document being analyzed, with the same terminology.
            """

            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    # Run the synchronous API call in a thread pool to avoid blocking
                    # O3-mini model has specific requirements - it doesn't support max_tokens or temperature
                    # Only pass the parameters it accepts to avoid API errors
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="o3-mini",  # Using the specified o3-mini model
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
                        return analysis_result
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON response: {str(e)}")
                        # If JSON parsing fails, return the raw content in a structured format
                        return {
                            "analysis":
                            "Error parsing AI response as JSON. Raw response included.",
                            "raw_response": response_content,
                            "error": str(e)
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
            
            # Return a structured error response that can be used by the calling function
            return {
                "analysis": f"Analysis failed: {error_msg}",
                "key_points": [],
                "conflicts": [],
                "recommendations": ["Contact support if this issue persists."],
                "error": error_msg,
                "error_type": type(e).__name__
            }
