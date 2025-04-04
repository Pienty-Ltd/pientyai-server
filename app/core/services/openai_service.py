import os
import asyncio
from typing import List, Optional, Dict, Any
from openai import OpenAI
from app.core.config import config
import logging
import json

logger = logging.getLogger(__name__)


class OpenAIService:
    BATCH_SIZE = 20  # Maximum number of texts to process in one API call
    MAX_RETRIES = 3  # Maximum number of retries for failed API calls
    RETRY_DELAY = 1  # Delay between retries in seconds

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

    async def analyze_document(self,
                               document_chunk: str,
                               knowledge_base_chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze a document chunk against relevant knowledge base chunks using OpenAI API
        
        Args:
            document_chunk: The chunk of text from the document to analyze
            knowledge_base_chunks: List of relevant knowledge base chunks for context
            
        Returns:
            Dictionary containing the analysis result
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

            # Construct the prompt with both knowledge base context and the document to analyze
            system_prompt = """You are an expert legal document analysis assistant specializing in commercial contracts, regulations, and company policies. Your task is to analyze 
            the provided document in the context of the knowledge base information. 
            
            When analyzing, you MUST focus on:
            1. Checking EXPLICIT COMPLIANCE with the country's commercial law, regulations, and legal requirements
            2. Identifying terms that CONTRADICT company policies found in the knowledge base 
            3. Finding SPECIFIC instances where contract terms don't align with company interests (e.g., if company policy states 3% interest but contract shows 2%)
            4. Detecting clauses that could create legal or financial risks for the company
            
            Structure your response as a JSON with the following keys:
            - analysis: Overall analysis of the document with focus on legal compliance and company policy alignment
            - key_points: Array of important points discovered in the contract
            - conflicts: Array of SPECIFIC conflicts with commercial law or company policies (be precise and definitive)
            - recommendations: Array of CONCRETE changes that MUST be made to protect company interests (not vague suggestions)
            
            IMPORTANT RULES:
            - Be DECISIVE and SPECIFIC in your recommendations - don't use hedging language
            - Identify EXACT terms that need modification (e.g., "Section 4.2 interest rate must be changed from 2% to 3%")
            - Focus on company interests and protecting the company's position
            - Detect the language of the input document and provide your response in the SAME LANGUAGE
            - If the document is in Turkish, respond in Turkish. If it's in English, respond in English, etc.
            """

            # Format the knowledge base chunks as a JSON array for better structure
            kb_chunks_json = json.dumps(knowledge_base_chunks, ensure_ascii=False, indent=2)
            
            user_message = f"""
            ## KNOWLEDGE BASE CONTEXT (JSON ARRAY):
            ```json
            {kb_chunks_json}
            ```
            
            ## DOCUMENT TO ANALYZE:
            {document_chunk}
            
            Analyze this document against the knowledge base context provided above as a JSON array.
            Pay special attention to company policies, commercial law requirements, and specific contractual terms.
            Focus on finding EXACT inconsistencies between the document and company policies or legal requirements.
            Remember to respond in the same language as the document being analyzed.
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
