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
                        retry_count += 1
                        if retry_count == self.MAX_RETRIES:
                            logger.error(
                                f"Failed to generate embeddings after {self.MAX_RETRIES} retries: {str(e)}"
                            )
                            raise

                        logger.warning(
                            f"Retry {retry_count}/{self.MAX_RETRIES} for batch {i//self.BATCH_SIZE + 1}: {str(e)}"
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
                               knowledge_base_chunks: List[str],
                               max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Analyze a document chunk against relevant knowledge base chunks using OpenAI API
        
        Args:
            document_chunk: The chunk of text from the document to analyze
            knowledge_base_chunks: List of relevant knowledge base chunks for context
            max_tokens: Maximum tokens for the response
            
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
            system_prompt = """You are a legal document analysis assistant. Your task is to analyze 
            the provided document in the context of the knowledge base information. 
            Focus on identifying:
            1. Key legal implications and concerns
            2. Potential conflicts with existing documents
            3. Important clauses or terms that require attention
            4. Recommended actions or considerations
            
            Structure your response as a JSON with the following keys:
            - analysis: Overall analysis of the document
            - key_points: Array of important points discovered
            - conflicts: Any conflicts with existing knowledge base documents
            - recommendations: Suggested actions based on your analysis
            
            IMPORTANT: Detect the language of the input document and provide your response in the SAME LANGUAGE.
            If the document is in Turkish, respond in Turkish. If it's in English, respond in English, etc.
            Always match the language of your response to the language of the document being analyzed.
            """

            # Construct the user message with knowledge base context
            kb_context = "\n\n".join([
                f"KNOWLEDGE BASE DOCUMENT CHUNK {i+1}:\n{chunk}"
                for i, chunk in enumerate(knowledge_base_chunks)
            ])

            user_message = f"""
            ## KNOWLEDGE BASE CONTEXT:
            {kb_context}
            
            ## DOCUMENT TO ANALYZE:
            {document_chunk}
            
            Analyze this document against the knowledge base context provided above.
            Remember to respond in the same language as the document being analyzed.
            """

            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    # Run the synchronous API call in a thread pool to avoid blocking
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
                        temperature=
                        0.1,  # Low temperature for more factual responses
                        max_tokens=max_tokens,
                        response_format={"type":
                                         "json_object"}  # Ensure JSON response
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
                    retry_count += 1
                    if retry_count == self.MAX_RETRIES:
                        logger.error(
                            f"Failed to get analysis after {self.MAX_RETRIES} retries: {str(e)}"
                        )
                        raise

                    logger.warning(
                        f"Retry {retry_count}/{self.MAX_RETRIES} for document analysis: {str(e)}"
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
