import os
import asyncio
from typing import List, Optional
from openai import OpenAI
from app.core.config import config
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    BATCH_SIZE = 20  # Maximum number of texts to process in one API call
    MAX_RETRIES = 3  # Maximum number of retries for failed API calls
    RETRY_DELAY = 1  # Delay between retries in seconds

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    async def create_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
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
                        logger.debug(f"Attempting to create embeddings for batch of size {len(batch)}")

                        # Run the synchronous API call in a thread pool to avoid blocking
                        response = await asyncio.to_thread(
                            self.client.embeddings.create,
                            model="text-embedding-ada-002",
                            input=batch
                        )

                        logger.debug(f"Response type: {type(response)}")
                        logger.debug(f"Response content: {response}")

                        # Extract embeddings from response data
                        batch_embeddings = [item.embedding for item in response.data]
                        logger.debug(f"Successfully extracted {len(batch_embeddings)} embeddings")
                        logger.debug(f"Sample embedding length: {len(batch_embeddings[0]) if batch_embeddings else 0}")

                        all_embeddings.extend(batch_embeddings)
                        logger.info(f"Successfully generated embeddings for batch {i//self.BATCH_SIZE + 1}")
                        break  # Success, exit retry loop

                    except Exception as e:
                        retry_count += 1
                        if retry_count == self.MAX_RETRIES:
                            logger.error(f"Failed to generate embeddings after {self.MAX_RETRIES} retries: {str(e)}")
                            raise

                        logger.warning(f"Retry {retry_count}/{self.MAX_RETRIES} for batch {i//self.BATCH_SIZE + 1}: {str(e)}")
                        await asyncio.sleep(self.RETRY_DELAY * retry_count)  # Exponential backoff

                # Add a small delay between batches to avoid rate limits
                if i + self.BATCH_SIZE < len(text_chunks):
                    await asyncio.sleep(0.5)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise