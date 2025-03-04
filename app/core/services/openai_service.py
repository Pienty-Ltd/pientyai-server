import os
from typing import List, Optional
import openai
from app.core.config import settings

class OpenAIService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY

    async def create_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of text chunks using OpenAI's API
        """
        try:
            embeddings = []
            # Process chunks in batches to avoid rate limits
            for chunk in text_chunks:
                response = await openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            # Log the error appropriately
            print(f"Error creating embeddings: {str(e)}")
            raise
