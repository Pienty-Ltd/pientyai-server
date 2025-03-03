import redis.asyncio as redis
import asyncio
from typing import Optional
import logging
from app.core.config import config

logger = logging.getLogger(__name__)

class RedisService:
    _client: Optional[redis.Redis] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.redis_url = config.REDIS_URL

    async def get_client(self) -> redis.Redis:
        """Get Redis client with connection pooling and error handling."""
        async with self._lock:
            if self._client is None:
                logger.debug("Initializing new Redis connection...")
                try:
                    self._client = await redis.from_url(
                        self.redis_url,
                        decode_responses=True,
                        health_check_interval=30
                    )
                    # Test connection
                    await asyncio.wait_for(self._client.ping(), timeout=5)
                    logger.info("Successfully established Redis connection")
                except asyncio.TimeoutError:
                    logger.error("Redis connection timeout!")
                    await self.close()
                    raise Exception("Redis connection timeout!")
                except Exception as e:
                    logger.error(f"Redis connection failed: {e}")
                    await self.close()
                    raise Exception(f"Redis connection failed: {e}")
        return self._client

    async def set_value(self,
                       key: str,
                       value: str,
                       expire: Optional[int] = None) -> None:
        """
        Set value for the specified key with error handling
        :param key: Key name
        :param value: Value to set
        :param expire: (Optional) Expiration time in seconds
        """
        try:
            client = await self.get_client()
            await client.set(name=key, value=value, ex=expire)
            logger.debug(f"Successfully set Redis key: {key}")
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {e}")
            raise

    async def get_value(self, key: str) -> Optional[str]:
        """
        Get value for the specified key with error handling
        :param key: Key name
        :return: Stored value (returns None if not found)
        """
        try:
            client = await self.get_client()
            value = await client.get(name=key)
            if value:
                logger.debug(f"Successfully retrieved Redis key: {key}")
            else:
                logger.debug(f"No value found for Redis key: {key}")
            return value
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {e}")
            return None

    async def delete_value(self, key: str) -> None:
        """
        Delete specified key and its value with error handling
        :param key: Key name
        """
        try:
            client = await self.get_client()
            await client.delete(key)
            logger.debug(f"Successfully deleted Redis key: {key}")
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection with proper cleanup"""
        async with self._lock:
            if self._client:
                try:
                    await self._client.close()
                    self._client = None
                    logger.debug("Redis connection closed successfully")
                except Exception as e:
                    logger.error(f"Error closing Redis connection: {e}")
                    self._client = None

    @classmethod
    async def get_instance(cls) -> "RedisService":
        """Returns singleton instance with error handling"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
            logger.debug("Created new RedisService instance")
        return cls._instance
