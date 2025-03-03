import redis.asyncio as redis
import asyncio
from typing import Optional
from app.core.config import config

class RedisService:
    _client: Optional[redis.Redis] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.redis_url = config.REDIS_URL

    async def get_client(self) -> redis.Redis:
        async with self._lock:
            if self._client is None or self._client.closed:
                self._client = await redis.from_url(self.redis_url, decode_responses=True)
                try:
                    await asyncio.wait_for(self._client.ping(), timeout=5)
                except asyncio.TimeoutError:
                    await self._client.close()
                    raise Exception("Redis connection timeout!")
                except Exception as e:
                    await self._client.close()
                    raise Exception(f"Redis connection failed: {e}")
        return self._client

    async def set_value(self, key: str, value: str, expire: Optional[int] = None) -> None:
        """
        Set value for the specified key
        :param key: Key name
        :param value: Value to set
        :param expire: (Optional) Expiration time in seconds
        """
        client = await self.get_client()
        await client.set(name=key, value=value, ex=expire)

    async def get_value(self, key: str) -> Optional[str]:
        """
        Get value for the specified key
        :param key: Key name
        :return: Stored value (returns None if not found)
        """
        client = await self.get_client()
        return await client.get(name=key)

    async def delete_value(self, key: str) -> None:
        """
        Delete specified key and its value
        :param key: Key name
        """
        client = await self.get_client()
        await client.delete(key)

    async def close(self) -> None:
        """Close Redis connection"""
        async with self._lock:
            if self._client and not self._client.closed:
                await self._client.close()
                self._client = None

    @classmethod
    async def get_instance(cls) -> "RedisService":
        """Returns singleton instance"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance
