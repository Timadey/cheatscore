"""
Redis client utilities.
"""
import redis.asyncio as aioredis
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Get or create Redis connection pool.
    
    Returns:
        Redis client instance
    """
    global _redis_pool
    
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False
        )
        logger.info(f"Connected to Redis: {settings.redis_url}")
    
    return _redis_pool


async def close_redis():
    """Close Redis connection pool."""
    global _redis_pool
    
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None
        logger.info("Redis connection closed")

