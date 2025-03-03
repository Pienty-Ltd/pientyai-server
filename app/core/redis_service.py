# This file is deprecated and moved to app/core/services/redis_service.py
# Left here temporarily to ensure clean migration
from app.core.services.redis_service import RedisService

# Re-export for backward compatibility
__all__ = ['RedisService']