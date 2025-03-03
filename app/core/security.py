from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import jwt, JWTError
from app.core.config import config
import logging
from app.core.services.redis_service import RedisService
import json

logger = logging.getLogger(__name__)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    try:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'),
                              hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False


def create_access_token(data: dict,
                        expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=config.TOKEN_EXPIRE)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode,
                                 config.SECRET_KEY,
                                 algorithm=config.TOKEN_ALGORITHM)
        logger.debug(f"Generated JWT token for user: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT access token."""
    try:
        decoded_token = jwt.decode(token,
                                   config.SECRET_KEY,
                                   algorithms=[config.TOKEN_ALGORITHM])
        logger.debug(
            f"Successfully decoded token for user: {decoded_token.get('sub')}")
        return decoded_token
    except JWTError as e:
        logger.warning(f"Failed to decode token: {str(e)}")
        return None


async def cache_user_data(user_data: dict) -> None:
    """Cache user data in Redis."""
    try:
        redis_service = await RedisService.get_instance()
        user_key = f"user:{user_data['fp']}"
        # Cache for 24 hours
        await redis_service.set_value(key=user_key,
                                      value=json.dumps(user_data),
                                      expire=86400)
        logger.debug(f"Cached user data for fp: {user_data['fp']}")
    except Exception as e:
        logger.error(f"Error caching user data: {str(e)}")
        # Don't raise the exception as caching failure shouldn't break the app
        pass


async def get_cached_user_data(user_fp: str) -> Optional[dict]:
    """Get cached user data from Redis."""
    try:
        redis_service = await RedisService.get_instance()
        user_key = f"user:{user_fp}"
        cached_data = await redis_service.get_value(user_key)
        if cached_data:
            logger.info(
                f"Retrieved user data from Redis cache for fp: {user_fp}")
            return json.loads(cached_data)
        logger.debug(f"No cached data found in Redis for fp: {user_fp}")
    except Exception as e:
        logger.error(f"Error retrieving cached user data: {str(e)}")
    return None
