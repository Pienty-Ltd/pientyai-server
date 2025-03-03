import logging
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.schemas.response import BaseResponse, ErrorResponse
from app.schemas.request import LoginRequest, RegisterRequest
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from app.core.security import (
    get_password_hash, verify_password, create_access_token, 
    decode_access_token, cache_user_data, get_cached_user_data
)
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/register", response_model=BaseResponse[LoginResponse])
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user_repo = UserRepository(db)
        existing_user = await user_repo.get_user_by_email(request.email)

        if existing_user:
            logger.warning(f"Registration attempt with existing email: {request.email}")
            return BaseResponse(
                success=False,
                message="Registration failed",
                error=ErrorResponse(message="Email already registered")
            )

        # Create new user with hashed password
        hashed_password = get_password_hash(request.password)
        user_data = {
            "email": request.email,
            "hashed_password": hashed_password,
            "full_name": request.full_name,
            "is_active": True
        }

        user = await user_repo.insert_user(user_data)
        logger.info(f"New user registered successfully: {request.email} with fp: {user.fp}")

        # Cache user data
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp
        }
        await cache_user_data(user_cache_data)

        # Generate access token
        access_token = create_access_token(
            data={"sub": user.email, "fp": user.fp}
        )

        return BaseResponse(
            data=LoginResponse(access_token=access_token),
            message="Registration successful"
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

@router.post("/login", response_model=BaseResponse[LoginResponse])
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_email(request.email)

        if not user or not verify_password(request.password, user.hashed_password):
            logger.warning(f"Failed login attempt for email: {request.email}")
            return BaseResponse(
                success=False,
                message="Login failed",
                error=ErrorResponse(message="Invalid credentials")
            )

        # Cache user data after successful login
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp
        }
        await cache_user_data(user_cache_data)

        access_token = create_access_token(
            data={"sub": user.email, "fp": user.fp}
        )

        logger.info(f"Successful login for user: {request.email}")
        return BaseResponse(
            data=LoginResponse(access_token=access_token),
            message="Login successful"
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_access_token(token)
    if not token_data or "sub" not in token_data or "fp" not in token_data:
        logger.warning("Invalid token data during authentication")
        raise credentials_exception

    # First try to get user from Redis
    cached_user = await get_cached_user_data(token_data["fp"])
    if cached_user:
        logger.debug(f"Retrieved user from cache: {cached_user['email']}")
        return type('User', (), cached_user)  # Create a simple object from dict

    # If not in cache, try database
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_fp(token_data["fp"])
    if not user:
        logger.warning(f"User not found for token fp: {token_data['fp']}")
        raise credentials_exception

    # Cache user data for future requests
    user_cache_data = {
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "fp": user.fp
    }
    await cache_user_data(user_cache_data)
    logger.debug(f"Successfully authenticated user: {user.email}")
    return user

@router.get("/me", response_model=BaseResponse[dict])
async def get_user_me(current_user = Depends(get_current_user)):
    return BaseResponse(
        data={
            "email": current_user.email,
            "full_name": current_user.full_name,
            "is_active": current_user.is_active,
            "fp": current_user.fp
        },
        message="User details retrieved successfully"
    )