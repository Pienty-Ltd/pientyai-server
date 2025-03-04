import logging
from fastapi import APIRouter, Depends, status, Request, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.response import LoginResponse
from app.schemas.request import LoginRequest, RegisterRequest
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from app.core.security import (get_password_hash, verify_password,
                             create_access_token, decode_access_token,
                             cache_user_data, get_cached_user_data)
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import UserRole

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}}
)

class CustomAuthException(HTTPException):
    """Custom authentication exception that will be handled by the auth exception handler"""
    pass

class CustomOAuth2PasswordBearer(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        try:
            return await super().__call__(request)
        except HTTPException as exc:
            logger.warning(f"OAuth2 authentication failed: {exc.detail}")
            raise CustomAuthException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"}
            )

oauth2_scheme = CustomOAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme),
                         db: AsyncSession = Depends(get_db)):
    """
    Get current authenticated user
    :param token: JWT token
    :param db: Database session
    :return: User object
    """
    try:
        token_data = decode_access_token(token)
        if not token_data or "sub" not in token_data or "fp" not in token_data:
            logger.warning("Invalid token data during authentication")
            raise CustomAuthException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # First try to get user from Redis cache
        cached_user = await get_cached_user_data(token_data["fp"])
        user = None

        if cached_user:
            logger.debug(f"Retrieved user from cache: {cached_user['email']}")
            # Create user object from cached data
            user = type('User', (), {
                'email': cached_user['email'],
                'full_name': cached_user['full_name'],
                'is_active': cached_user['is_active'],
                'fp': cached_user['fp'],
                'role': cached_user['role'],
                'id': cached_user['id']
            })
        else:
            # If not in cache, get from database
            user_repo = UserRepository(db)
            user = await user_repo.get_user_by_fp(token_data["fp"])
            if not user:
                logger.warning(f"User not found for token fp: {token_data['fp']}")
                raise CustomAuthException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Cache user data for future requests
            user_cache_data = {
                'email': user.email,
                'full_name': user.full_name,
                'is_active': user.is_active,
                'fp': user.fp,
                'role': user.role.value,
                'id': user.id
            }
            await cache_user_data(user_cache_data)

        logger.debug(f"Successfully authenticated user: {user.email}")
        return user

    except CustomAuthException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise CustomAuthException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

def admin_required(user = Depends(get_current_user)):
    """
    Dependency wrapper that checks if the user has admin role
    This doesn't add any parameters to the OpenAPI schema
    """
    if user.role != UserRole.ADMIN:
        logger.warning(f"Non-admin user attempted to access admin endpoint: {user.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user

@router.post("/register", response_model=BaseResponse[LoginResponse], 
            summary="Register new user",
            description="Register a new user with email and password. Creates a trial subscription automatically.")
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user with the following information:
    - email: Valid email address
    - password: Minimum 8 characters
    - full_name: User's full name

    Returns:
    - access_token: JWT token for authentication
    """
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
            "is_active": True,
            "role": UserRole.USER  # Default role is USER
        }

        # Create user first
        user = await user_repo.insert_user(user_data)
        logger.info(f"New user registered successfully: {request.email} with fp: {user.fp}")

        try:
            # Start trial period
            subscription_repo = SubscriptionRepository(db)
            await subscription_repo.create_trial_subscription(user.id)
            logger.info(f"Trial subscription created for user: {user.email}")
        except Exception as e:
            logger.error(f"Error creating trial subscription: {str(e)}")
            # Continue with user creation even if subscription creation fails
            # We can handle this case later through admin panel or background job

        # Cache user data
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp,
            "role": user.role.value,
            "id": user.id
        }
        await cache_user_data(user_cache_data)

        # Generate access token
        access_token = create_access_token(data={
            "sub": user.email,
            "fp": user.fp
        })

        return BaseResponse(
            data=LoginResponse(access_token=access_token),
            message="Registration successful"
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return BaseResponse(success=False, error=ErrorResponse(message=str(e)))

@router.post("/login", response_model=BaseResponse[LoginResponse],
            summary="User login",
            description="Authenticate user with email and password")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user with:
    - email: Registered email address
    - password: User's password

    Returns:
    - access_token: JWT token for authentication
    """
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
            "fp": user.fp,
            "role": user.role.value,
            "id": user.id
        }
        await cache_user_data(user_cache_data)

        access_token = create_access_token(data={
            "sub": user.email,
            "fp": user.fp
        })

        logger.info(f"Successful login for user: {request.email}")
        return BaseResponse(
            data=LoginResponse(access_token=access_token),
            message="Login successful"
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return BaseResponse(success=False, error=ErrorResponse(message=str(e)))

@router.get("/me", response_model=BaseResponse[dict],
           summary="Get current user",
           description="Get details of currently authenticated user")
async def get_user_me(current_user=Depends(get_current_user)):
    """
    Get current authenticated user's details:
    - email: User's email
    - full_name: User's full name
    - is_active: Account status
    - role: User role (admin/user)
    """
    return BaseResponse(
        data={
            "email": current_user.email,
            "full_name": current_user.full_name,
            "is_active": current_user.is_active,
            "fp": current_user.fp,
            "role": getattr(current_user, 'role', UserRole.USER)
        },
        message="User details retrieved successfully"
    )