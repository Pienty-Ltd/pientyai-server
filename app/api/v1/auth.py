import logging
from fastapi import APIRouter, Depends, status, Request, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.response import LoginResponse, TokenResponse
from app.schemas.request import LoginRequest, RegisterRequest
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from app.database.repositories.invitation_repository import InvitationRepository
from app.core.security import (get_password_hash, verify_password,
                             create_access_token, decode_access_token,
                             cache_user_data, get_cached_user_data, create_tokens, decode_refresh_token)
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import UserRole
from sqlalchemy import func

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}}
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme),
                       db: AsyncSession = Depends(get_db)):
    """Get current authenticated user"""
    try:
        token_data = decode_access_token(token)
        if not token_data or "sub" not in token_data or "fp" not in token_data:
            logger.warning("Invalid token data during authentication")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Authentication required",
                    "logout": True,
                    "details": [{"msg": "Authentication required"}]
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

        # First, try to get user data from Redis
        cached_data = await get_cached_user_data(token_data["fp"])
        if cached_data:
            logger.debug(f"Using cached user data for fp: {token_data['fp']}")
            # Get the complete user from database using fp
            user_repo = UserRepository(db)
            user = await user_repo.get_user_by_fp(token_data["fp"])
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "message": "User not found",
                        "logout": True,
                        "details": [{"msg": "User not found"}]
                    }
                )
            return user

        # If not in cache, get from database
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_fp(token_data["fp"])
        if not user:
            logger.warning(f"User not found for token fp: {token_data['fp']}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "logout": True,
                    "details": [{"msg": "User not found"}]
                },
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

        return user

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Authentication failed",
                "logout": True,
                "details": [{"msg": "Authentication failed"}]
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@router.post("/register", response_model=BaseResponse[LoginResponse])
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    try:
        # Check invitation code first
        invitation_repo = InvitationRepository(db)
        invitation_code = await invitation_repo.get_invitation_code(request.invitation_code)
        
        if not invitation_code:
            logger.warning(f"Invalid invitation code used during registration: {request.invitation_code}")
            return BaseResponse(
                success=False,
                message="Registration failed",
                error=ErrorResponse(message="Invalid invitation code")
            )
            
        if invitation_code.is_used:
            logger.warning(f"Attempt to use already used invitation code: {request.invitation_code}")
            return BaseResponse(
                success=False,
                message="Registration failed",
                error=ErrorResponse(message="This invitation code has already been used")
            )
        
        user_repo = UserRepository(db)

        # Create new user
        user_data = {
            "email": request.email,
            "hashed_password": get_password_hash(request.password),
            "full_name": request.full_name,
            "is_active": True,
            "role": UserRole.USER
        }

        # Attempt to create the user
        user = await user_repo.insert_user(user_data)
        if not user:
            logger.warning(f"Failed to create user with email: {request.email}")
            return BaseResponse(
                success=False,
                message="Registration failed",
                error=ErrorResponse(message="Email already registered or database error")
            )
            
        # Mark invitation code as used
        await invitation_repo.mark_as_used(request.invitation_code, user.id)
        
        # Create a trial subscription for the new user
        subscription_repo = SubscriptionRepository(db)
        trial_subscription = await subscription_repo.create_trial_subscription(user.id)
        if not trial_subscription:
            logger.warning(f"Failed to create trial subscription for user: {user.email}")
            # Continue with the registration process even if subscription creation fails
            # This will be handled later when the user tries to access subscription-based features

        # Cache user data
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp,
            "role": user.role.value,
            "id": user.id
        }
        
        # Add subscription data to cache
        if trial_subscription:
            is_active, subscription_data = await subscription_repo.check_subscription_active(user.id)
            if is_active:
                user_cache_data["subscription_status"] = subscription_data.get("subscription_status")
                if "trial_end" in subscription_data:
                    user_cache_data["trial_end"] = subscription_data.get("trial_end")
        
        await cache_user_data(user_cache_data)

        # Generate token pair
        access_token, refresh_token = create_tokens(data={
            "sub": user.email,
            "fp": user.fp
        })

        logger.info(f"Successfully registered user: {user.email} with invitation code: {request.invitation_code}")
        return BaseResponse(
            success=True,
            data=LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token
            ),
            message="Registration successful"
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Registration failed due to server error")
        )

@router.post("/login", response_model=BaseResponse[LoginResponse])
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user"""
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

        # Cache user data
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp,
            "role": user.role.value,
            "id": user.id
        }
        
        # Add subscription status to cache
        subscription_repo = SubscriptionRepository(db)
        if user.id:
            is_active, subscription_data = await subscription_repo.check_subscription_active(user.id)
            if is_active:
                user_cache_data["subscription_status"] = subscription_data.get("subscription_status")
                if "trial_end" in subscription_data:
                    user_cache_data["trial_end"] = subscription_data.get("trial_end")
                elif "subscription_end" in subscription_data:
                    user_cache_data["subscription_end"] = subscription_data.get("subscription_end")
        
        await cache_user_data(user_cache_data)

        # Generate token pair
        access_token, refresh_token = create_tokens(data={
            "sub": user.email,
            "fp": user.fp
        })

        logger.info(f"Successful login for user: {request.email}")
        return BaseResponse(
            success=True,
            data=LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token
            ),
            message="Login successful"
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Login failed")
        )

@router.post("/refresh", response_model=BaseResponse[TokenResponse])
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        token_data = decode_refresh_token(request.refresh_token)
        if not token_data:
            logger.warning("Invalid refresh token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Invalid refresh token",
                    "logout": True,
                    "details": [{"msg": "Invalid refresh token"}]
                }
            )

        # Get user from database
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_fp(token_data["fp"])
        if not user:
            logger.warning(f"User not found for refresh token fp: {token_data['fp']}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "logout": True,
                    "details": [{"msg": "User not found"}]
                }
            )

        # Cache user data with subscription info
        user_cache_data = {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "fp": user.fp,
            "role": user.role.value,
            "id": user.id
        }
        
        # Add subscription status to cache
        subscription_repo = SubscriptionRepository(db)
        if user.id:
            is_active, subscription_data = await subscription_repo.check_subscription_active(user.id)
            if is_active:
                user_cache_data["subscription_status"] = subscription_data.get("subscription_status")
                if "trial_end" in subscription_data:
                    user_cache_data["trial_end"] = subscription_data.get("trial_end")
                elif "subscription_end" in subscription_data:
                    user_cache_data["subscription_end"] = subscription_data.get("subscription_end")
        
        await cache_user_data(user_cache_data)
        
        # Generate new token pair
        access_token, refresh_token = create_tokens(data={
            "sub": user.email,
            "fp": user.fp
        })

        logger.info(f"Token refreshed for user: {user.email}")
        return BaseResponse(
            success=True,
            data=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token
            ),
            message="Token refreshed successfully"
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Token refresh failed")
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
            detail={
                "message": "Admin privileges required",
                "logout": True,
                "details": [{"msg": "Admin privileges required"}]
            }
        )
    return user

@router.get("/me", response_model=BaseResponse[dict],
           summary="Get current user",
           description="Get details of currently authenticated user")
async def get_user_me(request: Request, 
                     current_user=Depends(get_current_user),
                     db: AsyncSession = Depends(get_db)):
    """
    Get current authenticated user's details:
    - email: User's email
    - full_name: User's full name
    - is_active: Account status
    - role: User role (admin/user)
    - subscription: Subscription status and details
    """
    # Get user's subscription status
    subscription_repo = SubscriptionRepository(db)
    subscription_info = None
    
    if current_user.id:
        subscription = await subscription_repo.get_user_subscription(current_user.id)
        if subscription:
            is_active, subscription_data = await subscription_repo.check_subscription_active(current_user.id)
            subscription_info = {
                "status": subscription.status.value,
                "is_active": is_active,
                **subscription_data
            }
    
    return BaseResponse.from_request(
        request=request,
        data={
            "email": current_user.email,
            "full_name": current_user.full_name,
            "is_active": current_user.is_active,
            "fp": current_user.fp,
            "role": getattr(current_user, 'role', UserRole.USER),
            "subscription": subscription_info
        },
        message="User details retrieved successfully"
    )