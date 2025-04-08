from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.database.database_factory import get_db
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import User, SubscriptionStatus
from app.api.v1.auth import get_current_user
from app.core.services.stripe_service import StripeService
from app.core.config import config
from app.schemas.base import BaseResponse, ErrorResponse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/subscriptions")

class SubscriptionStatusResponse(BaseModel):
    status: str
    is_active: bool
    expiration_date: Optional[str] = None
    trial_end: Optional[str] = None
    days_remaining: Optional[int] = None

class CreateCheckoutSessionRequest(BaseModel):
    price_id: Optional[str] = None
    success_url: str
    cancel_url: str

@router.get("/status", response_model=BaseResponse[SubscriptionStatusResponse])
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the current user's subscription status"""
    try:
        subscription_repo = SubscriptionRepository(db)
        subscription = await subscription_repo.get_user_subscription(current_user.id)
        
        if not subscription:
            return BaseResponse(
                success=True,
                data=SubscriptionStatusResponse(
                    status="none",
                    is_active=False
                ),
                message="No subscription found"
            )
        
        now = datetime.utcnow()
        is_active = False
        days_remaining = None
        expiration_date = None
        trial_end = None
        
        if subscription.status == SubscriptionStatus.ACTIVE and subscription.subscription_end:
            is_active = subscription.subscription_end > now
            if is_active:
                days_remaining = (subscription.subscription_end - now).days
            expiration_date = subscription.subscription_end.isoformat()
        
        elif subscription.status == SubscriptionStatus.TRIAL and subscription.trial_end:
            is_active = subscription.trial_end > now
            if is_active:
                days_remaining = (subscription.trial_end - now).days
            trial_end = subscription.trial_end.isoformat()
        
        return BaseResponse(
            success=True,
            data=SubscriptionStatusResponse(
                status=subscription.status.value,
                is_active=is_active,
                expiration_date=expiration_date,
                trial_end=trial_end,
                days_remaining=days_remaining
            ),
            message="Subscription status retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to get subscription status")
        )

@router.post("/create-checkout-session", response_model=BaseResponse[Dict[str, Any]])
async def create_checkout_session(
    request: CreateCheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a Stripe Checkout session for subscription purchase"""
    try:
        stripe_service = StripeService.get_instance()
        
        # Use the default price ID if none provided
        price_id = request.price_id or config.STRIPE_PRICE_ID
        
        # Prepare metadata for the checkout session
        metadata = {
            "user_id": str(current_user.id),
            "user_email": current_user.email,
            "user_fp": current_user.fp,
            "subscription_create": "true"
        }
        
        # Create checkout session
        # Make sure price_id is always a string to fix LSP issue
        # Also handle the case where price_id might be None
        if price_id is None:
            price_id = config.STRIPE_PRICE_ID
            
        if not price_id:
            logger.error("No price_id provided or configured in environment")
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="No price ID configured for subscription")
            )
            
        checkout_session = stripe_service.create_checkout_session(
            price_id=str(price_id),
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            customer_email=current_user.email,
            metadata=metadata
        )
        
        if not checkout_session or not checkout_session.get('id'):
            logger.error("Failed to create Stripe Checkout session")
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Failed to create checkout session")
            )
        
        # Create comprehensive response with all available data
        response_data = {
            "checkout_url": checkout_session.get('url', ''),
            "session_id": checkout_session.get('id', ''),
            "status": checkout_session.get('status', ''),
            "payment_status": checkout_session.get('payment_status', ''),
            "customer_email": checkout_session.get('customer_email', current_user.email)
        }
        
        # Ensure we've got the important data
        if not response_data["checkout_url"] or not response_data["session_id"]:
            logger.warning(f"Missing critical data in checkout session: {checkout_session}")
            
        logger.info(f"Checkout session created with data: {response_data}")
            
        return BaseResponse(
            success=True,
            data=response_data,
            message="Checkout session created successfully"
        )
    
    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create checkout session")
        )