import logging
from datetime import datetime
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import User, SubscriptionStatus
from app.api.v1.auth import get_current_user
from app.core.security import get_cached_user_data, cache_user_data

logger = logging.getLogger(__name__)

async def check_user_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Dependency to check if the user has an active subscription.
    Will raise HTTPException if the user doesn't have a valid subscription.
    """
    try:
        # First, check if subscription info exists in cache
        cached_data = await get_cached_user_data(current_user.fp)
        if cached_data and "subscription_status" in cached_data:
            logger.debug(f"Using cached subscription data for user: {current_user.fp}")
            
            if cached_data["subscription_status"] == SubscriptionStatus.ACTIVE.value:
                # Check if we have an expiration date and if it's still valid
                if "subscription_end" in cached_data:
                    subscription_end = datetime.fromisoformat(cached_data["subscription_end"])
                    if subscription_end > datetime.utcnow():
                        return current_user
            
            if cached_data["subscription_status"] == SubscriptionStatus.TRIAL.value:
                # Check if trial is still valid
                if "trial_end" in cached_data:
                    trial_end = datetime.fromisoformat(cached_data["trial_end"])
                    if trial_end > datetime.utcnow():
                        return current_user
            
            # If we're here, cached subscription is not valid
            # We should fetch from database to double-check (maybe it was updated)
        
        # If not in cache or invalid, get from database
        subscription_repo = SubscriptionRepository(db)
        subscription = await subscription_repo.get_user_subscription(current_user.id)
        
        # No subscription record found
        if not subscription:
            logger.warning(f"No subscription found for user: {current_user.email}")
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "message": "Subscription required",
                    "logout": True,
                    "details": [{"msg": "Active subscription required to access this resource"}]
                }
            )
        
        now = datetime.utcnow()
        
        # Check subscription status
        if subscription.status == SubscriptionStatus.ACTIVE:
            if subscription.subscription_end and subscription.subscription_end > now:
                # Active and not expired
                
                # Update cache with subscription data
                cached_data = await get_cached_user_data(current_user.fp)
                if cached_data:
                    cached_data["subscription_status"] = SubscriptionStatus.ACTIVE.value
                    cached_data["subscription_end"] = subscription.subscription_end.isoformat()
                    await cache_user_data(cached_data)
                
                return current_user
        
        elif subscription.status == SubscriptionStatus.TRIAL:
            if subscription.trial_end and subscription.trial_end > now:
                # Trial and not expired
                
                # Update cache with subscription data
                cached_data = await get_cached_user_data(current_user.fp)
                if cached_data:
                    cached_data["subscription_status"] = SubscriptionStatus.TRIAL.value
                    cached_data["trial_end"] = subscription.trial_end.isoformat()
                    await cache_user_data(cached_data)
                
                return current_user
        
        # If we're here, subscription is not valid
        logger.warning(f"Invalid subscription for user: {current_user.email}, status: {subscription.status.value}")
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "message": "Subscription required",
                "logout": True,
                "details": [{"msg": "Active subscription required to access this resource"}]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking subscription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Internal server error",
                "details": [{"msg": "Error checking subscription status"}]
            }
        )