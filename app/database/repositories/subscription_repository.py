from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import UserSubscription, SubscriptionStatus
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SubscriptionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_trial_subscription(self, user_id: int) -> UserSubscription:
        """Create a new trial subscription for user"""
        trial_end = datetime.utcnow() + timedelta(days=14)  # 14-day trial period
        subscription = UserSubscription(
            user_id=user_id,
            status=SubscriptionStatus.TRIAL,
            trial_end=trial_end
        )
        async with self.db.begin():
            self.db.add(subscription)
        await self.db.refresh(subscription)
        return subscription

    async def get_user_subscription(self, user_id: int) -> Optional[UserSubscription]:
        """Get user's subscription details"""
        result = await self.db.execute(
            select(UserSubscription).filter(UserSubscription.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_subscription(self, subscription_id: int, updates: dict) -> Optional[UserSubscription]:
        """Update subscription details"""
        result = await self.db.execute(
            select(UserSubscription).filter(UserSubscription.id == subscription_id)
        )
        subscription = result.scalar_one_or_none()
        if subscription:
            async with self.db.begin():
                for key, value in updates.items():
                    setattr(subscription, key, value)
            await self.db.refresh(subscription)
        return subscription

    async def activate_subscription(self, subscription_id: int, stripe_subscription_id: str) -> Optional[UserSubscription]:
        """Activate a subscription after successful payment"""
        now = datetime.utcnow()
        updates = {
            "status": SubscriptionStatus.ACTIVE,
            "subscription_start": now,
            "subscription_end": now + timedelta(days=30),  # Default to 30 days
            "stripe_subscription_id": stripe_subscription_id
        }
        return await self.update_subscription(subscription_id, updates)
        
    async def get_subscription_by_stripe_id(self, stripe_subscription_id: str) -> Optional[UserSubscription]:
        """Get subscription by Stripe subscription ID"""
        result = await self.db.execute(
            select(UserSubscription).filter(UserSubscription.stripe_subscription_id == stripe_subscription_id)
        )
        return result.scalar_one_or_none()
        
    async def check_subscription_status(self, user_id: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user has an active subscription
        
        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - First value: True if subscription is active, False otherwise
                - Second value: Subscription details dictionary for caching
        """
        subscription = await self.get_user_subscription(user_id)
        if not subscription:
            return False, {}
            
        now = datetime.utcnow()
        subscription_data = {
            "subscription_status": subscription.status.value,
        }
        
        if subscription.status == SubscriptionStatus.ACTIVE:
            subscription_data["subscription_end"] = subscription.subscription_end.isoformat() if subscription.subscription_end else None
            if subscription.subscription_end and subscription.subscription_end > now:
                return True, subscription_data
                
        elif subscription.status == SubscriptionStatus.TRIAL:
            subscription_data["trial_end"] = subscription.trial_end.isoformat() if subscription.trial_end else None
            if subscription.trial_end and subscription.trial_end > now:
                return True, subscription_data
                
        return False, subscription_data
        
    async def check_subscription_active(self, user_id: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user has an active subscription (alias for check_subscription_status)
        This method is an alias of check_subscription_status to maintain API compatibility.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - First value: True if subscription is active, False otherwise
                - Second value: Subscription details dictionary
        """
        return await self.check_subscription_status(user_id)
        
    async def create_subscription(
        self,
        user_id: int,
        stripe_subscription_id: str,
        stripe_customer_id: str,
        price_id: str
    ) -> Optional[UserSubscription]:
        """
        Create a full subscription after payment
        
        Args:
            user_id: User ID
            stripe_subscription_id: Stripe Subscription ID
            stripe_customer_id: Stripe Customer ID
            price_id: Stripe Price ID
            
        Returns:
            Optional[UserSubscription]: Created subscription
        """
        try:
            # First, check if user already has a subscription
            existing_subscription = await self.get_user_subscription(user_id)
            
            now = datetime.utcnow()
            subscription_end = now + timedelta(days=30)  # Default to 30 days
            
            if existing_subscription:
                # Update existing subscription
                updates = {
                    "status": SubscriptionStatus.ACTIVE,
                    "subscription_start": now,
                    "subscription_end": subscription_end,
                    "stripe_subscription_id": stripe_subscription_id
                }
                return await self.update_subscription(existing_subscription.id, updates)
            else:
                # Create new subscription
                subscription = UserSubscription(
                    user_id=user_id,
                    status=SubscriptionStatus.ACTIVE,
                    subscription_start=now,
                    subscription_end=subscription_end,
                    stripe_subscription_id=stripe_subscription_id,
                    stripe_customer_id=stripe_customer_id,
                    price_id=price_id
                )
                async with self.db.begin():
                    self.db.add(subscription)
                await self.db.refresh(subscription)
                return subscription
                
        except Exception as e:
            logger.error(f"Error creating subscription: {str(e)}")
            return None