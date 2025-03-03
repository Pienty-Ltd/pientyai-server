from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import UserSubscription, SubscriptionStatus
from typing import Optional

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