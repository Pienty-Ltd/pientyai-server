import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.services.stripe_service import StripeService
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import PaymentStatus
from app.core.config import config
import stripe

logger = logging.getLogger(__name__)

async def handle_payment_intent_succeeded(
    payment_intent: Dict[str, Any],
    payment_repo: PaymentRepository,
    subscription_repo: SubscriptionRepository,
    db: AsyncSession
) -> None:
    """Handle successful payment intent"""
    try:
        async with db.begin():
            # Update payment status
            payment = await payment_repo.get_payment_by_intent_id(payment_intent['id'])
            if not payment:
                logger.error(f"Payment not found for intent: {payment_intent['id']}")
                return

            await payment_repo.update_payment_status(payment.id, PaymentStatus.SUCCEEDED)
            logger.info(f"Updated payment status to SUCCEEDED for intent: {payment_intent['id']}")

            # Create subscription if metadata indicates it's a subscription payment
            if payment_intent.get('metadata', {}).get('subscription_create'):
                stripe_service = StripeService.get_instance()

                # Create or get customer
                customer = None
                if payment_intent.get('customer'):
                    customer = stripe.Customer.retrieve(payment_intent['customer'])
                else:
                    # Create new customer if not exists
                    customer = stripe_service.create_customer(
                        email=payment_intent['metadata'].get('user_email'),
                        metadata={
                            'user_id': str(payment.user_id),
                            'user_fp': payment_intent['metadata'].get('user_fp')
                        }
                    )

                # Create Stripe subscription
                subscription = stripe_service.create_subscription(
                    customer_id=customer['id'],
                    price_id=config.STRIPE_PRICE_ID,
                    metadata={
                        'user_id': str(payment.user_id),
                        'payment_intent_id': payment_intent['id']
                    }
                )

                # Store subscription in database
                await subscription_repo.create_subscription(
                    user_id=payment.user_id,
                    stripe_subscription_id=subscription['id'],
                    stripe_customer_id=customer['id'],
                    price_id=config.STRIPE_PRICE_ID
                )
                logger.info(f"Created subscription {subscription['id']} for user {payment.user_id}")

    except Exception as e:
        logger.error(f"Error handling payment_intent.succeeded: {str(e)}")
        raise

async def handle_payment_intent_failed(
    payment_intent: Dict[str, Any],
    payment_repo: PaymentRepository,
    db: AsyncSession
) -> None:
    """Handle failed payment intent"""
    try:
        async with db.begin():
            payment = await payment_repo.get_payment_by_intent_id(payment_intent['id'])
            if payment:
                await payment_repo.update_payment_status(payment.id, PaymentStatus.FAILED)
                logger.warning(f"Payment failed for intent: {payment_intent['id']}")
    except Exception as e:
        logger.error(f"Error handling payment_intent.failed: {str(e)}")
        raise

async def handle_subscription_deleted(
    subscription: Dict[str, Any],
    subscription_repo: SubscriptionRepository,
    db: AsyncSession
) -> None:
    """Handle subscription cancellation"""
    try:
        async with db.begin():
            sub = await subscription_repo.get_subscription_by_stripe_id(subscription['id'])
            if sub:
                await subscription_repo.update_subscription(
                    sub.id,
                    {'status': 'canceled', 'ended_at': subscription.get('ended_at')}
                )
                logger.info(f"Subscription cancelled: {subscription['id']}")
    except Exception as e:
        logger.error(f"Error handling subscription.deleted: {str(e)}")
        raise