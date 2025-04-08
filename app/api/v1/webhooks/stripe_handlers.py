import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.services.stripe_service import StripeService
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import PaymentStatus, SubscriptionStatus
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

                # Ensure we have a price_id
                price_id = config.STRIPE_PRICE_ID
                if not price_id:
                    logger.error("No price ID configured for subscription")
                    return
                
                # Create Stripe subscription
                subscription = stripe_service.create_subscription(
                    customer_id=str(customer['id']),
                    price_id=str(price_id),
                    metadata={
                        'user_id': str(payment.user_id),
                        'payment_intent_id': payment_intent['id']
                    }
                )

                # Get the price ID with safety check
                price_id = config.STRIPE_PRICE_ID
                if not price_id:
                    price_id = "price_unknown"  # Fallback
                
                # Store subscription in database
                await subscription_repo.create_subscription(
                    user_id=payment.user_id,
                    stripe_subscription_id=str(subscription['id']),
                    stripe_customer_id=str(customer['id']),
                    price_id=str(price_id)
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
        
async def handle_checkout_session_completed(
    session: Dict[str, Any],
    payment_repo: PaymentRepository,
    subscription_repo: SubscriptionRepository,
    db: AsyncSession
) -> None:
    """
    Handle completed checkout session
    
    This is called when a customer completes the Stripe Checkout process.
    For subscription checkouts, we need to create the subscription in our database.
    """
    try:
        logger.info(f"Processing checkout session: {session['id']}")
        
        # Check if this is a subscription checkout
        is_subscription = session.get('mode') == 'subscription'
        if not is_subscription:
            logger.info(f"Checkout session {session['id']} is not a subscription, skipping")
            return
            
        # Get user_id from metadata
        metadata = session.get('metadata', {})
        user_id = metadata.get('user_id')
        
        if not user_id:
            logger.error(f"No user_id in checkout session metadata: {session['id']}")
            return
            
        user_id = int(user_id)
        
        # Get subscription and customer from session
        subscription_id = session.get('subscription')
        customer_id = session.get('customer')
        
        if not subscription_id or not customer_id:
            logger.error(f"Missing subscription or customer ID: {session['id']}")
            return
        
        # Retrieve the subscription details from Stripe
        stripe_service = StripeService.get_instance()
        subscription = stripe.Subscription.retrieve(subscription_id)
        
        # Set subscription period based on Stripe subscription details
        current_period_start = None
        current_period_end = None
        
        if subscription.get('current_period_start'):
            current_period_start = datetime.fromtimestamp(subscription['current_period_start'])
        
        if subscription.get('current_period_end'):
            current_period_end = datetime.fromtimestamp(subscription['current_period_end'])
        
        # Create or update subscription in database
        async with db.begin():
            existing_sub = await subscription_repo.get_user_subscription(user_id)
            
            if existing_sub:
                # Update existing subscription
                updates = {
                    'status': SubscriptionStatus.ACTIVE.value,
                    'stripe_subscription_id': subscription_id,
                    'stripe_customer_id': customer_id,
                    'price_id': session.get('line_items', {}).get('data', [{}])[0].get('price', {}).get('id') or config.STRIPE_PRICE_ID,
                    'subscription_start': current_period_start,
                    'subscription_end': current_period_end
                }
                
                await subscription_repo.update_subscription(existing_sub.id, updates)
                logger.info(f"Updated subscription for user {user_id}: {subscription_id}")
            else:
                # Get price ID with a safer approach to avoid None values
                price_id = session.get('line_items', {}).get('data', [{}])[0].get('price', {}).get('id')
                if not price_id:
                    price_id = config.STRIPE_PRICE_ID
                if not price_id:
                    price_id = "price_unknown"  # Fallback
                
                # Create new subscription
                await subscription_repo.create_subscription(
                    user_id=user_id,
                    stripe_subscription_id=str(subscription_id),
                    stripe_customer_id=str(customer_id),
                    price_id=str(price_id)
                )
                logger.info(f"Created new subscription for user {user_id}: {subscription_id}")
            
    except Exception as e:
        logger.error(f"Error handling checkout.session.completed: {str(e)}")
        raise