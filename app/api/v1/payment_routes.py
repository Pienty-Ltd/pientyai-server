from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
from app.database.database_factory import get_db
from app.core.services.stripe_service import StripeService
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.database.models.db_models import PaymentStatus
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.request import PaymentIntentRequest
from app.api.v1.auth import get_current_user
import logging
import stripe
from app.core.config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments")

@router.post("/create-payment-intent")
async def create_payment_intent(
    request: PaymentIntentRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        stripe_service = StripeService.get_instance()

        # Create PaymentIntent with Stripe
        payment_intent = stripe_service.create_payment_intent(
            amount=request.amount,
            currency=request.currency,
            metadata={
                "user_fp": current_user.fp,
                "user_email": current_user.email,
                **request.metadata if request.metadata else {}
            }
        )

        if not payment_intent or not payment_intent.get('id'):
            logger.error("Failed to create Stripe PaymentIntent")
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Failed to create payment intent")
            )

        # Create payment record in database
        payment_repo = PaymentRepository(db)
        await payment_repo.create_payment(
            user_id=current_user.id,
            amount=Decimal(request.amount) / 100,  # Convert cents to dollars
            stripe_payment_intent_id=payment_intent['id'],
            currency=request.currency
        )

        return BaseResponse(
            data={
                "client_secret": payment_intent['client_secret'],
                "public_key": stripe_service.get_public_key()
            }
        )
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )
    except Exception as e:
        logger.error(f"Error creating payment intent: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create payment intent")
        )

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Get the Stripe signature from headers
        stripe_signature = request.headers.get('stripe-signature')
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="No Stripe signature found")

        # Get the raw request body
        payload = await request.body()

        # Verify Stripe webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload,
                stripe_signature,
                config.STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid Stripe signature: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Handle different event types
        if event['type'] == 'payment_intent.succeeded':
            payment_intent = event['data']['object']
            payment_repo = PaymentRepository(db)
            subscription_repo = SubscriptionRepository(db)

            # Update payment status
            payment = await payment_repo.get_payment_by_intent_id(payment_intent['id'])
            if payment:
                await payment_repo.update_payment_status(payment.id, PaymentStatus.SUCCEEDED)

                # Get or create Stripe subscription
                try:
                    stripe_service = StripeService.get_instance()
                    subscription = await subscription_repo.get_user_subscription(payment.user_id)

                    if subscription:
                        # Create Stripe subscription
                        stripe_subscription = stripe_service.create_subscription(
                            customer_id=payment_intent['customer'],
                            price_id=config.STRIPE_PRICE_ID  # Monthly subscription price ID
                        )

                        # Activate subscription in our database
                        await subscription_repo.activate_subscription(
                            subscription.id,
                            stripe_subscription['id']
                        )
                        logger.info(f"Subscription activated for payment: {payment_intent['id']}")
                except Exception as e:
                    logger.error(f"Error activating subscription: {str(e)}")
                    # Don't raise the error, we'll handle it through admin panel

        elif event['type'] == 'payment_intent.payment_failed':
            payment_intent = event['data']['object']
            payment_repo = PaymentRepository(db)

            # Update payment status to failed
            payment = await payment_repo.get_payment_by_intent_id(payment_intent['id'])
            if payment:
                await payment_repo.update_payment_status(payment.id, PaymentStatus.FAILED)
                logger.warning(f"Payment failed for intent: {payment_intent['id']}")

        # Handle subscription events
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            subscription_repo = SubscriptionRepository(db)

            # Update subscription status to cancelled
            user_subscription = await subscription_repo.get_subscription_by_stripe_id(subscription['id'])
            if user_subscription:
                await subscription_repo.update_subscription(user_subscription.id, {
                    'status': 'canceled'
                })
                logger.info(f"Subscription cancelled: {subscription['id']}")

        return BaseResponse(message="Webhook processed successfully")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to process webhook")
        )