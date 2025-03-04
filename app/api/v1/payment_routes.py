from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.request import PaymentIntentRequest
from app.api.v1.auth import get_current_user
from app.api.v1.webhooks.stripe_handlers import (
    handle_payment_intent_succeeded,
    handle_payment_intent_failed,
    handle_subscription_deleted
)
from app.api.v1.utils.payment_utils import create_payment_record
import logging
import stripe
from app.core.config import config
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments")

class PaymentIntentWithPromoRequest(PaymentIntentRequest):
    promo_code: Optional[str] = None

@router.post("/create-payment-intent")
async def create_payment_intent(
    request: PaymentIntentWithPromoRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a payment intent with optional promo code"""
    try:
        # Prepare metadata
        metadata = {
            "user_fp": current_user.fp,
            "user_email": current_user.email,
        }
        # Update metadata with additional data if provided
        if request.metadata:
            metadata.update(request.metadata)

        payment_repo = PaymentRepository(db)
        promo_repo = None
        if request.promo_code:
            promo_repo = PromoCodeRepository(db)

        return await create_payment_record(
            user_id=current_user.id,
            amount=request.amount,
            currency=request.currency,
            metadata=metadata,
            payment_repo=payment_repo,
            promo_code=request.promo_code,
            promo_repo=promo_repo
        )

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )
    except Exception as e:
        logger.error(f"Error creating payment intent: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create payment intent")
        )

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Handle Stripe webhook events"""
    try:
        # Get the webhook signature from headers
        signature = request.headers.get('stripe-signature')
        logger.info(f"Received webhook request with signature: {signature}")

        if not signature:
            logger.error("Missing Stripe signature")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Stripe signature"
            )

        # Get raw request body
        payload = await request.body()
        logger.debug(f"Received webhook payload: {payload.decode()}")

        # Verify webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                config.STRIPE_WEBHOOK_SECRET
            )
            logger.info(f"Successfully verified webhook signature for event: {event.type}")
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid Stripe signature: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signature"
            )

        # Initialize repositories
        payment_repo = PaymentRepository(db)
        subscription_repo = SubscriptionRepository(db)

        # Handle different event types
        event_handlers = {
            'payment_intent.succeeded': lambda: handle_payment_intent_succeeded(
                event.data.object,
                payment_repo,
                subscription_repo,
                db
            ),
            'payment_intent.failed': lambda: handle_payment_intent_failed(
                event.data.object,
                payment_repo,
                db
            ),
            'customer.subscription.deleted': lambda: handle_subscription_deleted(
                event.data.object,
                subscription_repo,
                db
            )
        }

        # Execute appropriate handler for event type
        if event.type in event_handlers:
            await event_handlers[event.type]()
            logger.info(f"Successfully processed webhook event: {event.type}")
        else:
            logger.info(f"Unhandled webhook event type: {event.type}")

        # Return 200 response to acknowledge receipt
        return BaseResponse(
            success=True,
            message=f"Webhook event processed: {event.type}"
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        # Return 200 even for errors to prevent Stripe from retrying
        response.status_code = status.HTTP_200_OK
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Error processing webhook")
        )