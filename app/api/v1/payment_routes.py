from fastapi import APIRouter, Depends, HTTPException, status
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

        # Create PaymentIntent
        payment_intent = stripe_service.create_payment_intent(
            amount=request.amount,
            currency=request.currency,
            metadata={"user_fp": current_user.fp}
        )

        # Create payment record
        payment_repo = PaymentRepository(db)
        await payment_repo.create_payment(
            user_id=current_user.id,
            amount=Decimal(request.amount) / 100,  # Convert cents to dollars
            stripe_payment_intent_id=payment_intent.id
        )

        return BaseResponse(
            data={
                "client_secret": payment_intent.client_secret,
                "public_key": stripe_service.get_public_key()
            }
        )
    except Exception as e:
        logger.error(f"Error creating payment intent: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create payment intent")
        )

@router.post("/webhook")
async def stripe_webhook(
    db: AsyncSession = Depends(get_db)
):
    try:
        # TODO: Verify Stripe signature
        # For now, we'll just handle the basic success case
        payment_repo = PaymentRepository(db)
        subscription_repo = SubscriptionRepository(db)

        # Update payment status
        payment = await payment_repo.get_payment_by_intent_id("pi_123")  # Replace with actual ID
        if payment:
            await payment_repo.update_payment_status(payment.id, PaymentStatus.SUCCEEDED)

            # Activate subscription
            subscription = await subscription_repo.get_user_subscription(payment.user_id)
            if subscription:
                await subscription_repo.activate_subscription(
                    subscription.id,
                    "sub_123"  # Replace with actual Stripe subscription ID
                )

        return BaseResponse(message="Webhook processed successfully")
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to process webhook")
        )