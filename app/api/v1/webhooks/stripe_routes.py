import logging
import stripe
import json
from fastapi import APIRouter, Request, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.subscription_repository import SubscriptionRepository
from app.core.config import config
from app.api.v1.webhooks.stripe_handlers import (
    handle_payment_intent_succeeded,
    handle_payment_intent_failed,
    handle_subscription_deleted,
    handle_checkout_session_completed
)
from app.schemas.base import BaseResponse

router = APIRouter(prefix="/api/v1/webhooks")
logger = logging.getLogger(__name__)

@router.post("/stripe", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events
    
    This endpoint receives webhook events from Stripe and processes them accordingly.
    Events include payment_intent.succeeded, payment_intent.failed, etc.
    
    The endpoint verifies the webhook signature to ensure it's coming from Stripe.
    """
    try:
        # Get raw payload from request
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        
        if not sig_header:
            logger.warning("Missing Stripe signature header")
            return BaseResponse(success=False, message="Missing signature header")
        
        # Verify webhook signature
        try:
            if config.STRIPE_TEST_MODE:
                webhook_secret = config.STRIPE_TEST_WEBHOOK_SECRET
            else:
                webhook_secret = config.STRIPE_LIVE_WEBHOOK_SECRET
                
            if not webhook_secret:
                raise ValueError("Stripe webhook secret not configured")
                
            event = stripe.Webhook.construct_event(
                payload, sig_header, str(webhook_secret)
            )
            
            # Log which mode we're using (for debugging purposes)
            logger.info(f"Using Stripe in {'TEST' if config.STRIPE_TEST_MODE else 'PRODUCTION'} mode")
        except (stripe.error.SignatureVerificationError, ValueError) as e:
            logger.warning(f"Invalid Stripe signature or configuration: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signature or webhook configuration"
            )
        
        # Initialize repositories
        payment_repo = PaymentRepository(db)
        subscription_repo = SubscriptionRepository(db)
        
        # Process event based on type
        event_type = event["type"]
        event_object = event["data"]["object"]
        logger.info(f"Processing Stripe webhook event: {event_type}")
        
        if event_type == "payment_intent.succeeded":
            await handle_payment_intent_succeeded(event_object, payment_repo, subscription_repo, db)
        
        elif event_type == "payment_intent.failed":
            await handle_payment_intent_failed(event_object, payment_repo, db)
        
        elif event_type == "customer.subscription.deleted":
            await handle_subscription_deleted(event_object, subscription_repo, db)
            
        elif event_type == "checkout.session.completed":
            await handle_checkout_session_completed(event_object, payment_repo, subscription_repo, db)
        
        # For other event types, just log them
        else:
            logger.info(f"Unhandled Stripe webhook event: {event_type}")
        
        return BaseResponse(success=True, message=f"Webhook event processed: {event_type}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing Stripe webhook: {str(e)}", exc_info=True)
        # Don't return errors to Stripe, they'll retry
        return BaseResponse(success=True, message="Webhook received")
        
@router.get("/stripe/success")
async def stripe_success(
    session_id: str = "",
    db: AsyncSession = Depends(get_db)
):
    """
    Handle successful Stripe Checkout redirect
    
    This endpoint is called when a customer successfully completes the checkout process.
    It can retrieve the session details to display confirmation information.
    """
    try:
        if session_id:
            # You can verify the session here if needed
            return BaseResponse(
                success=True,
                message="Payment successful",
                data={"session_id": session_id, "status": "success"}
            )
        else:
            return BaseResponse(
                success=True,
                message="Payment completed",
                data={"status": "success"}
            )
    except Exception as e:
        logger.error(f"Error handling success redirect: {str(e)}")
        return BaseResponse(
            success=False,
            message="Error processing payment confirmation"
        )

@router.get("/stripe/cancel")
async def stripe_cancel():
    """
    Handle cancelled Stripe Checkout redirect
    
    This endpoint is called when a customer cancels the checkout process.
    """
    return BaseResponse(
        success=True,
        message="Payment cancelled",
        data={"status": "cancelled"}
    )