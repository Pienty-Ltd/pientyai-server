import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from app.core.services.stripe_service import StripeService
from app.database.repositories.payment_repository import PaymentRepository
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.schemas.base import BaseResponse, ErrorResponse

logger = logging.getLogger(__name__)

async def create_payment_record(
    user_id: int,
    amount: int,
    currency: str,
    metadata: Dict[str, Any],
    payment_repo: PaymentRepository,
    promo_code: Optional[str] = None,
    promo_repo: Optional[PromoCodeRepository] = None
) -> Dict[str, Any]:
    """
    Create a payment record in Stripe and local database
    :param user_id: User ID
    :param amount: Amount in cents
    :param currency: Currency code
    :param metadata: Payment metadata
    :param payment_repo: Payment repository instance
    :param promo_code: Optional promo code to apply
    :param promo_repo: Promo code repository instance
    :return: Dictionary containing payment intent details
    """
    try:
        stripe_service = StripeService.get_instance()
        logger.info(f"Creating payment intent for user {user_id}")

        # Apply promo code if provided
        final_amount = amount
        if promo_code and promo_repo:
            promo = await promo_repo.validate_and_use_code(promo_code)
            if promo:
                discount_amount = int(promo.calculate_discount(amount / 100) * 100)  # Convert to cents
                final_amount = max(0, amount - discount_amount)
                metadata["promo_code"] = promo_code
                metadata["discount_amount"] = str(discount_amount)
                logger.info(f"Applied promo code {promo_code}, discount: {discount_amount} cents")

        # Create PaymentIntent with Stripe
        payment_intent = stripe_service.create_payment_intent(
            amount=final_amount,
            currency=currency,
            metadata=metadata
        )

        if not payment_intent or not payment_intent.get('id'):
            logger.error("Failed to create Stripe PaymentIntent")
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Failed to create payment intent")
            )

        # Create payment record in database
        await payment_repo.create_payment(
            user_id=user_id,
            amount=Decimal(final_amount) / 100,  # Convert cents to dollars
            stripe_payment_intent_id=payment_intent['id'],
            currency=currency,
            metadata=metadata
        )

        logger.info(f"Successfully created payment intent: {payment_intent['id']}")
        return BaseResponse(
            success=True,
            data={
                "client_secret": payment_intent['client_secret'],
                "public_key": stripe_service.get_public_key()
            }
        )

    except Exception as e:
        logger.error(f"Error creating payment record: {str(e)}", exc_info=True)
        raise