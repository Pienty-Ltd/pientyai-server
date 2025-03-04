import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from app.database.models.db_models import PaymentHistory, PaymentStatus
from decimal import Decimal
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class PaymentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_payment(
        self,
        user_id: int,
        amount: Decimal,
        stripe_payment_intent_id: str,
        currency: str = "usd",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PaymentHistory:
        """
        Create a new payment record
        :param user_id: User ID
        :param amount: Payment amount
        :param stripe_payment_intent_id: Stripe PaymentIntent ID
        :param currency: Currency code
        :param description: Payment description
        :param metadata: Additional metadata
        :return: Created payment record
        """
        try:
            payment = PaymentHistory(
                user_id=user_id,
                amount=amount,
                currency=currency,
                stripe_payment_intent_id=stripe_payment_intent_id,
                status=PaymentStatus.PENDING,
                description=description,
                metadata=metadata
            )
            async with self.db.begin():
                self.db.add(payment)
                await self.db.flush()
                await self.db.refresh(payment)
                logger.info(f"Created payment record: {payment.id} for user: {user_id}")
            return payment
        except Exception as e:
            logger.error(f"Error creating payment record: {str(e)}")
            raise

    async def get_payment_by_intent_id(self, stripe_payment_intent_id: str) -> Optional[PaymentHistory]:
        """
        Get payment by Stripe PaymentIntent ID
        :param stripe_payment_intent_id: Stripe PaymentIntent ID
        :return: Payment record if found
        """
        try:
            result = await self.db.execute(
                select(PaymentHistory)
                .filter(PaymentHistory.stripe_payment_intent_id == stripe_payment_intent_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching payment by intent ID: {str(e)}")
            raise

    async def get_user_payments(
        self,
        user_id: int,
        limit: int = 10,
        offset: int = 0,
        status: Optional[PaymentStatus] = None
    ) -> List[PaymentHistory]:
        """
        Get all payments for a user with pagination
        :param user_id: User ID
        :param limit: Number of records to return
        :param offset: Number of records to skip
        :param status: Optional status filter
        :return: List of payment records
        """
        try:
            query = select(PaymentHistory).filter(PaymentHistory.user_id == user_id)

            if status:
                query = query.filter(PaymentHistory.status == status)

            query = query.order_by(PaymentHistory.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await self.db.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching user payments: {str(e)}")
            raise

    async def update_payment_status(
        self,
        payment_id: int,
        status: PaymentStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[PaymentHistory]:
        """
        Update payment status and metadata
        :param payment_id: Payment ID
        :param status: New payment status
        :param metadata: Optional metadata to update
        :return: Updated payment record
        """
        try:
            async with self.db.begin():
                stmt = update(PaymentHistory).where(
                    PaymentHistory.id == payment_id
                ).values(
                    status=status,
                    **({"metadata": metadata} if metadata else {})
                ).returning(PaymentHistory)

                result = await self.db.execute(stmt)
                payment = result.scalar_one_or_none()

                if payment:
                    logger.info(f"Updated payment {payment_id} status to {status}")
                else:
                    logger.warning(f"Payment {payment_id} not found for status update")

                return payment
        except Exception as e:
            logger.error(f"Error updating payment status: {str(e)}")
            raise