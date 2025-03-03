import stripe
from typing import Optional, Dict, Any
import logging
from app.core.config import config

logger = logging.getLogger(__name__)

class StripeService:
    _instance = None

    def __init__(self):
        self.is_production = config.API_PRODUCTION
        if self.is_production:
            self.public_key = config.STRIPE_LIVE_PUBLIC_KEY
            stripe.api_key = config.STRIPE_LIVE_SECRET_KEY
            logger.info("Initialized Stripe in PRODUCTION mode")
        else:
            self.public_key = config.STRIPE_TEST_PUBLIC_KEY
            stripe.api_key = config.STRIPE_TEST_SECRET_KEY
            logger.info("Initialized Stripe in TEST mode")

    @classmethod
    def get_instance(cls) -> "StripeService":
        """Returns singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_payment_intent(self, amount: int, currency: str = 'usd', metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a PaymentIntent
        :param amount: Amount in cents (e.g., 1000 for $10.00)
        :param currency: Three-letter currency code
        :param metadata: Optional metadata to attach to the payment
        :return: Stripe PaymentIntent object
        """
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                metadata=metadata
            )
            logger.info(f"Created PaymentIntent: {intent.id}")
            return intent
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating PaymentIntent: {str(e)}")
            raise

    def create_customer(self, email: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a Stripe Customer
        :param email: Customer's email address
        :param metadata: Optional metadata to attach to the customer
        :return: Stripe Customer object
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                metadata=metadata
            )
            logger.info(f"Created Customer: {customer.id}")
            return customer
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating Customer: {str(e)}")
            raise

    def create_subscription(self, customer_id: str, price_id: str) -> Dict[str, Any]:
        """
        Create a subscription for a customer
        :param customer_id: Stripe Customer ID
        :param price_id: Stripe Price ID
        :return: Stripe Subscription object
        """
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            logger.info(f"Created Subscription: {subscription.id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating Subscription: {str(e)}")
            raise

    def get_public_key(self) -> str:
        """Get the appropriate Stripe public key based on environment"""
        return self.public_key
