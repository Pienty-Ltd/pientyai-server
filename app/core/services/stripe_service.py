import stripe
from typing import Optional, Dict, Any
import logging
from app.core.config import config

logger = logging.getLogger(__name__)

class StripeError(Exception):
    """Custom exception for Stripe-related errors"""
    pass

class StripeService:
    _instance = None

    def __init__(self):
        """Initialize Stripe configuration"""
        self.is_test_mode = config.STRIPE_TEST_MODE
        
        # Initialize key variables with safety checks
        if self.is_test_mode:
            self.public_key = config.STRIPE_TEST_PUBLIC_KEY
            api_key = config.STRIPE_TEST_SECRET_KEY
            env_type = "TEST"
        else:
            self.public_key = config.STRIPE_LIVE_PUBLIC_KEY
            api_key = config.STRIPE_LIVE_SECRET_KEY
            env_type = "PRODUCTION"
            
        # Verify we have the required keys
        if not api_key:
            error_msg = f"Stripe API key not configured for {env_type} mode"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Set the API key after validation
        stripe.api_key = api_key
        logger.info(f"Initialized Stripe in {env_type} mode")

        # Configure Stripe client
        stripe.api_version = "2023-10-16"  # Lock API version
        stripe.max_network_retries = 2  # Enable automatic retries

    @classmethod
    def get_instance(cls) -> "StripeService":
        """Returns singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_payment_intent(
        self,
        amount: int,
        currency: str = 'usd',
        metadata: Optional[Dict[str, Any]] = None,
        customer: Optional[str] = None,
        payment_method_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Create a PaymentIntent
        :param amount: Amount in cents (e.g., 1000 for $10.00)
        :param currency: Three-letter currency code
        :param metadata: Optional metadata to attach to the payment
        :param customer: Optional Stripe customer ID
        :param payment_method_types: Optional list of payment method types
        :return: Stripe PaymentIntent object
        """
        try:
            intent_data = {
                'amount': amount,
                'currency': currency,
                'metadata': metadata or {},
                'automatic_payment_methods': {'enabled': True}
            }

            if customer:
                intent_data['customer'] = customer

            if payment_method_types:
                intent_data['payment_method_types'] = payment_method_types

            intent = stripe.PaymentIntent.create(**intent_data)
            logger.info(f"Created PaymentIntent: {intent.id}")
            return intent

        except stripe.error.CardError as e:
            logger.error(f"Card error creating PaymentIntent: {str(e)}")
            raise StripeError(f"Card error: {str(e)}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid request creating PaymentIntent: {str(e)}")
            raise StripeError(f"Invalid request: {str(e)}")
        except stripe.error.AuthenticationError as e:
            logger.error("Authentication with Stripe failed")
            raise StripeError("Authentication with Stripe failed")
        except stripe.error.APIConnectionError as e:
            logger.error(f"Network error creating PaymentIntent: {str(e)}")
            raise StripeError("Network error occurred")
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating PaymentIntent: {str(e)}")
            raise StripeError(f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating PaymentIntent: {str(e)}")
            raise StripeError("An unexpected error occurred")

    def create_customer(
        self,
        email: str,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Stripe Customer
        :param email: Customer's email address
        :param metadata: Optional metadata to attach to the customer
        :param name: Optional customer name
        :param payment_method: Optional default payment method ID
        :return: Stripe Customer object
        """
        try:
            customer_data = {
                'email': email,
                'metadata': metadata or {}
            }

            if name:
                customer_data['name'] = name

            if payment_method:
                customer_data['payment_method'] = payment_method

            customer = stripe.Customer.create(**customer_data)
            logger.info(f"Created Customer: {customer.id}")
            return customer

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating Customer: {str(e)}")
            raise StripeError(f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating Customer: {str(e)}")
            raise StripeError("An unexpected error occurred")

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        trial_period_days: Optional[int] = None,
        payment_behavior: str = 'default_incomplete'
    ) -> Dict[str, Any]:
        """
        Create a subscription for a customer
        :param customer_id: Stripe Customer ID
        :param price_id: Stripe Price ID
        :param metadata: Optional metadata
        :param trial_period_days: Optional trial period in days
        :param payment_behavior: Payment behavior setting
        :return: Stripe Subscription object
        """
        try:
            subscription_data = {
                'customer': customer_id,
                'items': [{'price': price_id}],
                'payment_behavior': payment_behavior,
                'metadata': metadata or {},
                'expand': ['latest_invoice.payment_intent']
            }

            if trial_period_days:
                subscription_data['trial_period_days'] = trial_period_days

            subscription = stripe.Subscription.create(**subscription_data)
            logger.info(f"Created Subscription: {subscription.id}")
            return subscription

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating Subscription: {str(e)}")
            raise StripeError(f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating Subscription: {str(e)}")
            raise StripeError("An unexpected error occurred")

    def get_public_key(self) -> str:
        """Get the appropriate Stripe public key based on environment"""
        if not self.public_key:
            raise StripeError("Stripe public key not configured")
        return self.public_key
        
    def create_checkout_session(
        self,
        price_id: str,
        success_url: str,
        cancel_url: str,
        customer_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = 'subscription'
    ) -> Dict[str, Any]:
        """
        Create a Stripe Checkout Session
        :param price_id: Stripe Price ID
        :param success_url: URL to redirect after successful payment
        :param cancel_url: URL to redirect after cancelled payment
        :param customer_email: Customer email (optional)
        :param metadata: Optional metadata
        :param mode: Checkout mode (subscription, payment, or setup)
        :return: Stripe Checkout Session object
        """
        try:
            session_data = {
                'line_items': [{'price': price_id, 'quantity': 1}],
                'mode': mode,
                'success_url': success_url,
                'cancel_url': cancel_url,
                'metadata': metadata or {}
            }
            
            if customer_email:
                session_data['customer_email'] = customer_email
                
            checkout_session = stripe.checkout.Session.create(**session_data)
            logger.info(f"Created Checkout Session: {checkout_session.id}")
            
            # Return a dictionary with the necessary information
            return {
                'id': checkout_session.id,
                'url': checkout_session.url,
                'status': checkout_session.status,
                'customer_email': checkout_session.customer_email,
                'payment_status': checkout_session.payment_status
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating Checkout Session: {str(e)}")
            raise StripeError(f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating Checkout Session: {str(e)}")
            raise StripeError("An unexpected error occurred")

    def cancel_subscription(
        self,
        subscription_id: str,
        invoice_now: bool = False,
        prorate: bool = True
    ) -> Dict[str, Any]:
        """
        Cancel a subscription
        :param subscription_id: The ID of the subscription to cancel
        :param invoice_now: Whether to create an invoice now for the final payment
        :param prorate: Whether to prorate the subscription
        :return: The canceled subscription object
        """
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
                proration_behavior='create_prorations' if prorate else 'none'
            )

            if invoice_now:
                stripe.Invoice.create(
                    customer=subscription.customer,
                    subscription=subscription_id
                )

            logger.info(f"Cancelled subscription: {subscription_id}")
            return subscription

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error cancelling subscription: {str(e)}")
            raise StripeError(f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error cancelling subscription: {str(e)}")
            raise StripeError("An unexpected error occurred")