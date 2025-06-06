from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any

class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")
    full_name: str = Field(..., min_length=1, description="User's full name")
    invitation_code: str = Field(..., min_length=8, max_length=8, description="Invitation code required for registration")

class PaymentIntentRequest(BaseModel):
    amount: int = Field(..., gt=0, description="Amount in cents to charge", example=1000)
    currency: str = Field(default="usd", description="Currency code", example="usd")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the payment")
    promo_code: Optional[str] = Field(default=None, description="Optional promo code to apply to the payment")