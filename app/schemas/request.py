from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")
    full_name: str = Field(..., min_length=1, description="User's full name")

class PaymentIntentRequest(BaseModel):
    amount: int = Field(..., gt=0, description="Amount in cents to charge", example=1000)
    currency: str = Field(default="usd", description="Currency code", example="usd")