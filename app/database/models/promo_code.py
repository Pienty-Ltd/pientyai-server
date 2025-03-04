from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SQLEnum, Numeric
from sqlalchemy.sql import func
from app.database.database_factory import Base

class DiscountType(str, Enum):
    PERCENTAGE = "percentage"  # e.g., 20% off
    FIXED = "fixed"           # e.g., $10 off
    FULL = "full"            # 100% off (free)

class PromoCode(Base):
    __tablename__ = "promo_codes"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(200))
    discount_type = Column(SQLEnum(DiscountType), nullable=False)
    discount_value = Column(Numeric(10, 2))  # Amount or percentage of discount
    max_uses = Column(Integer, default=1)    # How many times this code can be used
    times_used = Column(Integer, default=0)   # How many times this code has been used
    valid_from = Column(DateTime, default=func.now())
    valid_until = Column(DateTime)           # When the code expires
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def is_valid(self) -> bool:
        """Check if the promo code is valid for use"""
        now = datetime.utcnow()
        return (
            self.is_active and
            (self.max_uses == 0 or self.times_used < self.max_uses) and
            self.valid_from <= now and
            (self.valid_until is None or self.valid_until > now)
        )

    def calculate_discount(self, original_amount: float) -> float:
        """Calculate the discount amount based on the discount type and value"""
        if not self.is_valid():
            return 0.0

        if self.discount_type == DiscountType.FULL:
            return original_amount
        elif self.discount_type == DiscountType.PERCENTAGE:
            return original_amount * (float(self.discount_value) / 100)
        else:  # FIXED
            return min(float(self.discount_value), original_amount)
