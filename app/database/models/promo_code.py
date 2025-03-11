from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SQLEnum, Numeric, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
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
    discount_value = Column(Numeric(10, 2))
    max_uses = Column(Integer, default=1)
    max_uses_per_user = Column(Integer, default=1)
    times_used = Column(Integer, default=0)
    valid_from = Column(DateTime, default=func.now())
    valid_until = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship with usage history - set to dynamic for optimized loading
    usage_history = relationship("PromoCodeUsage", back_populates="promo_code", lazy="dynamic")

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

class PromoCodeUsage(Base):
    __tablename__ = "promo_code_usage"

    id = Column(Integer, primary_key=True, index=True)
    promo_code_id = Column(Integer, ForeignKey("promo_codes.id"), nullable=False)
    user_id = Column(Integer, nullable=False, index=True)
    used_at = Column(DateTime, default=func.now(), nullable=False)
    amount = Column(Numeric(10, 2))
    discount_amount = Column(Numeric(10, 2))
    usage_metadata = Column(String)

    # Set lazy loading for the relationship
    promo_code = relationship("PromoCode", back_populates="usage_history", lazy="select")