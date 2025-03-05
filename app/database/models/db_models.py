from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, Boolean, Enum, Numeric, Text, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database_factory import Base
from app.core.utils import create_random_key
from pgvector.sqlalchemy import Vector
import enum

# Association table for User-Organization many-to-many relationship
user_organizations = Table(
    'user_organizations',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('organization_id', Integer, ForeignKey('organizations.id'), primary_key=True)
)

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"

class SubscriptionStatus(enum.Enum):
    TRIAL = "trial"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELED = "canceled"

class PaymentStatus(enum.Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"

class FileStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False, 
                default=lambda: f"user_{create_random_key()}")
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    organizations = relationship(
        "Organization",
        secondary=user_organizations,
        back_populates="users"
    )

    subscription = relationship("UserSubscription", back_populates="user", uselist=False)
    payment_history = relationship("PaymentHistory", back_populates="user")
    files = relationship("File", back_populates="user")
    dashboard_stats = relationship("DashboardStats", back_populates="user")

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False,
                default=lambda: f"org_{create_random_key()}")
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    users = relationship(
        "User",
        secondary=user_organizations,
        back_populates="organizations"
    )
    files = relationship("File", back_populates="organization")
    knowledge_base = relationship("KnowledgeBase", back_populates="organization")
    dashboard_stats = relationship("DashboardStats", back_populates="organization")

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False,
                default=lambda: f"file_{create_random_key(25)}")
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, docx, etc.
    s3_key = Column(String, unique=True, nullable=False)
    status = Column(Enum(FileStatus), nullable=False, default=FileStatus.PENDING)
    file_size = Column(Integer)  # Size in bytes
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    user = relationship("User", back_populates="files")
    organization = relationship("Organization", back_populates="files")
    knowledge_base = relationship("KnowledgeBase", back_populates="file")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False,
                default=lambda: f"kb_{create_random_key(25)}")
    chunk_index = Column(Integer, nullable=False)  # Sıra numarası
    content = Column(Text, nullable=False)  # Chunk içeriği
    embedding = Column(Vector(1536), nullable=False)  # OpenAI ada-002 embedding vektörü
    meta_info = Column(String)  # JSON olarak ek bilgiler (metadata yerine meta_info kullanıyoruz)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    file = relationship("File", back_populates="knowledge_base")
    organization = relationship("Organization", back_populates="knowledge_base")

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    status = Column(Enum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.TRIAL)
    trial_start = Column(DateTime(timezone=True), server_default=func.now())
    trial_end = Column(DateTime(timezone=True), nullable=False)
    subscription_start = Column(DateTime(timezone=True))
    subscription_end = Column(DateTime(timezone=True))
    stripe_subscription_id = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="subscription")

class PaymentHistory(Base):
    __tablename__ = "payment_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String, nullable=False, default="usd")
    stripe_payment_intent_id = Column(String, nullable=False)
    status = Column(Enum(PaymentStatus), nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="payment_history")


class DashboardStats(Base):
    __tablename__ = "dashboard_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    total_knowledge_base_count = Column(Integer, default=0)
    total_file_count = Column(Integer, default=0)
    total_storage_used = Column(Integer, default=0)  # in bytes
    last_activity_date = Column(DateTime(timezone=True))
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # A row can be either for a user or an organization, but not both
    __table_args__ = (
        CheckConstraint('NOT(user_id IS NULL AND organization_id IS NULL)'),
        CheckConstraint('NOT(user_id IS NOT NULL AND organization_id IS NOT NULL)'),
    )

    user = relationship("User", back_populates="dashboard_stats")
    organization = relationship("Organization", back_populates="dashboard_stats")

User.dashboard_stats = relationship("DashboardStats", back_populates="user")
Organization.dashboard_stats = relationship("DashboardStats", back_populates="organization")