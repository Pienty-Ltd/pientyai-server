from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database_factory import Base
from app.core.utils import create_random_key

# Association table for User-Organization many-to-many relationship
user_organizations = Table(
    'user_organizations',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('organization_id', Integer, ForeignKey('organizations.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False, 
                default=lambda: f"user_{create_random_key()}")
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship with organizations
    organizations = relationship(
        "Organization",
        secondary=user_organizations,
        back_populates="users"
    )

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False,
                default=lambda: f"org_{create_random_key()}")
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship with users
    users = relationship(
        "User",
        secondary=user_organizations,
        back_populates="organizations"
    )