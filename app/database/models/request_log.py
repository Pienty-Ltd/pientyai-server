from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database_factory import Base
from app.core.utils import create_random_key

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False, 
                default=lambda: f"req_{create_random_key()}")
    request_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ip_address = Column(String, nullable=True)
    method = Column(String, nullable=False)
    path = Column(String, nullable=False)
    query_params = Column(JSON, nullable=True)
    request_headers = Column(JSON, nullable=True)
    request_body = Column(Text, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Duration in milliseconds
    error = Column(Text, nullable=True)  # Store error message if request failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship with user (if authenticated)
    user = relationship("User", backref="request_logs")