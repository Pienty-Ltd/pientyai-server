from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.database.database_factory import Base
from app.core.utils import create_random_key

class ErrorLog(Base):
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, index=True)
    fp = Column(String, unique=True, index=True, nullable=False, 
                default=lambda: f"err_{create_random_key()}")
    # Error type tracking
    error_type = Column(String, nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    error_traceback = Column(Text, nullable=True)
    
    # Location tracking
    component = Column(String, nullable=True, index=True)  # Module/file where error occurred
    function = Column(String, nullable=True)  # Function/method where error occurred
    line_number = Column(Integer, nullable=True)
    
    # Request context if available
    request_id = Column(String, index=True, nullable=True)  # Link to request_logs if applicable
    user_fp = Column(String, index=True, nullable=True)  # User who experienced the error
    ip_address = Column(String, nullable=True)
    path = Column(String, nullable=True)
    method = Column(String, nullable=True)
    
    # Environment context
    host = Column(String, nullable=True)  # Server hostname
    environment = Column(String, nullable=True)  # dev, staging, production
    
    # Additional information
    context_data = Column(JSON, nullable=True)  # Additional context as JSON
    
    # Status tracking
    is_resolved = Column(Integer, default=0, nullable=False)  # 0: unresolved, 1: resolved, 2: ignored
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ErrorLog(id={self.id}, error_type='{self.error_type}', created_at='{self.created_at}')>"