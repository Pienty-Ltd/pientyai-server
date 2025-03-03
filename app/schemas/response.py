from pydantic import BaseModel, Field
from typing import Optional, Generic, TypeVar, Dict, Any, List
import uuid

T = TypeVar('T')

class ErrorResponse(BaseModel):
    message: Optional[str] = None
    logout: bool = False
    details: Optional[List[Dict[str, Any]]] = None

class BaseResponse(BaseModel, Generic[T]):
    data: Optional[T] = None
    success: bool = True
    message: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    error: Optional[ErrorResponse] = None