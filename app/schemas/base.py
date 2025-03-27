from pydantic import BaseModel, Field
from typing import Optional, Generic, TypeVar, Dict, Any, List
from fastapi import Request
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
    
    @classmethod
    def from_request(cls, 
                   request: Request,
                   data: Optional[T] = None,
                   success: bool = True,
                   message: Optional[str] = None,
                   error: Optional[ErrorResponse] = None):
        """
        Create a BaseResponse with request_id from the request state.
        This helps to link the response to the logged request in the database.
        """
        # Get request_id from state if available, otherwise generate a new one
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        return cls(
            data=data,
            success=success,
            message=message,
            request_id=request_id,
            error=error
        )
