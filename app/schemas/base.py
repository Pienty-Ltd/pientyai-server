from pydantic import BaseModel, Field
from typing import Optional

class ItemBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class ItemCreate(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    detail: str
