from pydantic import BaseModel
from app.schemas.base import BaseResponse, ErrorResponse

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"