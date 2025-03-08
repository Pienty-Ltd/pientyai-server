from pydantic import BaseModel
from app.schemas.base import BaseResponse, ErrorResponse

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class LoginResponse(TokenResponse):
    pass