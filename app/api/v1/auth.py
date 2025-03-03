from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.response import BaseResponse, ErrorResponse
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/auth")

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/login", response_model=BaseResponse[LoginResponse])
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_email(request.email)
        
        if not user:
            return BaseResponse(
                success=False,
                message="Invalid credentials",
                error=ErrorResponse(message="User not found")
            )
            
        # TODO: Add password verification
        # For now, return a mock token
        return BaseResponse(
            data=LoginResponse(
                access_token="mock_token",
                token_type="bearer"
            ),
            message="Login successful"
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

@router.get("/me", response_model=BaseResponse[dict])
async def get_current_user():
    # TODO: Implement user verification from token
    return BaseResponse(
        data={"username": "test_user"},
        message="User details retrieved successfully"
    )
