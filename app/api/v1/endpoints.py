from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import logging
from app.schemas.base import (
    ItemCreate,
    ItemResponse,
    MessageResponse
)
from app.database.database_factory import get_db
from app.core.redis_service import RedisService
from app.database.repositories.user_repository import UserRepository
from app.database.repositories.organization_repository import OrganizationRepository
from app.database.models.db_models import User, Organization

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demonstration
items = []

@router.get("/health", response_model=MessageResponse)
async def check_connections(db: AsyncSession = Depends(get_db)):
    try:
        # Test database connection
        logger.info("Testing database connection...")
        await db.execute("SELECT 1")
        logger.info("Database connection successful")

        # Test Redis connection
        logger.info("Testing Redis connection...")
        redis_service = await RedisService.get_instance()
        await redis_service.set_value("test_key", "test_value")
        test_value = await redis_service.get_value("test_key")
        if test_value != "test_value":
            raise Exception("Redis test value mismatch")
        await redis_service.delete_value("test_key")
        logger.info("Redis connection successful")

        return {"message": "Database and Redis connections are working"}
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")

@router.get("/items", response_model=List[ItemResponse])
async def get_items():
    """
    Get all items
    """
    return items

@router.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate):
    """
    Create a new item
    """
    new_item = {
        "id": len(items) + 1,
        "name": item.name,
        "description": item.description
    }
    items.append(new_item)
    return new_item

@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """
    Get a specific item by ID
    """
    if item_id <= 0 or item_id > len(items):
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id - 1]

@router.delete("/items/{item_id}", response_model=MessageResponse)
async def delete_item(item_id: int):
    """
    Delete a specific item by ID
    """
    if item_id <= 0 or item_id > len(items):
        raise HTTPException(status_code=404, detail="Item not found")
    items.pop(item_id - 1)
    return {"message": "Item deleted successfully"}

@router.post("/users/test", response_model=MessageResponse)
async def test_user_creation(db: AsyncSession = Depends(get_db)):
    try:
        # Create a test user repository
        user_repo = UserRepository(db)

        # Test user data
        test_user = {
            "email": "test@example.com",
            "full_name": "Test User",
            "fp": "test_fp_123",
            "hashed_password": "hashed_password_here",
            "is_active": True
        }

        # Create user
        user = await user_repo.insert_user(test_user)
        logger.info(f"Created test user with ID: {user.id}")

        # Test organization repository
        org_repo = OrganizationRepository(db)

        # Create test organization
        org_data = {
            "name": "Test Law Firm",
            "description": "A test law firm organization"
        }

        org = await org_repo.create_organization(org_data)
        logger.info(f"Created test organization with ID: {org.id}")

        # Add user to organization
        await org_repo.add_user_to_organization(user, org)
        logger.info(f"Added user {user.id} to organization {org.id}")

        return {"message": "Test data created successfully"}
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))