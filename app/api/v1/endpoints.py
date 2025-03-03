from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.base import (
    ItemCreate,
    ItemResponse,
    MessageResponse
)

router = APIRouter()

# In-memory storage for demonstration
items = []

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
