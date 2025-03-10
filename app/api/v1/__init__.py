from fastapi import APIRouter
from app.api.v1.organization_routes import router as organization_router

# Create main router for v1 API
router = APIRouter()

# Include organization routes
router.include_router(organization_router)