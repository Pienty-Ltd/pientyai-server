from fastapi import APIRouter

# Create the admin router
router = APIRouter()

# Import all admin routers
from app.api.v1.admin.user_routes import router as user_admin_router

# Include all admin routes
router.include_router(user_admin_router)