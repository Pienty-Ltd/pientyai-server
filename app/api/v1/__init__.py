from fastapi import APIRouter
from app.api.v1.organization_routes import router as organization_router
from app.api.v1.admin_routes import router as admin_router  # Bu mevcut admin routeları için
from app.api.v1.admin import router as admin_package_router  # Yeni admin paketi
from app.api.v1.auth import router as auth_router
from app.api.v1.dashboard_routes import router as dashboard_router
from app.api.v1.document_routes import router as document_router
from app.api.v1.invitation_routes import router as invitation_router
from app.api.v1.payment_routes import router as payment_router

# Create main router for v1 API
router = APIRouter()

# Include all routes
router.include_router(organization_router)
router.include_router(admin_router)  # Eski admin rotaları (promo kodları, stats vb.)
router.include_router(admin_package_router, prefix="/api/v1/admin")  # Yeni admin paketi (kullanıcı yönetimi vb.)
router.include_router(auth_router)
router.include_router(dashboard_router)
router.include_router(document_router)
router.include_router(invitation_router)
router.include_router(payment_router)