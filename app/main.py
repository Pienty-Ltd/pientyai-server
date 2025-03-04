import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.v1.auth import router as auth_router, CustomAuthException
from app.api.v1.payment_routes import router as payment_router
from app.api.v1.admin_routes import router as admin_router
from app.core.config import config
from app.database.database_factory import create_tables
from app.schemas.base import BaseResponse, ErrorResponse

project_name = "Pienty.AI"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f'{project_name} %(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{project_name} API",
    description=f"{project_name} API Documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User authentication and authorization operations"
        },
        {
            "name": "Admin",
            "description": "Administrative operations"
        },
        {
            "name": "Promo Codes",
            "description": "Promo code management and usage"
        },
        {
            "name": "Payments",
            "description": "Payment processing and management"
        }
    ]
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(payment_router)
app.include_router(admin_router)

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content=BaseResponse(
            success=False,
            message="Validation Error",
            error=ErrorResponse(message="Invalid request data",
                              details=[{
                                  "loc": err["loc"],
                                  "msg": err["msg"]
                              } for err in exc.errors()])
        ).dict()
    )

# Authentication exception handler
@app.exception_handler(CustomAuthException)
async def auth_exception_handler(request: Request, exc: CustomAuthException):
    logger.warning(f"Authentication exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse(
            success=False,
            message="Authentication failed",
            error=ErrorResponse(message=str(exc.detail),
                              details=[{"msg": str(exc.detail)}])
        ).dict(),
        headers=exc.headers
    )

# Handle FastAPI's built-in HTTPException (including 401 Unauthorized)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse(
            success=False,
            message="Request failed",
            error=ErrorResponse(message=str(exc.detail),
                              details=[{"msg": str(exc.detail)}])
        ).dict(),
        headers=getattr(exc, 'headers', None)
    )

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=BaseResponse(
            success=False,
            message="Internal Server Error",
            error=ErrorResponse(message="An unexpected error occurred")
        ).dict()
    )

# Root endpoint
@app.get("/")
async def root():
    return BaseResponse(message=f"Welcome to {project_name} API")

# Health check endpoint
@app.get("/health")
async def health_check():
    return BaseResponse(data={"status": "healthy", "version": app.version})

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up {project_name} server...")
    try:
        await create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    # Ensure consistency with the port number used in the workflow
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)