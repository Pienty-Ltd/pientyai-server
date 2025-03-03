import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.v1.auth import router as auth_router
from app.core.config import config
from app.database.database_factory import create_tables
from app.schemas.response import BaseResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI Backend",
              description="FastAPI Backend API",
              version="1.0.0",
              docs_url="/docs",
              redoc_url="/redoc")

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
            error=ErrorResponse(
                message="Invalid request data",
                details=[{
                    "loc": err["loc"],
                    "msg": err["msg"]
                } for err in exc.errors()]
            )
        ).dict()
    )

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
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
    return BaseResponse(message="Welcome to FastAPI Backend API")

# Health check endpoint
@app.get("/health")
async def health_check():
    return BaseResponse(
        data={"status": "healthy", "version": app.version}
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    try:
        await create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)