import logging
import os
import asyncio
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from app.api.v1.auth import router as auth_router
from app.api.v1.payment_routes import router as payment_router
from app.api.v1.admin_routes import router as admin_router
from app.api.v1.document_routes import router as document_router
from app.api.v1.dashboard_routes import router as dashboard_router
from app.api.v1.invitation_routes import router as invitation_router
from app.api.v1.document_analysis_routes import router as document_analysis_router
from app.api.v1.subscription_routes import router as subscription_router
from app.api.v1 import router as v1_router
from app.core.config import config
from app.database.database_factory import create_tables, get_db
from app.schemas.base import BaseResponse, ErrorResponse
from app.api.v1.middlewares.request_logging_middleware import log_request_middleware
from app.core.services.error_logger_service import ErrorLoggerService

project_name = "Pienty.AI"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=
    f'{project_name} %(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title=f"{project_name} API",
              description=f"{project_name} API Documentation",
              version="1.0.0",
              docs_url="/docs",
              redoc_url="/redoc")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=
    False,  # Disable credentials since we're using allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.middleware("http")(log_request_middleware)

# Include routers
app.include_router(auth_router)
app.include_router(payment_router)
app.include_router(admin_router)
app.include_router(document_router)
app.include_router(dashboard_router)
app.include_router(invitation_router)
app.include_router(document_analysis_router)
app.include_router(subscription_router)
app.include_router(v1_router)


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {str(exc)}")
    
    # Log validation errors to help identify common input problems
    # We collect the error details to make them searchable in the database
    error_details = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors()]
    error_message = "; ".join(error_details)
    
    # Create a structured context with all validation errors
    context_data = {
        "validation_errors": [
            {
                "location": err["loc"],
                "message": err["msg"],
                "type": err.get("type", "unknown")
            } for err in exc.errors()
        ],
        "path": str(request.url),
        "method": request.method
    }
    
    # Log the validation error
    asyncio.create_task(
        ErrorLoggerService.log_error(
            error=exc,
            error_type="VALIDATION_ERROR",
            request_id=getattr(request.state, 'request_id', None),
            user_fp=getattr(request.state, 'user', {}).get('fp', None) if hasattr(request.state, 'user') else None,
            ip_address=request.client.host if request.client else None,
            path=str(request.url),
            method=request.method,
            context_data=context_data
        )
    )
    
    return JSONResponse(status_code=422,
                        content=BaseResponse(
                            success=False,
                            message="Validation Error",
                            error=ErrorResponse(message="Invalid request data",
                                                details=[{
                                                    "loc": err["loc"],
                                                    "msg": err["msg"]
                                                } for err in exc.errors()
                                                         ])).dict())


# Handle unauthorized access and authentication errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    status_code = exc.status_code
    headers = getattr(exc, 'headers', None)

    # Log level depends on status code
    if status_code >= 500:
        logger.error(f"Server error: {exc.detail}")
        
        # Log server errors (5xx) to the database
        # These are important to track as they indicate server-side issues
        asyncio.create_task(
            ErrorLoggerService.log_error(
                error=exc,
                error_type=f"HTTP_{status_code}",
                request_id=getattr(request.state, 'request_id', None),
                user_fp=getattr(request.state, 'user', {}).get('fp', None) if hasattr(request.state, 'user') else None,
                ip_address=request.client.host if request.client else None,
                path=str(request.url),
                method=request.method
            )
        )
    elif status_code >= 400:
        logger.warning(f"Client error: {exc.detail}")
        
        # Only log certain 4xx errors that might indicate problems with the application
        # like 405 Method Not Allowed or 415 Unsupported Media Type
        if status_code in [405, 415, 422, 429]:
            asyncio.create_task(
                ErrorLoggerService.log_error(
                    error=exc,
                    error_type=f"HTTP_{status_code}",
                    request_id=getattr(request.state, 'request_id', None),
                    user_fp=getattr(request.state, 'user', {}).get('fp', None) if hasattr(request.state, 'user') else None,
                    ip_address=request.client.host if request.client else None,
                    path=str(request.url),
                    method=request.method
                )
            )
    else:
        logger.info(f"HTTP exception: {exc.detail}")

    # If detail is a dict, preserve its structure; otherwise create a standard format
    if isinstance(exc.detail, dict):
        error_detail = exc.detail
        logger.debug(f"Using original error detail structure: {error_detail}")
    else:
        error_message = str(exc.detail)
        if status_code in [401, 403]:  # Authentication/Authorization errors
            error_message = "Authentication required" if status_code == 401 else "Permission denied"
            error_detail = {
                "message": error_message,
                "logout": True,
                "details": [{
                    "msg": error_message
                }]
            }
        else:
            error_detail = {
                "message": error_message,
                "logout": False,
                "details": [{
                    "msg": error_message
                }]
            }
        logger.debug(
            f"Created standard error detail structure: {error_detail}")

    return JSONResponse(status_code=status_code,
                        content=BaseResponse(
                            success=False,
                            message="Request failed",
                            error=ErrorResponse(**error_detail)).dict(),
                        headers=headers)


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Generate a unique error ID for tracking
    error_id = f"err_{uuid.uuid4().hex[:10]}"
    
    # Get request information
    request_id = getattr(request.state, 'request_id', None)
    user_fp = None
    
    # Try to get user from request state
    if hasattr(request.state, 'user'):
        user = getattr(request.state, 'user', None)
        if user and hasattr(user, 'fp'):
            user_fp = user.fp
    
    # Log the error to the database asynchronously
    # We use create_task to avoid waiting for the DB operation
    asyncio.create_task(
        ErrorLoggerService.log_error(
            error=exc,
            request_id=request_id,
            user_fp=user_fp,
            ip_address=request.client.host if request.client else None,
            path=str(request.url),
            method=request.method,
            context_data={
                "headers": dict(request.headers),
                "query_params": dict(request.query_params)
            }
        )
    )
    
    return JSONResponse(
        status_code=500,
        content=BaseResponse(
            success=False,
            message="Internal Server Error",
            error=ErrorResponse(
                message=f"An unexpected error occurred (Error ID: {error_id})")
        ).dict())


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

        # Setup dashboard stats cron jobs
        """try:
            db = await get_db().__anext__()
            await db.execute(text("SELECT manage_dashboard_stats_cron_jobs()"))
            await db.commit()
            logger.info("Dashboard stats cron jobs setup completed")
        except Exception as e:
            logger.error(
                f"Error setting up dashboard stats cron jobs: {str(e)}")
        finally:
            if 'db' in locals():
                await db.close()"""

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app",
                host="0.0.0.0",
                port=3348,
                reload=True,
                log_level="info")
