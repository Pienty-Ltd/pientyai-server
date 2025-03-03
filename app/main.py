import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.api.v1.auth import router as auth_router
from app.core.config import config
from app.database.database_factory import create_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal AI SaaS Backend",
              description="Backend server for Legal AI SaaS platform",
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
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "errors": [{
                "loc": err["loc"],
                "msg": err["msg"]
            } for err in exc.errors()]
        })

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Legal AI SaaS Backend"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": app.version}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    try:
        await create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

# Only used when running directly with Python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)