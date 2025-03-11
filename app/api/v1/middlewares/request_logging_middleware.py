from fastapi import Request
import logging
import time
from typing import Callable
from datetime import datetime

logger = logging.getLogger(__name__)

async def log_request_middleware(request: Request, call_next: Callable):
    """Log request details and timing for document operations"""
    start_time = time.time()
    
    # Get request path and method
    path = request.url.path
    method = request.method
    
    # Log request start
    logger.info(f"Request started: {method} {path}")
    
    # Get request headers and query params (excluding sensitive data)
    headers = dict(request.headers)
    if "authorization" in headers:
        headers["authorization"] = "Bearer [REDACTED]"
    
    query_params = dict(request.query_params)
    logger.debug(f"Request details: headers={headers}, query_params={query_params}")
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log request completion
        logger.info(f"Request completed: {method} {path} - Status: {response.status_code} (Duration: {duration:.2f}s)")
        
        return response
        
    except Exception as e:
        # Log any unhandled exceptions
        logger.error(f"Request failed: {method} {path} - Error: {str(e)}", exc_info=True)
        raise
