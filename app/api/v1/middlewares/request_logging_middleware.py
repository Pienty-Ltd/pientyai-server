from fastapi import Request
import logging
import time
import json
import uuid
import asyncio
from typing import Callable, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.request_log_repository import RequestLogRepository
from app.core.security import decode_access_token

logger = logging.getLogger(__name__)

async def get_request_body(request: Request) -> str:
    """Get request body as string, safely"""
    body = b""
    try:
        # Read and restore the request body
        body = await request.body()
        # Restore the request body for downstream use
        request._body = body
    except Exception as e:
        logger.warning(f"Error reading request body: {str(e)}")
    
    # Convert to string, truncate if too large
    try:
        body_str = body.decode() if body else ""
        # Truncate if too large (limit to 10KB)
        if len(body_str) > 10240:
            body_str = body_str[:10240] + "... [truncated]"
        
        # Try to parse as JSON for better output in logs
        try:
            parsed = json.loads(body_str)
            # Redact sensitive fields
            if isinstance(parsed, dict):
                sensitive_fields = ["password", "token", "secret", "key"]
                for field in sensitive_fields:
                    if field in parsed:
                        parsed[field] = "[REDACTED]"
                    # Check for nested dicts
                    for key, value in parsed.items():
                        if isinstance(value, dict):
                            for nested_field in sensitive_fields:
                                if nested_field in value:
                                    value[nested_field] = "[REDACTED]"
                body_str = json.dumps(parsed)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not valid JSON, keep as is
            pass
            
        return body_str
    except UnicodeDecodeError:
        return "[binary data]"

async def extract_user_fp_from_token(request: Request) -> Optional[str]:
    """Extract user fingerprint from authorization token"""
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
            
        token = auth_header.replace("Bearer ", "")
        token_data = decode_access_token(token)
        
        if token_data and "fp" in token_data:
            return token_data["fp"]
    except Exception as e:
        logger.warning(f"Error extracting user fingerprint from token: {str(e)}")
        
    return None

class ResponseCapture:
    """Capture response body from StreamingResponse"""
    def __init__(self, response):
        self.response = response
        self.response_body = b""
        
    async def __call__(self, scope, receive, send):
        # Modified send to capture response body
        async def send_wrapper(message):
            if message["type"] == "http.response.body":
                self.response_body += message["body"]
            await send(message)
            
        # Call the original response with our send_wrapper
        await self.response(scope, receive, send_wrapper)
    
    def get_body(self) -> str:
        """Convert captured body bytes to string"""
        try:
            # Decode response body
            body_str = self.response_body.decode()
            # Truncate if too large (limit to 10KB)
            if len(body_str) > 10240:
                body_str = body_str[:10240] + "... [truncated]"
            return body_str
        except UnicodeDecodeError:
            return "[binary data]"

async def log_request_middleware(request: Request, call_next: Callable):
    """
    Log request details and timing for all operations
    Store logs in database and associate with users when authenticated
    """
    start_time = time.time()
    
    # Generate a unique request ID that will link to the BaseResponse
    request_id = str(uuid.uuid4())
    
    # Set the request ID in state for use in BaseResponse
    request.state.request_id = request_id
    
    # Get request path and method
    path = request.url.path
    method = request.method
    
    # Get client IP
    client_host = request.client.host if request.client else None
    
    # Log request start
    logger.info(f"Request started: {method} {path} - RequestID: {request_id}")
    
    # Get request headers (excluding sensitive data)
    headers = dict(request.headers)
    safe_headers = {**headers}
    
    # Redact sensitive headers
    sensitive_headers = ["authorization", "cookie", "x-api-key"]
    for header in sensitive_headers:
        if header in safe_headers:
            safe_headers[header] = "[REDACTED]"
    
    # Get query params
    query_params = dict(request.query_params)
    
    # Get request body (for non-GET requests)
    request_body = None
    if method not in ["GET", "HEAD"]:
        request_body = await get_request_body(request)
    
    # Extract user fingerprint from token
    user_fp = await extract_user_fp_from_token(request)
    
    log_data = {
        "request_id": request_id,
        "user_fp": user_fp,
        "ip_address": client_host,
        "method": method,
        "path": path,
        "query_params": query_params,
        "request_headers": safe_headers,
        "request_body": request_body
    }
    
    response_body = None
    error_message = None
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Capture response body for logging
        response_capture = ResponseCapture(response)
        
        # Calculate request duration
        duration = time.time() - start_time
        duration_ms = int(duration * 1000)  # Convert to milliseconds
        
        # Update log data with response info
        log_data.update({
            "response_status": response.status_code,
            "duration_ms": duration_ms,
        })
        
        # Log request completion
        logger.info(f"Request completed: {method} {path} - Status: {response.status_code} " +
                   f"(Duration: {duration:.2f}s) - RequestID: {request_id}")
        
        # Store log in database asynchronously
        asyncio.create_task(store_log_in_db(log_data))
        
        return response_capture
        
    except Exception as e:
        # Calculate request duration
        duration = time.time() - start_time
        duration_ms = int(duration * 1000)
        
        # Update log data with error info
        error_message = str(e)
        log_data.update({
            "error": error_message,
            "duration_ms": duration_ms
        })
        
        # Log any unhandled exceptions
        logger.error(f"Request failed: {method} {path} - Error: {error_message} - " +
                    f"RequestID: {request_id}", exc_info=True)
        
        # Store log in database asynchronously
        asyncio.create_task(store_log_in_db(log_data))
        
        # Re-raise the exception
        raise

async def store_log_in_db(log_data: Dict[str, Any]):
    """Store request log in database"""
    try:
        # Get a database session
        async for db in get_db():
            # Create repository
            repo = RequestLogRepository(db)
            # Store log
            await repo.create_request_log(log_data)
            break
    except Exception as e:
        # Just log the error but don't re-raise to avoid affecting the request flow
        logger.error(f"Failed to store request log in database: {str(e)}", exc_info=True)
