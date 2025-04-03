import logging
import traceback
import sys
import socket
import asyncio
import inspect
from typing import Optional, Dict, Any, Type, Callable
from functools import wraps

from app.database.repositories.error_log_repository import ErrorLogRepository
from app.database.database_factory import get_db
from app.core.config import config

logger = logging.getLogger(__name__)

class ErrorLoggerService:
    """Service for logging errors to the database"""
    
    @staticmethod
    async def log_error(
        error: Exception,
        error_type: Optional[str] = None,
        component: Optional[str] = None,
        function: Optional[str] = None,
        line_number: Optional[int] = None,
        request_id: Optional[str] = None,
        user_fp: Optional[str] = None,
        ip_address: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log an error to the database
        
        Args:
            error: The exception to log
            error_type: Custom error type categorization (defaults to exception class name)
            component: Module/file where error occurred
            function: Function/method where error occurred
            line_number: Line number where error occurred
            request_id: Request ID if error occurred during request processing
            user_fp: User fingerprint if available
            ip_address: Client IP address if available
            path: Request path if available
            method: Request method if available
            context_data: Additional context data as dictionary
            
        Returns:
            Error fingerprint if logging was successful, None otherwise
        """
        try:
            # Get error details - use provided error_type or fallback to class name
            error_type = error_type or error.__class__.__name__
            error_message = str(error)
            error_traceback = ''.join(traceback.format_exception(
                type(error), error, error.__traceback__))
            
            # Auto-detect component and function if not provided
            if not component or not function or not line_number:
                frame_info = ErrorLoggerService._get_error_frame_info(error)
                component = component or frame_info.get('component')
                function = function or frame_info.get('function')
                line_number = line_number or frame_info.get('line_number')
            
            # Prepare error data
            error_data = {
                "error_type": error_type,
                "error_message": error_message,
                "error_traceback": error_traceback,
                "component": component,
                "function": function,
                "line_number": line_number,
                "request_id": request_id,
                "user_fp": user_fp,
                "ip_address": ip_address,
                "path": path,
                "method": method,
                "host": socket.gethostname(),
                "environment": "production" if config.API_PRODUCTION else "development",
                "context_data": context_data or {},
                "is_resolved": 0  # Default to unresolved
            }
            
            # Create database session and repository
            async for db in get_db():
                repo = ErrorLogRepository(db)
                error_log = await repo.create_error_log(error_data)
                if error_log:
                    return error_log.fp
                return None
            
        except Exception as e:
            # If error logging fails, just log to console but don't raise
            # to avoid infinite error recursion
            logger.error(f"Failed to log error to database: {str(e)}")
            return None
    
    @staticmethod
    def _get_error_frame_info(error: Exception) -> Dict[str, Any]:
        """
        Extract frame information from an exception
        
        Args:
            error: The exception to analyze
            
        Returns:
            Dictionary with component, function and line_number
        """
        try:
            # Get traceback frames
            tb = traceback.extract_tb(error.__traceback__)
            if not tb:
                return {"component": None, "function": None, "line_number": None}
            
            # Get the last frame (where the error occurred)
            last_frame = tb[-1]
            
            # Extract information
            filename = last_frame.filename
            component = filename.split('/')[-1]  # Get just the file name
            function = last_frame.name
            line_number = last_frame.lineno
            
            return {
                "component": component,
                "function": function,
                "line_number": line_number
            }
        except Exception:
            return {"component": None, "function": None, "line_number": None}
    
    @staticmethod
    def _extract_request_info(args) -> Dict[str, Any]:
        """
        Extract request information from function arguments
        Attempts to find FastAPI Request objects
        
        Args:
            args: Function arguments
            
        Returns:
            Dictionary with request information
        """
        request_info = {
            "request_id": None,
            "user_fp": None,
            "ip_address": None,
            "path": None,
            "method": None
        }
        
        # Check if any argument is a FastAPI Request
        for arg in args:
            # Check if it has typical FastAPI Request attributes
            if hasattr(arg, 'url') and hasattr(arg, 'method') and hasattr(arg, 'client'):
                request_info["path"] = str(getattr(arg, 'url'))
                request_info["method"] = getattr(arg, 'method')
                
                # Try to get client IP
                client = getattr(arg, 'client', None)
                if client:
                    request_info["ip_address"] = getattr(client, 'host', None)
                
                # Check for request ID in state
                state = getattr(arg, 'state', None)
                if state:
                    request_info["request_id"] = getattr(state, 'request_id', None)
                
                # Check for user in state
                if state and hasattr(state, 'user'):
                    user = getattr(state, 'user', None)
                    if user and hasattr(user, 'fp'):
                        request_info["user_fp"] = user.fp
        
        return request_info

# Decorator for error logging
def log_errors(component: Optional[str] = None):
    """
    Decorator to automatically log errors from a function
    
    Args:
        component: Optional component name to override the automatic detection
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get function info
                function_name = func.__name__
                module_name = func.__module__
                
                # Determine component if not provided
                actual_component = component or f"{module_name}.{function_name}"
                
                # Extract request information if available
                request_info = ErrorLoggerService._extract_request_info(args)
                
                # Log the error
                asyncio.create_task(ErrorLoggerService.log_error(
                    error=e,
                    component=actual_component,
                    function=function_name,
                    request_id=request_info.get('request_id'),
                    user_fp=request_info.get('user_fp'),
                    ip_address=request_info.get('ip_address'),
                    path=request_info.get('path'),
                    method=request_info.get('method')
                ))
                
                # Re-raise the exception
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function info
                function_name = func.__name__
                module_name = func.__module__
                
                # Determine component if not provided
                actual_component = component or f"{module_name}.{function_name}"
                
                # Extract request information if available
                request_info = ErrorLoggerService._extract_request_info(args)
                
                # We can't await in a sync function, so create a background task
                # to log the error
                logger.error(
                    f"Error in {actual_component}.{function_name}: {str(e)}",
                    exc_info=True
                )
                
                # Create a detached task to log the error
                asyncio.create_task(ErrorLoggerService.log_error(
                    error=e,
                    component=actual_component,
                    function=function_name,
                    request_id=request_info.get('request_id'),
                    user_fp=request_info.get('user_fp'),
                    ip_address=request_info.get('ip_address'),
                    path=request_info.get('path'),
                    method=request_info.get('method')
                ))
                
                # Re-raise the exception
                raise
        
        # Choose the right wrapper based on whether the original function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

