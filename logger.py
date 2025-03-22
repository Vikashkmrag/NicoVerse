"""
Logging module for Document Retrieval Application

This module provides structured logging capabilities that can be used
for dashboard visualization in tools like Kibana, Grafana, etc.
"""

import os
import json
import logging
import time
from datetime import datetime
import socket
import uuid
import traceback
from logging.handlers import RotatingFileHandler

# Constants
LOG_DIR = './logs'
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
JSON_LOG_FILE = os.path.join(LOG_DIR, 'app.json.log')
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure standard logger
standard_logger = logging.getLogger('standard_logger')
standard_logger.setLevel(logging.INFO)

# File handler for standard logs
file_handler = RotatingFileHandler(
    LOG_FILE, 
    maxBytes=MAX_LOG_SIZE, 
    backupCount=BACKUP_COUNT
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
standard_logger.addHandler(file_handler)

# Configure JSON logger for structured logging
json_logger = logging.getLogger('json_logger')
json_logger.setLevel(logging.INFO)

# File handler for JSON logs
json_file_handler = RotatingFileHandler(
    JSON_LOG_FILE, 
    maxBytes=MAX_LOG_SIZE, 
    backupCount=BACKUP_COUNT
)
json_logger.addHandler(json_file_handler)

# Get hostname for logging
HOSTNAME = socket.gethostname()

class AppLogger:
    """
    Logger class for the Document Retrieval Application.
    Provides structured logging suitable for dashboard visualization.
    """
    
    def __init__(self, component=None):
        """
        Initialize logger with optional component name.
        
        Args:
            component (str, optional): Component name for categorizing logs
        """
        self.component = component or 'app'
        self.session_id = str(uuid.uuid4())
    
    def _log_structured(self, level, message, event_type=None, **kwargs):
        """
        Log a structured message in JSON format.
        
        Args:
            level (str): Log level (info, warning, error, etc.)
            message (str): Log message
            event_type (str, optional): Type of event being logged
            **kwargs: Additional fields to include in the log
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'component': self.component,
            'session_id': self.session_id,
            'hostname': HOSTNAME,
            'message': message,
            'event_type': event_type or 'general',
        }
        
        # Add additional fields
        log_data.update(kwargs)
        
        # Log as JSON
        json_logger.info(json.dumps(log_data))
        
        # Also log to standard logger
        log_method = getattr(standard_logger, level.lower(), standard_logger.info)
        log_method(f"[{self.component}] {message}")
    
    def info(self, message, event_type=None, **kwargs):
        """Log an info message"""
        self._log_structured('info', message, event_type, **kwargs)
    
    def warning(self, message, event_type=None, **kwargs):
        """Log a warning message"""
        self._log_structured('warning', message, event_type, **kwargs)
    
    def error(self, message, event_type=None, exception=None, **kwargs):
        """
        Log an error message with optional exception details
        
        Args:
            message (str): Error message
            event_type (str, optional): Type of error event
            exception (Exception, optional): Exception object to log
            **kwargs: Additional fields to include in the log
        """
        error_details = {}
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        self._log_structured('error', message, event_type, error_details=error_details, **kwargs)
    
    def critical(self, message, event_type=None, exception=None, **kwargs):
        """Log a critical message"""
        error_details = {}
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        self._log_structured('critical', message, event_type, error_details=error_details, **kwargs)
    
    def model_usage(self, model_name, query_length, response_length, duration_ms, success=True, error=None):
        """
        Log model usage statistics
        
        Args:
            model_name (str): Name of the model used
            query_length (int): Length of the query in characters
            response_length (int): Length of the response in characters
            duration_ms (float): Duration of the model call in milliseconds
            success (bool): Whether the model call was successful
            error (str, optional): Error message if the call failed
        """
        self._log_structured(
            'info',
            f"Model usage: {model_name}",
            event_type='model_usage',
            model_name=model_name,
            query_length=query_length,
            response_length=response_length,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def document_processing(self, doc_count, doc_sources, chunk_count, model_name, duration_ms, success=True, error=None):
        """
        Log document processing statistics
        
        Args:
            doc_count (int): Number of documents processed
            doc_sources (list): List of document sources
            chunk_count (int): Number of chunks created
            model_name (str): Name of the model used for embeddings
            duration_ms (float): Duration of the processing in milliseconds
            success (bool): Whether the processing was successful
            error (str, optional): Error message if the processing failed
        """
        self._log_structured(
            'info',
            f"Processed {doc_count} documents",
            event_type='document_processing',
            doc_count=doc_count,
            doc_sources=doc_sources,
            chunk_count=chunk_count,
            model_name=model_name,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def user_activity(self, user_id, action, thread_id=None, details=None):
        """
        Log user activity
        
        Args:
            user_id (str): User identifier
            action (str): Action performed by the user
            thread_id (str, optional): Thread ID if applicable
            details (dict, optional): Additional details about the action
        """
        self._log_structured(
            'info',
            f"User activity: {action}",
            event_type='user_activity',
            user_id=user_id,
            action=action,
            thread_id=thread_id,
            details=details or {}
        )
    
    def thread_activity(self, thread_id, thread_name, action, message_count=None, model=None, documents=None):
        """
        Log thread activity
        
        Args:
            thread_id (str): Thread identifier
            thread_name (str): Thread name
            action (str): Action performed on the thread
            message_count (int, optional): Number of messages in the thread
            model (str, optional): Model used in the thread
            documents (list, optional): Documents used in the thread
        """
        self._log_structured(
            'info',
            f"Thread activity: {action} - {thread_name}",
            event_type='thread_activity',
            thread_id=thread_id,
            thread_name=thread_name,
            action=action,
            message_count=message_count,
            model=model,
            documents=documents
        )
    
    def system_info(self, info_type, details):
        """
        Log system information
        
        Args:
            info_type (str): Type of system information
            details (dict): Details about the system
        """
        self._log_structured(
            'info',
            f"System info: {info_type}",
            event_type='system_info',
            info_type=info_type,
            details=details
        )

# Create default logger instance
logger = AppLogger()

def get_logger(component=None):
    """
    Get a logger instance for a specific component
    
    Args:
        component (str, optional): Component name
    
    Returns:
        AppLogger: Logger instance
    """
    return AppLogger(component) 