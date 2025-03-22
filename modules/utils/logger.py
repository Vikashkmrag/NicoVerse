import logging
import datetime
import json
import os
import traceback

class AppLogger:
    """
    Centralized logging class for the document retrieval application.
    Handles application logs, JSON logs for analytics, and model usage tracking.
    """
    
    def __init__(self, name="app"):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler if not already added
        if not self.logger.handlers:
            file_handler = logging.FileHandler(f'logs/{name}.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Set up JSON logger for analytics
        self.json_logger = logging.getLogger('json_logger')
        self.json_logger.setLevel(logging.INFO)
        
        # Add JSON file handler if not already added
        if not self.json_logger.handlers:
            json_handler = logging.FileHandler('logs/app.json.log')
            self.json_logger.addHandler(json_handler)
    
    def info(self, message, event_type=None, **kwargs):
        """Log an info message with optional event type and additional data."""
        self.logger.info(message)
        
        if event_type:
            self._log_json(message, "info", event_type, **kwargs)
    
    def warning(self, message, event_type=None, **kwargs):
        """Log a warning message with optional event type and additional data."""
        self.logger.warning(message)
        
        if event_type:
            self._log_json(message, "warning", event_type, **kwargs)
    
    def error(self, message, event_type=None, exception=None, **kwargs):
        """Log an error message with optional event type, exception details, and additional data."""
        self.logger.error(message)
        
        if event_type:
            extra_data = kwargs.copy()
            if exception:
                extra_data["exception"] = str(exception)
                extra_data["traceback"] = traceback.format_exc()
            
            self._log_json(message, "error", event_type, **extra_data)
    
    def _log_json(self, message, level, event_type, **kwargs):
        """Internal method to log structured JSON data."""
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message,
            "event_type": event_type
        }
        
        # Add any additional data
        if kwargs:
            log_data["details"] = kwargs
        
        self.json_logger.info(json.dumps(log_data))
    
    def model_usage(self, model_name, query_length, response_length, duration_ms, success=True, error=None):
        """Log model usage statistics."""
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": "model_usage",
            "model_name": model_name,
            "query_length": query_length,
            "response_length": response_length,
            "duration_ms": duration_ms,
            "success": success
        }
        
        if error:
            log_data["error"] = error
        
        self.json_logger.info(json.dumps(log_data))
    
    def thread_activity(self, thread_id, thread_name, action, message_count, model=None):
        """Log thread-related activities."""
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": "thread_activity",
            "thread_id": thread_id,
            "thread_name": thread_name,
            "action": action,
            "message_count": message_count
        }
        
        if model:
            log_data["model"] = model
        
        self.json_logger.info(json.dumps(log_data))

# Singleton instance for app-wide use
_logger_instance = None

def get_logger(name="app"):
    """Get or create a logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AppLogger(name)
    return _logger_instance 