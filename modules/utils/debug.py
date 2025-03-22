import inspect
import os
import sys
from modules.utils.config import *
import config

def debug_print(message, *args, **kwargs):
    """
    Print debug messages with file name and line number information.
    
    Args:
        message: The debug message to print
        *args: Additional arguments to format the message
        **kwargs: Additional keyword arguments for formatting
    """
    if not config.DEBUG_MODE:
        return
    
    # Get caller frame information
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    line_number = frame.f_lineno
    
    # Format the message
    if args or kwargs:
        formatted_message = message.format(*args, **kwargs)
    else:
        formatted_message = message
    
    # Print the debug message with file and line information
    print(config.DEBUG_FORMAT.format(file=filename, line=line_number, message=formatted_message)) 