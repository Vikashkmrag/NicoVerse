import streamlit as st
import threading
import time

def show_temp_message(message, type="info", duration=3):
    """
    Display a temporary message that disappears after a specified duration.
    
    Args:
        message (str): The message to display
        type (str): The type of message (info, success, warning, error)
        duration (int): How long to display the message in seconds
    """
    # Create an empty container that we can clear later
    message_container = st.empty()
    
    # Display the message in the container
    with message_container:
        if type == "info":
            st.info(message)
        elif type == "success":
            st.success(message)
        elif type == "warning":
            st.warning(message)
        elif type == "error":
            st.error(message)
    
    # Use a separate thread to clear the message after the duration
    def clear_message():
        import time
        time.sleep(duration)
        try:
            message_container.empty()
        except Exception:
            # Ignore errors when trying to clear messages after session has ended
            pass
    
    # Start the thread to clear the message
    threading.Thread(target=clear_message, daemon=True).start() 