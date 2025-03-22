"""
NicoVerse

NicoVerse is a sleek and sophisticated platform, inspired by Nico Robin's enigmatic intellect and unyielding pursuit of knowledge.
"""

import os
import streamlit as st
from datetime import datetime

# Import modules
import config
from modules.ui.sidebar import Sidebar
from modules.ui.chat_interface import ChatInterface
from modules.models.query_handler import QueryHandler
from modules.utils.logger import get_logger
from modules.utils.debug import debug_print

# Initialize logger
logger = get_logger("main")

# Page configuration
st.set_page_config(
    page_title="NicoVerse",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize the session state with default values."""
    debug_print("Initializing session state")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        debug_print("Initialized messages")
    
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
        debug_print("Initialized documents")
    
    if "current_thread" not in st.session_state:
        st.session_state["current_thread"] = {
            "id": None,
            "name": f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
        debug_print("Initialized current_thread")
    
    if "model_sizes" not in st.session_state:
        st.session_state["model_sizes"] = {}
        debug_print("Initialized model_sizes")
    
    if "embedding_models_support" not in st.session_state:
        st.session_state["embedding_models_support"] = {}
        debug_print("Initialized embedding_models_support")
    
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = config.DEFAULT_MODEL
        debug_print(f"Initialized selected_model: {config.DEFAULT_MODEL}")
    
    # Initialize selected_image if not exists
    if "selected_image" not in st.session_state:
        st.session_state["selected_image"] = []
        debug_print("Initialized selected_image as empty list")
    
    # Initialize model_changed_for_image if not exists
    if "model_changed_for_image" not in st.session_state:
        st.session_state["model_changed_for_image"] = False
        debug_print("Initialized model_changed_for_image")
    
    # Initialize current_embedding_model if not exists
    if "current_embedding_model" not in st.session_state:
        st.session_state["current_embedding_model"] = config.FALLBACK_EMBEDDING_MODEL
        debug_print(f"Initialized current_embedding_model: {config.FALLBACK_EMBEDDING_MODEL}")

# Initialize components
def init_components():
    """Initialize the application components."""
    debug_print("Initializing application components")
    sidebar = Sidebar()
    chat_interface = ChatInterface()
    query_handler = QueryHandler()
    debug_print("Components initialized")
    
    return sidebar, chat_interface, query_handler

# Main function
def main():
    """Main application function."""
    debug_print("Starting main application function")
    # Initialize session state
    init_session_state()
    
    # Initialize components
    sidebar, chat_interface, query_handler = init_components()
    
    # Add custom CSS
    chat_interface.add_custom_css()
    
    # Display sidebar
    sidebar.display()
    
    # Display thread name editor
    chat_interface.display_thread_name_editor()
    
    # Display chat messages
    chat_interface.display_chat_messages()
    
    # Create a process_query function that uses the initialized query_handler
    def process_query(user_message, user_msg, model):
        """Process a user query using the query handler."""
        debug_print(f"process_query called with model: {model}")
        debug_print(f"user_msg: {user_msg}")
        return query_handler.process_query(user_message, user_msg, model)
    
    # Display chat input
    chat_interface.display_chat_input(process_query)
    
    # Log application start
    logger.info(
        "Application loaded",
        event_type="app_load",
        thread_id=st.session_state.get("current_thread", {}).get("id"),
        thread_name=st.session_state.get("current_thread", {}).get("name"),
        message_count=len(st.session_state.get("messages", []))
    )
    debug_print("Application loaded successfully")

# Run the application
if __name__ == "__main__":
    main() 