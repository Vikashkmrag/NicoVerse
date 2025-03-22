import streamlit as st
import time
from datetime import datetime

from modules.utils.logger import get_logger
from modules.utils.debug import debug_print
from modules.utils.temp_message import show_temp_message
from sql_db import ThreadDB

logger = get_logger("sidebar")

class Sidebar:
    """
    Handles the sidebar UI for thread management and navigation.
    """
    
    def __init__(self):
        self.db = ThreadDB()
    
    def display(self):
        """Display the sidebar with thread management options."""
        with st.sidebar:
            st.title("NicoVerse")
            
            # New chat button
            if st.button("New Chat", use_container_width=True):
                self._create_new_thread()
            
            st.divider()
            
            # Thread management
            st.subheader("Your Threads")
            
            # Get all threads from the database - always refresh to get the latest names
            threads = self.db.get_all_threads()
            debug_print("Found {} threads in database", len(threads) if threads else 0)
            
            # Store the clicked thread ID in session state
            if "clicked_thread_id" not in st.session_state:
                st.session_state["clicked_thread_id"] = None
            
            # Check if a thread was clicked in the previous render
            if st.session_state.get("clicked_thread_id") is not None:
                thread_id = st.session_state.get("clicked_thread_id")
                # Find the thread in the list
                for thread in threads:
                    if thread["id"] == thread_id:
                        debug_print("Loading previously clicked thread from list: {}", thread["name"])
                        self._load_thread(thread)
                        # Reset the clicked thread ID
                        st.session_state["clicked_thread_id"] = None
                        break
            
            # Display threads
            if threads:
                for thread in threads:
                    # Use a stable key based only on the thread ID
                    thread_key = f"thread_{thread['id']}"
                    debug_print("Creating button for thread: {} with key: {}", thread['name'], thread_key)
                    
                    # Create a row for the thread with buttons
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Thread button
                        if st.button(
                            thread["name"] or f"Thread {thread['id']}",
                            key=f"load_{thread_key}",
                            use_container_width=True
                        ):
                            debug_print("Thread button clicked: {}", thread['name'])
                            # Store the thread ID in session state instead of loading immediately
                            st.session_state["clicked_thread_id"] = thread["id"]
                            # Force a rerun to handle the click in the next render
                            # This rerun is necessary to trigger the thread loading logic
                            st.rerun()
                    
                    with col2:
                        # Delete button
                        if st.button(
                            "🗑️",
                            key=f"delete_{thread_key}",
                            help="Delete this thread"
                        ):
                            debug_print("Delete button clicked for thread: {}", thread['name'])
                            self._delete_thread(thread["id"])
            else:
                show_temp_message("No threads yet. Start a new chat!", type="info")
            
            # Add some space
            st.divider()
            
            # App info
            with st.expander("About"):
                st.markdown("""
                **NicoVerse**
                is a sleek and sophisticated platform, 
                inspired by Nico Robin’s enigmatic intellect 
                and unyielding pursuit of knowledge.
                            
                This application allows you to:
                - Chat with various AI models
                - Upload and query documents
                - Process images with multimodal models
                
                Built with Streamlit, Ollama and  LangChain.
                """)
    
    def _create_new_thread(self):
        """Create a new thread."""
        # Generate a default thread name with timestamp
        default_name = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Initialize session state for a new thread
        st.session_state["messages"] = []
        st.session_state["documents"] = []
        st.session_state["current_thread"] = {
            "id": None,
            "name": default_name
        }
        
        # Clear any selected image
        if "selected_image" in st.session_state:
            st.session_state["selected_image"] = []
        
        # Log the action
        logger.thread_activity(
            thread_id=None,
            thread_name=default_name,
            action="create",
            message_count=0
        )
        
        # Force a rerun to update the UI
        debug_print("New thread created, UI will update on next render")
        # st.rerun()  # Commented out to prevent potential infinite loops
    
    def _load_thread(self, thread):
        """
        Load a thread from the database.
        
        Args:
            thread: The thread object to load
        """
        debug_print("Loading thread: {}", thread)
        
        try:
            # Load the thread data
            thread_data = self.db.get_thread(thread["id"])
            debug_print("Thread data loaded: {}", thread_data is not None)
            
            if thread_data is None:
                debug_print("Thread data is None for thread ID: {}", thread["id"])
                show_temp_message(f"Failed to load thread: {thread['name']}. Thread data not found.", type="error")
                return
            
            # Update session state
            if "messages" in thread_data and thread_data["messages"] is not None:
                st.session_state["messages"] = thread_data["messages"]
                debug_print("Messages loaded: {} messages", len(thread_data["messages"]))
            else:
                st.session_state["messages"] = []
                debug_print("No messages found in thread data, initialized empty messages list")
            
            # Use the name from thread_data instead of the thread list
            thread_name = thread_data.get('name', thread["name"])
            st.session_state["current_thread"] = {
                "id": thread["id"],
                "name": thread_name
            }
            debug_print("Current thread set to: {}", thread_name)
            
            # Set the selected model if available
            if thread_data.get("model"):
                st.session_state["selected_model"] = thread_data["model"]
                debug_print("Selected model set to: {}", thread_data["model"])
            else:
                debug_print("No model found in thread data")
            
            # Clear any selected image
            if "selected_image" in st.session_state:
                st.session_state["selected_image"] = []
                debug_print("Cleared selected image")
            
            # Clear documents
            st.session_state["documents"] = []
            debug_print("Cleared documents")
            
            # Log the action
            logger.info(
                f"Loaded thread: {thread_name} (ID: {thread['id']}) with {len(thread_data.get('messages', []))} messages",
                event_type="thread_activity",
                thread_id=thread["id"],
                thread_name=thread_name,
                action="load",
                message_count=len(thread_data.get("messages", []))
            )
            
            # Force a rerun to update the UI
            debug_print("Thread loaded successfully")
            # st.rerun()  # Removed to prevent infinite loop
            
        except Exception as e:
            debug_print("Error loading thread: {}", str(e))
            show_temp_message(f"Error loading thread: {str(e)}", type="error")
            # Reset clicked thread ID to avoid infinite loop
            if "clicked_thread_id" in st.session_state:
                st.session_state["clicked_thread_id"] = None
    
    def _delete_thread(self, thread_id):
        """
        Delete a thread from the database.
        
        Args:
            thread_id: The ID of the thread to delete
        """
        debug_print("Deleting thread with ID: {}", thread_id)
        
        try:
            # Get the thread name for logging
            thread = self.db.get_thread(thread_id)
            thread_name = thread["name"] if thread else f"Thread {thread_id}"
            message_count = len(thread.get("messages", [])) if thread else 0
            
            # Delete the thread
            success = self.db.delete_thread(thread_id)
            
            if success:
                debug_print("Thread deleted successfully: {}", thread_name)
                
                # If the current thread is the one being deleted, create a new thread
                if st.session_state.get("current_thread", {}).get("id") == thread_id:
                    debug_print("Deleted current thread, creating new thread")
                    self._create_new_thread()
                else:
                    # Force a rerun to update the UI
                    debug_print("Thread deleted successfully, UI will update on next render")
                    # st.rerun()  # Commented out to prevent potential infinite loops
                
                # Log the action
                logger.info(
                    f"Deleted thread: {thread_name} (ID: {thread_id}) with {message_count} messages",
                    event_type="thread_activity",
                    thread_id=thread_id,
                    thread_name=thread_name,
                    action="delete",
                    message_count=message_count
                )
            else:
                debug_print("Failed to delete thread: {}", thread_id)
                show_temp_message(f"Failed to delete thread: {thread_name}", type="error")
                
        except Exception as e:
            debug_print("Error deleting thread: {}", str(e))
            show_temp_message(f"Error deleting thread: {str(e)}", type="error") 