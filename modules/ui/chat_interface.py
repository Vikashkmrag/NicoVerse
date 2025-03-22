import streamlit as st
import os
import config
from modules.utils.logger import get_logger
from modules.models.model_manager import ModelManager
from sql_db import ThreadDB
from modules.utils.debug import debug_print
from modules.utils.temp_message import show_temp_message
import time
import hashlib
import threading

logger = get_logger("chat_interface")

class ChatInterface:
    """
    Handles the Streamlit UI for the chat functionality.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.db = ThreadDB()
        # Import processors here to avoid circular imports
        from modules.data.document_processor import DocumentProcessor
        from modules.data.image_processor import ImageProcessor
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
    
    def display_chat_messages(self):
        """Display all chat messages in the session state."""
        chat_container = st.container()
        with chat_container:
            # Display all messages
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                        # Display image if present
                        if "image" in message and message["image"]:
                            try:
                                image_path = message["image"]
                                st.image(image_path, caption="Uploaded Image")
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        # Display sources if available
                        if "sources" in message and message["sources"]:
                            with st.expander("Sources"):
                                st.write(", ".join(message["sources"]))
    
    def display_thread_name_editor(self):
        """Display the thread name editor."""
        # Get the current thread name
        current_thread_name = st.session_state.get("current_thread", {}).get("name", "")
        
        # Create a text input for the thread name
        new_thread_name = st.text_input("Thread Name", value=current_thread_name)
        
        # Update the thread name if it changed
        if new_thread_name != current_thread_name:
            st.session_state["current_thread"]["name"] = new_thread_name
            
            # Update the thread in the database if it exists
            if st.session_state.get("current_thread", {}).get("id"):
                self.db.update_thread(
                    st.session_state["current_thread"]["id"],
                    new_thread_name,
                    st.session_state["messages"],
                    model=st.session_state.get("selected_model"),
                    documents=[]  # Don't save documents with threads for storage optimization
                )
    
    def display_document_list(self):
        """Display the list of documents and allow users to remove them."""
        documents = st.session_state.get("documents", [])
        
        if documents:
            with st.expander(f"📄 Documents ({len(documents)})", expanded=False):
                # Create a more compact document list with smaller delete buttons
                for i, doc in enumerate(documents):
                    doc_name = os.path.basename(doc)
                    col1, col2 = st.columns([9, 1])
                    with col1:
                        st.markdown(f"**{i+1}.** {doc_name}")
                    with col2:
                        if st.button("🗑️", key=f"remove_doc_{i}", help="Remove this document"):
                            # Remove the document from the list
                            documents.pop(i)
                            # Update the session state
                            st.session_state["documents"] = documents
                            debug_print("Removed document: {}", doc)
                            # Show a temporary message
                            show_temp_message(f"Removed: {doc_name}", "info", 1.0)
                            # Force a rerun to update the UI
                            st.rerun()
    
    def display_chat_input(self, query_handler):
        """
        Display the chat input controls.
        
        Args:
            query_handler: Function to handle user queries
        """
        st.divider()
        
        # Create a sleek chat interface
        with st.container():
            # Top row with model selector and file uploader
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col3:
                # Get available models
                available_models = self.model_manager.get_available_models()
                
                # Get the current model from session state
                current_model = st.session_state.get("selected_model", available_models[0] if available_models else config.DEFAULT_MODEL)
                
                # Make sure the current model is in the available models list
                if current_model not in available_models and available_models:
                    current_model = available_models[0]
                    st.session_state["selected_model"] = current_model
                
                # Select model - use a unique key that includes the current model to force UI refresh
                selected_model = st.selectbox(
                    "Model",
                    options=available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0,
                    key=f"model_selector_{current_model}"
                )
                
                # Update selected model in session state
                if selected_model != current_model:
                    previous_model = current_model
                    st.session_state["selected_model"] = selected_model
                    # Reset model context when switching models
                    self.model_manager.reset_model_context(selected_model)
                    # Force a rerun to update the UI
                    st.rerun()
                
                # Check if there's an incompatible attachment
                if len(st.session_state.get("selected_image", [])) > 0:
                    # If switching from a multimodal model to a non-multimodal model
                    if self.model_manager.check_model_multimodal_support(selected_model): # and not self.model_manager.check_model_multimodal_support(selected_model):
                        # Clear the selected image if the new model doesn't support images
                        show_temp_message("Selected model doesn't support images. Your image attachment has been cleared.", "warning", 1.5)
                        st.session_state["selected_image"] = []
                        st.rerun()
                
                # If we have documents and switched to a multimodal model, suggest a better model
                documents = st.session_state.get("documents", [])
                if documents and len(documents) > 0:
                    # Find a better model for document processing
                    embedding_model = self.model_manager.get_best_embedding_model(selected_model)
                    if embedding_model and embedding_model != selected_model:
                        show_temp_message(f"The selected model '{selected_model}' does not support document processing. Using '{embedding_model}' instead.", "warning", 1.5)
            
            with col2:
                # Document list as a compact horizontal list
                documents = st.session_state.get("documents", [])
                if documents:
                    # Create a header for documents
                    st.markdown("**Documents:**")
                    
                    # Create a container for the document items
                    st.markdown('<div class="document-list">', unsafe_allow_html=True)
                    
                    # Generate HTML for document items
                    doc_items_html = ""
                    for doc in documents:
                        doc_name = os.path.basename(doc)
                        doc_items_html += f'<div class="document-item">📄 {doc_name}</div>'
                    
                    # Display the document items
                    st.markdown(doc_items_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add a small clear all button
                    if st.button("🗑️ Clear All", key="clear_all_docs", help="Remove all documents"):
                        st.session_state["documents"] = []
                        show_temp_message("All documents removed", "info", 1.0)
                        st.rerun()
            
            with col1:
                # Simple attachment button
                st.markdown("""
                <style>
                /* Custom file uploader that looks like a button */
                .attachment-button {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                </style>
                <div class="attachment-button">
                """, unsafe_allow_html=True)
                
                # File attachment with a more compact design
                uploaded_file = st.file_uploader(
                    "Upload File",
                    type=["pdf", "txt"] + list(config.SUPPORTED_IMAGE_TYPES),
                    key="file_uploader",
                    label_visibility="collapsed"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Process the uploaded file automatically
                if uploaded_file:
                    self._process_uploaded_file(uploaded_file, available_models)
            
            # Chat input with embedded send button
            with st.form(key="chat_form", clear_on_submit=True):
                # Chat input
                user_message = st.text_area(
                    "Your message", 
                    height=80, 
                    placeholder="Type your message here...", 
                    label_visibility="collapsed", 
                    key="user_input"
                )
                
                # Submit button
                col1, col2 = st.columns([6, 1])
                with col2:
                    submit_button = st.form_submit_button(
                        "➤", 
                        type="primary",
                        help="Send message"
                    )
                
                if submit_button and user_message:
                    self._handle_send_button(user_message, query_handler)
    
    def _process_uploaded_file(self, uploaded_file, available_models):
        """
        Process an uploaded file based on its type.
        
        Args:
            uploaded_file: The uploaded file object
            available_models: List of available models
        """
        debug_print("Processing uploaded file: {}", uploaded_file.name)
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        current_model = st.session_state.get("selected_model")
        debug_print("File extension: {}, Current model: {}", file_extension, current_model)
        
        # Check if it's an image
        if file_extension.lower() in [f".{ext}" for ext in config.SUPPORTED_IMAGE_TYPES]:
            debug_print("File is an image. Supported image types: {}", config.SUPPORTED_IMAGE_TYPES)
            # Create images directory if it doesn't exist
            os.makedirs(config.IMAGES_DIR, exist_ok=True)
            
            # First, save the image regardless of model support
            try:
                # Save the image
                debug_print("Saving image to disk")
                image_path = self.image_processor.save_uploaded_image(uploaded_file)
                debug_print("Image saved at: {}", image_path)
                st.session_state["selected_image"] = image_path
                debug_print("Updated session state with selected_image: {}", image_path)
                
                # Display the image
                st.image(uploaded_file, caption=uploaded_file.name, width=150)
            except Exception as e:
                debug_print("Error processing image: {}", str(e))
                show_temp_message(f"Error processing image: {str(e)}", "error", 2.0)
                return
            
            # Check if current model supports images
            debug_print("Checking if current model {} supports images", current_model)
            multimodal_support = self.model_manager.check_model_multimodal_support(current_model)
            debug_print("Multimodal support for {}: {}", current_model, multimodal_support)
            
            if not multimodal_support:
                # Find multimodal models
                multimodal_models = [model for model in available_models if self.model_manager.check_model_multimodal_support(model)]
                debug_print("Available multimodal models: {}", multimodal_models)
                
                if multimodal_models:
                    # Automatically switch to a multimodal model
                    new_model = multimodal_models[0]
                    debug_print("Automatically switching to {} for image support", new_model)
                    
                    # Update session state
                    st.session_state["selected_model"] = new_model
                    st.session_state["model_changed_for_image"] = True
                    model = new_model
                    
                    # Display warning message
                    show_temp_message(f"Switched to {new_model} for image support", "info", 1.5)
                    
                    # Force a rerun to update the UI with the new model
                    time.sleep(0.5)  # Small delay to ensure the session state is updated
                    st.rerun()
                    return  # Exit the function to allow the rerun to take effect
                else:
                    # Warn the user that no multimodal models are available
                    show_temp_message(f"No multimodal models available. Your image has been attached, but the current model ({current_model}) cannot process it. Please install a multimodal model like gemma:2b-it or llava:7b.", "warning", 2.0)
                    debug_print("No multimodal models available. Image attached but cannot be processed.")
            
            debug_print("Selected image: {}", st.session_state.get('selected_image'))
        
        # Check if it's a document
        if file_extension.lower() in [f".{ext}" for ext in config.SUPPORTED_TEXT_TYPES]:
            # Create documents directory if it doesn't exist
            os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
            
            try:
                # Clear any previously selected image since we're now using a document
                # if "selected_image" in st.session_state and st.session_state["selected_image"]:
                #     st.info("Cleared previous image attachment as you're now using a document.")
                #     st.session_state["selected_image"] = None
                
                # Save the file
                file_path = self.document_processor.save_uploaded_document(uploaded_file)
                
                # Process the document
                docs = self.document_processor.load_documents(directory=None, uploaded_files=[uploaded_file])
                
                # Get the best embedding model for this document type
                current_model = st.session_state.get("selected_model")
                embedding_model = st.session_state.get("current_embedding_model")
                
                # If the current model doesn't support embeddings, suggest switching
                if embedding_model != current_model and not st.session_state.embedding_models_support.get(current_model, False):
                    show_temp_message(f"Using {embedding_model} for document embeddings as the current model doesn't support them.", "info", 1.5)
                
                # Create embeddings
                vectorstore, doc_names = self.document_processor.create_embeddings(
                    docs,
                    embedding_model=embedding_model,
                    user_selected_model=current_model
                )
                
                # Update session state with document names
                if doc_names:
                    st.session_state["documents"] = list(set(st.session_state.get("documents", []) + doc_names))
                    
                    # Update thread in database if it exists
                    # if st.session_state.get("current_thread", {}).get("id"):
                    #     self.db.update_thread(
                    #         st.session_state["current_thread"]["id"],
                    #         st.session_state["current_thread"]["name"],
                    #         st.session_state["messages"],
                    #         model=st.session_state.get("selected_model"),
                    #         documents=[]  # Don't save documents with threads for storage optimization
                    #     )
                    
                    show_temp_message(f"Added: {uploaded_file.name}", "success", 1.0)
            except Exception as e:
                debug_print("Error processing document: {}", str(e))
                show_temp_message(f"Error processing document: {str(e)}", "error", 2.0)
    
    def _handle_send_button(self, user_message, query_handler):
        """
        Handle the send button click.
        
        Args:
            user_message: The user's message
            query_handler: Function to handle user queries
        """
        debug_print("Handle send button clicked with message: {}...", user_message[:30])
        
        # Get selected model
        model = st.session_state.get("selected_model", config.DEFAULT_MODEL)
        debug_print("Selected model: {}", model)
        send_request = True
        
        # Check if we have an image but the model doesn't support it
        if st.session_state.get("selected_image"):
            debug_print("Image detected in session state: {}", st.session_state.get("selected_image"))
            
            if not self.model_manager.check_model_multimodal_support(model):
                debug_print("Model {} does not support images", model)
                
                # Find multimodal models
                available_models = self.model_manager.get_available_models()
                multimodal_models = [m for m in available_models if self.model_manager.check_model_multimodal_support(m)]
                
                if multimodal_models:
                    # Automatically switch to a multimodal model
                    new_model = multimodal_models[0]
                    st.session_state["selected_model"] = new_model
                    model = new_model
                    show_temp_message(f"Switched to {new_model} for image support", "warning", 1.5)
                    debug_print("Automatically switched to {} for image support", new_model)
                else:
                    show_temp_message(f"The current model ({model}) cannot process images. Please install a multimodal model like gemma:2b-it or llava:7b.", "error", 2.0)
                    debug_print("No multimodal models available. Cannot process image.")
                    send_request = False
            else:
                debug_print("Model {} supports images", model)
        else:
            debug_print("No image in session state")

        # Check if we have documents but the model might not be optimal for text
        documents = st.session_state.get("documents", [])
        if documents and len(documents) > 0:
            debug_print("Documents detected: {}", len(documents))
            # Find a better model for document processing
            send_request = True
            embedding_model = st.session_state.get("current_embedding_model", config.FALLBACK_EMBEDDING_MODEL)
            debug_print("Using embedding model: {}", embedding_model)
        
        if send_request:
            debug_print("Sending request to query handler")
            # Add user message to chat history
            user_msg = {"role": "user", "content": user_message}
            
            # Add image to message if selected
            if st.session_state.get("selected_image"):
                debug_print("Adding image to message: {}", st.session_state.get("selected_image"))
                user_msg["image"] = st.session_state["selected_image"]
                # Clear the selected image after sending
                # st.session_state["selected_image"] = []
            
            st.session_state.messages.append(user_msg)
            
            # Process message
            try:
                debug_print("Calling query_handler with message and model: {}", model)
                # Call the query handler to process the message
                response, sources = query_handler(user_message, user_msg, model)
                debug_print("Received response from query_handler: {}...", response[:30])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
                # Save thread to database
                if not st.session_state["current_thread"]["id"]:
                    # Create new thread
                    thread_id = self.db.save_thread(
                        st.session_state["current_thread"]["name"],
                        st.session_state["messages"],
                        model=model,
                        # documents=[]  # Don't save documents with threads for storage optimization
                    )
                    st.session_state["current_thread"]["id"] = thread_id
                else:
                    # Update existing thread
                    self.db.update_thread(
                        st.session_state["current_thread"]["id"],
                        st.session_state["current_thread"]["name"],
                        st.session_state["messages"],
                        model=model,
                        # documents=[]  # Don't save documents with threads for storage optimization
                    )
                
                # Log message
                logger.info(
                    f"Chat message from user in thread {st.session_state['current_thread']['name']}",
                    event_type="chat_message",
                    thread_id=st.session_state["current_thread"]["id"],
                    thread_name=st.session_state["current_thread"]["name"],
                    role="user",
                    content=user_message,
                    model=model
                )
                
                # Rerun to update UI
                st.rerun()
            except Exception as e:
                debug_print("Error processing message: {}", str(e))
                show_temp_message(f"Error: {str(e)}", "error", 2.0)
                logger.error(f"Error processing message: {str(e)}")
    
    def add_custom_css(self):
        """Add custom CSS styling to the UI."""
        st.markdown("""
        <style>
        /* General UI improvements */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Dark mode text fixes */
        .stMarkdown, .stText, .stTextArea textarea, .stSelectbox, .stFileUploader {
            color: var(--text-color) !important;
        }
        
        /* Chat message styling for dark mode compatibility */
        [data-testid="stChatMessage"] {
            background-color: var(--background-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        [data-testid="stChatMessage"]:hover {
            transform: translateY(-1px);
        }
        
        /* Ensure chat message content is visible */
        [data-testid="stChatMessage"] p {
            color: var(--text-color) !important;
        }
        
        /* Force text color in chat messages */
        .element-container [data-testid="stChatMessage"] div.stMarkdown {
            color: var(--text-color) !important;
        }
        
        .element-container [data-testid="stChatMessage"] div.stMarkdown p {
            color: var(--text-color) !important;
        }
        
        /* Ensure expander text is visible */
        .streamlit-expanderContent {
            color: var(--text-color) !important;
        }
        
        /* Button styling */
        .stButton button {
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 10px;
            font-size: 16px;
            transition: border 0.2s ease;
            background-color: var(--input-bg);
        }
        .stTextArea textarea:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
        }
        
        /* Select box styling */
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 8px;
            background-color: var(--input-bg);
        }
        
        /* File uploader styling - completely hide everything except the button */
        div.stFileUploader {
            width: 40px !important;
            height: 40px !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Hide all file uploader text and elements */
        .stFileUploader > div:first-child,
        .stFileUploader > div > div:first-child,
        .stFileUploader > div > div > div:first-child,
        .stFileUploader > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div:first-child,
        .stFileUploader > div > div > div > div > div > div > div > div > div > div > div > div > div > div:first-child {
            display: none !important;
        }
        
        /* Hide all text elements in the file uploader */
        .stFileUploader p, 
        .stFileUploader span, 
        .stFileUploader label, 
        .stFileUploader div[data-testid="stFileUploadDropzoneInput"],
        .stFileUploader div[data-testid="stMarkdownContainer"],
        .stFileUploader small,
        .stFileUploader div[data-testid="stText"] {
            display: none !important;
        }
        
        /* Make the file uploader button more compact */
        div.stFileUploader button {
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            min-width: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 20px !important;
            margin: 0 !important;
            background-color: transparent !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
        }
        
        /* Replace the button text with an icon */
        div.stFileUploader button p,
        div.stFileUploader button span {
            display: none !important;
        }
        
        div.stFileUploader button::before {
            content: "📎" !important;
            font-size: 20px !important;
        }
        
        /* Submit button styling */
        button[kind="primaryFormSubmit"] {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            min-width: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 20px !important;
            margin: 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            transition: all 0.2s ease !important;
        }
        
        button[kind="primaryFormSubmit"]:hover {
            transform: scale(1.1) !important;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3) !important;
        }
        
        /* Chat container styling */
        div.chat-container {
            margin-bottom: 20px;
        }
        
        /* Document list styling */
        .document-list {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .document-item {
            background-color: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.3);
            color: var(--text-color);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 12px;
            display: inline-flex;
            align-items: center;
        }
        
        /* Style for notification messages */
        div[data-baseweb="notification"] {
            margin-bottom: 10px !important;
        }
        
        /* Style for info messages */
        div[data-baseweb="notification"][kind="info"] {
            background-color: rgba(3, 169, 244, 0.1) !important;
            border-color: #03A9F4 !important;
        }
        
        /* Style for success messages */
        div[data-baseweb="notification"][kind="positive"] {
            background-color: rgba(76, 175, 80, 0.1) !important;
            border-color: #4CAF50 !important;
        }
        
        /* Style for warning messages */
        div[data-baseweb="notification"][kind="warning"] {
            background-color: rgba(255, 152, 0, 0.1) !important;
            border-color: #FF9800 !important;
        }
        
        /* Style for error messages */
        div[data-baseweb="notification"][kind="negative"] {
            background-color: rgba(244, 67, 54, 0.1) !important;
            border-color: #F44336 !important;
        }
        
        /* CSS variables for theme compatibility */
        :root {
            --text-color: #262730;
            --background-color: #ffffff;
            --border-color: #dddddd;
            --input-bg: #f9f9f9;
            --hover-color: #e0e0e0;
        }
        
        /* Dark theme variables */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-color: #fafafa;
                --background-color: #1e1e1e;
                --border-color: #444444;
                --input-bg: #2d2d2d;
                --hover-color: #3d3d3d;
            }
        }
        
        /* Force dark mode if Streamlit theme is dark */
        .dark {
            --text-color: #fafafa !important;
            --background-color: #1e1e1e !important;
            --border-color: #444444 !important;
            --input-bg: #2d2d2d !important;
            --hover-color: #3d3d3d !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Add theme detection script
        st.components.v1.html("""
        <script>
            if (document.body.classList.contains('dark')) {
                document.documentElement.classList.add('dark');
            }
            
            // Watch for theme changes
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.attributeName === 'class') {
                        if (document.body.classList.contains('dark')) {
                            document.documentElement.classList.add('dark');
                        } else {
                            document.documentElement.classList.remove('dark');
                        }
                    }
                });
            });
            
            observer.observe(document.body, { attributes: true });
            
            // Additional script to hide file uploader text
            document.addEventListener('DOMContentLoaded', function() {
                // Hide all text elements in the file uploader
                const hideFileUploaderText = () => {
                    const fileUploaders = document.querySelectorAll('.stFileUploader');
                    fileUploaders.forEach(uploader => {
                        // Hide all child elements except the button
                        Array.from(uploader.querySelectorAll('*')).forEach(el => {
                            if (el.tagName !== 'BUTTON') {
                                el.style.display = 'none';
                            } else {
                                // For buttons, hide any text inside
                                Array.from(el.querySelectorAll('*')).forEach(child => {
                                    child.style.display = 'none';
                                });
                            }
                        });
                    });
                };
                
                // Run initially and set up a mutation observer to catch dynamically added elements
                hideFileUploaderText();
                const observer = new MutationObserver(mutations => {
                    hideFileUploaderText();
                });
                observer.observe(document.body, { childList: true, subtree: true });
                
                // Run it again after a short delay to catch any elements that might be added later
                setTimeout(hideFileUploaderText, 500);
                setTimeout(hideFileUploaderText, 1000);
                setTimeout(hideFileUploaderText, 2000);
            });
        </script>
        """, height=0) 