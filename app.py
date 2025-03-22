import os
import streamlit as st
import json
import subprocess
import glob
import time
import datetime
import base64
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from sql_db import ThreadDB, backup_database, restore_database
from PyPDF2 import PdfReader
from langchain.schema import Document
from logger import get_logger
import traceback
import shutil
import config  # Import the config module directly
import logging
import uuid
import requests
from PIL import Image
import io
from modules.ui.chat_interface import ChatInterface
from modules.utils.temp_message import show_temp_message

# Initialize logger using the custom AppLogger
logger = get_logger("app")

# Configure JSON logger for analytics
json_logger = logging.getLogger('json_logger')
json_logger.setLevel(logging.INFO)
json_handler = logging.FileHandler('logs/app.json.log')
json_logger.addHandler(json_handler)

# Constants
VECTORSTORE_DIR = config.VECTORSTORE_DIR
DOCUMENTS_DIR = config.DOCUMENTS_DIR
db = ThreadDB()

# Create necessary directories
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Log system startup
json_logger.info(
    json.dumps({
        "timestamp": datetime.datetime.now().isoformat(), 
        "event_type": "system_startup", 
        "details": {
            "vectorstore_dir": VECTORSTORE_DIR, 
            "documents_dir": DOCUMENTS_DIR
        }
    })
)

# Function to load documents from the specified directory or uploaded files
def load_documents(directory=None, uploaded_files=None):
    docs = []
    if directory:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if filename.endswith('.pdf'):
                reader = PdfReader(filepath)
                text = "".join(page.extract_text() for page in reader.pages)
                docs.append({"page_content": text, "metadata": {"source": filename}})
            elif filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    docs.append({"page_content": text, "metadata": {"source": filename}})
    elif uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.pdf'):
                reader = PdfReader(uploaded_file)
                text = "".join(page.extract_text() for page in reader.pages)
                docs.append({"page_content": text, "metadata": {"source": uploaded_file.name}})
            elif uploaded_file.name.endswith('.txt'):
                text = uploaded_file.read().decode('utf-8')
                docs.append({"page_content": text, "metadata": {"source": uploaded_file.name}})
    return docs

# Function to reset model context - moved to the top of the file
def reset_model_context(model_name):
    """
    Reset a model's context using the Ollama API.
    This is a streamlined implementation focused on speed.
    """
    # Create a progress bar
    progress_text = f"Resetting model context..."
    progress_bar = st.progress(0, text=progress_text)
    
    try:
        logger.info(f"Resetting model context for {model_name}")
        
        # Use a single, fast approach to reset context
        with st.spinner(f"Resetting model context..."):
            start_time = time.time()
            
            try:
                # Direct API call with minimal parameters for speed
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name, 
                        "prompt": "",  # Empty prompt is fastest
                        "keep_alive": 0  # Don't keep model loaded
                    },
                    timeout=1  # Ultra short timeout - we don't need the response
                )
                success = True
            except requests.exceptions.Timeout:
                # Timeout is expected and fine - the request was sent
                success = True
            except Exception as e:
                logger.warning(f"Error in fast context reset: {str(e)}")
                success = False
            
            actual_time = time.time() - start_time
            
            # Update progress bar to completion
            progress_bar.progress(1.0, 
                                 text=f"Reset complete in {actual_time:.1f}s")
            
            # Log result
            logger.info(f"Model context reset completed for {model_name}")
            
            # Clear the progress bar immediately
            time.sleep(0.2)  # Just enough time to see completion
            progress_bar.empty()
            
            return success
            
    except Exception as e:
        # Clear the progress bar in case of error
        progress_bar.empty()
        
        logger.error(f"Error during model context reset: {str(e)}")
        return False

# Function to clear the vectorstore directory
def clear_vectorstore():
    """Clear all files in the vectorstore directory."""
    if os.path.exists(VECTORSTORE_DIR):
        logger.info("Clearing vectorstore directory")
        for file in os.listdir(VECTORSTORE_DIR):
            file_path = os.path.join(VECTORSTORE_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error clearing file {file_path}: {e}")
        logger.info("Vectorstore directory cleared")
    else:
        logger.info("Vectorstore directory does not exist, creating it")
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Function to fix or recreate the vectorstore
def fix_vectorstore():
    """
    Attempt to fix a corrupted vectorstore by clearing it.
    This will force the system to recreate the vectorstore next time documents are processed.
    """
    try:
        logger.info("Attempting to fix corrupted vectorstore", event_type="vectorstore_fix")
        
        # Check if vectorstore directory exists
        if os.path.exists(VECTORSTORE_DIR):
            # Clear all files in the vectorstore directory
            for file in os.listdir(VECTORSTORE_DIR):
                file_path = os.path.join(VECTORSTORE_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Removed file: {file_path}", event_type="vectorstore_fix")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Removed directory: {file_path}", event_type="vectorstore_fix")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}", 
                                event_type="vectorstore_fix_error",
                                exception=e)
            
            logger.info("Vectorstore cleared successfully", event_type="vectorstore_fix_success")
            return True
        else:
            # Create the directory if it doesn't exist
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            logger.info("Created new vectorstore directory", event_type="vectorstore_fix_success")
            return True
            
    except Exception as e:
        logger.error(f"Failed to fix vectorstore: {str(e)}", 
                    event_type="vectorstore_fix_error",
                    exception=e,
                    traceback=traceback.format_exc())
        return False

# Function to reprocess documents for a thread
def reprocess_thread_documents(document_names, model="deepseek-r1:8b"):
    """
    Reprocess documents for a thread to ensure proper context isolation.
    This loads the documents from the documents directory and creates new embeddings.
    """
    try:
        logger.info(f"Reprocessing documents for thread: {document_names}", 
                   event_type="thread_document_reprocessing")
        
        # Load documents from the document names
        docs = []
        found_doc_names = []  # Keep track of which documents were actually found
        
        for doc_name in document_names:
            # Try to find the document in the documents directory
            filepath = os.path.join(DOCUMENTS_DIR, doc_name)
            if os.path.exists(filepath):
                if doc_name.endswith('.pdf'):
                    reader = PdfReader(filepath)
                    text = "".join(page.extract_text() for page in reader.pages)
                    docs.append({"page_content": text, "metadata": {"source": doc_name}})
                    found_doc_names.append(doc_name)
                elif doc_name.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        docs.append({"page_content": text, "metadata": {"source": doc_name}})
                        found_doc_names.append(doc_name)
            else:
                logger.warning(f"Document not found: {doc_name}", 
                              event_type="document_not_found")
        
        # If documents were found, create embeddings
        if docs:
            # Check if the specified model supports embeddings
            model_supports_embeddings = st.session_state.embedding_models_support.get(model, False)
            
            # Use the model directly if it supports embeddings, otherwise find a model that does
            embedding_model = None
            if model_supports_embeddings:
                embedding_model = model
            else:
                # Find a model that supports embeddings
                embedding_model = get_best_embedding_model()
                if embedding_model and embedding_model != model:
                    logger.info(f"Using {embedding_model} for embeddings instead of {model}")
            
            # If we don't have an embedding model, try to get any available model
            if embedding_model is None:
                available_models = manage_ollama_models()
                if available_models:
                    embedding_model = available_models[0]
                    logger.warning(f"No embedding-supported models found. Trying with {embedding_model} anyway.")
                else:
                    logger.error("No models available for document processing")
                    return False
            
            # Clear existing vectorstore to ensure clean processing
            fix_vectorstore()
            
            # Create embeddings with our enhanced function
            vectorstore, doc_names = create_embeddings(
                docs, 
                embedding_model=embedding_model,
                user_selected_model=model
            )
            
            if doc_names:
                # Update the session state with the documents that were actually found and processed
                # This ensures we don't keep references to missing documents
                if set(found_doc_names) != set(document_names):
                    logger.warning(f"Some documents were not found. Updating session state with found documents only: {found_doc_names}")
                    st.session_state["documents"] = found_doc_names
                    
                    # Update the thread in the database if it exists
                    if st.session_state.get("current_thread", {}).get("id"):
                        db.update_thread(
                            st.session_state["current_thread"]["id"],
                            st.session_state["current_thread"]["name"],
                            st.session_state["messages"],
                            model=st.session_state.get("selected_model"),
                            documents=[]  # Don't save documents with threads for storage optimization
                        )
                
                logger.info(f"Successfully reprocessed {len(doc_names)} documents for thread", 
                           event_type="thread_document_reprocessing_success")
                return True
            else:
                logger.warning("Documents were found but processing failed", 
                              event_type="thread_document_reprocessing_warning")
                return False
        else:
            logger.warning("No documents found to reprocess", 
                          event_type="thread_document_reprocessing_warning")
            return False
    except Exception as e:
        logger.error(f"Error reprocessing documents: {str(e)}", 
                    event_type="thread_document_reprocessing_error",
                    exception=e,
                    traceback=traceback.format_exc())
        return False

# Function to manage Ollama models and check sizes
def manage_ollama_models():
    """
    Retrieve available Ollama models and their sizes.
    Filter out models larger than 12GB.
    Also check and store embedding support information.
    """
    try:
        # Get list of available models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error retrieving Ollama models: {result.stderr}")
            return []
        
        # Parse the output to get model names and sizes
        models = []
        model_sizes = {}
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                model_name = parts[0]
                size_str = parts[2]
                size_unit = parts[3]
                
                # Convert size to GB for comparison
                try:
                    size_value = float(size_str)
                    if size_unit == 'MB':
                        size_gb = size_value / 1024
                    elif size_unit == 'GB':
                        size_gb = size_value
                    else:
                        size_gb = 0  # Unknown unit
                    
                    # Filter out models larger than 12GB
                    if size_gb <= 12:
                        models.append(model_name)
                        model_sizes[model_name] = size_gb
                        logger.info(f"Model {model_name} size: {size_gb:.2f} GB")
                    else:
                        logger.info(f"Skipping model {model_name} (size: {size_gb:.2f} GB) - exceeds 12GB limit")
                except ValueError:
                    logger.warning(f"Could not parse size for model {model_name}: {size_str} {size_unit}")
        
        # Store model sizes in session state for later use
        st.session_state['model_sizes'] = model_sizes
        
        # Load cached embedding support information from database
        cached_embedding_info = db.get_all_model_embedding_support()
        
        # Initialize embedding support dictionary in session state if not exists
        if 'embedding_models_support' not in st.session_state:
            st.session_state.embedding_models_support = {}
        
        # Update session state with cached information
        for model_name, info in cached_embedding_info.items():
            # Check if the cached info is recent (less than 24 hours old)
            last_checked = datetime.datetime.fromisoformat(info['last_checked'])
            time_diff = datetime.datetime.now() - last_checked
            
            if time_diff.total_seconds() < 86400:  # 24 hours in seconds
                st.session_state.embedding_models_support[model_name] = info['supports_embeddings']
                logger.info(f"Using cached embedding support info for {model_name}: {info['supports_embeddings']}")
        
        # Check which models need to be tested for embedding support
        models_to_check = [model for model in models 
                          if model not in st.session_state.embedding_models_support]
        
        # Check embedding support for models that don't have cached information
        if models_to_check:
            with st.spinner(f"Checking embedding support for {len(models_to_check)} models..."):
                for model in models_to_check:
                    supports_embeddings = check_model_embeddings_support(model)
                    st.session_state.embedding_models_support[model] = supports_embeddings
                    # This will also save to the database inside check_model_embeddings_support
        
        return models
    
    except Exception as e:
        logger.error(f"Error managing Ollama models: {str(e)}")
        return []

# Function to check if a model supports embeddings
def check_model_embeddings_support(model_name):
    """
    Check if a model supports embeddings by first checking the database,
    and if not found or outdated, making a test API call.
    Returns True if embeddings are supported, False otherwise.
    """
    try:
        # First, check if we have this information in the database
        db_info = db.get_model_embedding_support(model_name)
        
        # If we have recent information in the database (less than 24 hours old), use it
        if db_info:
            last_checked = datetime.datetime.fromisoformat(db_info['last_checked'])
            time_diff = datetime.datetime.now() - last_checked
            
            # If the information is less than 24 hours old, use it
            if time_diff.total_seconds() < 86400:  # 24 hours in seconds
                logger.info(f"Using cached embedding support info for {model_name}: {db_info['supports_embeddings']}")
                return db_info['supports_embeddings']
        
        # If not in database or outdated, make a test API call
        import requests
        import json
        
        # Make a test API call to check embedding support
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model_name, "prompt": "test embedding support"},
            timeout=5
        )
        
        supports_embeddings = False
        if response.status_code == 200:
            result = response.json()
            # Check if embedding is not None and has values
            if result.get('embedding') is not None and len(result.get('embedding', [])) > 0:
                logger.info(f"Model {model_name} supports embeddings")
                supports_embeddings = True
            else:
                logger.info(f"Model {model_name} does not support embeddings (null or empty response)")
                supports_embeddings = False
        else:
            logger.warning(f"Embedding test for {model_name} failed with status {response.status_code}")
            supports_embeddings = False
        
        # Save the result to the database
        db.save_model_embedding_support(model_name, supports_embeddings)
        
        return supports_embeddings
            
    except Exception as e:
        logger.error(f"Error checking embedding support for {model_name}: {str(e)}")
        return False

# Function to get the best available embedding model
def get_best_embedding_model(user_selected_model=None):
    """
    Determine the best available model for embeddings.
    First checks if the current model supports embeddings,
    then falls back to other available models.
    """
    # If a specific model is provided and it supports embeddings, use it
    if user_selected_model and check_model_embeddings_support(user_selected_model):
        return user_selected_model
    
    # Try the fallback model from config
    fallback_model = config.FALLBACK_EMBEDDING_MODEL
    if check_model_embeddings_support(fallback_model):
        return fallback_model
    
    # Get all available models under 12GB
    available_models = manage_ollama_models()
    
    # Test each available model for embedding support
    for model in available_models:
        if check_model_embeddings_support(model):
            return model
    
    # If no models support embeddings, return None
    logger.error("No models available that support embeddings")
    return None

# Function to create embeddings for documents
def create_embeddings(docs, embedding_model=None, user_selected_model=None):
    """
    Create embeddings for documents using the specified model.
    If the model doesn't support embeddings, fall back to a default.
    Returns: 
        - vectorstore: FAISS vectorstore object
        - doc_names: List of document names that were processed
    """
    try:
        # Initialize progress bar
        progress_text = "Creating embeddings..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Determine the best embedding model to use
        if embedding_model is None:
            # Use the user selected model if it supports embeddings
            if user_selected_model and st.session_state.embedding_models_support.get(user_selected_model, False):
                embedding_model = user_selected_model
            else:
                # Otherwise find the best available embedding model
                embedding_model = get_best_embedding_model()
            
            # If user selected a model that doesn't support embeddings, notify them
            if user_selected_model and embedding_model != user_selected_model:
                logger.warning(f"Using '{embedding_model}' for embeddings instead of '{user_selected_model}'")
        
        if embedding_model is None:
            show_temp_message("No models available that support document processing. Please check your Ollama installation.", type="error")
            logger.error("Failed to create embeddings: No suitable embedding model available")
            progress_bar.empty()
            return None, []
            
        logger.info(f"Creating embeddings using model: {embedding_model}")
        
        # Create embeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Test the embeddings to ensure they work
        try:
            test_embedding = embeddings.embed_query("Test query to verify embeddings are working")
            if test_embedding is None or len(test_embedding) == 0:
                raise ValueError(f"Embedding model {embedding_model} returned null or empty embeddings. The model may not be running correctly.")
        except Exception as test_error:
            show_temp_message(f"Error testing embeddings: {str(test_error)}", type="error")
            logger.error(f"Embedding test failed: {str(test_error)}")
            progress_bar.empty()
            return None, []
        
        # Extract document names for return value
        doc_names = [doc["metadata"]["source"] for doc in docs if "metadata" in doc and "source" in doc["metadata"]]
        
        # Convert raw docs to LangChain Document objects
        langchain_docs = []
        for doc in docs:
            langchain_docs.append(Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ))
    
    # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        chunks = text_splitter.split_documents(langchain_docs)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(
            chunks,
            embedding=embeddings
        )
        
        # Update progress bar
        progress_bar.progress(1.0, text="Embeddings created successfully!")
        time.sleep(0.5)  # Show completion briefly
        progress_bar.empty()
        
        # Save the vectorstore to disk
    vectorstore.save_local(VECTORSTORE_DIR)
        
        logger.info(f"Processed {len(doc_names)} documents")
        return vectorstore, doc_names
        
    except Exception as e:
        show_temp_message(f"Error creating embeddings: {str(e)}", type="error")
        logger.error(f"Error in create_embeddings: {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        return None, []

# Function to load vectorstore and perform retrieval
def perform_retrieval(query, model="deepseek-r1:8b", use_documents=True):
    start_time = time.time()
    success = True
    error_msg = None
    
    try:
        # Check if we should use documents or just chat with the model directly
        if use_documents:
            # Check if vectorstore exists and has files
            if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
                # If no documents but user wants to chat anyway, fall back to direct chat
                logger.info("No vectorstore found, falling back to direct chat", 
                           event_type="retrieval_fallback")
                return perform_direct_chat(query, model)
            
            try:
                # Create a status message for the user
                status_message = st.empty()
                status_message.info("Preparing to search documents...")
                
                # Create embeddings with error handling
                logger.info(f"Creating embeddings for query using model {model}", event_type="debug")
                status_message.info(f"Creating query embeddings...")
                
                # Check if the current model supports embeddings
                current_model_supports_embeddings = st.session_state.embedding_models_support.get(model, False)
                
                # Use the current model if it supports embeddings, otherwise find a model that does
                embedding_model = None
                if current_model_supports_embeddings:
                    embedding_model = model
                    logger.info(f"Using selected model {model} for embeddings")
                else:
                    # Find a model that supports embeddings
                    embedding_model = get_best_embedding_model()
                    if embedding_model:
                        logger.info(f"Selected model {model} doesn't support embeddings, using {embedding_model} instead")
                
                if embedding_model is None:
                    logger.error("No models available that support embeddings")
                    # Don't show error, just log it and fall back to direct chat
                    return perform_direct_chat(query, model)
                
                # Use a more subtle message
                status_message.info(f"Searching documents...")
                
                # Use a more subtle message
                status_message.info(f"Searching documents...")
                embeddings = OllamaEmbeddings(model=embedding_model)
                
                # Load vectorstore with error handling
                logger.info("Loading vectorstore", event_type="debug")
                status_message.info("Loading document database...")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings)
                
                # Create retriever with error handling
                logger.info("Creating retriever", event_type="debug")
                status_message.info("Setting up retrieval system...")
    retriever = vectorstore.as_retriever()
    
                # Create LLM with error handling
                logger.info(f"Creating Ollama LLM with model {model}", event_type="debug")
                status_message.info(f"Thinking...")
                llm = Ollama(model=model)
                
                # Create QA chain with error handling
                logger.info("Creating QA chain", event_type="debug")
                # Don't show this message to keep UI clean
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                # Get the retrieved documents for context first with error handling
                logger.info("Getting relevant documents", event_type="debug")
                # Keep the UI clean with a simple message
                status_message.info("Finding relevant information...")
                try:
                    retrieved_docs = retriever.get_relevant_documents(query)
                    
                    # IMPORTANT: Filter sources to only include documents from the current thread
                    # This ensures no leakage of sources between threads
                    current_thread_docs = st.session_state.get("documents", [])
                    sources = []
                    for doc in retrieved_docs:
                        source = doc.metadata.get("source", "Unknown")
                        # Only include sources that are in the current thread's document list
                        if source in current_thread_docs or not current_thread_docs:
                            sources.append(source)
                    
                    # Log the filtering process
                    logger.info(f"Retrieved {len(retrieved_docs)} docs, filtered to {len(sources)} sources from current thread", 
                               event_type="source_filtering",
                               thread_docs=current_thread_docs,
                               retrieved_sources=[doc.metadata.get("source", "Unknown") for doc in retrieved_docs],
                               filtered_sources=sources)
                    
                    # Don't show this message to keep UI clean
                    
                except ValueError as ve:
                    # Handle FAISS search error specifically
                    if "not enough values to unpack" in str(ve):
                        logger.error(f"FAISS search error: {str(ve)}", 
                                    event_type="faiss_error",
                                    traceback=traceback.format_exc())
                        # Try to recreate the vectorstore index
                        logger.info("Attempting to fix vectorstore by falling back to direct chat", 
                                   event_type="vectorstore_recovery")
                        # Don't show warning messages to the user
                        status_message.empty()
                        return perform_direct_chat(query, model)
                    else:
                        raise
                
                # Then run the query
                logger.info("Running QA chain", event_type="debug")
                status_message.info("Generating answer...")
    result = qa_chain.run(query)
                logger.info(f"QA chain result type: {type(result)}", 
                           event_type="debug", 
                           result_type=str(type(result)))
                
                # Clear the status message
                status_message.empty()
                
                # Make sure result is a string
                if not isinstance(result, str):
                    logger.info(f"Converting QA result from {type(result)} to string", 
                               event_type="debug")
                    result = str(result)
                    
                # Ensure sources is a list
                if not isinstance(sources, list):
                    logger.info(f"Converting sources from {type(sources)} to list", 
                               event_type="debug")
                    sources = []
                    
                # Return the result and sources
                return result, list(set(sources))
                
            except Exception as e:
                # Handle vectorstore/embedding errors by falling back to direct chat
                logger.error(f"Error in document retrieval: {str(e)}", 
                            event_type="vectorstore_error",
                            traceback=traceback.format_exc())
                logger.info("Falling back to direct chat due to vectorstore error", 
                           event_type="vectorstore_fallback")
                return perform_direct_chat(query, model)
        else:
            # Direct chat with the model without documents
            logger.info("Using direct chat mode", event_type="debug")
            direct_result = perform_direct_chat(query, model)
            
            # Debug the return value
            logger.info(f"Direct chat return value type: {type(direct_result)}, value: {direct_result}", 
                       event_type="debug")
            
            # Ensure we're returning a tuple with two values
            if isinstance(direct_result, tuple) and len(direct_result) == 2:
                return direct_result
            else:
                logger.error(f"Direct chat returned unexpected format: {direct_result}", 
                            event_type="format_error")
                # Fix the return value if it's not in the expected format
                if isinstance(direct_result, str):
                    return direct_result, []
                else:
                    return str(direct_result), []
        
    except Exception as e:
        success = False
        error_msg = str(e)
        
        # Log detailed error
        logger.error(f"Error performing retrieval: {error_msg}", 
                    event_type="retrieval_error", 
                    exception=e,
                    traceback=traceback.format_exc(),
                    use_documents=use_documents,
                    model=model)
        
        # Log model usage
        duration_ms = (time.time() - start_time) * 1000
        logger.model_usage(
            model_name=model,
            query_length=len(query),
            response_length=0,
            duration_ms=duration_ms,
            success=False,
            error=error_msg
        )
        
        # Return the error message and an empty list of sources
        return f"Error: {error_msg} (Traceback: {traceback.format_exc().splitlines()[-3:]})", []

# Function for direct chat with the model (no documents)
def perform_direct_chat(query, model="deepseek-r1:8b"):
    start_time = time.time()
    
    try:
        # Create a status message for the user
        status_message = st.empty()
        status_message.info(f"Preparing to chat with {model}...")
        
        # Use Ollama directly without retrieval
        status_message.info(f"Sending your question to {model}...")
        llm = Ollama(model=model)
        
        # Show a thinking message
        status_message.info(f"{model} is thinking about your question...")
        result = llm.invoke(query)
        
        # Clear the status message
        status_message.empty()
        
        # Debug logging
        logger.info(f"Direct chat result type: {type(result)}", 
                   event_type="debug", 
                   result_type=str(type(result)))
        
        # Make sure result is a string
        if not isinstance(result, str):
            logger.info(f"Converting result from {type(result)} to string", 
                       event_type="debug")
            result = str(result)
        
        # Log model usage
        duration_ms = (time.time() - start_time) * 1000
        logger.model_usage(
            model_name=model,
            query_length=len(query),
            response_length=len(result),
            duration_ms=duration_ms,
            success=True
        )
        
        # Explicitly return a tuple with two values
        return result, []
    except Exception as e:
        # Clear any status message
        try:
            status_message.empty()
        except:
            pass
            
        error_msg = str(e)
        logger.error(f"Direct chat error: {error_msg}", 
                    event_type="direct_chat_error", 
                    exception=e,
                    traceback=traceback.format_exc())
        
        # Log error
        duration_ms = (time.time() - start_time) * 1000
        logger.model_usage(
            model_name=model,
            query_length=len(query),
            response_length=0,
            duration_ms=duration_ms,
            success=False,
            error=error_msg
        )
        
        # Explicitly return a tuple with two values
        return f"Error: {error_msg}", []

# Generate default thread name based on timestamp
def generate_thread_name():
    now = datetime.datetime.now()
    return f"Thread {now.strftime('%Y-%m-%d %H:%M')}"

# Function to check if a model supports multimodal inputs
def check_model_multimodal_support(model_name):
    """
    Check if a model supports multimodal inputs (text + images).
    Currently only Gemma models are known to support this.
    
    Args:
        model_name (str): The name of the model to check
        
    Returns:
        bool: True if the model supports multimodal inputs, False otherwise
    """
    # Check if the model is in the list of known multimodal models
    if model_name in config.MULTIMODAL_MODELS:
        return True
        
    # Check if it's a Gemma model (which generally support multimodal)
    if model_name.startswith("gemma:"):
        return True
        
    # For other models, assume they don't support multimodal inputs
    return False

# Function to encode an image to base64
def encode_image_to_base64(image_file):
    """
    Encode an image file to base64 for use with multimodal models.
    
    Args:
        image_file: A file-like object containing the image data
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        # If it's a streamlit UploadedFile, read the bytes
        if hasattr(image_file, 'getvalue'):
            image_bytes = image_file.getvalue()
        # If it's a file path, open and read the file
        elif isinstance(image_file, str) and os.path.exists(image_file):
            with open(image_file, 'rb') as f:
                image_bytes = f.read()
        # If it's already bytes, use it directly
        elif isinstance(image_file, bytes):
            image_bytes = image_file
        else:
            raise ValueError(f"Unsupported image file type: {type(image_file)}")
            
        # Encode to base64
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"Successfully encoded image to base64 ({len(base64_encoded)} chars)")
        return base64_encoded
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

# Function to save an uploaded image
def save_uploaded_image(uploaded_file):
    """
    Save an uploaded image file to the images directory.
    
    Args:
        uploaded_file: A Streamlit UploadedFile object
        
    Returns:
        str: Path to the saved image file
    """
    try:
        # Create a unique filename
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(config.IMAGES_DIR, unique_filename)
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
            
        logger.info(f"Saved uploaded image to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded image: {str(e)}")
        raise

# Function to query Gemma with an image
def query_gemma_with_image(prompt, image_base64=None, model="gemma:latest"):
    """
    Query a Gemma model with text and an optional image.
    
    Args:
        prompt (str): The text prompt to send to the model
        image_base64 (str, optional): Base64 encoded image, or None for text-only queries
        model (str): The Gemma model to use
        
    Returns:
        str: The model's response
    """
    start_time = time.time()
    
    try:
        # Create a status message for the user
        status_message = st.empty()
        
        # Check if we're doing an image query or text-only query
        if image_base64:
            status_message.info(f"Processing image with {model}...")
            
            # Prepare the API request with image
            url = f"{config.OLLAMA_API_URL}/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
        else:
            # Text-only query
            status_message.info(f"Sending your question to {model}...")
            
            # Prepare the API request without image
            url = f"{config.OLLAMA_API_URL}/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        
        # Make the API request
        response = requests.post(url, json=data, timeout=config.LONG_TIMEOUT)
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            status_message.error(error_msg)
            return f"Error: {error_msg}", []
            
        # Parse the response
        result = response.json()
        
        # Clear the status message
        status_message.empty()
        
        # Extract the response text
        response_text = result.get("response", "No response received")
        
        # Log model usage
        duration_ms = (time.time() - start_time) * 1000
        logger.model_usage(
            model_name=model,
            query_length=len(prompt),
            response_length=len(response_text),
            duration_ms=duration_ms,
            success=True
        )
        
        # Return the response and an empty list of sources (no document retrieval for images)
        return response_text, []
    except Exception as e:
        # Clear any status message
        try:
            status_message.empty()
        except:
            pass
            
        error_msg = str(e)
        logger.error(f"Query processing error: {error_msg}", 
                    event_type="model_query_error", 
                    exception=e,
                    traceback=traceback.format_exc())
        
        # Log error
        duration_ms = (time.time() - start_time) * 1000
        logger.model_usage(
            model_name=model,
            query_length=len(prompt),
            response_length=0,
            duration_ms=duration_ms,
            success=False,
            error=error_msg
        )
        
        # Return error message and empty sources list
        return f"Error processing query: {error_msg}", []

# Streamlit UI layout
st.set_page_config(page_title="Document Retrieval App", layout="wide")

# Check if this is a new session by looking for a session ID
# Streamlit clears session state on server restart, so this will be True on each restart
if "session_id" not in st.session_state:
    # Generate a unique session ID
    st.session_state["session_id"] = str(uuid.uuid4())
    
    # Force a new thread on application startup
    # This is a new session, reset everything
    st.session_state["app_initialized"] = True
    
    # Generate a new thread name
    default_thread_name = generate_thread_name()
    
    # Reset the application state
    st.session_state["current_thread"] = {"id": None, "name": default_thread_name}
    st.session_state["messages"] = []
    st.session_state["documents"] = []
    
    # Clear the vectorstore
    fix_vectorstore()
    
    # Log the fresh start
    logger.info(f"Application started with a fresh thread. Session ID: {st.session_state['session_id']}")

# Initialize session state for threads and messages
if "current_thread" not in st.session_state:
    default_thread_name = generate_thread_name()
    st.session_state["current_thread"] = {"id": None, "name": default_thread_name}
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "documents" not in st.session_state:
    st.session_state["documents"] = []
if "images" not in st.session_state:
    st.session_state["images"] = []
if "embedding_models_support" not in st.session_state:
    st.session_state["embedding_models_support"] = {}
if "loaded_models" not in st.session_state:
    # Try to get available models
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        available_models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line:
                model_name = line.split()[0]
                available_models.append(model_name)
        
        # Load the first available model by default
        if available_models:
            st.session_state["loaded_models"] = [available_models[0]]
            if "selected_model" not in st.session_state:
                st.session_state["selected_model"] = available_models[0]
    except Exception:
        st.session_state["loaded_models"] = []

# Option to use documents or not
if "use_documents" not in st.session_state:
    st.session_state["use_documents"] = True

# Function to completely reset application state for a new thread
def reset_app_state_for_new_thread():
    """
    Completely reset the application state for a new thread.
    This ensures total isolation between threads.
    """
    # Get current model before resetting
    current_model = st.session_state.get("selected_model")
    
    # Remember document preference
    use_documents = st.session_state.get("use_documents", True)
    
    # Reset model context to clear any memory of previous documents
    if current_model:
        reset_model_context(current_model)
    
    # Clear the current thread context
    st.session_state["current_thread"] = {"id": None, "name": generate_thread_name()}
    st.session_state["messages"] = []
    
    # IMPORTANT: Clear document sources to ensure complete isolation
    st.session_state["documents"] = []
    
    # Restore document preference
    st.session_state["use_documents"] = use_documents
    
    # Clear the vectorstore to unload previous document context
    # This is critical for ensuring no documents from previous threads are accessible
    fix_vectorstore()
    
    # Log the reset
    logger.info("Complete application state reset for new thread")
    
    return current_model

# SIDEBAR - Thread Management
with st.sidebar:
    st.title("Document Retrieval")
    
    # New thread button at the top
    if st.button("+ New Thread", use_container_width=True):
        # Reset the entire application state
        reset_app_state_for_new_thread()
        
        # Log thread creation
        logger.thread_activity(
            thread_id="new",
            thread_name=st.session_state["current_thread"]["name"],
            action="create",
            message_count=0
        )
        
        st.rerun()
    
    st.divider()
    
    # Display saved threads in a simple list
    st.subheader("Your Threads")
    threads = db.load_threads()
    if threads:
        for thread_data in threads:
            thread_id = thread_data[0]
            name = thread_data[1]
            created_at = thread_data[2]
            model = thread_data[3] if len(thread_data) > 3 else None
            
            # Get first message if available
            thread_messages = db.load_thread_messages(thread_id)
            first_question = ""
            if thread_messages:
                for msg in thread_messages:
                    if msg.get("role") == "user":
                        first_question = msg.get("content", "")[:30] + "..." if len(msg.get("content", "")) > 30 else msg.get("content", "")
                        break
            
            # Format the thread display with name and first question
            thread_info = f"{name}"
            if first_question:
                thread_info += f"\n{first_question}"
            
            # Use a button with the thread name and first question
            if st.button(thread_info, key=f"thread_{thread_id}", use_container_width=True):
                # Get current model before switching
                current_model = st.session_state.get("selected_model")
                
                # Reset model context to clear any memory of previous documents
                if current_model:
                    reset_model_context(current_model)
                
                # Load thread messages
                st.session_state["current_thread"] = {"id": thread_id, "name": name}
                st.session_state["messages"] = db.load_thread_messages(thread_id)
    
                # IMPORTANT: Clear the vectorstore BEFORE loading thread documents
                # This ensures complete isolation between threads
                fix_vectorstore()
                
                # For storage optimization, we don't load documents from old threads
                # Users will need to add documents again when they open an old thread
                st.session_state["documents"] = []
                
                # Log thread selection
                logger.thread_activity(
                    thread_id=thread_id,
                    thread_name=name,
                    action="select",
                    message_count=len(st.session_state["messages"]),
                    model=model
                )
                
                # Load thread model if available
                thread_model = db.load_thread_model(thread_id)
                if thread_model:
                    # Set as active model
                    st.session_state["selected_model"] = thread_model
                    
                    # Add to loaded models if not already there
                    if thread_model not in st.session_state.get("loaded_models", []):
                        if len(st.session_state.get("loaded_models", [])) >= 3:
                            # Remove the oldest model
                            st.session_state["loaded_models"].pop(0)
                        st.session_state["loaded_models"].append(thread_model)
                
                st.rerun()
    else:
        show_temp_message("No previous threads found.", type="info")
    
    # Settings section at the bottom of sidebar
    st.divider()
    st.subheader("Settings")
    
    # Analytics dashboard link
    if st.button("📊 Analytics Dashboard", use_container_width=True):
        logger.info("Analytics dashboard opened")
        # This will open the dashboard in a new tab
        js = f"""
        <script>
        window.open("http://localhost:8501/dashboard", "_blank");
        </script>
        """
        st.components.v1.html(js, height=0)
    
    # Database management (hidden in expandable section)
    with st.expander("Database Management"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Backup DB", use_container_width=True):
                backup_path = backup_database()
                if backup_path:
                    show_temp_message(f"Backup created at: {backup_path}", type="success")
                else:
                    show_temp_message("Failed to create backup", type="error")
        
        with col2:
            if st.button("Restore DB", use_container_width=True):
                # List available backups
                backup_files = glob.glob("./backups/*.db")
                if backup_files:
                    latest_backup = max(backup_files, key=os.path.getctime)
                    if restore_database(latest_backup):
                        show_temp_message(f"Restored from: {latest_backup}", type="success")
                    else:
                        show_temp_message("Failed to restore", type="error")
                else:
                    show_temp_message("No backups found", type="error")

# MAIN CONTENT - Chat Interface
# Thread name editing directly in the header
col1, col2 = st.columns([4, 1])
with col1:
    # Editable thread name
    new_thread_name = st.text_input("Thread Name", 
                                   value=st.session_state["current_thread"]["name"],
                                   placeholder="Enter thread name...",
                                   label_visibility="collapsed")
    
    if new_thread_name != st.session_state["current_thread"]["name"]:
        # Update thread name in session state
        st.session_state["current_thread"]["name"] = new_thread_name
        
        # If thread exists in DB, update it
        if st.session_state["current_thread"]["id"]:
            db.update_thread(
                st.session_state["current_thread"]["id"],
                new_thread_name,
                st.session_state["messages"],
                model=st.session_state.get("selected_model"),
                documents=[]  # Don't save documents with threads for storage optimization
            )
            logger.thread_activity(
                thread_id=st.session_state["current_thread"]["id"],
                thread_name=new_thread_name,
                action="rename",
                message_count=len(st.session_state["messages"])
            )
            # Force a rerun to update the sidebar
            st.rerun()

with col2:
    # Add a "Go to bottom" button
    if st.button("⬇️ Go to chat", use_container_width=True):
        # Use JavaScript to scroll to the bottom
        st.components.v1.html(
            """
            <script>
            window.scrollTo(0, document.body.scrollHeight);
            </script>
            """,
            height=0
        )

# Add sleek styling
st.markdown("""
<style>
.stButton button {
    padding: 0.25rem 0.5rem;
    font-size: 0.8rem;
    border-radius: 4px;
}
.stTextArea textarea {
    border-radius: 4px;
    border: 1px solid #ddd;
}
.stSelectbox div[data-baseweb="select"] {
    border-radius: 4px;
}
div.stFileUploader {
    border-radius: 4px;
}
div.stFileUploader button {
    border-radius: 4px;
}
div.chat-container {
    margin-bottom: 20px;
}
div.chat-input-container {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Display chat messages
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
                        show_temp_message(f"Error displaying image: {str(e)}", type="error")
            else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        st.write(", ".join(message["sources"]))

# Streamlined chat input and controls
st.divider()

# Create a sleek chat interface
with st.container():
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Top row with model selector
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Get available models
        available_models = manage_ollama_models()
        
        # Select model
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(st.session_state.get("selected_model", available_models[0])) if available_models and st.session_state.get("selected_model") in available_models else 0,
            key="model_selector"
        )
        
        # Update selected model in session state
        if selected_model != st.session_state.get("selected_model"):
            previous_model = st.session_state.get("selected_model")
            st.session_state["selected_model"] = selected_model
            # Reset model context when switching models
            reset_model_context(selected_model)
            
            # Check if there's an incompatible attachment
            if "selected_image" in st.session_state and st.session_state["selected_image"]:
                # If switching from a multimodal model to a non-multimodal model
                if check_model_multimodal_support(previous_model) and not check_model_multimodal_support(selected_model):
                    # Clear the selected image if the new model doesn't support images
                    show_temp_message("Selected model doesn't support images. Your image attachment has been cleared.", type="warning")
                    st.session_state["selected_image"] = None
                    st.rerun()
            
            # If we have documents and switched to a multimodal model, suggest a better model
            documents = st.session_state.get("documents", [])
            if documents and len(documents) > 0:
                # Find a better model for document processing
                embedding_model = get_best_embedding_model(selected_model)
                if embedding_model and embedding_model != selected_model:
                    show_temp_message(f"The selected model '{selected_model}' might not be optimal for document processing. Consider using '{embedding_model}' instead.", type="warning")
    
    # File attachment and chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Chat input
        # Add a unique key that changes when messages are sent to force the input to clear
        input_key = f"user_input_{len(st.session_state.messages)}"
        user_message = st.text_area("Your message", height=80, placeholder="Type your message here...", label_visibility="collapsed", key=input_key)
    
    with col2:
        # File attachment
        uploaded_file = st.file_uploader(
            "Attach File",
            type=["pdf", "txt"] + list(config.SUPPORTED_IMAGE_TYPES),
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        # Process the uploaded file automatically
        if uploaded_file:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            current_model = st.session_state.get("selected_model")
            
            # Check if it's an image
            if file_extension.lower() in [f".{ext}" for ext in config.SUPPORTED_IMAGE_TYPES]:
                # Create images directory if it doesn't exist
                os.makedirs(config.IMAGES_DIR, exist_ok=True)
                
                # Check if current model supports images
                if not check_model_multimodal_support(current_model):
                    # Find multimodal models
                    multimodal_models = [model for model in available_models if check_model_multimodal_support(model)]
                    
                    if multimodal_models:
                        # Automatically switch to a multimodal model
                        new_model = multimodal_models[0]
                        st.session_state["selected_model"] = new_model
                        st.session_state["model_changed_for_image"] = True
                        model = new_model
                        
                        # Display warning message
                        show_temp_message(f"Switched to {multimodal_models[0]} for image support", type="info")
            else:
                # It's a document (PDF or TXT)
                # Check if current model is a multimodal model (which might not be optimal for text)
                if check_model_multimodal_support(current_model):
                    # Find text-optimized models
                    text_models = [model for model in available_models if not check_model_multimodal_support(model)]
                    
                    if text_models:
                        # Find a model that supports embeddings
                        embedding_model = get_best_embedding_model()
                        if embedding_model in text_models:
                            # Switch to the best embedding model
                            st.session_state["selected_model"] = embedding_model
                            show_temp_message(f"Switched to {embedding_model} for document processing", type="info")
                        else:
                            # If best embedding model is not in text_models, use the first text model
                            st.session_state["selected_model"] = text_models[0]
                            show_temp_message(f"Switched to {text_models[0]} for document processing", type="info")
                
                # Create documents directory if it doesn't exist
                os.makedirs(DOCUMENTS_DIR, exist_ok=True)
                
                try:
                    # Clear any previously selected image since we're now using a document
                    if "selected_image" in st.session_state and st.session_state["selected_image"]:
                        show_temp_message("Cleared previous image attachment as you're now using a document.", type="info")
                        st.session_state["selected_image"] = None
                    
                    # Save the file
                    file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process the document
                    docs = load_documents(directory=None, uploaded_files=[uploaded_file])
                    
                    # Get the best embedding model for this document type
                    current_model = st.session_state.get("selected_model")
                    embedding_model = get_best_embedding_model(current_model)
                    
                    # If the current model doesn't support embeddings, suggest switching
                    if embedding_model != current_model and not st.session_state.embedding_models_support.get(current_model, False):
                        show_temp_message(f"Using {embedding_model} for document embeddings as the current model doesn't support them.", type="info")
                    
                    # Create embeddings
                    vectorstore, doc_names = create_embeddings(
                        docs,
                        embedding_model=embedding_model,
                        user_selected_model=current_model
                    )
                    
                    # Update session state with document names
                    if doc_names:
                        st.session_state["documents"] = list(set(st.session_state.get("documents", []) + doc_names))
                        
                        # Update thread in database if it exists
                        if st.session_state.get("current_thread", {}).get("id"):
                            db.update_thread(
                                st.session_state["current_thread"]["id"],
                                st.session_state["current_thread"]["name"],
                                st.session_state["messages"],
                                model=st.session_state.get("selected_model"),
                                documents=[]  # Don't save documents with threads for storage optimization
                            )
                        
                        show_temp_message(f"Added: {uploaded_file.name}", type="success")
                except Exception as e:
                    show_temp_message(f"Error processing document: {str(e)}", type="error")
                    logger.error(f"Document processing error: {str(e)}")
    
    # Send button
    if st.button("Send", use_container_width=True) and user_message:
        # Get selected model
        model = st.session_state.get("selected_model", "deepseek-r1:8b")
        
        # Check if we have an image but the model doesn't support it
        if st.session_state.get("selected_image") and not check_model_multimodal_support(model):
            show_temp_message(f"The selected model '{model}' doesn't support image processing. Please select a multimodal model like Gemma.", type="error")
            # Don't proceed with sending
            st.rerun()
        # Check if we have documents but the model might not be optimal for text
        elif len(st.session_state.get("documents", [])) > 0 and check_model_multimodal_support(model):
            # Find a better model for document processing
            embedding_model = get_best_embedding_model()
            if embedding_model and embedding_model != model:
                show_temp_message(f"The selected model '{model}' might not be optimal for document processing. Consider switching to '{embedding_model}'.", type="warning")
        else:
            # Add user message to chat history
            user_msg = {"role": "user", "content": user_message}
            
            # Add image to message if selected
            if st.session_state.get("selected_image"):
                user_msg["image"] = st.session_state["selected_image"]
                # Clear the selected image after sending
                st.session_state["selected_image"] = None
            
            st.session_state.messages.append(user_msg)
            
            # Process message
            try:
                # Check if we're using an image
                if "image" in user_msg and user_msg["image"]:
                    # Process with image
                    image_path = user_msg["image"]
                    
                    # Encode image to base64
                    with open(image_path, "rb") as img_file:
                        image_base64 = encode_image_to_base64(img_file.read())
                    
                    # Query model with image
                    response, sources = query_gemma_with_image(user_message, image_base64, model=model)
                else:
                    # Check if we're using a multimodal model for text-only query
                    if check_model_multimodal_support(model):
                        # Use the same function but without an image
                        response, sources = query_gemma_with_image(user_message, image_base64=None, model=model)
                    else:
                        # Regular text query with non-multimodal model
                        response, sources = perform_retrieval(
                            user_message,
                            model=model,
                            use_documents=True  # Always use documents if available
                        )
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
                # Save thread to database
                if not st.session_state["current_thread"]["id"]:
                    # Create new thread
                    thread_id = db.save_thread(
                        st.session_state["current_thread"]["name"],
                        st.session_state["messages"],
                        model=model,
                        documents=[]  # Don't save documents with threads for storage optimization
                    )
                    st.session_state["current_thread"]["id"] = thread_id
                else:
                    # Update existing thread
                    db.update_thread(
                        st.session_state["current_thread"]["id"],
                        st.session_state["current_thread"]["name"],
                        st.session_state["messages"],
                        model=model,
                        documents=[]  # Don't save documents with threads for storage optimization
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
                show_temp_message(f"Error: {str(e)}", type="error")
                logger.error(f"Error processing message: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    # Don't call main() directly - the original app flow will handle everything
    # We've already added our dynamic embedding model selection to the existing code
    pass
