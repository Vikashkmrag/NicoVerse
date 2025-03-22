import os
import time
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from PyPDF2 import PdfReader
import shutil
import config
from modules.utils.logger import get_logger
from modules.models.model_manager import ModelManager
from modules.utils.debug import debug_print
from modules.utils.temp_message import show_temp_message
import tempfile
import threading

logger = get_logger("document_processor")

class DocumentProcessor:
    """
    Handles document loading, processing, and vector embeddings.
    """
    
    def __init__(self):
        self.vectorstore_dir = config.VECTORSTORE_DIR
        self.documents_dir = config.DOCUMENTS_DIR
        self.model_manager = ModelManager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Create necessary directories
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        os.makedirs(self.documents_dir, exist_ok=True)
    
    def load_documents(self, directory=None, uploaded_files=None):
        """
        Load documents from a directory or uploaded files.
        
        Args:
            directory (str, optional): Directory path to load documents from
            uploaded_files (list, optional): List of uploaded file objects
            
        Returns:
            list: List of document dictionaries with page_content and metadata
        """
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
    
    def save_uploaded_document(self, uploaded_file):
        """
        Save an uploaded document to the documents directory.
        
        Args:
            uploaded_file: A Streamlit UploadedFile object
            
        Returns:
            str: Path to the saved document
        """
        file_path = os.path.join(self.documents_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    
    def create_embeddings(self, docs, embedding_model=None, user_selected_model=None):
        """
        Create embeddings for documents using the specified model.
        
        Args:
            docs (list): List of document dictionaries
            embedding_model (str, optional): Model to use for embeddings
            user_selected_model (str, optional): User's selected model
            
        Returns:
            tuple: (vectorstore, doc_names) - FAISS vectorstore and list of document names
        """
        try:
            # Initialize progress bar
            progress_text = "Creating embeddings..."
            progress_bar = st.progress(0, text=progress_text)
            
            # Determine the best embedding model to use
            if embedding_model is None:
                # Use the user selected model if it supports embeddings
                embedding_model = self.model_manager.get_best_embedding_model(user_selected_model)
                
                # If no suitable model found, show error
                if embedding_model is None:
                    show_temp_message("No models available that support document processing. Please check your Ollama installation.", type="error")
                    logger.error("Failed to create embeddings: No suitable embedding model available")
                    progress_bar.empty()
                    return None, []
                
                # If user selected a model that doesn't support embeddings, notify them
                if user_selected_model and embedding_model != user_selected_model:
                    show_temp_message(f"Using '{embedding_model}' for embeddings instead of '{user_selected_model}' for better performance.", type="info")
                    logger.info(f"Using '{embedding_model}' for embeddings instead of '{user_selected_model}'")
            
            # Update progress bar
            progress_bar.progress(0.1, text=f"Creating embeddings using {embedding_model}...")
            logger.info(f"Creating embeddings using model: {embedding_model}")
            
            # Create embeddings
            embeddings = OllamaEmbeddings(model=embedding_model)
            
            # Test the embeddings to ensure they work
            try:
                progress_bar.progress(0.2, text=f"Testing embedding model {embedding_model}...")
                test_embedding = embeddings.embed_query("Test query to verify embeddings are working")
                if test_embedding is None or len(test_embedding) == 0:
                    raise ValueError(f"Embedding model {embedding_model} returned null or empty embeddings. The model may not be running correctly.")
                
                # Update progress
                progress_bar.progress(0.3, text=f"Embedding test successful. Processing documents...")
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
            
            # Update progress
            progress_bar.progress(0.4, text=f"Splitting documents into chunks...")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            chunks = text_splitter.split_documents(langchain_docs)
            
            # Update progress
            progress_bar.progress(0.6, text=f"Creating vector embeddings for {len(chunks)} chunks...")
            
            # Create vectorstore
            vectorstore = FAISS.from_documents(
                chunks,
                embedding=embeddings
            )
            
            # Update progress
            progress_bar.progress(0.8, text=f"Saving vectorstore to disk...")
            
            # Save the vectorstore to disk
            vectorstore.save_local(self.vectorstore_dir)
            
            # Update progress bar
            progress_bar.progress(1.0, text="Embeddings created successfully!")
            time.sleep(0.5)  # Show completion briefly
            progress_bar.empty()
            
            # Store the embedding model used in session state for reference
            st.session_state['current_embedding_model'] = embedding_model
            
            # Show success message with model info
            show_temp_message(f"Successfully processed {len(doc_names)} documents using {embedding_model} for embeddings.", type="success")
            
            logger.info(f"Processed {len(doc_names)} documents with {embedding_model}")
            return vectorstore, doc_names
            
        except Exception as e:
            show_temp_message(f"Error creating embeddings: {str(e)}", type="error")
            logger.error(f"Error in create_embeddings: {str(e)}")
            if 'progress_bar' in locals():
                progress_bar.empty()
            return None, []
    
    def clear_vectorstore(self):
        """Clear all files in the vectorstore directory."""
        if os.path.exists(self.vectorstore_dir):
            logger.info("Clearing vectorstore directory")
            for file in os.listdir(self.vectorstore_dir):
                file_path = os.path.join(self.vectorstore_dir, file)
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
            os.makedirs(self.vectorstore_dir, exist_ok=True)
    
    def fix_vectorstore(self):
        """
        Attempt to fix a corrupted vectorstore by clearing it.
        This will force the system to recreate the vectorstore next time documents are processed.
        """
        try:
            logger.info("Attempting to fix corrupted vectorstore", event_type="vectorstore_fix")
            
            # Check if vectorstore directory exists
            if os.path.exists(self.vectorstore_dir):
                # Clear all files in the vectorstore directory
                for file in os.listdir(self.vectorstore_dir):
                    file_path = os.path.join(self.vectorstore_dir, file)
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
                os.makedirs(self.vectorstore_dir, exist_ok=True)
                logger.info("Created new vectorstore directory", event_type="vectorstore_fix_success")
                return True
                
        except Exception as e:
            logger.error(f"Failed to fix vectorstore: {str(e)}", 
                        event_type="vectorstore_fix_error",
                        exception=e)
            return False 