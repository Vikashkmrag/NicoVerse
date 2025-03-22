import os
import json
import time
import base64
import requests
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

import config
from modules.utils.logger import get_logger
from modules.utils.debug import debug_print
from modules.models.model_manager import ModelManager
from modules.data.document_processor import DocumentProcessor
from modules.data.image_processor import ImageProcessor

logger = get_logger("query_handler")

class QueryHandler:
    """
    Handles processing of user queries, document retrieval, and model interactions.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
    
    def process_query(self, user_message, user_msg, model):
        """
        Process a user query and return a response.
        
        Args:
            user_message: The text content of the user's message
            user_msg: The full user message object (may include image)
            model: The model to use for the query
            
        Returns:
            tuple: (response, sources)
        """
        debug_print("process_query called with model: {}", model)
        debug_print("user_msg: {}", user_msg)
        debug_print("user_msg keys: {}", user_msg.keys())
        
        # Log the query
        logger.info(
            f"Processing query with model {model}",
            event_type="query_processing",
            model=model,
            has_image="image" in user_msg and user_msg["image"] is not None,
            has_documents=len(st.session_state.get("documents", [])) > 0
        )
        
        # Check if the message contains an image
        if "image" in user_msg and user_msg["image"]:
            debug_print("Image detected in message: {}", user_msg['image'])
            
            # Verify the image file exists
            if not os.path.exists(user_msg["image"]):
                error_msg = f"Image file not found: {user_msg['image']}"
                debug_print("{}", error_msg)
                return error_msg, []
                
            debug_print("Image file exists at: {}", user_msg['image'])
            
            # Check if the model supports multimodal inputs
            if not self.model_manager.check_model_multimodal_support(model):
                debug_print("Model {} does not support multimodal inputs", model)
                
                # Find a better model that supports multimodal inputs
                available_models = self.model_manager.get_available_models()
                multimodal_models = [m for m in available_models if self.model_manager.check_model_multimodal_support(m)]
                debug_print("Available multimodal models: {}", multimodal_models)
                
                if multimodal_models:
                    # Automatically select a multimodal model
                    model = multimodal_models[0]
                    debug_print("Automatically selected multimodal model: {}", model)
                    
                    # Update the session state with the new model
                    if st.session_state.get("selected_model") != model:
                        st.session_state["selected_model"] = model
                        st.session_state["model_changed_for_image"] = True
                        debug_print("Updated session state with selected_model: {}", model)
                else:
                    return "Error: No multimodal models available to process the image. Please install a multimodal model like gemma:2b-it or llava:7b.", []
            else:
                debug_print("Model {} supports multimodal inputs", model)
            
            # Process the image query
            debug_print("Calling _process_image_query with model: {}", model)
            return self._process_image_query(user_message, user_msg["image"], model)
            
        # Check if there are documents to use for retrieval
        documents = st.session_state.get("documents", [])
        if documents and len(documents) > 0:
            debug_print("Documents detected: {}", len(documents))
            # Process the query with document retrieval
            return self._process_document_query(user_message, model)
        
        # Regular text query
        debug_print("Regular text query")
        return self._process_text_query(user_message, model)
    
    def _process_image_query(self, user_message, image_path, model):
        """
        Process a query that includes an image.
        
        Args:
            user_message: The text content of the user's message
            image_path: Path to the image file
            model: The model to use for the query
            
        Returns:
            tuple: (response, sources)
        """
        debug_print("_process_image_query called with image_path: {}, model: {}", image_path, model)
        
        # Verify the image file exists
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            debug_print("{}", error_msg)
            return error_msg, []
            
        debug_print("Image file exists at: {}", image_path)
        
        # Check if the model supports multimodal inputs
        multimodal_support = self.model_manager.check_model_multimodal_support(model)
        debug_print("Multimodal support for {}: {}", model, multimodal_support)
        
        if not multimodal_support:
            # Find a better model that supports multimodal inputs
            available_models = self.model_manager.get_available_models()
            multimodal_models = [m for m in available_models if self.model_manager.check_model_multimodal_support(m)]
            debug_print("Available multimodal models: {}", multimodal_models)
            
            if multimodal_models:
                better_model = multimodal_models[0]
                debug_print("Switching to multimodal model: {} instead of {}", better_model, model)
                
                # Update the model
                model = better_model
                
                # Update the session state with the new model
                if st.session_state.get("selected_model") != better_model:
                    st.session_state["selected_model"] = better_model
                    st.session_state["model_changed_for_image"] = True
                    debug_print("Updated session state with selected_model: {}", better_model)
            else:
                error_msg = "No multimodal models available to process the image. Please install a multimodal model like gemma:2b-it or llava:7b."
                debug_print("{}", error_msg)
                return error_msg, []
        
        try:
            # Encode the image to base64
            debug_print("Encoding image to base64")
            image_base64 = self.image_processor.encode_image_to_base64(image_path)
        except Exception as e:
            error_msg = f"Error encoding image: {str(e)}"
            debug_print("{}", error_msg)
            return error_msg, []
            
        debug_print("Image encoded successfully (length: {})", len(image_base64))
        
        try:
            # Call the model with the image
            return self._call_model_with_image(user_message, image_base64, model)
        except Exception as e:
            error_msg = f"Error calling model with image: {str(e)}"
            debug_print("{}", error_msg)
            return error_msg, []
    
    def _process_document_query(self, user_message, model):
        """
        Process a query against documents.
        
        Args:
            user_message: The text content of the user's message
            model: The model to use for the query
            
        Returns:
            tuple: (response, sources)
        """
        try:
            # Load the vectorstore
            try:
                # Create the embeddings object
                from langchain.embeddings import OllamaEmbeddings
                
                # Get the embedding model from session state or use a default
                embedding_model = st.session_state.get('current_embedding_model', config.FALLBACK_EMBEDDING_MODEL)
                debug_print("Using embedding model: {}", embedding_model)
                
                # Use the embedding model that was used to create the embeddings
                embeddings = OllamaEmbeddings(model=embedding_model)
                
                # Load the vectorstore with allow_dangerous_deserialization=True
                vectorstore = FAISS.load_local(
                    config.VECTORSTORE_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Create the retriever
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": config.RETRIEVER_K}
                )
            except Exception as e:
                error_msg = f"Error loading vectorstore: {str(e)}"
                logger.error(error_msg)
                return f"Error: {error_msg}", []
            
            # Create the QA chain
            try:
                # Define the prompt template
                template = """
                    You are an AI assistant that should answer questions based on provided context.

                    Context:
                    -------------------------
                    {context}
                    -------------------------

                    Question:
                    {question}

                    Instructions:
                    - If the question is relevant to the provided context, give a fine-grained, detailed, and structured answer.
                    - If the question is irrelevant to the context, ignore the provided context and answer generically.

                    Answer:
                    """
                
                # Create the prompt
                prompt = PromptTemplate(
                    template=template.strip(),
                    input_variables=["context", "question"]
                )
                
                # Create the QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self._get_langchain_llm(model),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
            except Exception as e:
                error_msg = f"Error creating QA chain: {str(e)}"
                logger.error(error_msg)
                return f"Error: {error_msg}", []
            
            # Run the QA chain
            start_time = time.time()
            result = qa_chain({"query": user_message})
            response_time = time.time() - start_time
            
            # Extract the sources
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.append(doc.metadata["source"])
            
            # Log the response time
            logger.info(
                f"Received response from QA chain in {response_time:.2f} seconds",
                event_type="qa_response",
                model=model,
                response_time=response_time,
                source_count=len(sources)
            )
            
            # Return the response and sources
            return result["result"], list(set(sources))
        
        except Exception as e:
            # Log the error
            error_msg = f"Error processing document query: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}", []
    
    def _process_text_query(self, user_message, model):
        """
        Process a regular text query.
        
        Args:
            user_message: The text content of the user's message
            model: The model to use for the query
            
        Returns:
            tuple: (response, sources)
        """
        try:
            # Prepare the API request
            url = f"{config.OLLAMA_API_BASE}/api/generate"
            
            # Create the request payload
            payload = {
                "model": model,
                "prompt": user_message,
                "stream": False
            }
            
            # Log the request
            logger.info(
                f"Sending text query to model {model}",
                event_type="text_query",
                model=model
            )
            
            # Make the API request
            start_time = time.time()
            response = requests.post(url, json=payload)
            response_time = time.time() - start_time
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response
                result = response.json()
                
                # Log the response time
                logger.info(
                    f"Received response from model {model} in {response_time:.2f} seconds",
                    event_type="model_response",
                    model=model,
                    response_time=response_time,
                    tokens=result.get("eval_count", 0)
                )
                
                # Return the response text and empty sources
                return result["response"], []
            else:
                # Log the error
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}", []
        
        except Exception as e:
            # Log the error
            error_msg = f"Error processing text query: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}", []
    
    def _get_langchain_llm(self, model_name):
        """
        Get a LangChain LLM for the specified model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            LangChain LLM
        """
        from langchain.llms import Ollama
        
        return Ollama(model=model_name, base_url=config.OLLAMA_API_BASE)
    
    def _call_model_with_image(self, user_message, image_base64, model):
        """
        Call the model API with an image and return the response.
        
        Args:
            user_message: The text content of the user's message
            image_base64: Base64 encoded image data
            model: The model to use for the query
            
        Returns:
            tuple: (response, sources)
        """
        debug_print("Calling model with image and model: {}", model)
        
        try:
            # We don't need to display the image here as it's already displayed in the chat interface
            # The image path is stored in the session state and displayed with the user message
            
            # Prepare the API request
            url = f"{config.OLLAMA_API_BASE}/api/generate"
            
            # Create the request payload
            payload = {
                "model": model,
                "prompt": user_message,
                "stream": False,
                "images": [image_base64]
            }
            
            # Log the request (excluding the image data for brevity)
            logger.info(
                f"Sending image query to model {model}",
                event_type="image_query",
                model=model
            )
            
            debug_print("Sending request to Ollama API")
            
            # Show a spinner while processing
            with st.spinner(f"Processing image with {model}..."):
                # Make the API request
                start_time = time.time()
                response = requests.post(url, json=payload)
                response_time = time.time() - start_time
                
                debug_print("Received response from Ollama API in {:.2f} seconds", response_time)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response
                    result = response.json()
                    
                    # Log the response time
                    logger.info(
                        f"Received response from model {model} in {response_time:.2f} seconds",
                        event_type="model_response",
                        model=model,
                        response_time=response_time,
                        tokens=result.get("eval_count", 0)
                    )
                    
                    debug_print("Response successfully received and parsed")
                    
                    # Return the response text and empty sources
                    return result["response"], []
                else:
                    # Log the error
                    error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    debug_print("API error: {}", error_msg)
                    return f"Error: {error_msg}", []
        
        except Exception as e:
            # Log the error
            error_msg = f"Error calling model with image: {str(e)}"
            logger.error(error_msg)
            debug_print("Exception: {}", error_msg)
            return f"Error: {error_msg}", [] 