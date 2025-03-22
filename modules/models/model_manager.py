import subprocess
import datetime
import requests
import time
import streamlit as st
import config
from modules.utils.logger import get_logger
from modules.utils.debug import debug_print

logger = get_logger("model_manager")

class ModelManager:
    """
    Manages Ollama models, including checking for embedding support and multimodal capabilities.
    """
    
    def __init__(self):
        # Initialize session state for model information if not exists
        if 'model_sizes' not in st.session_state:
            st.session_state['model_sizes'] = {}
        if 'embedding_models_support' not in st.session_state:
            st.session_state.embedding_models_support = {}
    
    def get_available_models(self):
        """
        Retrieve available Ollama models and their sizes.
        Filter out models larger than 12GB.
        Also check and store embedding support information.
        
        Returns:
            list: List of available model names
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
            from sql_db import ThreadDB
            db = ThreadDB()
            cached_embedding_info = db.get_all_model_embedding_support()
            
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
                        supports_embeddings = self.check_model_embeddings_support(model)
                        st.session_state.embedding_models_support[model] = supports_embeddings
                        # This will also save to the database inside check_model_embeddings_support
            
            return models
        
        except Exception as e:
            logger.error(f"Error managing Ollama models: {str(e)}")
            return []
    
    def check_model_embeddings_support(self, model_name):
        """
        Check if a model supports embeddings by first checking the database,
        and if not found or outdated, making a test API call.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if embeddings are supported, False otherwise
        """
        try:
            # First, check if we have this information in the database
            from sql_db import ThreadDB
            db = ThreadDB()
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
    
    def get_best_embedding_model(self, user_selected_model=None):
        """
        Determine the best available model for embeddings.
        First checks if the current model supports embeddings,
        then falls back to other available models.
        
        Args:
            user_selected_model (str, optional): User's selected model
            
        Returns:
            str: Name of the best embedding model, or None if none available
        """
        # If a specific model is provided and it supports embeddings, use it
        if user_selected_model and self.check_model_embeddings_support(user_selected_model):
            logger.info(f"Using user selected model for embeddings: {user_selected_model}")
            # Update the UI to show which model is being used for embeddings
            if 'current_embedding_model' not in st.session_state:
                st.session_state['current_embedding_model'] = user_selected_model
            return user_selected_model
        
        # Try the fallback model from config
        fallback_model = config.FALLBACK_EMBEDDING_MODEL
        if self.check_model_embeddings_support(fallback_model):
            logger.info(f"Using fallback model for embeddings: {fallback_model}")
            # Update the UI to show which model is being used for embeddings
            if 'current_embedding_model' not in st.session_state:
                st.session_state['current_embedding_model'] = fallback_model
            return fallback_model
        
        # Get all available models under 12GB
        available_models = self.get_available_models()
        
        # Check if we have any models with embedding support in session state
        embedding_models = [model for model in available_models 
                           if model in st.session_state.embedding_models_support 
                           and st.session_state.embedding_models_support[model]]
        
        if embedding_models:
            # Use the first available model that supports embeddings
            selected_model = embedding_models[0]
            logger.info(f"Selected embedding model from available models: {selected_model}")
            # Update the UI to show which model is being used for embeddings
            if 'current_embedding_model' not in st.session_state:
                st.session_state['current_embedding_model'] = selected_model
            return selected_model
        
        # If no models in session state support embeddings, test each available model
        for model in available_models:
            if self.check_model_embeddings_support(model):
                logger.info(f"Found model with embedding support after testing: {model}")
                # Update the UI to show which model is being used for embeddings
                if 'current_embedding_model' not in st.session_state:
                    st.session_state['current_embedding_model'] = model
                return model
        
        # If no models support embeddings, return None
        logger.error("No models available that support embeddings")
        return None
    
    def check_model_multimodal_support(self, model_name):
        """
        Check if a model supports multimodal inputs (images).
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if the model supports multimodal inputs, False otherwise
        """
        debug_print("Checking multimodal support for {}", model_name)
        
        # Check if the model is in the list of known multimodal models
        if model_name in config.MULTIMODAL_MODELS:
            debug_print("Model {} is in the list of known multimodal models", model_name)
            return True
        
        # Check if it's a Gemma model with 'it' in the name (instruction tuned)
        if "gemma" in model_name.lower() and "it" in model_name.lower():
            debug_print("Model {} is a Gemma model, assuming multimodal support", model_name)
            return True
            
        # Check if it's a LLaVA model
        if "llava" in model_name.lower():
            debug_print("Model {} is a LLaVA model, assuming multimodal support", model_name)
            return True
            
        # Check if it's a Bakllava model
        if "bakllava" in model_name.lower():
            debug_print("Model {} is a Bakllava model, assuming multimodal support", model_name)
            return True
            
        # Check if it's a Moondream model
        if "moondream" in model_name.lower():
            debug_print("Model {} is a Moondream model, assuming multimodal support", model_name)
            return True
            
        # Default to False for unknown models
        return False
    
    def reset_model_context(self, model_name):
        """
        Reset a model's context using the Ollama API.
        This is a streamlined implementation focused on speed.
        
        Args:
            model_name (str): Name of the model to reset
            
        Returns:
            bool: True if successful, False otherwise
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
                        timeout=100  # Ultra short timeout - we don't need the response
                    )
                    success = True
                # except requests.exceptions.Timeout:
                #     # Timeout is expected and fine - the request was sent
                #     success = True
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