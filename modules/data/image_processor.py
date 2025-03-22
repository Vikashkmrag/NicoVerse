import os
import base64
import uuid
import config
from modules.utils.logger import get_logger
from modules.utils.debug import debug_print

logger = get_logger("image_processor")

class ImageProcessor:
    """
    Handles image operations, including saving and encoding images.
    """
    
    def __init__(self):
        self.images_dir = config.IMAGES_DIR
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
    
    def save_uploaded_image(self, uploaded_file):
        """
        Save an uploaded image file to the images directory.
        
        Args:
            uploaded_file: A Streamlit UploadedFile object
            
        Returns:
            str: Path to the saved image file
        """
        try:
            debug_print("save_uploaded_image called with file: {}", uploaded_file.name)
            # Create a unique filename
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(self.images_dir, unique_filename)
            debug_print("Generated unique filename: {}", unique_filename)
            debug_print("Full file path: {}", file_path)
            
            # Save the file
            with open(file_path, 'wb') as f:
                file_content = uploaded_file.getvalue()
                debug_print("File content size: {} bytes", len(file_content))
                f.write(file_content)
                
            debug_print("File saved successfully to: {}", file_path)
            logger.info(f"Saved uploaded image to {file_path}")
            return file_path
        except Exception as e:
            debug_print("Error saving uploaded image: {}", str(e))
            logger.error(f"Error saving uploaded image: {str(e)}")
            raise
    
    def encode_image_to_base64(self, image_file):
        """
        Encode an image file to base64 for use with multimodal models.
        
        Args:
            image_file: A file-like object containing the image data
            
        Returns:
            str: Base64 encoded string of the image
        """
        try:
            debug_print("encode_image_to_base64 called with image_file type: {}", type(image_file))
            
            # If it's a streamlit UploadedFile, read the bytes
            if hasattr(image_file, 'getvalue'):
                debug_print("Image file is a Streamlit UploadedFile")
                image_bytes = image_file.getvalue()
            # If it's a file path, open and read the file
            elif isinstance(image_file, str) and os.path.exists(image_file):
                debug_print("Image file is a file path: {}", image_file)
                with open(image_file, 'rb') as f:
                    image_bytes = f.read()
            # If it's already bytes, use it directly
            elif isinstance(image_file, bytes):
                debug_print("Image file is already bytes")
                image_bytes = image_file
            else:
                error_msg = f"Unsupported image file type: {type(image_file)}"
                debug_print("{}", error_msg)
                raise ValueError(error_msg)
                
            debug_print("Image bytes size: {} bytes", len(image_bytes))
            
            # Encode to base64
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            debug_print("Base64 encoded string length: {} chars", len(base64_encoded))
            logger.info(f"Successfully encoded image to base64 ({len(base64_encoded)} chars)")
            return base64_encoded
        except Exception as e:
            debug_print("Error encoding image to base64: {}", str(e))
            logger.error(f"Error encoding image to base64: {str(e)}")
            raise 