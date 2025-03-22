# Nico Verse

A sleek and sophisticated platform, inspired by Nico Robin’s enigmatic intellect and unyielding pursuit of knowledge.

## Features

- Chat with various AI models through Ollama
- Upload and query PDF and text documents
- Process images with multimodal models
- Save and manage conversation threads
- Automatic model selection based on content type
- Detailed logging and analytics

## Modular Structure

The application has been organized into a modular structure for better maintainability:

```
document_retrieval/
├── app.py                  # Old Main application entry point
├── app_new.py              # New modular application entry point
├── config.py               # Global configuration settings
├── data/                   # Data storage directory
│   ├── database/           # SQLite database for threads
│   ├── documents/          # Uploaded documents
│   ├── images/             # Uploaded images
│   ├── logs/               # Application logs
│   └── vectorstore/        # Document embeddings
├── modules/                # Modular components
│   ├── data/               # Data processing modules
│   │   ├── document_processor.py  # Document handling
│   │   └── image_processor.py     # Image handling
│   ├── models/             # Model interaction modules
│   │   ├── model_manager.py       # Model management
│   │   └── query_handler.py       # Query processing
│   ├── ui/                 # User interface modules
│   │   ├── chat_interface.py      # Chat UI
│   │   └── sidebar.py             # Sidebar UI
│   └── utils/              # Utility modules
│       ├── config.py              # Configuration
│       └── logger.py              # Logging
└── sql_db.py               # Database operations
```


## Getting Started

1. Ensure you have Ollama installed and running on your system.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app_new.py
   ```

## Usage

1. Start a new chat or select an existing thread from the sidebar.
2. Upload documents or images using the file uploader.
3. Type your message and click "Send" to interact with the AI model.
4. The application will automatically select the appropriate model based on your content.

## Configuration

The application's behavior can be customized by modifying the settings in `config.py`. Key settings include:

- `DEFAULT_MODEL`: The default model to use for text queries.
- `MULTIMODAL_MODELS`: List of models that support image processing.
- `EMBEDDING_MODEL_NAME`: The model to use for document embeddings.
- `SUPPORTED_FILE_TYPES`: List of supported document file types.
- `SUPPORTED_IMAGE_TYPES`: List of supported image file types.

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- PyTorch
- Ollama (running locally or on a remote server)
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd NicoVerse
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Install Ollama from https://ollama.ai/

4. Pull models using Ollama:
```
ollama pull llama3:8b
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b
```
Or use the built-in model management in the app to pull models.

## Usage

1. Start the Streamlit application:
```
streamlit run app_new.py
```

2. Upload your documents:
   - Use the "Upload files" option to upload PDF or text files
   - Or specify a directory path containing your documents

3. Manage and select models:
   - View available models
   - Load up to 3 models at once
   - Pull new models directly from the app
   - Select an active model for queries

4. Start asking questions about your documents

5. Create new threads or switch between existing threads
   - Each thread remembers which model was used
   - Documents used in each thread are tracked

## Analytics Dashboard

The application includes a comprehensive analytics dashboard that visualizes usage metrics and logs:

1. Start the dashboard:
```
streamlit run run_dashboard.py
```

2. Or click the "Open Analytics Dashboard" button in the main application

The dashboard provides:
- Overview of application usage
- Model usage statistics
- Document processing metrics
- Thread activity visualization
- Error tracking and monitoring
- Filtering by date range, event type, and component

## Logging System

The application includes a structured logging system that records:
- Model usage (queries, response times, success/failure)
- Document processing (count, sources, processing time)
- Thread activity (creation, updates, selection)
- User actions and system events
- Errors and exceptions

Logs are stored in both human-readable and JSON formats in the `logs` directory:
- `app.log`: Human-readable logs
- `app.json.log`: Structured JSON logs for dashboard visualization

## Command-line Document Processing

You can also process documents from the command line:

```
python ingest.py --model deepseek-r1:8b --dir ./my_documents
```



## Troubleshooting

- If you encounter issues with Ollama models, ensure Ollama is running locally
- For document processing issues, check that your PDFs are text-based (not scanned images)
- If the application fails to start, verify all dependencies are installed correctly
- If you get connection errors, make sure Ollama is running with `ollama serve`
- Check the logs directory for detailed error information

