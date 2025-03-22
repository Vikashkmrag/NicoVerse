# ingest.py

import os
import subprocess
import argparse
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from PyPDF2 import PdfReader
from langchain.schema import Document

DATA_DIR = './documents'
VECTORSTORE_DIR = './vectorstore'

def get_available_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        available_models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line:
                model_name = line.split()[0]
                available_models.append(model_name)
        return available_models
    except Exception as e:
        print(f"Error getting model list: {str(e)}")
        return []

def load_documents(directory=DATA_DIR):
    docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            reader = PdfReader(filepath)
            text = "".join(page.extract_text() for page in reader.pages)
            docs.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

def create_embeddings(docs, model=None):
    # If no model specified, use the first available model
    if not model:
        available_models = get_available_models()
        if available_models:
            model = available_models[0]
        else:
            raise ValueError("No Ollama models available. Please install Ollama and pull a model.")
    
    print(f"Using model: {model}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    
    return [doc.metadata["source"] for doc in docs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process documents and create embeddings')
    parser.add_argument('--model', type=str, help='Ollama model to use for embeddings')
    parser.add_argument('--dir', type=str, default=DATA_DIR, help='Directory containing documents')
    args = parser.parse_args()
    
    # Create documents directory if it doesn't exist
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        print(f"Created directory: {args.dir}")
        print(f"Please add your documents to {args.dir} and run this script again.")
    else:
        docs = load_documents(args.dir)
        if docs:
            try:
                sources = create_embeddings(docs, model=args.model)
                print(f"Successfully processed {len(docs)} documents and saved embeddings to {VECTORSTORE_DIR}")
                print(f"Documents processed: {', '.join(sources)}")
            except Exception as e:
                print(f"Error creating embeddings: {str(e)}")
        else:
            print(f"No documents found in {args.dir}. Please add documents and run again.")
