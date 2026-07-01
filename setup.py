"""Setup script for Local RAG System"""

import subprocess
import sys
import os
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        import ollama
        print("✓ Ollama Python client is installed")
        return True
    except ImportError:
        print("✗ Ollama not installed. Visit https://ollama.ai to install")
        return False


def check_ollama_models():
    """Check if required models are available"""
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]
        
        required_models = ['nomic-embed-text', 'llama2']
        missing_models = [m for m in required_models if not any(m in name for name in model_names)]
        
        if missing_models:
            print(f"\nMissing models: {', '.join(missing_models)}")
            print("Pull them with:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            return False
        
        print("✓ Required Ollama models are available")
        return True
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False


def create_sample_document():
    """Create a sample document for testing"""
    sample_path = Path("documents/sample.txt")
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_content = """# Sample Document for RAG Testing

This is a sample document that demonstrates how the Local RAG system works.

## Section 1: Overview
The Retrieval-Augmented Generation (RAG) system combines document retrieval with language models
to provide accurate, contextual answers based on your documents.

## Section 2: Features
- PDF document ingestion
- Local embedding generation with nomic-embed-text
- Vector-based semantic search
- LLM-powered answer generation using Llama 3
- No internet or API keys required
- Fully offline operation

## Section 3: How It Works
1. Documents are uploaded and converted to text
2. Text is split into manageable chunks
3. Each chunk is embedded using nomic-embed-text
4. Embeddings are stored in a vector database
5. User queries are embedded and compared against stored embeddings
6. Top matching chunks are retrieved
7. Llama 3 generates answers based on retrieved context

## Section 4: Usage
Simply place your PDF documents in the 'documents' folder and run the system.
The RAG pipeline will automatically ingest them and make them queryable.
"""
        with open(sample_path, 'w') as f:
            f.write(sample_content)
        print(f"✓ Sample document created at {sample_path}")


def main():
    """Run setup"""
    print("="*60)
    print("Local RAG-Based Document Q&A System - Setup")
    print("="*60 + "\n")
    
    # Check Ollama
    if not check_ollama_installed():
        print("\nSetup cannot continue without Ollama")
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check models
    if not check_ollama_models():
        print("\nSetup incomplete - please pull the required models")
        return False
    
    # Create sample document
    create_sample_document()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your PDF documents to the 'documents' folder")
    print("2. Run: python rag_system.py")
    print("3. Ask questions about your documents!")
    print("\nExample queries:")
    print("  - What does this document say about...?")
    print("  - Summarize the key points about...")
    print("  - Explain the concept of...")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
