#!/usr/bin/env python3
"""
Quick Start Guide for Local RAG System
Interactive setup and testing
"""

import os
import sys
from pathlib import Path
from rag_system import LocalRAGSystem


def print_banner():
    print("\n" + "="*70)
    print("  Local RAG-Based Document Q&A System (Ollama)")
    print("  " + "="*66)
    print()


def test_ollama_connection():
    """Test if Ollama is accessible"""
    print("🔍 Testing Ollama connection...")
    try:
        import ollama
        response = ollama.list()
        print("✓ Ollama is running")
        
        models = [m['name'] for m in response['models']]
        print(f"  Available models: {', '.join(models)}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def setup_system():
    """Interactive setup"""
    print("\n📋 Setup Configuration")
    print("-" * 70)
    
    # Check requirements
    if not test_ollama_connection():
        return None
    
    # Initialize RAG system
    print("\n🚀 Initializing RAG System...")
    rag_system = LocalRAGSystem(
        vector_store_path="vector_store",
        embedding_model="nomic-embed-text",
        query_model="llama2"
    )
    print("✓ RAG System initialized")
    
    return rag_system


def load_sample_documents(rag_system):
    """Load sample documents"""
    print("\n📄 Loading Sample Documents...")
    print("-" * 70)
    
    documents_folder = "documents"
    if not Path(documents_folder).exists():
        Path(documents_folder).mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(Path(documents_folder).glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in 'documents' folder")
        print("   Please add PDF files to continue with examples")
        return False
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    results = rag_system.load_documents_from_folder(documents_folder)
    
    for filename, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")
    
    return any(results.values())


def interactive_qa(rag_system):
    """Interactive Q&A loop"""
    print("\n" + "="*70)
    print("  Interactive Q&A Session")
    print("="*70)
    print("\nCommands:")
    print("  'quit'   - Exit the program")
    print("  'stats'  - Show collection statistics")
    print("  'reload' - Reload documents from folder")
    print("\n" + "-"*70 + "\n")
    
    stats = rag_system.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("⚠️  No documents in the vector store!")
        print("   Please add PDF files to the 'documents' folder and try again")
        return
    
    while True:
        try:
            query = input("❓ Ask a question: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("\n👋 Thank you for using Local RAG System!")
                break
            
            if query.lower() == 'stats':
                stats = rag_system.get_collection_stats()
                print(f"\n📊 Collection Statistics:")
                print(f"   Total chunks: {stats['total_chunks']}")
                print(f"   Embedding model: {stats['embedding_model']}")
                print(f"   Query model: {stats['query_model']}\n")
                continue
            
            if query.lower() == 'reload':
                print("Reloading documents...")
                rag_system.load_documents_from_folder("documents")
                stats = rag_system.get_collection_stats()
                print(f"✓ Reloaded. Total chunks: {stats['total_chunks']}\n")
                continue
            
            print("\n🔄 Generating answer...\n")
            answer = rag_system.answer_query(query, num_context_chunks=3)
            print(f"💬 Answer:\n{answer}\n")
            print("-" * 70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main function"""
    print_banner()
    
    # Setup
    rag_system = setup_system()
    if not rag_system:
        print("\n❌ Setup failed. Please fix the issues above.")
        sys.exit(1)
    
    # Show stats
    stats = rag_system.get_collection_stats()
    print(f"\n📊 Current Status:")
    print(f"   Chunks in vector store: {stats['total_chunks']}")
    
    # Load documents
    if stats['total_chunks'] == 0:
        print("\n📚 No documents found. Attempting to load from 'documents' folder...")
        has_docs = load_sample_documents(rag_system)
        if not has_docs:
            print("\n⚠️  No documents to process.")
            print("   1. Add PDF files to the 'documents' folder")
            print("   2. Run this script again")
            sys.exit(1)
    else:
        print(f"\n✓ Found {stats['total_chunks']} chunks in vector store")
    
    # Interactive session
    interactive_qa(rag_system)


if __name__ == "__main__":
    main()
