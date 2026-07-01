#!/usr/bin/env python
"""
Simplified demo script for Local RAG System
Tests ingestion and Q&A without PDF requirements
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def demo_rag_system():
    """Run a demo of the RAG system"""
    from rag_system_simple import SimpleRAGSystem
    
    print("="*70)
    print("🚀 Local RAG-Based Document Q&A System - DEMO")
    print("="*70)
    
    # Initialize system
    print("\n📦 Initializing RAG system...")
    try:
        rag_system = SimpleRAGSystem(
            vector_store_path="vector_store",
            embedding_model="nomic-embed-text",
            query_model="llama2"
        )
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing RAG system: {e}")
        return False
    
    # Load sample documents
    print("\n📄 Loading sample documents...")
    documents_folder = "documents"
    if os.path.exists(documents_folder):
        # Manually process text files as demo
        txt_files = list(Path(documents_folder).glob("*.txt"))
        if txt_files:
            for txt_file in txt_files:
                print(f"  Processing: {txt_file.name}")
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Chunk the text
                    chunks = rag_system.chunk_text(text, chunk_size=500, chunk_overlap=50)
                    print(f"    Created {len(chunks)} chunks")
                    
                    # Generate embeddings
                    for i, chunk in enumerate(chunks):
                        try:
                            embedding = rag_system.get_embedding(chunk)
                            chunk_id = f"{txt_file.stem}_chunk_{i}"
                            
                            rag_system.embeddings[chunk_id] = embedding
                            rag_system.metadata[chunk_id] = {
                                "source": txt_file.stem,
                                "chunk_index": i,
                                "document_path": str(txt_file),
                                "text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                            }
                            
                            if (i + 1) % 3 == 0:
                                print(f"    ✓ Processed {i + 1} chunks")
                        except Exception as e:
                            print(f"    ✗ Error on chunk {i}: {e}")
                            continue
                    
                    # Save
                    rag_system._save_embeddings()
                    rag_system._save_metadata()
                    print(f"  ✓ Saved {txt_file.name}")
                except Exception as e:
                    print(f"  ✗ Error processing {txt_file.name}: {e}")
        else:
            print("  No .txt files found in documents folder")
    else:
        print(f"  Documents folder not found: {documents_folder}")
        return False
    
    # Show stats
    stats = rag_system.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   Query model: {stats['query_model']}")
    
    if stats['total_chunks'] == 0:
        print("\n⚠ No documents ingested. Add PDF files or .txt files to documents/")
        return False
    
    # Interactive Q&A
    print("\n" + "="*70)
    print("💬 Ask Questions About Your Documents")
    print("="*70)
    print("Type 'exit' to quit, 'stats' for stats\n")
    
    sample_questions = [
        "What are the key features of Python?",
        "Explain object-oriented programming",
        "What are the best practices in Python?"
    ]
    
    print("📝 Sample questions you can ask:")
    for q in sample_questions:
        print(f"   - {q}")
    
    print()
    
    while True:
        try:
            query = input("❓ Ask a question: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("\n👋 Thank you for using Local RAG System!")
                break
            
            if query.lower() == 'stats':
                stats = rag_system.get_collection_stats()
                print(f"\n📊 Collection Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                print()
                continue
            
            print("\n🔄 Searching documents and generating answer...")
            answer = rag_system.answer_query(query, num_context_chunks=3)
            print(f"\n💬 Answer:\n{answer}\n")
            print("-"*70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = demo_rag_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
