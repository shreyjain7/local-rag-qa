"""
Test script for Local RAG System
Verify all components are working correctly
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = {
        'ollama': 'ollama',
        'chromadb': 'chromadb',
        'pypdf': 'pypdf',
        'langchain': 'langchain',
    }
    
    missing = []
    for name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All imports successful\n")
    return True


def test_ollama():
    """Test Ollama connection and models"""
    print("Testing Ollama connection...")
    
    try:
        import ollama
        
        # Check if Ollama is running
        try:
            response = ollama.list()
            print("  ✓ Ollama service is running")
        except Exception as e:
            print(f"  ✗ Cannot connect to Ollama: {e}")
            print("    Start Ollama with: ollama serve")
            return False
        
        # Check for required models
        models = {m['name']: m for m in response['models']}
        model_names = list(models.keys())
        
        required_models = ['nomic-embed-text', 'llama2']
        for model in required_models:
            found = any(model in name for name in model_names)
            if found:
                print(f"  ✓ {model} is available")
            else:
                print(f"  ✗ {model} not found")
                print(f"    Pull with: ollama pull {model}")
                return False
        
        print("\n✓ Ollama tests passed\n")
        return True
    
    except Exception as e:
        print(f"  ✗ Error testing Ollama: {e}")
        return False


def test_rag_system():
    """Test RAG system initialization"""
    print("Testing RAG System...")
    
    try:
        from rag_system import LocalRAGSystem
        
        rag = LocalRAGSystem(
            vector_store_path="test_vector_store",
            embedding_model="nomic-embed-text",
            query_model="llama2"
        )
        print("  ✓ RAG system initialized")
        
        stats = rag.get_collection_stats()
        print(f"  ✓ Collection accessible (chunks: {stats['total_chunks']})")
        
        print("\n✓ RAG System tests passed\n")
        return True
    
    except Exception as e:
        print(f"  ✗ Error testing RAG system: {e}")
        return False


def test_documents_folder():
    """Test documents folder setup"""
    print("Testing documents folder...")
    
    docs_folder = Path("documents")
    if not docs_folder.exists():
        docs_folder.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {docs_folder} folder")
    else:
        print(f"  ✓ {docs_folder} folder exists")
    
    pdf_files = list(docs_folder.glob("*.pdf"))
    if pdf_files:
        print(f"  ✓ Found {len(pdf_files)} PDF file(s)")
    else:
        print(f"  ⚠ No PDF files found in {docs_folder}")
        print("    Add PDF files here to test the system")
    
    print()
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Local RAG System - Comprehensive Test")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Ollama", test_ollama),
        ("RAG System", test_rag_system),
        ("Documents Folder", test_documents_folder),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}\n")
            results[test_name] = False
    
    # Summary
    print("="*60)
    print("  Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Add PDF files to the 'documents' folder")
        print("  2. Run: python quickstart.py")
        print("  3. Or run: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
