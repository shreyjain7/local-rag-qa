#!/usr/bin/env python
"""Quick test of Ollama integration - Fixed"""
import sys

try:
    import ollama
    print("Testing Ollama connection...")
    
    response = ollama.list()
    # Handle both dict and object responses
    if hasattr(response, 'models'):
        models = response.models
    else:
        models = response.get("models", [])
    
    model_list = [m.model if hasattr(m, 'model') else m.get("name", "Unknown") for m in models]
    
    print(f"✓ Ollama is running!")
    print(f"Available models ({len(model_list)}): {model_list}")
    
    # Check for required models
    has_embed = any('nomic-embed-text' in str(m) for m in model_list)
    has_llama = any('llama' in str(m) for m in model_list)
    
    if has_embed:
        print("✓ nomic-embed-text found")
    else:
        print("✗ nomic-embed-text not found")
    
    if has_llama:
        print("✓ Llama model found")
    else:
        print("✗ Llama model not found")
    
    if has_embed and has_llama:
        print("\n✓✓✓ SYSTEM IS READY! ✓✓✓")
        sys.exit(0)
    else:
        print("\n⚠ Waiting for models to finish downloading...")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure Ollama is running: ollama serve")
    sys.exit(1)
