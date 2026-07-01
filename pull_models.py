#!/usr/bin/env python
"""Script to pull required Ollama models"""
import subprocess
import sys
from pathlib import Path

ollama_exe = Path("C:\\Users\\Shrey Jain\\AppData\\Local\\Programs\\Ollama\\ollama.exe")

if not ollama_exe.exists():
    print(f"❌ Ollama not found at {ollama_exe}")
    sys.exit(1)

models_to_pull = ["nomic-embed-text", "llama2"]

for model in models_to_pull:
    print(f"\n📥 Pulling {model}...")
    try:
        result = subprocess.run(
            [str(ollama_exe), "pull", model],
            capture_output=False,
            timeout=600
        )
        if result.returncode == 0:
            print(f"✓ Successfully pulled {model}")
        else:
            print(f"✗ Failed to pull {model}")
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout pulling {model}")
    except Exception as e:
        print(f"✗ Error pulling {model}: {e}")

print("\n" + "="*60)
print("Model download process complete!")
print("="*60)
