#!/usr/bin/env python3
"""
Setup script for ChatInsights Python backend.
Run this ONCE before starting the server.
"""

import subprocess
import sys
import os

def run(cmd, desc):
    print(f"\n→ {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ✗ Failed: {cmd}")
        sys.exit(1)
    print(f"  ✓ Done")

print("=" * 60)
print("  ChatInsights — Backend Setup")
print("=" * 60)

# 1. Install requirements
run(
    f"{sys.executable} -m pip install -r requirements.txt",
    "Installing Python dependencies"
)

# 2. Download the sentiment model
print("\n→ Downloading sentiment model (~260MB, one-time)...")
try:
    from transformers import pipeline
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    print("  ✓ Model downloaded and cached")
except Exception as e:
    print(f"  ⚠  Model download failed: {e}")
    print("     The server will use rule-based sentiment as fallback.")

print()
print("=" * 60)
print("  Setup complete! Start the server with:")
print()
print("    python server.py")
print()
print("  The API will be available at http://localhost:8000")
print("=" * 60)
