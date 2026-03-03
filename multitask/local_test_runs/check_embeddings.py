#!/usr/bin/env python3
"""
Check the dimensions of openai_embeddings.npy and print the first couple rows.
"""

import numpy as np
from pathlib import Path

def check_embeddings():
    """Load and inspect the openai_embeddings.npy file."""
    # Get the path to the embeddings file
    embeddings_path = Path(__file__).parent / "openai_embeddings.npy"
    
    if not embeddings_path.exists():
        print(f"File not found: {embeddings_path}")
        return
    
    # Load the embeddings
    embeddings = np.load(embeddings_path)
    
    print(f"File: {embeddings_path.name}")
    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"\nFirst 2 rows:")
    print(embeddings[:2])

if __name__ == "__main__":
    check_embeddings()
