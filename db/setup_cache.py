#!/usr/bin/env python3
"""
Setup script to create cache directory structure.
Run this once before using the preprocessing system.
"""
from pathlib import Path
import config

def setup_cache_dirs():
    """Create cache directory structure."""
    print("Setting up cache directories...")
    
    # Create main cache directory
    Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {config.CACHE_DIR}")
    
    # Create subdirectories
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {config.EMBEDDINGS_DIR}")
    
    config.TAGGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {config.TAGGRAMS_DIR}")
    
    config.PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {config.PROJECTIONS_DIR}")
    
    print("\nCache directory structure ready!")
    print(f"\nDirectory tree:")
    print(f"{config.CACHE_DIR}/")
    print(f"  ├── embeddings/")
    print(f"  ├── taggrams/")
    print(f"  ├── projections/")
    print(f"  └── audio_cache.db (will be created on first use)")

if __name__ == '__main__':
    setup_cache_dirs()

