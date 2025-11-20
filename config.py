"""
Configuration for audio embedding preprocessing and caching system.
"""
from pathlib import Path
import os

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory paths (absolute paths)
CACHE_DIR = str(PROJECT_ROOT / 'db')
DATABASE_PATH = str(PROJECT_ROOT / 'db' / 'audio_cache.db')
AUDIO_DIR = str(PROJECT_ROOT / 'audio')
CSV_PATH = str(PROJECT_ROOT / 'audio' / 'selected_songs.csv')
SONGS_PATH = str(PROJECT_ROOT / 'audio')
MODEL_DIR = str(PROJECT_ROOT / 'ML' / 'models')

# Model configurations
CONV_MODELS = ['musicnn', 'vgg']
DATASETS = ['msd', 'mtat']

TRANF_MODELS = ['whisper','mert']
MODEL_SIZES = ['base', '95m']

# Model weight paths (absolute paths)
MODEL_WEIGHTS = {
    'musicnn': {
        'msd': str(PROJECT_ROOT / 'ML' / 'pesos' / 'msd' / 'musicnn.pth'),
        'mtat': str(PROJECT_ROOT / 'ML' / 'pesos' / 'mtat' / 'musicnn.pth')
    },
    'vgg': {
        'msd': str(PROJECT_ROOT / 'ML' / 'pesos' / 'msd' / 'vgg.pth'),
        'mtat': str(PROJECT_ROOT / 'ML' / 'pesos' / 'mtat' / 'vgg.pth')
    },
    'whisper': {
        'base': 'base'
    },
    'mert': {
        '95m': 'm-a-p/MERT-v1-95M'
    }
}

# Preprocessing settings
# SEGMENT_SIZE is deprecated - now processing full songs (1 embedding per song)
SAMPLE_RATE = 16000  # Hz

# Projection methods
ON_DEMAND_PROJECTION_METHODS = ['tsne', 'umap']  # Computed on request

