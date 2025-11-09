"""
Configuration for audio embedding preprocessing and caching system.
"""
from pathlib import Path

# Directory paths
CACHE_DIR = Path('./cache')
DATABASE_PATH = './cache/audio_cache.db'
AUDIO_DIR = './audio/'

# Cache subdirectories
EMBEDDINGS_DIR = Path(CACHE_DIR) / 'embeddings'
TAGGRAMS_DIR = Path(CACHE_DIR) / 'taggrams'
PROJECTIONS_DIR = Path(CACHE_DIR) / 'projections'

# Model configurations
MODELS = ['musicnn', 'vgg']
DATASETS = ['msd', 'mtat']

# Model weight paths
MODEL_WEIGHTS = {
    'musicnn': {
        'msd': './ML/pesos/msd/musicnn.pth',
        'mtat': './ML/pesos/mtat/musicnn.pth'
    },
    'vgg': {
        'msd': './ML/pesos/msd/vgg.pth',
        'mtat': './ML/pesos/mtat/vgg.pth'
    }
}

# Preprocessing settings
# SEGMENT_SIZE is deprecated - now processing full songs (1 embedding per song)
SAMPLE_RATE = 16000  # Hz

# Projection methods
CACHED_PROJECTION_METHODS = ['pca']  # Only PCA is pre-cached
ON_DEMAND_PROJECTION_METHODS = ['tsne', 'umap']  # Computed on request

