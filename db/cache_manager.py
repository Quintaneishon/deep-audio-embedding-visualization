"""
Cache manager for loading and saving embeddings, taggrams, and projections.
"""
import numpy as np
from pathlib import Path
import config


def _ensure_cache_dirs():
    """Ensure all cache directories exist."""
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    config.TAGGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_filename(filename, model, dataset, cache_type, method=None):
    """
    Generate cache filename based on parameters.
    
    Args:
        filename: Audio filename (e.g., '1.mp3')
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
        cache_type: Type of cache ('embedding', 'taggram', 'projection')
        method: Projection method (only for projections, e.g., 'pca')
    
    Returns:
        Path object for the cache file
    """
    # Remove extension from filename
    base_name = Path(filename).stem
    
    if cache_type == 'embedding':
        cache_filename = f"{model}_{dataset}_{base_name}_emb.npy"
        return config.EMBEDDINGS_DIR / cache_filename
    elif cache_type == 'taggram':
        cache_filename = f"{model}_{dataset}_{base_name}_tag.npy"
        return config.TAGGRAMS_DIR / cache_filename
    elif cache_type == 'projection':
        if method is None:
            raise ValueError("Method must be specified for projection cache")
        cache_filename = f"{model}_{dataset}_{base_name}_{method}.npy"
        return config.PROJECTIONS_DIR / cache_filename
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")


def get_cached_embedding(filename, model, dataset):
    """
    Load cached embedding for a track.
    
    Args:
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
    
    Returns:
        numpy array or None if not cached
    """
    cache_path = _get_cache_filename(filename, model, dataset, 'embedding')
    if cache_path.exists():
        return np.load(cache_path)
    return None


def get_cached_taggram(filename, model, dataset):
    """
    Load cached taggram for a track.
    
    Args:
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
    
    Returns:
        numpy array or None if not cached
    """
    cache_path = _get_cache_filename(filename, model, dataset, 'taggram')
    if cache_path.exists():
        return np.load(cache_path)
    return None


def get_cached_projection(filename, model, dataset, method):
    """
    Load cached projection for a track.
    
    Args:
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
        method: Projection method ('pca', 'tsne', 'umap')
    
    Returns:
        numpy array or None if not cached
    """
    cache_path = _get_cache_filename(filename, model, dataset, 'projection', method)
    if cache_path.exists():
        return np.load(cache_path)
    return None


def save_to_cache(data, cache_type, filename, model, dataset, method=None):
    """
    Save data to cache.
    
    Args:
        data: numpy array to save
        cache_type: Type of cache ('embedding', 'taggram', 'projection')
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
        method: Projection method (only for projections)
    
    Returns:
        Path to the saved file
    """
    _ensure_cache_dirs()
    cache_path = _get_cache_filename(filename, model, dataset, cache_type, method)
    np.save(cache_path, data)
    return str(cache_path)


def get_cache_path(filename, model, dataset, cache_type, method=None):
    """
    Get the cache path without loading the file.
    Useful for database storage.
    
    Args:
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
        cache_type: Type of cache ('embedding', 'taggram', 'projection')
        method: Projection method (only for projections)
    
    Returns:
        String path to the cache file
    """
    return str(_get_cache_filename(filename, model, dataset, cache_type, method))


def is_cached(filename, model, dataset, cache_type, method=None):
    """
    Check if data is cached without loading it.
    
    Args:
        filename: Audio filename
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
        cache_type: Type of cache ('embedding', 'taggram', 'projection')
        method: Projection method (only for projections)
    
    Returns:
        Boolean indicating if data is cached
    """
    cache_path = _get_cache_filename(filename, model, dataset, cache_type, method)
    return cache_path.exists()

