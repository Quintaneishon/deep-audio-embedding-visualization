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

TRANF_MODELS = ['whisper', 'mert', 'whisper_contrastive', 'vggish', 'lightweight_adapter_v1','lightweight_adapter_v2']
MODEL_SIZES = ['base', '95m', 'base', 'pretrained', 'base', 'base']

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
    },
    'whisper_contrastive': {
        'base': str(PROJECT_ROOT / 'ML' / 'checkpoints' / 'whisper_contrastive_20251128_085448/best_model.pth')
    },
    'vggish': {
        'pretrained': 'vggish-10086976.pth'
    },
    'lightweight_adapter_v1': {
        'base': str(PROJECT_ROOT / 'ML' / 'checkpoints' / 'lightweight_adapter_20251217_142925/best_model.pth')
    },
    'lightweight_adapter_v2': {
        'base': str(PROJECT_ROOT / 'ML' / 'checkpoints' / 'lightweight_adapter_20251218_050912/best_model.pth')
    }
}

# Preprocessing settings
# SEGMENT_SIZE is deprecated - now processing full songs (1 embedding per song)
SAMPLE_RATE = 16000  # Hz

# Projection methods
ON_DEMAND_PROJECTION_METHODS = ['tsne', 'umap']  # Computed on request

# ============================================================================
# Contrastive Learning Configuration
# ============================================================================

# Dataset paths for MTG-Jamendo
MTG_JAMENDO_ROOT = '/home/ar/Data/Ajitzi/mtg-jamendo-dataset'
MTG_JAMENDO_AUDIO_DIR = str(Path(MTG_JAMENDO_ROOT) / 'songs')
MTG_JAMENDO_SPLITS_DIR = str(Path(MTG_JAMENDO_ROOT) / 'data' / 'splits' / 'split-0')

# Training hyperparameters
CONTRASTIVE_TRAINING = {
    # Model configuration
    'model_name': 'base',  # Whisper model size: 'tiny', 'base', 'small', 'medium'
    'projection_dim': 128,  # Dimension of projection head output
    
    # Training parameters
    'batch_size': 16,  # Batch size (adjust based on GPU memory)
    'num_epochs': 50,  # Maximum number of epochs
    'learning_rate': 1e-3,  # Initial learning rate
    'weight_decay': 1e-4,  # L2 regularization
    
    # Loss function
    'temperature': 0.07,  # Temperature for contrastive loss (0.07 is standard)
    
    # Data loading
    'num_workers': 4,  # Number of parallel data loading workers
    'audio_duration': 30.0,  # Audio duration in seconds
    'sample_rate': 16000,  # Audio sample rate (Hz)
    'balanced_sampling': False,  # Use weighted sampling for class balance
    
    # Optimization
    'scheduler': 'plateau',  # LR scheduler: 'plateau', 'cosine', 'step', 'none'
    'early_stopping_patience': 10,  # Epochs to wait before early stopping
    
    # Checkpointing
    'checkpoint_dir': str(PROJECT_ROOT / 'ML' / 'checkpoints'),
    'log_dir': str(PROJECT_ROOT / 'ML' / 'logs'),
    'save_frequency': 1,  # Save checkpoint every N epochs
}

# Model checkpoint path for trained contrastive model
WHISPER_CONTRASTIVE_WEIGHTS = str(PROJECT_ROOT / 'ML' / 'checkpoints' / 'best_model.pth')

# ============================================================================
# Lightweight Adapter Configuration
# ============================================================================

# Lightweight Adapter training hyperparameters
LIGHTWEIGHT_ADAPTER = {
    # Model configuration
    'model_name': 'base',  # Whisper model size: 'tiny', 'base', 'small'
    'feature_dim': 6,  # Number of acoustic features (centroid, bandwidth, rolloff, zcr, rms, tempo)
    'projection_dim': 32,  # Dimension to project features to
    'output_dim': 128,  # Final embedding dimension
    
    # Training parameters
    'batch_size': 12,  # Batch size (optimized for GPU memory with feature extraction)
    'num_epochs': 35,  # Maximum number of epochs (fewer than WhisperContrastive due to fewer parameters)
    'learning_rate': 1e-3,  # Initial learning rate
    'weight_decay': 1e-4,  # L2 regularization
    
    # Mixed precision training
    'mixed_precision': True,  # Use FP16 to reduce memory usage by ~40%
    
    # Loss function
    'temperature': 0.07,  # Temperature for contrastive loss (same as WhisperContrastive)
    
    # Data loading
    'num_workers': 4,  # Number of parallel data loading workers
    'audio_duration': 30.0,  # Audio duration in seconds
    'sample_rate': 16000,  # Audio sample rate (Hz)
    'balanced_sampling': False,  # Use weighted sampling for class balance
    
    # Feature extraction
    'extract_on_fly': False,  # Extract features on-the-fly (slower) or use cache (faster)
    'feature_cache_dir': str(PROJECT_ROOT / 'ML' / 'features_cache'),  # Directory for cached features
    'feature_cache_file': str(PROJECT_ROOT / 'ML' / 'features_cache' / 'acoustic_features.h5'),  # HDF5 cache
    'feature_stats_file': str(PROJECT_ROOT / 'ML' / 'features_cache' / 'feature_stats.json'),  # Normalization stats
    
    # Optimization
    'scheduler': 'plateau',  # LR scheduler: 'plateau', 'cosine', 'step', 'none'
    'early_stopping_patience': 10,  # Epochs to wait before early stopping
    
    # Checkpointing
    'checkpoint_dir': str(PROJECT_ROOT / 'ML' / 'checkpoints'),
    'log_dir': str(PROJECT_ROOT / 'ML' / 'logs'),
    'save_frequency': 1,  # Save checkpoint every N epochs
}

# Expected trainable parameters: ~70K (vs 262K for WhisperContrastive)
# Expected VRAM usage: ~1.5-1.8 GB with mixed precision, ~2.5-3.0 GB without
# Expected training time: 3-4 hours for 35 epochs on CUDA GPU

