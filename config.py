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

TRANF_MODELS = ['whisper', 'mert', 'whisper_contrastive', 'vggish']
MODEL_SIZES = ['base', '95m', 'base', 'pretrained']

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
        'base': str(PROJECT_ROOT / 'ML' / 'checkpoints' / 'best_model.pth')
    },
    'vggish': {
        'pretrained': 'vggish-10086976.pth'
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

