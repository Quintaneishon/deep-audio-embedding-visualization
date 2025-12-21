import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
import torch
import librosa
import soundfile as sf
import warnings
from models.MusiCNN import Musicnn
from models.VGG import VGG_Res
from models.VGGish import VGGish
from models.Whisper import WhisperEmbedding, pad_or_trim, log_mel_spectrogram
from models.MERT import MERTEmbedding, resample_audio
from models.WhisperContrastive import WhisperContrastive
from models.LightweightAdapter import LightweightAdapter
import config

MSD_W_MUSICNN = './pesos/msd/musicnn.pth'
MTAT_W_MUSICNN = './pesos/mtat/musicnn.pth'

MSD_W_VGG = './pesos/msd/vgg.pth'
MTAT_W_VGG = './pesos/mtat/vgg.pth'

N_TAGS = 50      # Número de etiquetas (tags)
EMBEDDING_DIM = 200  # Dimensión de los Embeddings (la clase Musicnn lo define internamente) 
SR_MUSICNN = 16000     # Tasa de muestreo que MusiCNN espera

DC = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_audio_safe(audio_path, sr=None):
    """
    Load audio safely without soundfile/audioread warnings.
    Tries soundfile first (supports WAV/FLAC/OGG), falls back to librosa for MP3/M4A.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (None = keep original)
    
    Returns:
        y: Audio waveform as numpy array
        sr: Sample rate
    """
    try:
        # Try soundfile first for supported formats (WAV, FLAC, OGG)
        y, orig_sr = sf.read(audio_path, dtype='float32')
        
        # Resample if needed
        if sr is not None and sr != orig_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            return y, sr
        return y, orig_sr
    except:
        # Fall back to librosa for MP3/M4A (suppress warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            y, loaded_sr = librosa.load(audio_path, sr=sr)
        return y, loaded_sr

def embeddings_y_taggrams_MusiCNN(pesos, audio, dataset_name='msd', segment_size=None, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using MusiCNN.
    
    Args:
        pesos: Path to model weights
        audio: Path to audio file
        dataset_name: Dataset name ('msd' or 'mtat')
        segment_size: Deprecated - kept for backward compatibility
        sr: Sample rate
    
    Returns:
        embeddings: (1, embedding_dim) - single vector per song
        taggrams: (1, n_tags) - single taggram per song
    """
    model = Musicnn(n_class=N_TAGS, dataset=dataset_name) 
    model.load_state_dict(torch.load(pesos, map_location=DC, weights_only=True))
    model.to(DC)  # Move model to GPU
    model.eval()

    # Load full audio
    y, _ = load_audio_safe(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        taggrams_tensor, embeddings_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGG(pesos, audio, dataset_name='msd', segment_size=None, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using VGG.
    
    Args:
        pesos: Path to model weights
        audio: Path to audio file
        dataset_name: Dataset name ('msd' or 'mtat')
        segment_size: Deprecated
        sr: Sample rate
    
    Returns:
        embeddings: (1, embedding_dim)
        taggrams: (1, n_tags)
    """
    use_simple = (dataset_name == 'mtat')
    model = VGG_Res(n_class=N_TAGS, use_simple_res=use_simple)
    model.load_state_dict(torch.load(pesos, map_location=DC, weights_only=True))
    model.to(DC)
    model.eval()

    # Load full audio
    y, _ = load_audio_safe(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        taggrams_tensor, embeddings_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_Whisper(model_name, audio, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using Whisper encoder.
    
    Args:
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium')
        audio: Path to audio file
        sr: Sample rate (should be 16000 for Whisper)
    
    Returns:
        embeddings: (1, embedding_dim) - final encoder output
        taggrams: (1, 50) - intermediate layer features
    """
    # Create Whisper model
    model = WhisperEmbedding(model_name=model_name)
    model.to(DC)
    model.eval()

    # Load full audio at 16kHz (Whisper's expected sample rate)
    y, _ = load_audio_safe(audio, sr=16000)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        
        # Get embeddings using forward (mean-pooled final layer)
        embeddings_tensor, _ = model(x)
        
        # Get raw intermediate features (same approach as WhisperContrastive)
        # Convert audio to mel spectrogram
        audio_padded = pad_or_trim(y)
        mel = log_mel_spectrogram(audio_padded, n_mels=model.n_mels)
        mel = mel.unsqueeze(0).to(DC)
        
        # Extract intermediate features for taggram (raw, not projected)
        _, taggrams_tensor = model.extract_encoder_features(mel)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()  # [n_ctx, n_audio_state]
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_ctx * n_audio_state) - flattened
    
    return embeddings, taggrams

def embeddings_y_taggrams_MERT(model_name, audio, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using MERT encoder.
    
    Args:
        model_name: MERT model name ('95M' or '330M')
        audio: Path to audio file
        sr: Sample rate (will be resampled to 24000 for MERT)
    
    Returns:
        embeddings: (1, embedding_dim) - final encoder output (768 for 95M, 1024 for 330M)
        taggrams: (1, 50) - intermediate layer features
    """
    # Create MERT model (will cache in config.MODEL_DIR)
    model = MERTEmbedding(model_name=model_name, cache_dir=config.MODEL_DIR)
    model.to(DC)
    model.eval()

    # Load audio at original sample rate first
    y, loaded_sr = load_audio_safe(audio, sr=None)
    
    # Convert to tensor
    y_tensor = torch.from_numpy(y).float()
    
    # Resample to 24kHz (MERT's expected sample rate)
    resample_audio(y_tensor, loaded_sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = y_tensor.unsqueeze(0).to(DC)
        embeddings_tensor, taggrams_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_WhisperContrastive(pesos, audio, model_name='base', projection_dim=128, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using trained WhisperContrastive model.
    
    Args:
        pesos: Path to trained model weights
        audio: Path to audio file
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium')
        projection_dim: Dimension of projection head (default: 128)
        sr: Sample rate (should be 16000 for Whisper)
    
    Returns:
        embeddings: (1, projection_dim) - L2 normalized contrastive embeddings
        taggrams: (1, 50) - intermediate layer features from frozen encoder
    """
    
    # Create WhisperContrastive model
    model = WhisperContrastive(model_name=model_name, projection_dim=projection_dim)
    
    # Load trained weights (projection head)
    checkpoint = torch.load(pesos, map_location=DC, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DC)
    model.eval()

    # Load full audio at 16kHz (Whisper's expected sample rate)
    y, _ = load_audio_safe(audio, sr=16000)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        
        # Get contrastive embeddings (normalized)
        embeddings_tensor = model(x)  # [1, projection_dim]
        
        # Also get taggrams from the frozen encoder if needed
        # Convert audio to mel spectrogram
        audio_padded = pad_or_trim(y)
        mel = log_mel_spectrogram(audio_padded, n_mels=model.n_mels)
        mel = mel.unsqueeze(0).to(DC)
        
        # Extract intermediate features for taggram
        _, taggrams_tensor = model.extract_encoder_features(mel)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, projection_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGGish(model_name, audio, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using pretrained VGGish.
    
    Args:
        model_name: 'pretrained' (will auto-download Google's pretrained weights)
        audio: Path to audio file
        sr: Sample rate (should be 16000 for VGGish)
    
    Returns:
        embeddings: (1, 128) - VGGish embeddings
        taggrams: (1, 50) - classifier output (randomly initialized, can be fine-tuned)
    """
    # Create VGGish model with pretrained weights
    model = VGGish(pretrained=True, n_class=N_TAGS)
    model.to(DC)
    model.eval()

    # Load full audio at 16kHz (VGGish's expected sample rate)
    y, _ = load_audio_safe(audio, sr=16000)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        
        # Get embeddings and taggrams
        taggrams_tensor, embeddings_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, 128)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams


def embeddings_y_taggrams_LightweightAdapter(pesos, audio, feature_stats_path=None, model_name='base', 
                                            output_dim=128, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using trained LightweightAdapter model.
    
    This function works with both v1 (frozen early layers) and v2 (fine-tuned early layers) models.
    The model configuration is automatically loaded from the checkpoint.
    
    Architecture layers:
    - Embeddings: Penultimate layer (544-dim) = Whisper features + projected acoustic features
    - Taggrams: Final dense layer (128-dim) = Fusion layer output before L2 normalization
    
    Args:
        pesos: Path to trained model checkpoint (.pth file)
        audio: Path to audio file
        feature_stats_path: Path to feature normalization stats JSON (if None, looks in checkpoint dir)
        model_name: Whisper model name ('tiny', 'base', 'small') - should match training
        output_dim: Output embedding dimension (default: 128)
        sr: Sample rate (should be 16000)
    
    Returns:
        embeddings: (1, 544) - Penultimate layer (Whisper + acoustic features concatenated)
        taggrams: (1, 128) - Final dense layer output (before L2 normalization)
    """
    from pathlib import Path
    from acoustic_features import AcousticFeatureExtractor
    
    # Load checkpoint to get model configuration
    checkpoint = torch.load(pesos, map_location=DC)
    model_config = checkpoint.get('model_config', {})
    
    # Extract configuration from checkpoint
    feature_dim = model_config.get('feature_dim', 6)
    projection_dim = model_config.get('projection_dim', 32)
    output_dim = model_config.get('output_dim', output_dim)
    model_name = model_config.get('model_name', model_name)
    
    # Determine feature stats path if not provided
    if feature_stats_path is None:
        # Try to find feature_stats.json in the checkpoint directory
        checkpoint_dir = Path(pesos).parent
        feature_stats_path = checkpoint_dir / 'feature_stats.json'
        if not feature_stats_path.exists():
            # Fall back to default location
            feature_stats_path = Path(config.PROJECT_ROOT) / 'ML' / 'features_cache' / 'feature_stats.json'
    
    # Create LightweightAdapter model (configuration will be loaded from checkpoint)
    # Note: num_transformer_layers and finetune_early_layers don't affect inference
    # since we're loading trained weights
    model = LightweightAdapter(
        model_name=model_name,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
        output_dim=output_dim,
        device=DC,
        feature_stats_path=str(feature_stats_path) if feature_stats_path.exists() else None,
        num_transformer_layers=2,  # Default for loading
        finetune_early_layers=False  # Doesn't matter for inference
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DC)
    model.eval()
    
    # Initialize feature extractor for extracting acoustic features
    feature_extractor = AcousticFeatureExtractor(sample_rate=sr)
    if feature_stats_path and Path(feature_stats_path).exists():
        feature_extractor.load_stats(str(feature_stats_path))

    # Load full audio at 16kHz
    y, _ = load_audio_safe(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        
        # Extract acoustic features (6-dim)
        features = feature_extractor.extract_batch(x, normalize=True)
        features = features.to(DC)
        
        # === EXTRACT EMBEDDINGS (PENULTIMATE LAYER) ===
        # Process through early Whisper layers
        batch_size = x.shape[0]
        mel_list = []
        for i in range(batch_size):
            audio_padded = pad_or_trim(x[i].cpu().numpy())
            mel = log_mel_spectrogram(audio_padded, n_mels=model.n_mels)
            mel_list.append(mel)
        
        mel = torch.stack(mel_list).to(DC)
        
        # Extract early layer features
        if model.finetune_early_layers:
            early_features = model.extract_early_layer_features(mel)
        else:
            with torch.no_grad():
                early_features = model.extract_early_layer_features(mel)
        
        # Average pooling to get Whisper embedding
        whisper_emb = torch.mean(early_features, dim=1)  # [batch, n_audio_state=512]
        
        # Project acoustic features
        features_projected = model.feature_projection(features)  # [batch, projection_dim=32]
        
        # Concatenate - THIS IS THE EMBEDDINGS (PENULTIMATE LAYER)
        embeddings_penultimate = torch.cat([whisper_emb, features_projected], dim=1)  # [batch, 544]
        
        # === EXTRACT TAGGRAMS (FINAL DENSE LAYER) ===
        # Apply fusion layer (before L2 normalization)
        taggrams_final_dense = model.fusion(embeddings_penultimate)  # [batch, output_dim=128]
        
        # Convert to numpy
        embeddings = embeddings_penultimate.squeeze(0).cpu().numpy()
        taggrams = taggrams_final_dense.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, 544) - Penultimate layer
    taggrams = taggrams.reshape(1, -1)      # (1, 128) - Final dense layer
    
    return embeddings, taggrams

