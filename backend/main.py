import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
import torch
import librosa
from models.MusiCNN import Musicnn
from models.VGG import VGG_Res
from models.VGGish import VGGish
from models.Whisper import WhisperEmbedding, pad_or_trim, log_mel_spectrogram
from models.MERT import MERTEmbedding, resample_audio
from models.WhisperContrastive import WhisperContrastive
from proyecciones import proyectar_embeddings
import torchaudio
import config

MSD_W_MUSICNN = './pesos/msd/musicnn.pth'
MTAT_W_MUSICNN = './pesos/mtat/musicnn.pth'

MSD_W_VGG = './pesos/msd/vgg.pth'
MTAT_W_VGG = './pesos/mtat/vgg.pth'

N_TAGS = 50      # Número de etiquetas (tags)
EMBEDDING_DIM = 200  # Dimensión de los Embeddings (la clase Musicnn lo define internamente) 
SR_MUSICNN = 16000     # Tasa de muestreo que MusiCNN espera

DC = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)  # Move model to GPU
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
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
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
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
    y, _ = librosa.load(audio, sr=16000)
    
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
    y, loaded_sr = librosa.load(audio, sr=None)
    
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
    checkpoint = torch.load(pesos, map_location=DC)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DC)
    model.eval()

    # Load full audio at 16kHz (Whisper's expected sample rate)
    y, _ = librosa.load(audio, sr=16000)
    
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
    y, _ = librosa.load(audio, sr=16000)
    
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
