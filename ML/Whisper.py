"""
Whisper-based audio embedding extractor.

This module wraps OpenAI's Whisper model to extract audio embeddings
from the encoder (Transformer-based) rather than using it for speech recognition.

The encoder embeddings capture temporal and sequential patterns through
self-attention mechanisms, complementing the convolutional approaches
of MusiCNN and VGG models.
"""

import sys
import os
from functools import lru_cache
from typing import Optional, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "whisper"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import whisper

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    This mirrors `whisper.audio.mel_filters` but resolves the asset path
    via the installed `whisper` package instead of the local file location.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(
        os.path.dirname(whisper.__file__), "assets", "mel_filters.npz"
    )
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram.

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        # In this project we only pass raw waveforms (NumPy arrays or tensors),
        # so we don't need to handle file paths here.
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEmbedding(nn.Module):
    """
    Whisper-based embedding extractor.
    
    Extracts two types of features:
    1. Final encoder embeddings (high-level temporal/sequential features)
    2. Intermediate layer activations (mid-level features as "taggram" proxy)
    
    Both are temporally pooled (max + avg) to get fixed-size representations.
    """
    
    def __init__(self, model_name='base', intermediate_layer=None):
        """
        Initialize Whisper embedding extractor.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium')
            intermediate_layer: Which encoder layer to extract for taggram
                              (default: middle layer)
        """
        super(WhisperEmbedding, self).__init__()
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Whisper model
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        self.whisper_model.eval()
        
        # Get model dimensions
        self.n_mels = self.whisper_model.dims.n_mels  # 80 for Whisper
        self.n_audio_state = self.whisper_model.dims.n_audio_state  # Embedding dimension
        self.n_audio_layer = self.whisper_model.dims.n_audio_layer  # Number of encoder layers
        
        # Determine intermediate layer for taggram extraction
        if intermediate_layer is None:
            # Use middle layer by default
            self.intermediate_layer = self.n_audio_layer // 2
        else:
            self.intermediate_layer = intermediate_layer
        
        # Projection layer to convert intermediate features to 50-dim taggram
        # Intermediate features have same dimension as final output
        self.taggram_projection = nn.Linear(self.n_audio_state, 50)
        nn.init.xavier_uniform_(self.taggram_projection.weight)
        nn.init.zeros_(self.taggram_projection.bias)
        self.taggram_projection.to(self.device)
    
    def extract_encoder_features(self, mel):
        """
        Extract features from encoder at multiple layers.
        
        Args:
            mel: Mel spectrogram tensor [batch, n_mels, n_frames]
        
        Returns:
            final_features: Final encoder output [batch, n_ctx, n_state]
            intermediate_features: Intermediate layer output [batch, n_ctx, n_state]
        """
        encoder = self.whisper_model.encoder
        
        # Initial convolutions
        x = torch.nn.functional.gelu(encoder.conv1(mel))
        x = torch.nn.functional.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, n_ctx, n_state]
        
        # Add positional embeddings
        x = (x + encoder.positional_embedding).to(x.dtype)
        
        # Pass through encoder blocks, capturing intermediate layer
        intermediate_features = None
        for i, block in enumerate(encoder.blocks):
            x = block(x)
            if i == self.intermediate_layer:
                intermediate_features = x.clone()
        
        # Final layer norm
        final_features = encoder.ln_post(x)
        
        return final_features, intermediate_features
    
    def temporal_pooling(self, features):
        """
        Apply temporal pooling (max + average) over time dimension.
        
        Args:
            features: [batch, time, feature_dim]
        
        Returns:
            pooled: [batch, feature_dim] - concatenation of max and avg pooling
        """
        # Max pooling over time
        max_pool = torch.max(features, dim=1)[0]  # [batch, feature_dim]
        
        # Average pooling over time
        avg_pool = torch.mean(features, dim=1)  # [batch, feature_dim]
        
        # Concatenate (similar to MusiCNN approach)
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # [batch, feature_dim * 2]
        
        return pooled
    
    def forward(self, x):
        """
        Extract embeddings and taggrams from audio.
        
        Args:
            x: Raw audio waveform tensor [batch, num_samples] at 16kHz
        
        Returns:
            embeddings: [batch, embedding_dim] - final encoder features
            taggrams: [batch, 50] - projected intermediate features
        """
        batch_size = x.shape[0]
        
        # Convert audio to mel spectrogram (Whisper format)
        mel_list = []
        for i in range(batch_size):
            # Pad or trim to 30 seconds
            audio_padded = pad_or_trim(x[i].cpu().numpy())
            
            # Convert to mel spectrogram
            mel = log_mel_spectrogram(audio_padded, n_mels=self.n_mels)
            mel_list.append(mel)
        
        # Stack into batch
        mel = torch.stack(mel_list).to(self.device)  # [batch, n_mels, n_frames]
        
        # Extract features from encoder
        with torch.no_grad():
            final_features, intermediate_features = self.extract_encoder_features(mel)
        
        # Apply temporal pooling to get fixed-size representations
        # Note: We use single pooling (avg) instead of concat for efficiency
        final_pooled = torch.mean(final_features, dim=1)  # [batch, n_audio_state]
        intermediate_pooled = torch.mean(intermediate_features, dim=1)  # [batch, n_audio_state]
        
        # Final embedding
        embeddings = final_pooled
        
        # Project intermediate features to 50-dim taggram
        taggrams = torch.sigmoid(self.taggram_projection(intermediate_pooled))
        
        return embeddings, taggrams
