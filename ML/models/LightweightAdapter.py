"""
Lightweight Adapter for audio embeddings with explicit acoustic features.

This model combines frozen Whisper embeddings (512 dim) with 6 explicit
acoustic features via a small trainable adapter (~70K parameters).

Architecture:
1. Frozen Whisper encoder (pre-trained) → 512 dim
2. Acoustic features extraction → 6 dim
3. Feature projection: Linear(6 → 32) + ReLU → 32 dim (trainable)
4. Concatenation: [512, 32] → 544 dim
5. Fusion layer: Linear(544 → 128) → 128 dim (trainable)
6. L2 normalization for contrastive learning

Total trainable parameters: ~70K (vs 262K in WhisperContrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ML.models.Whisper import WhisperEmbedding, log_mel_spectrogram, pad_or_trim
from ML.acoustic_features import AcousticFeatureExtractor


class LightweightAdapter(WhisperEmbedding):
    """
    Lightweight adapter that fuses Whisper embeddings with acoustic features.
    
    Based on parameter-efficient fine-tuning (PEFT) methodology.
    Only the adapter layers are trainable; Whisper encoder is frozen.
    """
    
    def __init__(
        self, 
        model_name='base', 
        feature_dim=6,
        projection_dim=32,
        output_dim=128,
        intermediate_layer=None, 
        device=None,
        feature_stats_path=None
    ):
        """
        Initialize Lightweight Adapter.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium')
            feature_dim: Dimension of acoustic features (default: 6)
            projection_dim: Dimension to project features to (default: 32)
            output_dim: Final embedding dimension (default: 128)
            intermediate_layer: Which encoder layer to extract for taggram
            device: Device to load model on (default: cuda if available, else cpu)
            feature_stats_path: Path to feature normalization statistics JSON file
        """
        super(LightweightAdapter, self).__init__(
            model_name=model_name,
            intermediate_layer=intermediate_layer,
            device=device
        )
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.output_dim = output_dim
        
        # Feature extractor
        self.feature_extractor = AcousticFeatureExtractor(sample_rate=16000)
        
        # Load normalization statistics if provided
        if feature_stats_path:
            self.feature_extractor.load_stats(feature_stats_path)
        
        # ADAPTER LAYERS (trainable)
        
        # 1. Feature projection: 6 → 32
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU()
        )
        
        # 2. Fusion layer: (n_audio_state + projection_dim) → output_dim
        # n_audio_state = 512 for Whisper base
        fusion_input_dim = self.n_audio_state + projection_dim
        self.fusion = nn.Linear(fusion_input_dim, output_dim)
        
        # Initialize adapter weights
        self._init_adapter_weights()
        
        # Move adapter to device
        self.feature_projection.to(self.device)
        self.fusion.to(self.device)
        
        # Freeze Whisper encoder (after creating adapter)
        self._freeze_encoder()
        
        print(f"\nLightweight Adapter initialized:")
        print(f"  Whisper model: {model_name} (frozen)")
        print(f"  Whisper embedding dim: {self.n_audio_state}")
        print(f"  Acoustic features: {feature_dim} dim")
        print(f"  Feature projection: {feature_dim} → {projection_dim} dim")
        print(f"  Fusion input: {fusion_input_dim} dim")
        print(f"  Output embedding: {output_dim} dim")
        print(f"  Trainable parameters: {self.count_trainable_parameters():,}")
    
    def _init_adapter_weights(self):
        """Initialize adapter weights using Xavier initialization."""
        for module in [self.feature_projection, self.fusion]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def _freeze_encoder(self):
        """Freeze all parameters in the Whisper encoder."""
        # Freeze the entire Whisper model
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        
        # Also freeze the taggram projection
        for param in self.taggram_projection.parameters():
            param.requires_grad = False
    
    def count_trainable_parameters(self):
        """Count number of trainable parameters in adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_features(self, x, features=None):
        """
        Extract fused embeddings from audio + acoustic features.
        
        Args:
            x: Raw audio waveform tensor [batch, num_samples] at 16kHz
            features: Pre-computed acoustic features [batch, 6] (optional)
                     If None, will be extracted from audio
        
        Returns:
            embeddings: [batch, output_dim] - L2 normalized embeddings
        """
        batch_size = x.shape[0]
        
        # 1. Extract Whisper embeddings (frozen)
        mel_list = []
        for i in range(batch_size):
            # Pad or trim to 30 seconds
            audio_padded = pad_or_trim(x[i].cpu().numpy())
            
            # Convert to mel spectrogram
            mel = log_mel_spectrogram(audio_padded, n_mels=self.n_mels)
            mel_list.append(mel)
        
        # Stack into batch
        mel = torch.stack(mel_list).to(self.device)  # [batch, n_mels, n_frames]
        
        # Extract features from intermediate encoder layer (frozen)
        with torch.no_grad():
            _, intermediate_features = self.extract_encoder_features(mel)
            # intermediate_features: [batch, time_frames, n_audio_state]
            
            # Average pooling over time to get fixed-size representation
            whisper_emb = torch.mean(intermediate_features, dim=1)  # [batch, n_audio_state]
        
        # 2. Extract acoustic features if not provided
        if features is None:
            features = self.feature_extractor.extract_batch(x, normalize=True)
        
        # Ensure features are on correct device
        features = features.to(self.device)
        
        # 3. ADAPTER: Project acoustic features (trainable)
        features_projected = self.feature_projection(features)  # [batch, projection_dim]
        
        # 4. ADAPTER: Concatenate Whisper embeddings + projected features
        combined = torch.cat([whisper_emb, features_projected], dim=1)  # [batch, n_audio_state + projection_dim]
        
        # 5. ADAPTER: Fusion layer (trainable)
        embeddings = self.fusion(combined)  # [batch, output_dim]
        
        # 6. L2 normalization for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, x, features=None):
        """
        Forward pass for contrastive learning.
        
        Args:
            x: Raw audio waveform tensor [batch, num_samples] at 16kHz
            features: Pre-computed acoustic features [batch, 6] (optional)
        
        Returns:
            embeddings: [batch, output_dim] - L2 normalized embeddings
        """
        return self.forward_features(x, features)
    
    def get_trainable_parameters(self):
        """Return only the trainable parameters (adapter layers)."""
        params = []
        params.extend(self.feature_projection.parameters())
        params.extend(self.fusion.parameters())
        return params
    
    def get_acoustic_features(self, x):
        """
        Extract acoustic features from audio batch.
        
        Useful for pre-computing features for faster training.
        
        Args:
            x: Raw audio waveform tensor [batch, num_samples]
        
        Returns:
            features: [batch, feature_dim] tensor
        """
        return self.feature_extractor.extract_batch(x, normalize=True)
    
    def save_feature_stats(self, filepath):
        """Save feature normalization statistics."""
        self.feature_extractor.save_stats(filepath)
    
    def load_feature_stats(self, filepath):
        """Load feature normalization statistics."""
        self.feature_extractor.load_stats(filepath)


def create_lightweight_adapter(
    model_name='base',
    feature_dim=6,
    projection_dim=32,
    output_dim=128,
    feature_stats_path=None,
    device=None
):
    """
    Factory function to create a LightweightAdapter model.
    
    Args:
        model_name: Whisper model size
        feature_dim: Number of acoustic features
        projection_dim: Dimension to project features to
        output_dim: Final embedding dimension
        feature_stats_path: Path to feature normalization stats
        device: Device to load model on
    
    Returns:
        model: LightweightAdapter instance
    """
    model = LightweightAdapter(
        model_name=model_name,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
        output_dim=output_dim,
        device=device,
        feature_stats_path=feature_stats_path
    )
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing LightweightAdapter creation...")
    
    model = create_lightweight_adapter(
        model_name='base',
        feature_dim=6,
        projection_dim=32,
        output_dim=128
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass with dummy data
    print(f"\nTesting forward pass...")
    dummy_audio = torch.randn(2, 480000)  # 2 samples, 30 seconds at 16kHz
    
    embeddings = model(dummy_audio)
    
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output is L2 normalized: {torch.allclose(torch.norm(embeddings, dim=1), torch.ones(2), atol=1e-5)}")
    
    print(f"\nTest passed!")

