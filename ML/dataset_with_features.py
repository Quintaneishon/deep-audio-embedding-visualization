"""
PyTorch Dataset with Acoustic Features for MTG-Jamendo.

Extends MTGJamendoDataset to include acoustic feature extraction,
supporting both on-the-fly extraction and cached features.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import h5py

from ML.dataset import MTGJamendoDataset, collate_fn as original_collate_fn
from ML.acoustic_features import AcousticFeatureExtractor


class MTGJamendoDatasetWithFeatures(MTGJamendoDataset):
    """
    MTG-Jamendo Dataset that also returns acoustic features.
    
    Extends MTGJamendoDataset to return (audio, label, features) tuples
    instead of (audio, label).
    """
    
    def __init__(
        self,
        tsv_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        transform=None,
        feature_cache_file: Optional[str] = None,
        feature_stats_file: Optional[str] = None,
        extract_on_fly: bool = False
    ):
        """
        Initialize MTG-Jamendo dataset with features.
        
        Args:
            tsv_file: Path to TSV file with track metadata and labels
            audio_dir: Root directory containing audio files
            sample_rate: Target sample rate for audio (default: 16000 Hz)
            duration: Maximum duration in seconds (None = use full track)
            transform: Optional transform to apply to audio
            feature_cache_file: Path to HDF5 file with cached features (optional)
            feature_stats_file: Path to JSON file with feature normalization stats
            extract_on_fly: If True, extract features on-the-fly instead of using cache
        """
        # Initialize parent dataset
        super().__init__(tsv_file, audio_dir, sample_rate, duration, transform)
        
        # Feature extraction setup
        self.feature_cache_file = feature_cache_file
        self.extract_on_fly = extract_on_fly
        
        # Initialize feature extractor
        self.feature_extractor = AcousticFeatureExtractor(sample_rate=sample_rate)
        
        # Load normalization statistics if available
        if feature_stats_file and os.path.exists(feature_stats_file):
            self.feature_extractor.load_stats(feature_stats_file)
            print(f"Loaded feature normalization stats from {feature_stats_file}")
        else:
            if not extract_on_fly:
                print("Warning: No feature stats file provided. Features will not be normalized.")
        
        # Check if cache exists
        self.use_cache = (feature_cache_file is not None and 
                         os.path.exists(feature_cache_file) and 
                         not extract_on_fly)
        
        if self.use_cache:
            print(f"Using cached features from {feature_cache_file}")
        elif extract_on_fly:
            print("Extracting features on-the-fly during training")
        else:
            print("Warning: No feature cache available and extract_on_fly=False")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the track
        
        Returns:
            audio: Audio waveform tensor [num_samples]
            label: Genre label as integer index
            features: Acoustic features tensor [6]
        """
        # Get audio and label from parent class
        row = self.metadata.iloc[idx]
        path = row['PATH']
        
        # Construct full audio path
        audio_path = self.audio_dir / path
        if not audio_path.exists():
            # Try with .low.mp3 extension
            audio_path_low = self.audio_dir / path.replace('.mp3', '.low.mp3')
            if audio_path_low.exists():
                audio_path = audio_path_low
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path} or {audio_path_low}")
        
        # Load audio
        try:
            import librosa
            waveform, sr = librosa.load(
                str(audio_path), 
                sr=self.sample_rate,
                mono=True
            )
            waveform = torch.from_numpy(waveform).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            num_samples = int(self.duration * self.sample_rate) if self.duration else 480000
            waveform = torch.zeros(num_samples)
        
        # Trim or pad to fixed duration if specified
        if self.duration is not None:
            target_length = int(self.duration * self.sample_rate)
            if waveform.shape[0] > target_length:
                waveform = waveform[:target_length]
            elif waveform.shape[0] < target_length:
                padding = target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        # Get genre label
        genre = self.genres[idx]
        label = self.genre_to_idx.get(genre, 0)
        
        # Extract or load features
        if self.use_cache:
            # Load from cache
            features = self._load_features_from_cache(path)
        else:
            # Extract on-the-fly
            features = self.feature_extractor.extract_features(
                waveform, 
                normalize=True
            )
        
        return waveform, label, features
    
    def _load_features_from_cache(self, path: str) -> torch.Tensor:
        """
        Load pre-computed features from HDF5 cache.
        
        Args:
            path: Path to audio file (used as key)
        
        Returns:
            features: Tensor of shape [6]
        """
        filename = Path(path).name
        
        try:
            with h5py.File(self.feature_cache_file, 'r') as f:
                if filename not in f:
                    # Fall back to on-the-fly extraction
                    print(f"Warning: {filename} not in cache, extracting on-the-fly")
                    audio_path = self.audio_dir / path
                    features = self.feature_extractor.extract_features(
                        str(audio_path),
                        normalize=True
                    )
                else:
                    features = torch.tensor(f[filename][:], dtype=torch.float32)
                    
                    # Normalize if stats are available
                    if self.feature_extractor.feature_mean is not None:
                        features = (features - self.feature_extractor.feature_mean) / \
                                  (self.feature_extractor.feature_std + 1e-8)
        except Exception as e:
            print(f"Error loading features from cache for {filename}: {e}")
            # Return zero features as fallback
            features = torch.zeros(6)
        
        return features


def collate_fn_with_features(batch: List[Tuple[torch.Tensor, int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length audio with features.
    
    Pads all audio to the same length within the batch and stacks features.
    
    Args:
        batch: List of (audio, label, features) tuples
    
    Returns:
        audios: Batched audio tensor [batch_size, max_length]
        labels: Batched label tensor [batch_size]
        features: Batched features tensor [batch_size, 6]
    """
    audios, labels, features = zip(*batch)
    
    # Find max length in batch
    max_length = max(audio.shape[0] for audio in audios)
    
    # Pad all audio to max length
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_audios.append(audio)
    
    # Stack into batch tensors
    audios_batch = torch.stack(padded_audios)
    labels_batch = torch.tensor(labels, dtype=torch.long)
    features_batch = torch.stack(features)
    
    return audios_batch, labels_batch, features_batch


def create_dataloaders_with_features(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    audio_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 30.0,
    balanced_sampling: bool = False,
    feature_cache_file: Optional[str] = None,
    feature_stats_file: Optional[str] = None,
    extract_on_fly: bool = False
):
    """
    Create train, validation, and test dataloaders with features.
    
    Args:
        train_tsv: Path to training split TSV
        val_tsv: Path to validation split TSV
        test_tsv: Path to test split TSV
        audio_dir: Root directory containing audio files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        sample_rate: Target sample rate
        duration: Duration to crop/pad audio to (seconds)
        balanced_sampling: Whether to use balanced sampling for training
        feature_cache_file: Path to HDF5 file with cached features
        feature_stats_file: Path to JSON file with normalization stats
        extract_on_fly: Extract features on-the-fly instead of using cache
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Create datasets
    train_dataset = MTGJamendoDatasetWithFeatures(
        train_tsv, audio_dir, sample_rate, duration,
        feature_cache_file=feature_cache_file,
        feature_stats_file=feature_stats_file,
        extract_on_fly=extract_on_fly
    )
    val_dataset = MTGJamendoDatasetWithFeatures(
        val_tsv, audio_dir, sample_rate, duration,
        feature_cache_file=feature_cache_file,
        feature_stats_file=feature_stats_file,
        extract_on_fly=extract_on_fly
    )
    test_dataset = MTGJamendoDatasetWithFeatures(
        test_tsv, audio_dir, sample_rate, duration,
        feature_cache_file=feature_cache_file,
        feature_stats_file=feature_stats_file,
        extract_on_fly=extract_on_fly
    )
    
    # Create sampler for balanced training if requested
    train_sampler = None
    shuffle = True
    if balanced_sampling:
        weights = train_dataset.get_weights_for_balanced_sampling()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        shuffle = False  # Mutually exclusive with sampler
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_features,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_features,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_features,
        pin_memory=True
    )
    
    # Dataset info for reference
    dataset_info = {
        'num_classes': len(train_dataset.unique_genres),
        'genre_to_idx': train_dataset.genre_to_idx,
        'idx_to_genre': train_dataset.idx_to_genre,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_distribution': train_dataset.get_genre_distribution(),
        'feature_dim': 6
    }
    
    return train_loader, val_loader, test_loader, dataset_info

