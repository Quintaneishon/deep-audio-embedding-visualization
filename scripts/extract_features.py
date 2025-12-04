"""
Feature Extraction Script for Lightweight Adapter.

Pre-computes acoustic features for all audio files in MTG-Jamendo dataset
and caches them to HDF5 file for fast loading during training.

Usage:
    python scripts/extract_features.py \
        --data-root /path/to/mtg-jamendo-dataset \
        --split split-0 \
        --output ML/features_cache/acoustic_features.h5

Expected runtime: 1-2 hours for full MTG-Jamendo dataset
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ML.acoustic_features import AcousticFeatureExtractor
from ML.dataset import MTGJamendoDataset
from config import MTG_JAMENDO_ROOT, LIGHTWEIGHT_ADAPTER


def extract_and_cache_features(
    train_tsv,
    val_tsv,
    test_tsv,
    audio_dir,
    output_file,
    stats_file,
    sample_rate=16000
):
    """
    Extract features from all splits and cache to HDF5.
    
    Args:
        train_tsv: Training split TSV
        val_tsv: Validation split TSV
        test_tsv: Test split TSV
        audio_dir: Audio directory
        output_file: Output HDF5 cache file
        stats_file: Output JSON stats file
        sample_rate: Sample rate for processing
    """
    print("="*80)
    print("FEATURE EXTRACTION AND CACHING")
    print("="*80 + "\n")
    
    # Initialize feature extractor
    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)
    
    # Load datasets to get file paths
    print("Loading datasets...")
    train_dataset = MTGJamendoDataset(train_tsv, audio_dir, sample_rate=sample_rate, duration=30.0)
    val_dataset = MTGJamendoDataset(val_tsv, audio_dir, sample_rate=sample_rate, duration=30.0)
    test_dataset = MTGJamendoDataset(test_tsv, audio_dir, sample_rate=sample_rate, duration=30.0)
    
    print(f"  Train: {len(train_dataset)} tracks")
    print(f"  Val: {len(val_dataset)} tracks")
    print(f"  Test: {len(test_dataset)} tracks")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} tracks\n")
    
    # Collect all audio files
    all_files = []
    all_datasets = [
        ('train', train_dataset),
        ('val', val_dataset),
        ('test', test_dataset)
    ]
    
    for split_name, dataset in all_datasets:
        for i in range(len(dataset)):
            row = dataset.metadata.iloc[i]
            path = row['PATH']
            audio_path = Path(audio_dir) / path
            
            # Try .low.mp3 extension if file doesn't exist
            if not audio_path.exists():
                audio_path_low = Path(audio_dir) / path.replace('.mp3', '.low.mp3')
                if audio_path_low.exists():
                    audio_path = audio_path_low
            
            if audio_path.exists():
                all_files.append(str(audio_path))
    
    print(f"Found {len(all_files)} audio files\n")
    
    # =========================================================================
    # STEP 1: Compute normalization statistics from training set
    # =========================================================================
    
    print("STEP 1: Computing normalization statistics from training set")
    print("-" * 80)
    
    train_files = []
    for i in range(len(train_dataset)):
        row = train_dataset.metadata.iloc[i]
        path = row['PATH']
        audio_path = Path(audio_dir) / path
        
        if not audio_path.exists():
            audio_path_low = Path(audio_dir) / path.replace('.mp3', '.low.mp3')
            if audio_path_low.exists():
                audio_path = audio_path_low
        
        if audio_path.exists():
            train_files.append(str(audio_path))
    
    # Compute stats (this will print progress)
    extractor.compute_normalization_stats(train_files)
    
    # Save stats
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats_path = Path(stats_file)
    extractor.save_stats(str(stats_path))
    
    print()
    
    # =========================================================================
    # STEP 2: Extract and cache features for all files
    # =========================================================================
    
    print("STEP 2: Extracting and caching features for all files")
    print("-" * 80)
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        successful = 0
        failed = 0
        
        for audio_file in tqdm(all_files, desc="Extracting features"):
            try:
                # Extract features
                features = extractor.extract_features(audio_file, normalize=False)
                
                # Use filename as key
                key = Path(audio_file).name
                f.create_dataset(key, data=features.numpy())
                
                successful += 1
                
            except Exception as e:
                failed += 1
                if failed <= 10:  # Only print first 10 errors
                    print(f"\n  Warning: Failed to extract features from {audio_file}: {e}")
    
    print(f"\nFeature extraction complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/(successful+failed)*100:.1f}%")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    print(f"Feature cache: {output_file}")
    print(f"  Total tracks cached: {successful}")
    print(f"  File size: {Path(output_file).stat().st_size / 1e6:.1f} MB")
    
    print(f"\nNormalization stats: {stats_file}")
    print(f"  Feature names: {extractor.feature_names}")
    print(f"  Mean: {extractor.feature_mean.tolist()}")
    print(f"  Std: {extractor.feature_std.tolist()}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    
    print("1. Run quick validation to check if features help:")
    print(f"   python scripts/quick_validation.py \\")
    print(f"       --data-root {Path(audio_dir).parent.parent}")
    print()
    
    print("2. If validation is positive, train LightweightAdapter:")
    print(f"   python ML/train_lightweight.py \\")
    print(f"       --data-root {Path(audio_dir).parent.parent} \\")
    print(f"       --feature-cache {output_file} \\")
    print(f"       --feature-stats {stats_file} \\")
    print(f"       --batch-size 12 \\")
    print(f"       --num-epochs 35 \\")
    print(f"       --mixed-precision")
    print()


def main():
    parser = argparse.ArgumentParser(description='Extract and cache acoustic features')
    
    parser.add_argument('--data-root', type=str, default=MTG_JAMENDO_ROOT,
                       help='MTG-Jamendo dataset root directory')
    parser.add_argument('--split', type=str, default='split-0',
                       help='Dataset split to use')
    parser.add_argument('--output', type=str,
                       default=LIGHTWEIGHT_ADAPTER['feature_cache_file'],
                       help='Output HDF5 cache file')
    parser.add_argument('--stats', type=str,
                       default=LIGHTWEIGHT_ADAPTER['feature_stats_file'],
                       help='Output JSON stats file')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate for audio processing')
    
    args = parser.parse_args()
    
    # Prepare paths
    data_root = Path(args.data_root)
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'
    
    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv = splits_dir / 'autotagging_genre-test.tsv'
    
    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Split: {args.split}")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Output cache: {args.output}")
    print(f"  Output stats: {args.stats}")
    print()
    
    # Check if files exist
    if not train_tsv.exists():
        print(f"Error: Training TSV not found: {train_tsv}")
        return
    
    if not val_tsv.exists():
        print(f"Error: Validation TSV not found: {val_tsv}")
        return
    
    if not test_tsv.exists():
        print(f"Error: Test TSV not found: {test_tsv}")
        return
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    # Run extraction
    extract_and_cache_features(
        str(train_tsv),
        str(val_tsv),
        str(test_tsv),
        str(audio_dir),
        args.output,
        args.stats,
        args.sample_rate
    )


if __name__ == '__main__':
    main()

