"""
Quick Validation Script for Lightweight Adapter.

This script validates whether acoustic features actually help before
committing to full model training.

Strategy:
1. Extract embeddings from existing WhisperContrastive model
2. Extract acoustic features from same audio files
3. Concatenate and evaluate with simple classifier (Logistic Regression)
4. Compare accuracy and MAP@10 vs baseline (WhisperContrastive alone)
5. Decision: If improvement >5%, proceed with full training

Expected runtime: 1-3 hours depending on dataset size
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ML.models.WhisperContrastive import WhisperContrastive
from ML.acoustic_features import AcousticFeatureExtractor
from ML.dataset import create_dataloaders
from config import PROJECT_ROOT, MTG_JAMENDO_ROOT, MTG_JAMENDO_AUDIO_DIR, MTG_JAMENDO_SPLITS_DIR

# Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def extract_whisper_embeddings(model, dataloader, device):
    """
    Extract embeddings from WhisperContrastive model.
    
    Args:
        model: WhisperContrastive model
        dataloader: DataLoader
        device: Device to run on
    
    Returns:
        embeddings: numpy array [N, 128]
        labels: numpy array [N]
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Extracting Whisper embeddings...")
    with torch.no_grad():
        for audio, labels in tqdm(dataloader):
            audio = audio.to(device)
            
            # Extract embeddings
            embeddings = model(audio)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels


def extract_acoustic_features_from_loader(dataloader, sample_rate=16000):
    """
    Extract acoustic features from DataLoader.
    
    Args:
        dataloader: DataLoader
        sample_rate: Sample rate
    
    Returns:
        features: numpy array [N, 6]
    """
    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)
    all_features = []
    
    print("Extracting acoustic features...")
    for audio, labels in tqdm(dataloader):
        # Extract features for batch
        features = extractor.extract_batch(audio, normalize=False)
        all_features.append(features.numpy())
    
    features = np.vstack(all_features)
    
    # Normalize features (z-score)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / (std + 1e-8)
    
    return features


def evaluate_retrieval_map(embeddings, labels, k=10):
    """
    Compute Mean Average Precision @ K.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] labels
        k: Number of neighbors
    
    Returns:
        map_score: Mean Average Precision @ K
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_samples = len(embeddings)
    ap_scores = []
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    for i in range(n_samples):
        # Get similarities to all other samples
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self
        
        # Top-K most similar
        top_k_indices = np.argsort(sims)[-k:][::-1]
        
        # Check if same label
        query_label = labels[i]
        retrieved_labels = labels[top_k_indices]
        relevance = (retrieved_labels == query_label).astype(int)
        
        # Average Precision
        if relevance.sum() > 0:
            precisions = []
            relevant_count = 0
            for j, rel in enumerate(relevance):
                if rel:
                    relevant_count += 1
                    precisions.append(relevant_count / (j + 1))
            ap = np.mean(precisions) if precisions else 0
        else:
            ap = 0
        
        ap_scores.append(ap)
    
    map_score = np.mean(ap_scores)
    return map_score


def main():
    parser = argparse.ArgumentParser(description='Quick validation of acoustic features')
    parser.add_argument('--checkpoint', type=str,
                       default='ML/checkpoints/whisper_contrastive_20251128_085448/best_model.pth',
                       help='Path to WhisperContrastive checkpoint')
    parser.add_argument('--model-name', type=str, default='base',
                       help='Whisper model name')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for extraction')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--data-root', type=str, default=MTG_JAMENDO_ROOT,
                       help='MTG-Jamendo dataset root')
    parser.add_argument('--split', type=str, default='split-0',
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Prepare dataset paths
    data_root = Path(args.data_root)
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'
    
    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    
    print(f"Loading data from {data_root}")
    print(f"  Split: {args.split}")
    print(f"  Train: {train_tsv}")
    print(f"  Val: {val_tsv}\n")
    
    # Create dataloaders
    train_loader, val_loader, _, dataset_info = create_dataloaders(
        str(train_tsv),
        str(val_tsv),
        str(val_tsv),  # Use val as test for now
        str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=16000,
        duration=30.0
    )
    
    print(f"Dataset loaded:")
    print(f"  Genres: {dataset_info['num_classes']}")
    print(f"  Train samples: {dataset_info['train_size']}")
    print(f"  Val samples: {dataset_info['val_size']}\n")
    
    # Load WhisperContrastive model
    print(f"Loading WhisperContrastive from {args.checkpoint}")
    model = WhisperContrastive(model_name=args.model_name, projection_dim=128, device=device)
    
    checkpoint_path = Path(PROJECT_ROOT) / args.checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print(f"Using freshly initialized model (not recommended for validation)\n")
    
    model.to(device)
    model.eval()
    
    # Extract Whisper embeddings
    train_whisper_emb, train_labels = extract_whisper_embeddings(model, train_loader, device)
    val_whisper_emb, val_labels = extract_whisper_embeddings(model, val_loader, device)
    
    print(f"\nWhisper embeddings extracted:")
    print(f"  Train: {train_whisper_emb.shape}")
    print(f"  Val: {val_whisper_emb.shape}")
    
    # Extract acoustic features
    train_acoustic_feat = extract_acoustic_features_from_loader(train_loader)
    val_acoustic_feat = extract_acoustic_features_from_loader(val_loader)
    
    print(f"\nAcoustic features extracted:")
    print(f"  Train: {train_acoustic_feat.shape}")
    print(f"  Val: {val_acoustic_feat.shape}")
    
    # Concatenate: Whisper + Acoustic
    train_hybrid = np.concatenate([train_whisper_emb, train_acoustic_feat], axis=1)
    val_hybrid = np.concatenate([val_whisper_emb, val_acoustic_feat], axis=1)
    
    print(f"\nHybrid embeddings created:")
    print(f"  Train: {train_hybrid.shape}")
    print(f"  Val: {val_hybrid.shape}")
    
    # =========================================================================
    # EVALUATION 1: Classification Accuracy
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("EVALUATION 1: GENRE CLASSIFICATION ACCURACY")
    print(f"{'='*80}\n")
    
    # Train simple classifier on Whisper embeddings only
    print("Training classifier on Whisper embeddings only...")
    clf_whisper = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf_whisper.fit(train_whisper_emb, train_labels)
    
    acc_whisper_train = accuracy_score(train_labels, clf_whisper.predict(train_whisper_emb))
    acc_whisper_val = accuracy_score(val_labels, clf_whisper.predict(val_whisper_emb))
    
    print(f"  Train accuracy: {acc_whisper_train:.4f}")
    print(f"  Val accuracy:   {acc_whisper_val:.4f}")
    
    # Train classifier on Hybrid embeddings
    print("\nTraining classifier on Hybrid embeddings (Whisper + Features)...")
    clf_hybrid = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf_hybrid.fit(train_hybrid, train_labels)
    
    acc_hybrid_train = accuracy_score(train_labels, clf_hybrid.predict(train_hybrid))
    acc_hybrid_val = accuracy_score(val_labels, clf_hybrid.predict(val_hybrid))
    
    print(f"  Train accuracy: {acc_hybrid_train:.4f}")
    print(f"  Val accuracy:   {acc_hybrid_val:.4f}")
    
    # =========================================================================
    # EVALUATION 2: Retrieval MAP@10
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("EVALUATION 2: RETRIEVAL PERFORMANCE (MAP@10)")
    print(f"{'='*80}\n")
    
    print("Computing MAP@10 for Whisper embeddings...")
    map_whisper_val = evaluate_retrieval_map(val_whisper_emb, val_labels, k=10)
    print(f"  Whisper MAP@10: {map_whisper_val:.4f}")
    
    print("\nComputing MAP@10 for Hybrid embeddings...")
    map_hybrid_val = evaluate_retrieval_map(val_hybrid, val_labels, k=10)
    print(f"  Hybrid MAP@10:  {map_hybrid_val:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    print("Classification Accuracy (Validation):")
    print(f"  Whisper only:         {acc_whisper_val:.4f}")
    print(f"  Hybrid (W + Features): {acc_hybrid_val:.4f}")
    acc_improvement = (acc_hybrid_val / acc_whisper_val - 1) * 100
    print(f"  Improvement:          {acc_improvement:+.2f}%")
    
    print(f"\nRetrieval MAP@10 (Validation):")
    print(f"  Whisper only:         {map_whisper_val:.4f}")
    print(f"  Hybrid (W + Features): {map_hybrid_val:.4f}")
    map_improvement = (map_hybrid_val / map_whisper_val - 1) * 100
    print(f"  Improvement:          {map_improvement:+.2f}%")
    
    # =========================================================================
    # DECISION
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("DECISION")
    print(f"{'='*80}\n")
    
    threshold = 5.0  # 5% improvement threshold
    
    if acc_improvement >= threshold or map_improvement >= threshold:
        print(f"✅ VERDICT: Features HELP significantly!")
        print(f"   Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"   MAP@10 improvement:   {map_improvement:+.2f}%")
        print(f"\n   → RECOMMENDATION: Proceed with LightweightAdapter training")
    elif acc_improvement >= threshold * 0.5 or map_improvement >= threshold * 0.5:
        print(f"⚠️  VERDICT: Features help moderately")
        print(f"   Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"   MAP@10 improvement:   {map_improvement:+.2f}%")
        print(f"\n   → RECOMMENDATION: Consider LightweightAdapter (marginal benefit)")
    else:
        print(f"❌ VERDICT: Features do NOT help significantly")
        print(f"   Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"   MAP@10 improvement:   {map_improvement:+.2f}%")
        print(f"\n   → RECOMMENDATION: Investigate why features don't help:")
        print(f"      - Check feature normalization")
        print(f"      - Try different features")
        print(f"      - Verify features are not redundant with Whisper")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
