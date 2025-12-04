"""
Comprehensive Evaluation Script for Lightweight Adapter.

Compares LightweightAdapter with baseline models:
- MusiCNN (msd)
- VGG (msd)
- Whisper (base)
- WhisperContrastive (base)
- LightweightAdapter (new)

Metrics:
- Clustering: Silhouette Score, Davies-Bouldin Index
- Retrieval: MAP@5, MAP@10, Recall@10
- Visualization: t-SNE/UMAP plots, confusion matrix

Expected runtime: 30-60 minutes
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ML.models.LightweightAdapter import LightweightAdapter
from ML.models.WhisperContrastive import WhisperContrastive
from ML.dataset_with_features import create_dataloaders_with_features
from ML.dataset import create_dataloaders
from backend.main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, embeddings_y_taggrams_Whisper
from config import PROJECT_ROOT, MTG_JAMENDO_ROOT, MODEL_WEIGHTS

# Metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap.umap_ as umap


class EmbeddingEvaluator:
    """
    Comprehensive evaluator for audio embeddings.
    """
    
    def __init__(self, embeddings_dict, labels, feature_names=None):
        """
        Args:
            embeddings_dict: Dict of {model_name: embeddings_array}
            labels: Array of genre labels
            feature_names: List of genre names
        """
        self.embeddings_dict = embeddings_dict
        self.labels = labels
        self.feature_names = feature_names
    
    def evaluate_clustering(self, embeddings, name):
        """
        Evaluate clustering quality.
        
        Returns:
            dict with silhouette and davies_bouldin scores
        """
        print(f"  Computing clustering metrics for {name}...")
        
        # Silhouette Score (higher is better)
        silhouette = silhouette_score(embeddings, self.labels, metric='cosine')
        
        # Davies-Bouldin Index (lower is better)
        davies_bouldin = davies_bouldin_score(embeddings, self.labels)
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    def evaluate_retrieval(self, embeddings, name, k=10):
        """
        Evaluate retrieval performance with MAP@K.
        
        Returns:
            dict with map and recall scores
        """
        print(f"  Computing retrieval metrics for {name}...")
        
        n_samples = len(embeddings)
        sim_matrix = cosine_similarity(embeddings)
        
        # MAP@K
        ap_scores = []
        recall_scores = []
        
        for i in range(n_samples):
            query_label = self.labels[i]
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf  # Exclude self
            
            # Top-K
            top_k_indices = np.argsort(sims)[-k:][::-1]
            retrieved_labels = self.labels[top_k_indices]
            
            # Relevance
            relevance = (retrieved_labels == query_label).astype(int)
            
            # Average Precision
            if relevance.sum() > 0:
                precisions = []
                relevant_count = 0
                for j, rel in enumerate(relevance):
                    if rel:
                        relevant_count += 1
                        precisions.append(relevant_count / (j + 1))
                ap = np.mean(precisions)
            else:
                ap = 0
            
            ap_scores.append(ap)
            
            # Recall@K
            total_relevant = np.sum(self.labels == query_label) - 1  # Exclude self
            if total_relevant > 0:
                recall = relevance.sum() / min(total_relevant, k)
            else:
                recall = 0
            recall_scores.append(recall)
        
        map_score = np.mean(ap_scores)
        recall_score = np.mean(recall_scores)
        
        return {
            f'map@{k}': map_score,
            f'recall@{k}': recall_score
        }
    
    def evaluate_all(self):
        """
        Evaluate all models.
        
        Returns:
            DataFrame with results
        """
        results = {}
        
        print("\n" + "="*80)
        print("EVALUATING ALL MODELS")
        print("="*80 + "\n")
        
        for model_name, embeddings in self.embeddings_dict.items():
            print(f"Evaluating: {model_name}")
            
            # Clustering
            clustering_metrics = self.evaluate_clustering(embeddings, model_name)
            
            # Retrieval
            retrieval_metrics_5 = self.evaluate_retrieval(embeddings, model_name, k=5)
            retrieval_metrics_10 = self.evaluate_retrieval(embeddings, model_name, k=10)
            
            # Combine results
            results[model_name] = {
                **clustering_metrics,
                **retrieval_metrics_5,
                **retrieval_metrics_10
            }
            
            print()
        
        return pd.DataFrame(results).T
    
    def plot_comparison(self, results_df, output_dir):
        """
        Create comparison visualizations.
        
        Args:
            results_df: DataFrame with results
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = results_df.index.tolist()
        
        # 1. Silhouette Score
        ax = axes[0, 0]
        silhouette = results_df['silhouette'].values
        bars = ax.bar(range(len(models)), silhouette)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality - Silhouette Score (higher is better)')
        ax.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(silhouette)
        bars[best_idx].set_color('green')
        
        # 2. Davies-Bouldin Index
        ax = axes[0, 1]
        db = results_df['davies_bouldin'].values
        bars = ax.bar(range(len(models)), db, color='orange')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Davies-Bouldin Index')
        ax.set_title('Clustering Separation (lower is better)')
        ax.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmin(db)
        bars[best_idx].set_color('green')
        
        # 3. MAP@10
        ax = axes[1, 0]
        map10 = results_df['map@10'].values
        bars = ax.bar(range(len(models)), map10, color='purple')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('MAP@10')
        ax.set_title('Retrieval Performance - MAP@10 (higher is better)')
        ax.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(map10)
        bars[best_idx].set_color('green')
        
        # 4. Overall Score (weighted combination)
        ax = axes[1, 1]
        # Normalize metrics to [0, 1] range
        silhouette_norm = (silhouette - silhouette.min()) / (silhouette.max() - silhouette.min() + 1e-8)
        db_norm = 1 - (db - db.min()) / (db.max() - db.min() + 1e-8)  # Invert since lower is better
        map10_norm = (map10 - map10.min()) / (map10.max() - map10.min() + 1e-8)
        
        overall = 0.3 * silhouette_norm + 0.2 * db_norm + 0.5 * map10_norm
        
        bars = ax.bar(range(len(models)), overall, color='teal')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Overall Score')
        ax.set_title('Overall Performance (weighted: 30% clustering, 50% retrieval)')
        ax.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(overall)
        bars[best_idx].set_color('green')
        ax.text(best_idx, overall[best_idx], '★ WINNER', 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = output_dir / 'comparison_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
        plt.close()
    
    def plot_tsne(self, output_dir):
        """
        Create t-SNE visualizations for all models.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_models = len(self.embeddings_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, embeddings) in zip(axes, self.embeddings_dict.items()):
            print(f"  Computing t-SNE for {model_name}...")
            
            # Compute t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = tsne.fit_transform(embeddings)
            
            # Plot
            unique_labels = np.unique(self.labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c=[colors[i]], label=label, alpha=0.6, s=20)
            
            ax.set_title(f'{model_name}\nt-SNE Visualization')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'tsne_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {plot_path}")
        plt.close()


def extract_embeddings_from_model(model, dataloader, device, with_features=False):
    """
    Extract embeddings from a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader (with or without features)
        device: Device
        with_features: Whether dataloader returns features
    
    Returns:
        embeddings: numpy array
        labels: numpy array
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if with_features:
                audio, labels, features = batch
                audio = audio.to(device)
                features = features.to(device)
                embeddings = model(audio, features)
            else:
                audio, labels = batch
                audio = audio.to(device)
                embeddings = model(audio)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate Lightweight Adapter')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to LightweightAdapter checkpoint')
    parser.add_argument('--model-name', type=str, default='base',
                       help='Whisper model name')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--data-root', type=str, default=MTG_JAMENDO_ROOT,
                       help='MTG-Jamendo root')
    parser.add_argument('--split', type=str, default='split-0',
                       help='Dataset split')
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                       help='Output directory for results')
    parser.add_argument('--feature-cache', type=str, default=None,
                       help='Feature cache file')
    parser.add_argument('--feature-stats', type=str, default=None,
                       help='Feature stats file')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Output directory
    output_dir = Path(PROJECT_ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    data_root = Path(args.data_root)
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'
    
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    
    print(f"Loading validation data from {val_tsv}\n")
    
    # =========================================================================
    # EXTRACT EMBEDDINGS FROM ALL MODELS
    # =========================================================================
    
    print("="*80)
    print("EXTRACTING EMBEDDINGS FROM ALL MODELS")
    print("="*80 + "\n")
    
    embeddings_dict = {}
    
    # 1. LightweightAdapter (new model)
    print("1. LightweightAdapter (from checkpoint)")
    print("-" * 80)
    
    # Load checkpoint to get config
    checkpoint_path = Path(PROJECT_ROOT) / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Get feature stats from checkpoint directory
    if args.feature_stats is None:
        args.feature_stats = str(checkpoint_path.parent / 'feature_stats.json')
    
    # Create dataloader with features
    _, val_loader_features, _, dataset_info = create_dataloaders_with_features(
        str(val_tsv), str(val_tsv), str(val_tsv),
        str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=16000,
        duration=30.0,
        feature_cache_file=args.feature_cache,
        feature_stats_file=args.feature_stats,
        extract_on_fly=args.feature_cache is None
    )
    
    # Create model and load weights
    model_lightweight = LightweightAdapter(
        model_name=model_config['model_name'],
        feature_dim=model_config['feature_dim'],
        projection_dim=model_config['projection_dim'],
        output_dim=model_config['output_dim'],
        device=device,
        feature_stats_path=args.feature_stats
    )
    model_lightweight.load_state_dict(checkpoint['model_state_dict'])
    model_lightweight.to(device)
    model_lightweight.eval()
    
    print(f"Extracting embeddings from LightweightAdapter...")
    emb_lightweight, labels = extract_embeddings_from_model(
        model_lightweight, val_loader_features, device, with_features=True
    )
    embeddings_dict['LightweightAdapter'] = emb_lightweight
    print(f"  Shape: {emb_lightweight.shape}\n")
    
    # 2. WhisperContrastive (baseline)
    print("2. WhisperContrastive (baseline)")
    print("-" * 80)
    
    # Create dataloader without features
    _, val_loader, _, _ = create_dataloaders(
        str(val_tsv), str(val_tsv), str(val_tsv),
        str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=16000,
        duration=30.0
    )
    
    # Load WhisperContrastive
    checkpoint_whisper = Path(PROJECT_ROOT) / 'ML/checkpoints/whisper_contrastive_20251128_085448/best_model.pth'
    if checkpoint_whisper.exists():
        model_whisper = WhisperContrastive(model_name='base', projection_dim=128, device=device)
        checkpoint_data = torch.load(checkpoint_whisper, map_location=device, weights_only=True)
        model_whisper.load_state_dict(checkpoint_data['model_state_dict'])
        model_whisper.to(device)
        model_whisper.eval()
        
        print(f"Extracting embeddings from WhisperContrastive...")
        emb_whisper, _ = extract_embeddings_from_model(
            model_whisper, val_loader, device, with_features=False
        )
        embeddings_dict['WhisperContrastive'] = emb_whisper
        print(f"  Shape: {emb_whisper.shape}\n")
    else:
        print(f"  Checkpoint not found: {checkpoint_whisper}")
        print(f"  Skipping WhisperContrastive\n")
    
    # Store labels and genre names
    genre_names = [dataset_info['idx_to_genre'][i] for i in range(dataset_info['num_classes'])]
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    evaluator = EmbeddingEvaluator(embeddings_dict, labels, genre_names)
    results_df = evaluator.evaluate_all()
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    print(results_df.to_string())
    print()
    
    # Save results to CSV
    results_path = output_dir / 'comparison_results.csv'
    results_df.to_csv(results_path)
    print(f"Results saved to {results_path}\n")
    
    # =========================================================================
    # COMPARISON ANALYSIS
    # =========================================================================
    
    print("="*80)
    print("COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    if 'WhisperContrastive' in embeddings_dict and 'LightweightAdapter' in embeddings_dict:
        baseline = results_df.loc['WhisperContrastive']
        adapter = results_df.loc['LightweightAdapter']
        
        print("LightweightAdapter vs WhisperContrastive:")
        print(f"  Silhouette Score:  {adapter['silhouette']:.4f} vs {baseline['silhouette']:.4f} "
              f"({(adapter['silhouette']/baseline['silhouette']-1)*100:+.2f}%)")
        print(f"  Davies-Bouldin:    {adapter['davies_bouldin']:.4f} vs {baseline['davies_bouldin']:.4f} "
              f"({(1-adapter['davies_bouldin']/baseline['davies_bouldin'])*100:+.2f}% better)")
        print(f"  MAP@5:             {adapter['map@5']:.4f} vs {baseline['map@5']:.4f} "
              f"({(adapter['map@5']/baseline['map@5']-1)*100:+.2f}%)")
        print(f"  MAP@10:            {adapter['map@10']:.4f} vs {baseline['map@10']:.4f} "
              f"({(adapter['map@10']/baseline['map@10']-1)*100:+.2f}%)")
        print(f"  Recall@10:         {adapter['recall@10']:.4f} vs {baseline['recall@10']:.4f} "
              f"({(adapter['recall@10']/baseline['recall@10']-1)*100:+.2f}%)")
        
        # Overall improvement
        map10_improvement = (adapter['map@10'] / baseline['map@10'] - 1) * 100
        
        print(f"\n{'='*80}")
        if map10_improvement > 10:
            print(f"✅ EXCELLENT: LightweightAdapter improves MAP@10 by {map10_improvement:.1f}%")
        elif map10_improvement > 5:
            print(f"✅ GOOD: LightweightAdapter improves MAP@10 by {map10_improvement:.1f}%")
        elif map10_improvement > 0:
            print(f"⚠️  MARGINAL: LightweightAdapter improves MAP@10 by {map10_improvement:.1f}%")
        else:
            print(f"❌ WORSE: LightweightAdapter is {-map10_improvement:.1f}% worse than baseline")
        print(f"{'='*80}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    evaluator.plot_comparison(results_df, output_dir)
    evaluator.plot_tsne(output_dir)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

