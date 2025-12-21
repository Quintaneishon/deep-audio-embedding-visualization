"""
Multi-Tag Evaluation Metrics for Audio Embeddings.

Computes Hubness (Skewness) and nDCG@K for instrument and mood/theme tags.
These metrics evaluate embedding quality on non-genre semantic dimensions.
"""

import numpy as np
from scipy.stats import skew
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Dict, Tuple


def compute_hubness_skewness(embeddings: np.ndarray, k: int = 10) -> Tuple[float, np.ndarray]:
    """
    Compute hubness as skewness of k-occurrence distribution.
    
    Hubness is a phenomenon in high-dimensional spaces where some points
    become hubs that appear in many k-nearest neighbor lists.
    
    Args:
        embeddings: [N, D] embedding matrix
        k: Number of nearest neighbors to consider
    
    Returns:
        skewness: Hubness measure (higher = more hubness problem, <0.2 is good)
        k_occurrences: Array of k-occurrence counts for each point
    """
    n_samples = len(embeddings)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # For each sample, find k nearest neighbors
    k_occurrences = np.zeros(n_samples)
    
    for i in range(n_samples):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self
        
        # Get k nearest neighbors
        top_k_indices = np.argsort(sims)[-k:]
        
        # Count occurrences
        k_occurrences[top_k_indices] += 1
    
    # Compute skewness
    skewness_value = skew(k_occurrences)
    
    return skewness_value, k_occurrences


def compute_ndcg_at_k(embeddings: np.ndarray, labels: List[Set], k: int = 10) -> float:
    """
    Compute nDCG@K for multi-label tag-based retrieval.
    
    For each query, retrieves k nearest neighbors and computes relevance
    based on tag overlap (Jaccard similarity).
    
    Args:
        embeddings: [N, D] embedding matrix
        labels: List of N sets, where each set contains tags for that sample
        k: Number of results to consider
    
    Returns:
        mean_ndcg: Average nDCG@K across all queries (0-1, higher is better)
    """
    n_samples = len(embeddings)
    sim_matrix = cosine_similarity(embeddings)
    
    ndcg_scores = []
    
    for i in range(n_samples):
        # Get query labels
        query_labels = labels[i]
        
        if len(query_labels) == 0:
            continue
        
        # Get similarities
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self
        
        # Get top-k results
        top_k_indices = np.argsort(sims)[-k:][::-1]
        
        # Compute relevance scores (Jaccard similarity)
        relevance = []
        for idx in top_k_indices:
            retrieved_labels = labels[idx]
            
            # Jaccard similarity
            if len(query_labels | retrieved_labels) > 0:
                jaccard = len(query_labels & retrieved_labels) / len(query_labels | retrieved_labels)
            else:
                jaccard = 0.0
            
            relevance.append(jaccard)
        
        relevance = np.array(relevance)
        
        # DCG@K
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
        
        # Ideal DCG (if we had perfect ranking)
        ideal_relevance = np.sort(relevance)[::-1]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
        
        # nDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def load_multi_label_tags_from_csv(csv_path: str) -> Tuple[List[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Load instrument and mood/theme tags from CSV file.
    
    Args:
        csv_path: Path to CSV with columns: track_id, filename, instrument_tags, mood_tags, all_tags
    
    Returns:
        track_ids: List of track IDs
        instrument_tags: Dict mapping track_id to set of instrument tags
        mood_tags: Dict mapping track_id to set of mood tags
    """
    import csv
    
    track_ids = []
    instrument_tags = {}
    mood_tags = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            track_id = row['track_id']
            track_ids.append(track_id)
            
            # Parse instrument tags
            inst_str = row.get('instrument_tags', '')
            if inst_str:
                instrument_tags[track_id] = set(inst_str.split('|'))
            else:
                instrument_tags[track_id] = set()
            
            # Parse mood tags
            mood_str = row.get('mood_tags', '')
            if mood_str:
                mood_tags[track_id] = set(mood_str.split('|'))
            else:
                mood_tags[track_id] = set()
    
    return track_ids, instrument_tags, mood_tags


def evaluate_embeddings_with_tags(
    embeddings: np.ndarray,
    track_ids: List[str],
    instrument_tags: Dict[str, Set[str]],
    mood_tags: Dict[str, Set[str]],
    k: int = 10
) -> Dict[str, float]:
    """
    Comprehensive evaluation of embeddings using instrument and mood tags.
    
    Args:
        embeddings: [N, D] embedding matrix
        track_ids: List of track IDs corresponding to embeddings
        instrument_tags: Dict mapping track_id to set of instrument tags
        mood_tags: Dict mapping track_id to set of mood tags
        k: Number of neighbors for metrics
    
    Returns:
        results: Dict with metric scores
    """
    # Prepare tag lists
    all_tags = [instrument_tags[tid] | mood_tags[tid] for tid in track_ids]
    instrument_only = [instrument_tags[tid] for tid in track_ids]
    mood_only = [mood_tags[tid] for tid in track_ids]
    
    # Filter samples with at least one tag
    valid_indices = [i for i, tags in enumerate(all_tags) if len(tags) > 0]
    
    if len(valid_indices) == 0:
        print("Warning: No tracks with instrument/mood tags found!")
        return {}
    
    filtered_embeddings = embeddings[valid_indices]
    filtered_all_tags = [all_tags[i] for i in valid_indices]
    filtered_instrument = [instrument_only[i] for i in valid_indices]
    filtered_mood = [mood_only[i] for i in valid_indices]
    
    # Compute Hubness
    skewness, k_occ = compute_hubness_skewness(filtered_embeddings, k=k)
    
    # Compute nDCG for combined tags
    ndcg_combined = compute_ndcg_at_k(filtered_embeddings, filtered_all_tags, k=k)
    
    # Compute nDCG for instrument-only (filter to samples with instrument tags)
    instrument_valid_idx = [i for i, tags in enumerate(filtered_instrument) if len(tags) > 0]
    if len(instrument_valid_idx) > 0:
        ndcg_instrument = compute_ndcg_at_k(
            filtered_embeddings[instrument_valid_idx],
            [filtered_instrument[i] for i in instrument_valid_idx],
            k=k
        )
    else:
        ndcg_instrument = 0.0
    
    # Compute nDCG for mood-only (filter to samples with mood tags)
    mood_valid_idx = [i for i, tags in enumerate(filtered_mood) if len(tags) > 0]
    if len(mood_valid_idx) > 0:
        ndcg_mood = compute_ndcg_at_k(
            filtered_embeddings[mood_valid_idx],
            [filtered_mood[i] for i in mood_valid_idx],
            k=k
        )
    else:
        ndcg_mood = 0.0
    
    results = {
        'hubness_skewness': float(skewness),
        'ndcg@10_combined': float(ndcg_combined),
        'ndcg@10_instrument': float(ndcg_instrument),
        'ndcg@10_mood': float(ndcg_mood),
        'n_samples_evaluated': len(filtered_embeddings),
        'n_with_instrument': len(instrument_valid_idx),
        'n_with_mood': len(mood_valid_idx),
        'mean_k_occurrence': float(np.mean(k_occ)),
        'std_k_occurrence': float(np.std(k_occ)),
        'max_k_occurrence': float(np.max(k_occ))
    }
    
    return results


def format_multi_tag_results(results: Dict[str, float]) -> str:
    """
    Format multi-tag evaluation results for reporting.
    
    Args:
        results: Results dict from evaluate_embeddings_with_tags
    
    Returns:
        formatted_str: Formatted string for display/logging
    """
    if not results:
        return "No multi-tag evaluation results available."
    
    output = []
    output.append("=== Multi-Tag Evaluation (Instrument & Mood/Theme) ===")
    output.append(f"Samples evaluated: {results['n_samples_evaluated']}")
    output.append(f"  With instrument tags: {results['n_with_instrument']}")
    output.append(f"  With mood tags: {results['n_with_mood']}")
    output.append("")
    output.append("HUBNESS (Skewness of k-occurrences):")
    output.append(f"  Skewness: {results['hubness_skewness']:.4f}")
    output.append(f"  (Lower is better, <0.2 is good, <0.5 is acceptable)")
    output.append(f"  Mean k-occurrence: {results['mean_k_occurrence']:.2f}")
    output.append(f"  Max k-occurrence: {results['max_k_occurrence']:.0f}")
    output.append("")
    output.append("nDCG@10 (Tag-based Retrieval):")
    output.append(f"  Combined (Instrument + Mood): {results['ndcg@10_combined']:.4f}")
    output.append(f"  Instrument only: {results['ndcg@10_instrument']:.4f}")
    output.append(f"  Mood/Theme only: {results['ndcg@10_mood']:.4f}")
    output.append(f"  (Higher is better, 0-1 range)")
    
    return '\n'.join(output)

