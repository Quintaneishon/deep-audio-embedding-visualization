"""
Utility functions for preprocessing operations.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import database
import gc
import sqlite3
import config


def load_embeddings_by_genre(embeddings_records, genre_map, genre_tags, model, dataset):
    """
    Load embeddings and taggrams from database, grouped by genre.
    
    Args:
        embeddings_records: List of embedding records from database
        genre_map: Dictionary mapping filename to genre
        genre_tags: List of unique genre tags
        model: Model name
        dataset: Dataset/model size name
    
    Returns:
        Tuple of (genre_song_embeddings, genre_song_taggrams, song_info, skipped_count)
    """
    genre_song_embeddings = {genre: [] for genre in genre_tags}
    genre_song_taggrams = {genre: [] for genre in genre_tags}
    song_info = []
    skipped_count = 0
    
    for record in embeddings_records:
        filename = record['filename']
        
        # Check if filename has genre in CSV
        if filename not in genre_map:
            skipped_count += 1
            continue
        
        genre = genre_map[filename]
        
        # Load embedding and taggram from database
        data = database.get_embedding_by_filename(filename, model, dataset)
        
        if data is None:
            print(f"  Warning: Missing data for {filename}, skipping...")
            skipped_count += 1
            continue
        
        embedding = data['embedding']
        taggram = data['taggram']
        
        # Store for centroid computation
        genre_song_embeddings[genre].append(embedding[0])  # [0] to get 1D array
        genre_song_taggrams[genre].append(taggram[0])
        song_info.append({
            'filename': filename,
            'embedding': embedding[0],
            'taggram': taggram[0],
            'genre': genre
        })
    
    return genre_song_embeddings, genre_song_taggrams, song_info, skipped_count


def compute_genre_centroids(genre_song_embeddings, genre_song_taggrams, genre_tags):
    """
    Compute mean embeddings and taggrams (centroids) for each genre.
    
    Args:
        genre_song_embeddings: Dictionary of genre -> list of embeddings
        genre_song_taggrams: Dictionary of genre -> list of taggrams
        genre_tags: List of unique genre tags
    
    Returns:
        Tuple of (genre_embedding_centroids, genre_taggram_centroids, genre_stats)
    """
    genre_embedding_centroids = {}
    genre_taggram_centroids = {}
    genre_stats = {}
    
    for genre in genre_tags:
        embeddings_list = genre_song_embeddings[genre]
        taggrams_list = genre_song_taggrams[genre]
        
        if len(embeddings_list) > 0:
            # Compute embedding centroid
            embedding_centroid = np.mean(embeddings_list, axis=0)
            genre_embedding_centroids[genre] = embedding_centroid
            
            # Compute taggram centroid
            taggram_centroid = np.mean(taggrams_list, axis=0)
            genre_taggram_centroids[genre] = taggram_centroid
            
            genre_stats[genre] = {
                'count': len(embeddings_list),
                'embedding_centroid_norm': float(np.linalg.norm(embedding_centroid)),
                'taggram_centroid_norm': float(np.linalg.norm(taggram_centroid))
            }
    
    return genre_embedding_centroids, genre_taggram_centroids, genre_stats


def compute_song_similarities(song_info, genre_embedding_centroids, genre_taggram_centroids):
    """
    Compute similarity scores for each song to genre centroids using both embeddings and taggrams.
    
    Args:
        song_info: List of song dictionaries with filename, embedding, taggram, genre
        genre_embedding_centroids: Dictionary of genre -> embedding centroid
        genre_taggram_centroids: Dictionary of genre -> taggram centroid
    
    Returns:
        List of song similarity dictionaries
    """
    song_similarities = []
    
    for song in song_info:
        song_embedding = song['embedding'].reshape(1, -1)
        song_taggram = song['taggram'].reshape(1, -1)
        genre = song['genre']
        
        # ===== EMBEDDING SIMILARITIES =====
        # Compute cosine similarity to own genre centroid (embedding)
        if genre in genre_embedding_centroids:
            own_genre_centroid = genre_embedding_centroids[genre].reshape(1, -1)
            emb_similarity_to_own_genre = float(
                cosine_similarity(song_embedding, own_genre_centroid)[0][0]
            )
        else:
            emb_similarity_to_own_genre = None
        
        # Compute embedding similarity to all genre centroids
        all_emb_genre_similarities = {}
        for g, centroid in genre_embedding_centroids.items():
            centroid_reshaped = centroid.reshape(1, -1)
            sim = float(cosine_similarity(song_embedding, centroid_reshaped)[0][0])
            all_emb_genre_similarities[g] = sim
        
        # Find most similar genre by embedding
        if all_emb_genre_similarities:
            emb_most_similar_genre = max(all_emb_genre_similarities, key=all_emb_genre_similarities.get)
            emb_max_similarity = all_emb_genre_similarities[emb_most_similar_genre]
        else:
            emb_most_similar_genre = None
            emb_max_similarity = None
        
        # ===== TAGGRAM SIMILARITIES =====
        # Compute cosine similarity to own genre centroid (taggram)
        if genre in genre_taggram_centroids:
            own_genre_taggram_centroid = genre_taggram_centroids[genre].reshape(1, -1)
            tag_similarity_to_own_genre = float(
                cosine_similarity(song_taggram, own_genre_taggram_centroid)[0][0]
            )
        else:
            tag_similarity_to_own_genre = None
        
        # Compute taggram similarity to all genre centroids
        all_tag_genre_similarities = {}
        for g, centroid in genre_taggram_centroids.items():
            centroid_reshaped = centroid.reshape(1, -1)
            sim = float(cosine_similarity(song_taggram, centroid_reshaped)[0][0])
            all_tag_genre_similarities[g] = sim
        
        # Find most similar genre by taggram
        if all_tag_genre_similarities:
            tag_most_similar_genre = max(all_tag_genre_similarities, key=all_tag_genre_similarities.get)
            tag_max_similarity = all_tag_genre_similarities[tag_most_similar_genre]
        else:
            tag_most_similar_genre = None
            tag_max_similarity = None
        
        song_similarities.append({
            'filename': song['filename'],
            'genre': genre,
            # Embedding-based metrics
            'emb_similarity_to_own_genre': emb_similarity_to_own_genre,
            'emb_most_similar_genre': emb_most_similar_genre,
            'emb_max_similarity': emb_max_similarity,
            'emb_agreement': genre == emb_most_similar_genre,
            'all_emb_similarities': all_emb_genre_similarities,
            # Taggram-based metrics
            'tag_similarity_to_own_genre': tag_similarity_to_own_genre,
            'tag_most_similar_genre': tag_most_similar_genre,
            'tag_max_similarity': tag_max_similarity,
            'tag_agreement': genre == tag_most_similar_genre,
            'all_tag_similarities': all_tag_genre_similarities
        })
    
    return song_similarities


def compute_aggregate_statistics(song_similarities):
    """
    Calculate aggregate statistics from song similarity scores.
    
    Args:
        song_similarities: List of song similarity dictionaries
    
    Returns:
        Dictionary with aggregate statistics for both embeddings and taggrams
    """
    # Calculate aggregate statistics for embeddings and taggrams
    emb_similarities_to_own = [s['emb_similarity_to_own_genre'] for s in song_similarities 
                               if s['emb_similarity_to_own_genre'] is not None]
    tag_similarities_to_own = [s['tag_similarity_to_own_genre'] for s in song_similarities 
                               if s['tag_similarity_to_own_genre'] is not None]
    
    # Count agreements
    emb_agreements = sum(1 for s in song_similarities if s['emb_agreement'])
    tag_agreements = sum(1 for s in song_similarities if s['tag_agreement'])
    
    return {
        # Embedding stats
        'emb_mean_similarity_to_own_genre': float(np.mean(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
        'emb_std_similarity_to_own_genre': float(np.std(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
        'emb_min_similarity': float(np.min(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
        'emb_max_similarity': float(np.max(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
        'emb_agreement_rate': emb_agreements / len(song_similarities) if song_similarities else 0.0,
        # Taggram stats
        'tag_mean_similarity_to_own_genre': float(np.mean(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
        'tag_std_similarity_to_own_genre': float(np.std(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
        'tag_min_similarity': float(np.min(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
        'tag_max_similarity': float(np.max(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
        'tag_agreement_rate': tag_agreements / len(song_similarities) if song_similarities else 0.0,
        # General stats
        'total_songs': len(song_similarities)
    }


def print_statistics(combo_key, aggregate_stats):
    """
    Print statistics in a formatted way.
    
    Args:
        combo_key: String identifier for the model/dataset combination
        aggregate_stats: Dictionary of aggregate statistics
    """
    print(f"\n  Statistics:")
    print(f"    EMBEDDING-BASED:")
    print(f"      Mean similarity to own genre: {aggregate_stats['emb_mean_similarity_to_own_genre']:.4f}")
    print(f"      Std deviation: {aggregate_stats['emb_std_similarity_to_own_genre']:.4f}")
    print(f"      Agreement rate: {aggregate_stats['emb_agreement_rate']:.2%}")
    print(f"    TAGGRAM-BASED:")
    print(f"      Mean similarity to own genre: {aggregate_stats['tag_mean_similarity_to_own_genre']:.4f}")
    print(f"      Std deviation: {aggregate_stats['tag_std_similarity_to_own_genre']:.4f}")
    print(f"      Agreement rate: {aggregate_stats['tag_agreement_rate']:.2%}")
    
    # Print MAP@K statistics if available
    if 'map_at_k_genre_precision' in aggregate_stats:
        k_value = aggregate_stats.get('map_at_k_k_value', 10)
        n_songs = aggregate_stats.get('map_at_k_n_songs_evaluated', 0)
        print(f"    MAP@{k_value} (K-NEAREST NEIGHBORS):")
        print(f"      Songs evaluated: {n_songs}")
        print(f"      Genre Precision: {aggregate_stats['map_at_k_genre_precision']:.4f} ± {aggregate_stats.get('map_at_k_std_genre_precision', 0):.4f}")
        print(f"      Physical Error (normalized): {aggregate_stats['map_at_k_mean_physical_error']:.4f} ± {aggregate_stats.get('map_at_k_std_physical_error', 0):.4f}")
        print(f"        └─ Centroid component: {aggregate_stats['map_at_k_centroid_component']:.4f}")
        print(f"        └─ Tempo component: {aggregate_stats['map_at_k_tempo_component']:.4f}")


def compute_centroids_streaming(embeddings_records, genre_map, genre_tags, model, dataset):
    """
    Compute genre centroids using streaming approach (memory efficient).
    Only keeps running sums, not all embeddings.
    
    Args:
        embeddings_records: List of embedding records from database
        genre_map: Dictionary mapping filename to genre
        genre_tags: List of unique genre tags
        model: Model name
        dataset: Dataset/model size name
    
    Returns:
        Tuple of (genre_embedding_centroids, genre_taggram_centroids)
    """
    # Initialize accumulators
    genre_emb_sums = {genre: None for genre in genre_tags}
    genre_tag_sums = {genre: None for genre in genre_tags}
    genre_counts = {genre: 0 for genre in genre_tags}
    
    for record in embeddings_records:
        filename = record['filename']
        
        if filename not in genre_map:
            continue
        
        genre = genre_map[filename]
        
        # Load embedding and taggram
        data = database.get_embedding_by_filename(filename, model, dataset)
        if data is None:
            continue
        
        embedding = data['embedding'][0]  # 1D array
        taggram = data['taggram'][0]
        
        # Accumulate sums (not storing individual embeddings!)
        if genre_emb_sums[genre] is None:
            genre_emb_sums[genre] = embedding.copy()
            genre_tag_sums[genre] = taggram.copy()
        else:
            genre_emb_sums[genre] += embedding
            genre_tag_sums[genre] += taggram
        
        genre_counts[genre] += 1
        
        # Clean up immediately after use
        del embedding, taggram, data
    
    # Compute centroids (averages)
    genre_embedding_centroids = {}
    genre_taggram_centroids = {}
    
    for genre in genre_tags:
        if genre_counts[genre] > 0:
            genre_embedding_centroids[genre] = genre_emb_sums[genre] / genre_counts[genre]
            genre_taggram_centroids[genre] = genre_tag_sums[genre] / genre_counts[genre]
    
    return genre_embedding_centroids, genre_taggram_centroids


def compute_single_song_similarity(filename, genre, song_embedding, song_taggram,
                                   genre_embedding_centroids, genre_taggram_centroids):
    """
    Compute similarity metrics for a single song.
    
    Args:
        filename: Song filename
        genre: Song's true genre
        song_embedding: Song's embedding vector (reshaped to 2D)
        song_taggram: Song's taggram vector (reshaped to 2D)
        genre_embedding_centroids: Dictionary of genre -> embedding centroid
        genre_taggram_centroids: Dictionary of genre -> taggram centroid
    
    Returns:
        Dictionary with similarity metrics
    """
    # ===== EMBEDDING SIMILARITIES =====
    if genre in genre_embedding_centroids:
        own_genre_centroid = genre_embedding_centroids[genre].reshape(1, -1)
        emb_similarity_to_own_genre = float(
            cosine_similarity(song_embedding, own_genre_centroid)[0][0]
        )
    else:
        emb_similarity_to_own_genre = None
    
    all_emb_genre_similarities = {}
    for g, centroid in genre_embedding_centroids.items():
        centroid_reshaped = centroid.reshape(1, -1)
        sim = float(cosine_similarity(song_embedding, centroid_reshaped)[0][0])
        all_emb_genre_similarities[g] = sim
    
    if all_emb_genre_similarities:
        emb_most_similar_genre = max(all_emb_genre_similarities, key=all_emb_genre_similarities.get)
        emb_max_similarity = all_emb_genre_similarities[emb_most_similar_genre]
    else:
        emb_most_similar_genre = None
        emb_max_similarity = None
    
    # ===== TAGGRAM SIMILARITIES =====
    if genre in genre_taggram_centroids:
        own_genre_taggram_centroid = genre_taggram_centroids[genre].reshape(1, -1)
        tag_similarity_to_own_genre = float(
            cosine_similarity(song_taggram, own_genre_taggram_centroid)[0][0]
        )
    else:
        tag_similarity_to_own_genre = None
    
    all_tag_genre_similarities = {}
    for g, centroid in genre_taggram_centroids.items():
        centroid_reshaped = centroid.reshape(1, -1)
        sim = float(cosine_similarity(song_taggram, centroid_reshaped)[0][0])
        all_tag_genre_similarities[g] = sim
    
    if all_tag_genre_similarities:
        tag_most_similar_genre = max(all_tag_genre_similarities, key=all_tag_genre_similarities.get)
        tag_max_similarity = all_tag_genre_similarities[tag_most_similar_genre]
    else:
        tag_most_similar_genre = None
        tag_max_similarity = None
    
    return {
        'filename': filename,
        'genre': genre,
        'emb_similarity_to_own_genre': emb_similarity_to_own_genre,
        'emb_most_similar_genre': emb_most_similar_genre,
        'emb_max_similarity': emb_max_similarity,
        'emb_agreement': genre == emb_most_similar_genre,
        'all_emb_similarities': all_emb_genre_similarities,
        'tag_similarity_to_own_genre': tag_similarity_to_own_genre,
        'tag_most_similar_genre': tag_most_similar_genre,
        'tag_max_similarity': tag_max_similarity,
        'tag_agreement': genre == tag_most_similar_genre,
        'all_tag_similarities': all_tag_genre_similarities
    }


def compute_similarities_streaming(embeddings_records, genre_map, 
                                   genre_embedding_centroids, genre_taggram_centroids, 
                                   model, dataset):
    """
    Compute similarities song by song (streaming, memory efficient).
    
    Args:
        embeddings_records: List of embedding records from database
        genre_map: Dictionary mapping filename to genre
        genre_embedding_centroids: Dictionary of genre -> embedding centroid
        genre_taggram_centroids: Dictionary of genre -> taggram centroid
        model: Model name
        dataset: Dataset/model size name
    
    Returns:
        List of song similarity dictionaries
    """
    song_similarities = []
    
    for i, record in enumerate(tqdm(embeddings_records, desc="  Computing similarities", unit="song")):
        filename = record['filename']
        
        if filename not in genre_map:
            continue
        
        genre = genre_map[filename]
        
        # Load one song at a time
        data = database.get_embedding_by_filename(filename, model, dataset)
        if data is None:
            continue
        
        song_embedding = data['embedding'][0].reshape(1, -1)
        song_taggram = data['taggram'][0].reshape(1, -1)
        
        # Compute similarities for this song
        song_sim = compute_single_song_similarity(
            filename, genre, song_embedding, song_taggram,
            genre_embedding_centroids, genre_taggram_centroids
        )
        
        song_similarities.append(song_sim)
        
        # Clean up immediately
        del data, song_embedding, song_taggram
        
        # Periodic garbage collection
        if (i + 1) % 1000 == 0:
            gc.collect()
    
    return song_similarities


def find_k_nearest_neighbors_indices(query_idx, embeddings_matrix, k=10):
    """
    Find K nearest neighbors for a query embedding using cosine similarity.
    
    Args:
        query_idx: Index of query embedding in the matrix
        embeddings_matrix: Numpy array of shape [n_samples, embedding_dim]
        k: Number of nearest neighbors to find
    
    Returns:
        List of indices of K nearest neighbors (excluding query itself)
    """
    query_embedding = embeddings_matrix[query_idx:query_idx+1]  # Keep 2D shape
    
    # Compute cosine similarities to all embeddings
    similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
    
    # Set self-similarity to -inf to exclude it
    similarities[query_idx] = -np.inf
    
    # Get indices of k largest similarities
    k_nearest_indices = np.argsort(similarities)[-k:][::-1]
    
    return k_nearest_indices.tolist()


def compute_physical_error(centroid_q, centroid_n, tempo_q, tempo_n, normalization_stats):
    """
    Compute normalized physical error between query and neighbor.
    
    Args:
        centroid_q: Spectral centroid of query song (Hz)
        centroid_n: Spectral centroid of neighbor song (Hz)
        tempo_q: Tempo of query song (BPM)
        tempo_n: Tempo of neighbor song (BPM)
        normalization_stats: Dict with 'centroid_range' and 'tempo_range'
    
    Returns:
        Float: Normalized physical error (sum of normalized distances)
    """
    # Normalize by range to make features contribute equally
    centroid_error = abs(centroid_q - centroid_n) / (normalization_stats['centroid_range'] + 1e-8)
    tempo_error = abs(tempo_q - tempo_n) / (normalization_stats['tempo_range'] + 1e-8)
    
    return centroid_error + tempo_error, centroid_error, tempo_error


def compute_map_at_k_with_physical_error(song_info_list, k=10):
    """
    Compute MAP@K with genre precision and physical error metrics.
    
    Args:
        song_info_list: List of dicts with keys: filename, genre, embedding, 
                       spectral_centroid, tempo
        k: Number of nearest neighbors to evaluate
    
    Returns:
        Dict with per-song results and aggregate statistics
    """
    print(f"  Computing MAP@{k} with physical error...")
    
    # Filter songs that have acoustic features
    valid_songs = [s for s in song_info_list 
                   if s.get('spectral_centroid') is not None 
                   and s.get('tempo') is not None]
    
    if len(valid_songs) == 0:
        print(f"    Warning: No songs with acoustic features found")
        return None
    
    if len(valid_songs) < k:
        print(f"    Warning: Only {len(valid_songs)} songs with features, using k={len(valid_songs)-1}")
        k = max(1, len(valid_songs) - 1)
    
    # Build embeddings matrix
    embeddings_matrix = np.vstack([s['embedding'].reshape(1, -1) for s in valid_songs])
    
    # Compute normalization stats for physical error
    centroids = [s['spectral_centroid'] for s in valid_songs]
    tempos = [s['tempo'] for s in valid_songs]
    
    normalization_stats = {
        'centroid_range': max(centroids) - min(centroids),
        'tempo_range': max(tempos) - min(tempos),
        'centroid_min': min(centroids),
        'centroid_max': max(centroids),
        'tempo_min': min(tempos),
        'tempo_max': max(tempos)
    }
    
    # Evaluate each song
    per_song_results = []
    genre_precisions = []
    physical_errors = []
    centroid_errors = []
    tempo_errors = []
    
    for i, query_song in enumerate(valid_songs):
        # Find k nearest neighbors
        neighbor_indices = find_k_nearest_neighbors_indices(i, embeddings_matrix, k)
        
        # Evaluate neighbors
        same_genre_count = 0
        neighbor_errors = []
        neighbor_centroid_errors = []
        neighbor_tempo_errors = []
        
        for neighbor_idx in neighbor_indices:
            neighbor = valid_songs[neighbor_idx]
            
            # Check genre match
            if neighbor['genre'] == query_song['genre']:
                same_genre_count += 1
            
            # Compute physical error
            phys_err, cent_err, temp_err = compute_physical_error(
                query_song['spectral_centroid'],
                neighbor['spectral_centroid'],
                query_song['tempo'],
                neighbor['tempo'],
                normalization_stats
            )
            neighbor_errors.append(phys_err)
            neighbor_centroid_errors.append(cent_err)
            neighbor_tempo_errors.append(temp_err)
        
        # Calculate metrics
        genre_precision = same_genre_count / k
        mean_physical_error = np.mean(neighbor_errors)
        mean_centroid_error = np.mean(neighbor_centroid_errors)
        mean_tempo_error = np.mean(neighbor_tempo_errors)
        
        per_song_results.append({
            'filename': query_song['filename'],
            'genre': query_song['genre'],
            'genre_precision': genre_precision,
            'physical_error': mean_physical_error,
            'centroid_error': mean_centroid_error,
            'tempo_error': mean_tempo_error
        })
        
        genre_precisions.append(genre_precision)
        physical_errors.append(mean_physical_error)
        centroid_errors.append(mean_centroid_error)
        tempo_errors.append(mean_tempo_error)
    
    # Aggregate statistics
    aggregate_stats = {
        'map_at_k_genre_precision': float(np.mean(genre_precisions)),
        'map_at_k_mean_physical_error': float(np.mean(physical_errors)),
        'map_at_k_centroid_component': float(np.mean(centroid_errors)),
        'map_at_k_tempo_component': float(np.mean(tempo_errors)),
        'map_at_k_std_genre_precision': float(np.std(genre_precisions)),
        'map_at_k_std_physical_error': float(np.std(physical_errors)),
        'map_at_k_n_songs_evaluated': len(valid_songs),
        'map_at_k_k_value': k
    }
    
    return {
        'per_song_results': per_song_results,
        'aggregate_stats': aggregate_stats,
        'normalization_stats': normalization_stats
    }


def process_model_dataset_combination(model, dataset, genre_map, genre_tags):
    """
    Process a single model/dataset combination for genre similarity analysis.
    
    Args:
        model: Model name
        dataset: Dataset or model size name
        genre_map: Dictionary mapping filename to genre
        genre_tags: List of unique genre tags
    
    Returns:
        Dictionary with genre centroids, song similarities, and aggregate stats, or None if no data
    """
    combo_key = f"{model}_{dataset}"
    print(f"\n{'='*60}")
    print(f"Processing: {model.upper()} / {dataset.upper()}")
    print(f"{'='*60}")
    
    # Get all tracks with embeddings for this combo
    db = database.get_db()
    embeddings_records = db.execute(
        '''SELECT e.*, t.filename, t.spectral_centroid, t.tempo 
           FROM embeddings e 
           JOIN tracks t ON e.track_id = t.id 
           WHERE e.model = ? AND e.dataset = ?''',
        (model, dataset)
    ).fetchall()
    
    if not embeddings_records:
        print(f"  No embeddings found for {combo_key}, skipping...")
        return None
    
    print(f"  Found {len(embeddings_records)} tracks")
    
    print(f"  Computing genre centroids...")
    genre_embedding_centroids, genre_taggram_centroids = compute_centroids_streaming(
        embeddings_records, genre_map, genre_tags, model, dataset
    )
    
    if not genre_embedding_centroids:
        print(f"  No valid genre centroids computed, skipping...")
        return None
    
    print(f"\n  Found {len(genre_embedding_centroids)} genres with songs\n")
    
    song_similarities = compute_similarities_streaming(
        embeddings_records, genre_map, genre_embedding_centroids, 
        genre_taggram_centroids, model, dataset
    )
    
    if not song_similarities:
        print(f"  No valid songs found, skipping...")
        return None
    
    print(f"  Processing {len(song_similarities)} tracks with genre labels")
    
    # Calculate aggregate statistics
    aggregate_stats = compute_aggregate_statistics(song_similarities)
    
    # Compute MAP@K with physical error
    # Build song_info_list with acoustic features
    song_info_list = []
    for record in embeddings_records:
        filename = record['filename']
        if filename not in genre_map:
            continue
        
        # Load embedding
        data = database.get_embedding_by_filename(filename, model, dataset)
        if data is None:
            continue
        
        song_info_list.append({
            'filename': filename,
            'genre': genre_map[filename],
            'embedding': data['embedding'][0],
            'spectral_centroid': record['spectral_centroid'],
            'tempo': record['tempo']
        })
    
    # Compute MAP@K metrics
    map_at_k_result = compute_map_at_k_with_physical_error(song_info_list, k=10)
    
    if map_at_k_result is not None:
        # Merge MAP@K stats into aggregate_stats
        aggregate_stats.update(map_at_k_result['aggregate_stats'])
    
    # Print statistics
    print_statistics(combo_key, aggregate_stats)
    
    # Clean up large data structures before returning
    del embeddings_records
    del genre_embedding_centroids
    del genre_taggram_centroids
    del song_similarities
    del song_info_list
    if map_at_k_result:
        del map_at_k_result
    gc.collect()
    
    return {
        'aggregate_stats': aggregate_stats
    }

