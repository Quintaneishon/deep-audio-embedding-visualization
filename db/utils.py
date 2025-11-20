"""
Utility functions for preprocessing operations.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import database


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
        '''SELECT e.*, t.filename 
           FROM embeddings e 
           JOIN tracks t ON e.track_id = t.id 
           WHERE e.model = ? AND e.dataset = ?''',
        (model, dataset)
    ).fetchall()
    
    if not embeddings_records:
        print(f"  No embeddings found for {combo_key}, skipping...")
        return None
    
    print(f"  Found {len(embeddings_records)} tracks")
    
    # Load embeddings and taggrams, group by genre from CSV
    genre_song_embeddings, genre_song_taggrams, song_info, skipped_count = load_embeddings_by_genre(
        embeddings_records, genre_map, genre_tags, model, dataset
    )
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} tracks (no genre in CSV or missing data)")
    
    print(f"  Processing {len(song_info)} tracks with genre labels")
    
    # Compute genre centroids (mean embeddings and taggrams)
    print(f"\n  Computing genre centroids...")
    genre_embedding_centroids, genre_taggram_centroids, genre_stats = compute_genre_centroids(
        genre_song_embeddings, genre_song_taggrams, genre_tags
    )
    
    print(f"\n  Found {len(genre_embedding_centroids)} genres with songs")
    
    # Compute similarity scores for each song
    print(f"\n  Computing embedding and taggram similarity scores...")
    song_similarities = compute_song_similarities(
        song_info, genre_embedding_centroids, genre_taggram_centroids
    )
    
    # Calculate aggregate statistics
    aggregate_stats = compute_aggregate_statistics(song_similarities)
    
    # Print statistics
    print_statistics(combo_key, aggregate_stats)
    
    return {
        'genre_embedding_centroids': genre_embedding_centroids,
        'genre_taggram_centroids': genre_taggram_centroids,
        'genre_stats': genre_stats,
        'song_similarities': song_similarities,
        'aggregate_stats': aggregate_stats
    }

