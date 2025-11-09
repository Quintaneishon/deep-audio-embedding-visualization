"""
Preprocessing module for batch extracting and caching embeddings, taggrams, and projections.
"""
import os
import librosa
from pathlib import Path
from flask import g
import config
import database
import cache_manager
from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG
from proyecciones import proyectar_embeddings


def index_audio_files():
    """
    Scan audio directory and add all audio files to the database.
    
    Returns:
        Number of new tracks indexed
    """
    audio_dir = Path(config.AUDIO_DIR)
    if not audio_dir.exists():
        print(f"Error: Audio directory {config.AUDIO_DIR} does not exist")
        return 0
    
    # Get all audio files
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(tuple(audio_extensions))
    ]
    
    indexed_count = 0
    for filename in audio_files:
        # Check if already in database
        existing = database.get_track_by_filename(filename)
        if existing is None:
            # Get audio duration
            try:
                audio_path = audio_dir / filename
                duration = librosa.get_duration(path=str(audio_path))
            except Exception as e:
                print(f"Warning: Could not get duration for {filename}: {e}")
                duration = None
            
            # Insert into database
            database.insert_track(filename, duration)
            indexed_count += 1
            print(f"Indexed: {filename}")
    
    print(f"\nIndexed {indexed_count} new tracks")
    return indexed_count


def extract_embeddings_for_track(track_id, model, dataset):
    """
    Extract and cache embeddings and taggrams for a specific track.
    
    Args:
        track_id: Database ID of the track
        model: Model name ('musicnn' or 'vgg')
        dataset: Dataset name ('msd' or 'mtat')
    
    Returns:
        Tuple of (embedding_id, success)
    """
    # Get track info
    db = database.get_db()
    track = db.execute('SELECT * FROM tracks WHERE id = ?', (track_id,)).fetchone()
    if track is None:
        print(f"Error: Track with id {track_id} not found")
        return None, False
    
    filename = track['filename']
    audio_path = os.path.join(config.AUDIO_DIR, filename)
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found")
        return None, False
    
    # Get model weights path
    try:
        weights_path = config.MODEL_WEIGHTS[model][dataset]
    except KeyError:
        print(f"Error: Invalid model/dataset combination: {model}/{dataset}")
        return None, False
    
    print(f"Extracting embeddings for {filename} using {model}/{dataset}...")
    
    try:
        # Extract embeddings and taggrams
        if model == 'musicnn':
            embeddings, taggrams = embeddings_y_taggrams_MusiCNN(
                weights_path, audio_path, dataset_name=dataset
            )
        elif model == 'vgg':
            embeddings, taggrams = embeddings_y_taggrams_VGG(
                weights_path, audio_path
            )
        else:
            print(f"Error: Unknown model {model}")
            return None, False
        
        # Save to cache
        embedding_path = cache_manager.save_to_cache(
            embeddings, 'embedding', filename, model, dataset
        )
        taggram_path = cache_manager.save_to_cache(
            taggrams, 'taggram', filename, model, dataset
        )
        
        # Insert into database
        embedding_id = database.insert_embedding(
            track_id, model, dataset, embedding_path, taggram_path
        )
        
        print(f"  ✓ Saved embeddings: {embeddings.shape}")
        print(f"  ✓ Saved taggrams: {taggrams.shape}")
        
        return embedding_id, True
        
    except Exception as e:
        print(f"Error extracting embeddings for {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def compute_pca_for_embedding(embedding_id):
    """
    Compute and cache PCA projection for an embedding.
    
    Args:
        embedding_id: Database ID of the embedding
    
    Returns:
        Tuple of (projection_id, success)
    """
    # Get embedding info
    db = database.get_db()
    embedding = db.execute('SELECT * FROM embeddings WHERE id = ?', (embedding_id,)).fetchone()
    if embedding is None:
        print(f"Error: Embedding with id {embedding_id} not found")
        return None, False
    
    track = db.execute('SELECT * FROM tracks WHERE id = ?', (embedding['track_id'],)).fetchone()
    filename = track['filename']
    model = embedding['model']
    dataset = embedding['dataset']
    
    print(f"Computing PCA for {filename} ({model}/{dataset})...")
    
    try:
        # Load embeddings from cache
        embeddings = cache_manager.get_cached_embedding(filename, model, dataset)
        if embeddings is None:
            print(f"Error: Embeddings not found in cache")
            return None, False
        
        # Compute PCA projection
        coords = proyectar_embeddings(embeddings, metodo='pca')
        
        # Save to cache
        projection_path = cache_manager.save_to_cache(
            coords, 'projection', filename, model, dataset, method='pca'
        )
        
        # Insert into database
        projection_id = database.insert_projection(
            embedding_id, 'pca', projection_path
        )
        
        print(f"  ✓ Saved PCA projection: {coords.shape}")
        
        return projection_id, True
        
    except Exception as e:
        print(f"Error computing PCA for embedding {embedding_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def process_all_tracks():
    """
    Batch process all tracks: extract embeddings and compute PCA projections.
    Processes for all model/dataset combinations.
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'tracks_processed': 0,
        'embeddings_created': 0,
        'projections_created': 0,
        'errors': 0
    }
    
    # Get all tracks
    tracks = database.get_all_tracks()
    
    if not tracks:
        print("No tracks found in database. Run 'flask index-audio' first.")
        return stats
    
    print(f"\nProcessing {len(tracks)} tracks...")
    print(f"Models: {config.MODELS}")
    print(f"Datasets: {config.DATASETS}")
    print("-" * 60)
    
    for track in tracks:
        track_id = track['id']
        filename = track['filename']
        
        print(f"\n[{track_id}] Processing: {filename}")
        
        # Process for each model/dataset combination
        for model in config.MODELS:
            for dataset in config.DATASETS:
                # Check if already processed
                existing = database.get_embedding(track_id, model, dataset)
                if existing:
                    print(f"  - {model}/{dataset}: Already processed (skipping)")
                    continue
                
                # Extract embeddings and taggrams
                embedding_id, success = extract_embeddings_for_track(
                    track_id, model, dataset
                )
                
                if success:
                    stats['embeddings_created'] += 1
                    
                    # Compute PCA projection
                    projection_id, pca_success = compute_pca_for_embedding(embedding_id)
                    if pca_success:
                        stats['projections_created'] += 1
                    else:
                        stats['errors'] += 1
                else:
                    stats['errors'] += 1
        
        # Mark track as processed
        database.mark_track_processed(track_id)
        stats['tracks_processed'] += 1
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Tracks processed: {stats['tracks_processed']}")
    print(f"Embeddings created: {stats['embeddings_created']}")
    print(f"Projections created: {stats['projections_created']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60)
    
    return stats


def process_single_track(filename):
    """
    Process a single audio file: extract embeddings and compute PCA projections.
    
    Args:
        filename: Audio filename (e.g., '1.mp3')
    
    Returns:
        Boolean indicating success
    """
    # Check if file exists
    audio_path = Path(config.AUDIO_DIR) / filename
    if not audio_path.exists():
        print(f"Error: Audio file {filename} not found in {config.AUDIO_DIR}")
        return False
    
    # Get or create track in database
    track = database.get_track_by_filename(filename)
    if track is None:
        # Get duration
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except Exception as e:
            print(f"Warning: Could not get duration: {e}")
            duration = None
        
        track_id = database.insert_track(filename, duration)
        print(f"Added {filename} to database")
    else:
        track_id = track['id']
    
    print(f"\nProcessing: {filename} (track_id={track_id})")
    print("-" * 60)
    
    success_count = 0
    total_count = len(config.MODELS) * len(config.DATASETS)
    
    # Process for each model/dataset combination
    for model in config.MODELS:
        for dataset in config.DATASETS:
            print(f"\n{model}/{dataset}:")
            
            # Extract embeddings and taggrams
            embedding_id, success = extract_embeddings_for_track(
                track_id, model, dataset
            )
            
            if success:
                # Compute PCA projection
                projection_id, pca_success = compute_pca_for_embedding(embedding_id)
                if pca_success:
                    success_count += 1
    
    # Mark track as processed
    database.mark_track_processed(track_id)
    
    print("\n" + "=" * 60)
    print(f"Completed {success_count}/{total_count} combinations for {filename}")
    print("=" * 60)
    
    return success_count > 0

def compute_genre_similarity_scores():
    """
    Final preprocessing step: compute genre-based similarity scores.
    
    For each model/dataset combination:
    1. Groups songs by their dominant genre tag (only genre--- tags)
    2. Computes genre centroids (mean embedding for each genre)
    3. Calculates similarity scores for each song
    
    Returns:
        Dictionary with genre statistics and similarity scores
    """
    import numpy as np
    from main import TAGS
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n" + "=" * 60)
    print("COMPUTING GENRE SIMILARITY SCORES")
    print("=" * 60)
    
    # Filter to get only genre tags
    genre_indices = [i for i, tag in enumerate(TAGS) if tag.startswith('genre---')]
    genre_tags = [TAGS[i] for i in genre_indices]
    
    print(f"\nFound {len(genre_tags)} genre tags")
    print(f"Genre tags: {genre_tags[:5]}... (showing first 5)")
    
    results = {}
    
    # Process each model/dataset combination
    for model in config.MODELS:
        for dataset in config.DATASETS:
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
                continue
            
            print(f"  Found {len(embeddings_records)} tracks")
            
            # Load embeddings and taggrams, group by dominant genre
            genre_song_embeddings = {genre: [] for genre in genre_tags}
            song_info = []  # Store (filename, embedding, dominant_genre)
            
            for record in embeddings_records:
                filename = record['filename']
                
                # Load embedding and taggram
                embedding = cache_manager.get_cached_embedding(filename, model, dataset)
                taggram = cache_manager.get_cached_taggram(filename, model, dataset)
                
                if embedding is None or taggram is None:
                    print(f"  Warning: Missing cache for {filename}, skipping...")
                    continue
                
                # Find dominant genre tag (same logic as server.py)
                genre_values = [taggram[0, i] for i in genre_indices]
                max_genre_idx = genre_values.index(max(genre_values))
                dominant_genre_tag = genre_tags[max_genre_idx]
                dominant_genre_value = genre_values[max_genre_idx]
                
                # Store for centroid computation
                genre_song_embeddings[dominant_genre_tag].append(embedding[0])  # [0] to get 1D array
                song_info.append({
                    'filename': filename,
                    'embedding': embedding[0],
                    'dominant_genre': dominant_genre_tag,
                    'genre_confidence': float(dominant_genre_value)
                })
            
            # Compute genre centroids (mean embeddings)
            genre_centroids = {}
            genre_stats = {}
            
            print(f"\n  Computing genre centroids...")
            for genre in genre_tags:
                embeddings_list = genre_song_embeddings[genre]
                if len(embeddings_list) > 0:
                    centroid = np.mean(embeddings_list, axis=0)
                    genre_centroids[genre] = centroid
                    genre_stats[genre] = {
                        'count': len(embeddings_list),
                        'centroid_norm': float(np.linalg.norm(centroid))
                    }
                    print(f"    {genre}: {len(embeddings_list)} songs")
            
            print(f"\n  Found {len(genre_centroids)} genres with songs")
            
            # Compute similarity scores for each song
            print(f"\n  Computing similarity scores...")
            song_similarities = []
            
            for song in song_info:
                song_embedding = song['embedding'].reshape(1, -1)
                dominant_genre = song['dominant_genre']
                
                # Compute cosine similarity to own genre centroid
                if dominant_genre in genre_centroids:
                    own_genre_centroid = genre_centroids[dominant_genre].reshape(1, -1)
                    similarity_to_own_genre = float(
                        cosine_similarity(song_embedding, own_genre_centroid)[0][0]
                    )
                else:
                    similarity_to_own_genre = None
                
                # Compute similarity to all genre centroids
                all_genre_similarities = {}
                for genre, centroid in genre_centroids.items():
                    centroid_reshaped = centroid.reshape(1, -1)
                    sim = float(cosine_similarity(song_embedding, centroid_reshaped)[0][0])
                    all_genre_similarities[genre] = sim
                
                # Find most similar genre (might be different from dominant)
                if all_genre_similarities:
                    most_similar_genre = max(all_genre_similarities, key=all_genre_similarities.get)
                    max_similarity = all_genre_similarities[most_similar_genre]
                else:
                    most_similar_genre = None
                    max_similarity = None
                
                song_similarities.append({
                    'filename': song['filename'],
                    'dominant_genre': dominant_genre,
                    'genre_confidence': song['genre_confidence'],
                    'similarity_to_own_genre': similarity_to_own_genre,
                    'most_similar_genre': most_similar_genre,
                    'max_similarity': max_similarity,
                    'agreement': dominant_genre == most_similar_genre,
                    'all_similarities': all_genre_similarities
                })
            
            # Calculate aggregate statistics
            similarities_to_own = [s['similarity_to_own_genre'] for s in song_similarities 
                                  if s['similarity_to_own_genre'] is not None]
            agreement_rate = sum(1 for s in song_similarities if s['agreement']) / len(song_similarities)
            
            results[combo_key] = {
                'genre_centroids': genre_centroids,
                'genre_stats': genre_stats,
                'song_similarities': song_similarities,
                'aggregate_stats': {
                    'mean_similarity_to_own_genre': float(np.mean(similarities_to_own)),
                    'std_similarity_to_own_genre': float(np.std(similarities_to_own)),
                    'min_similarity': float(np.min(similarities_to_own)),
                    'max_similarity': float(np.max(similarities_to_own)),
                    'genre_agreement_rate': float(agreement_rate),
                    'total_songs': len(song_similarities)
                }
            }
            
            # Save centroids and similarity scores to cache
            centroids_path = config.CACHE_DIR / 'centorids' / f'genre_centroids_{model}_{dataset}.npy'
            similarities_path = config.CACHE_DIR/ 'similarities'/ f'genre_similarities_{model}_{dataset}.npy'
            
            np.save(centroids_path, genre_centroids)
            np.save(similarities_path, song_similarities)
            
            print(f"\n  Statistics:")
            print(f"    Mean similarity to own genre: {results[combo_key]['aggregate_stats']['mean_similarity_to_own_genre']:.4f}")
            print(f"    Std deviation: {results[combo_key]['aggregate_stats']['std_similarity_to_own_genre']:.4f}")
            print(f"    Genre agreement rate: {agreement_rate*100:.1f}%")
            print(f"  Saved centroids to: {centroids_path}")
            print(f"  Saved similarities to: {similarities_path}")
    
    return results