"""
Preprocessing module for batch extracting embeddings, taggrams.
"""
import os
import sys
import csv
import librosa
from pathlib import Path
from flask import g
from tqdm import tqdm
import config
import database
from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, embeddings_y_taggrams_Whisper
from proyecciones import proyectar_embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check GPU availability at module load
if torch.cuda.is_available():
    print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    print(f"  Using device: cuda:0")
else:
    print("⚠ No GPU detected - using CPU (this will be slower)")

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
    
    print(f"\nIndexed {indexed_count} new tracks")
    return indexed_count


def extract_embeddings_for_track(track_id, model, dataset):
    """
    Extract and cache embeddings and taggrams for a specific track.
    
    Args:
        track_id: Database ID of the track
        model: Model
        dataset: Dataset
    
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
        return None, False
        
    try:
        # Extract embeddings and taggrams
        if model == 'musicnn':
            embeddings, taggrams = embeddings_y_taggrams_MusiCNN(
                weights_path, audio_path, dataset_name=dataset
            )
        elif model == 'vgg':
            embeddings, taggrams = embeddings_y_taggrams_VGG(
                weights_path, audio_path, dataset_name=dataset
            )
        elif model == 'whisper':
            # For whisper, weights_path is the model name (e.g., 'base', 'small')
            embeddings, taggrams = embeddings_y_taggrams_Whisper(
                weights_path, audio_path
            )
        else:
            print(f"Error: Unknown model {model}")
            return None, False
        
        # Insert into database with vectors
        embedding_id = database.insert_embedding(
            track_id, model, dataset, embeddings, taggrams
        )
        
        return embedding_id, True
        
    except Exception as e:
        print(f"Error extracting embeddings for {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def process_all_tracks():
    """
    Batch process all tracks: extract embeddings for all model/dataset combinations.
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'tracks_processed': 0,
        'embeddings_created': 0,
        'errors': 0
    }
    
    # Get all tracks
    tracks = database.get_all_tracks()
    
    if not tracks:
        print("No tracks found in database. Run 'flask index-audio' first.")
        return stats
    
    # Calculate total operations
    model_dataset_combinations = [(m, d) for m in config.MODELS for d in config.DATASETS]
    total_operations = len(tracks) * len(model_dataset_combinations)
    
    # Single progress bar for all operations
    with tqdm(total=total_operations, desc="Processing tracks", unit="operation") as pbar:
        for track in tracks:
            track_id = track['id']
            
            for model, dataset in model_dataset_combinations:
                # Check if already processed
                existing = database.get_embedding(track_id, model, dataset)
                if existing:
                    pbar.update(1)
                    continue
                
                # Extract embeddings and taggrams
                embedding_id, success = extract_embeddings_for_track(
                    track_id, model, dataset
                )
                
                if success:
                    stats['embeddings_created'] += 1
                else:
                    stats['errors'] += 1
                
                pbar.update(1)
            
            # Mark track as processed
            database.mark_track_processed(track_id)
            stats['tracks_processed'] += 1
    
    return stats


def process_single_track(filename):
    """
    Process a single audio file: extract embeddings for all model/dataset combinations.
    
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
    1. Groups songs by their genre tag from CSV file
    2. Computes genre centroids (mean embedding and taggram for each genre)
    3. Calculates similarity scores for each song using both embeddings and taggrams
    
    Returns:
        Dictionary with:
        - genre_embedding_centroids: Mean embeddings per genre
        - genre_taggram_centroids: Mean taggrams per genre
        - song_similarities: Per-song metrics including:
            * Embedding-based similarity to own genre and all genres
            * Taggram-based similarity to own genre and all genres
            * Agreement flags (whether predicted genre matches actual)
        - aggregate_stats: Overall statistics for both embedding and taggram similarities
    """
    print("\n" + "=" * 60)
    print("COMPUTING GENRE SIMILARITY SCORES")
    print("=" * 60)
    
    # Load genre mappings from CSV
    csv_path = Path(config.AUDIO_DIR) / 'selected_songs.csv'
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return {}
    
    print(f"\nLoading genre mappings from: {csv_path}")
    genre_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre_map[row['filename']] = row['genre']
    
    # Get unique genres from CSV
    genre_tags = sorted(set(genre_map.values()))
    
    print(f"Found {len(genre_map)} tracks with genre labels")
    print(f"Found {len(genre_tags)} unique genres")
    print(f"Genres: {genre_tags}")
    
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
            
            # Load embeddings and taggrams, group by genre from CSV
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
            
            if skipped_count > 0:
                print(f"  Skipped {skipped_count} tracks (no genre in CSV or missing data)")
            
            print(f"  Processing {len(song_info)} tracks with genre labels")
            
            # Compute genre centroids (mean embeddings and taggrams)
            genre_embedding_centroids = {}
            genre_taggram_centroids = {}
            genre_stats = {}
            
            print(f"\n  Computing genre centroids...")
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
            
            print(f"\n  Found {len(genre_embedding_centroids)} genres with songs")
            
            # Compute similarity scores for each song
            print(f"\n  Computing embedding and taggram similarity scores...")
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
            
            # Calculate aggregate statistics for embeddings and taggrams
            emb_similarities_to_own = [s['emb_similarity_to_own_genre'] for s in song_similarities 
                                       if s['emb_similarity_to_own_genre'] is not None]
            tag_similarities_to_own = [s['tag_similarity_to_own_genre'] for s in song_similarities 
                                       if s['tag_similarity_to_own_genre'] is not None]
            
            # Count agreements
            emb_agreements = sum(1 for s in song_similarities if s['emb_agreement'])
            tag_agreements = sum(1 for s in song_similarities if s['tag_agreement'])
            
            # Compute Silhouette scores
            print(f"\n  Computing Silhouette scores...")
            
            # Prepare data for Silhouette score computation
            embeddings_array = np.array([s['embedding'] for s in song_info])
            taggrams_array = np.array([s['taggram'] for s in song_info])
            genre_labels = [s['genre'] for s in song_info]
            
            # Compute Silhouette scores (only if we have at least 2 genres with 2+ samples each)
            unique_genres = set(genre_labels)
            emb_silhouette = None
            tag_silhouette = None
            
            if len(unique_genres) >= 2:
                try:
                    emb_silhouette = float(silhouette_score(embeddings_array, genre_labels, metric='cosine'))
                    print(f"    Embedding Silhouette score: {emb_silhouette:.4f}")
                except Exception as e:
                    print(f"    Warning: Could not compute embedding Silhouette score: {e}")
                
                try:
                    tag_silhouette = float(silhouette_score(taggrams_array, genre_labels, metric='cosine'))
                    print(f"    Taggram Silhouette score: {tag_silhouette:.4f}")
                except Exception as e:
                    print(f"    Warning: Could not compute taggram Silhouette score: {e}")
            else:
                print(f"    Skipping Silhouette scores (need at least 2 genres, found {len(unique_genres)})")
            
            results[combo_key] = {
                'genre_embedding_centroids': genre_embedding_centroids,
                'genre_taggram_centroids': genre_taggram_centroids,
                'genre_stats': genre_stats,
                'song_similarities': song_similarities,
                'aggregate_stats': {
                    # Embedding stats
                    'emb_mean_similarity_to_own_genre': float(np.mean(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
                    'emb_std_similarity_to_own_genre': float(np.std(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
                    'emb_min_similarity': float(np.min(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
                    'emb_max_similarity': float(np.max(emb_similarities_to_own)) if emb_similarities_to_own else 0.0,
                    'emb_agreement_rate': emb_agreements / len(song_similarities) if song_similarities else 0.0,
                    'emb_silhouette_score': emb_silhouette,
                    # Taggram stats
                    'tag_mean_similarity_to_own_genre': float(np.mean(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
                    'tag_std_similarity_to_own_genre': float(np.std(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
                    'tag_min_similarity': float(np.min(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
                    'tag_max_similarity': float(np.max(tag_similarities_to_own)) if tag_similarities_to_own else 0.0,
                    'tag_agreement_rate': tag_agreements / len(song_similarities) if song_similarities else 0.0,
                    'tag_silhouette_score': tag_silhouette,
                    # General stats
                    'total_songs': len(song_similarities)
                }
            }
            
            print(f"\n  Statistics:")
            print(f"    EMBEDDING-BASED:")
            print(f"      Mean similarity to own genre: {results[combo_key]['aggregate_stats']['emb_mean_similarity_to_own_genre']:.4f}")
            print(f"      Std deviation: {results[combo_key]['aggregate_stats']['emb_std_similarity_to_own_genre']:.4f}")
            print(f"      Agreement rate: {results[combo_key]['aggregate_stats']['emb_agreement_rate']:.2%}")
            if emb_silhouette is not None:
                print(f"      Silhouette score: {emb_silhouette:.4f}")
            print(f"    TAGGRAM-BASED:")
            print(f"      Mean similarity to own genre: {results[combo_key]['aggregate_stats']['tag_mean_similarity_to_own_genre']:.4f}")
            print(f"      Std deviation: {results[combo_key]['aggregate_stats']['tag_std_similarity_to_own_genre']:.4f}")
            print(f"      Agreement rate: {results[combo_key]['aggregate_stats']['tag_agreement_rate']:.2%}")
            if tag_silhouette is not None:
                print(f"      Silhouette score: {tag_silhouette:.4f}")
    
    return results