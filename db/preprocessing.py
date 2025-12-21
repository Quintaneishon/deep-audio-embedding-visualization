"""
Preprocessing module for batch extracting embeddings, taggrams.
"""
import os
import sys
import csv
import json
import subprocess
import librosa
from pathlib import Path
from tqdm import tqdm
import config
import database
from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, embeddings_y_taggrams_Whisper, embeddings_y_taggrams_MERT, embeddings_y_taggrams_WhisperContrastive, embeddings_y_taggrams_VGGish, embeddings_y_taggrams_LightweightAdapter
import torch
import gc
from db.utils import process_model_dataset_combination
from ML.acoustic_features import AcousticFeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check GPU availability at module load
if torch.cuda.is_available():
    print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    print(f"  Using device: cuda:0")
else:
    print("⚠ No GPU detected - using CPU (this will be slower)")

def get_audio_duration(audio_path):
    """
    Get audio duration using ffprobe (supports all formats: MP3, M4A, WAV, FLAC, OGG, etc.)
    This avoids the librosa soundfile/audioread deprecation warning.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds (float) or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        # Fallback to librosa if ffprobe fails
        try:
            return librosa.get_duration(path=str(audio_path))
        except:
            return None

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
                duration = get_audio_duration(audio_path)
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
        print(f"Error: No weights configured for model={model}, dataset={dataset}")
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
        elif model == 'mert':
            # For MERT, weights_path is the model name (e.g., '95m', '330m')
            embeddings, taggrams = embeddings_y_taggrams_MERT(
                weights_path, audio_path
            )
        elif model == 'whisper_contrastive':
            embeddings, taggrams = embeddings_y_taggrams_WhisperContrastive(
                weights_path, audio_path
            )
        elif model == 'vggish':
            embeddings, taggrams = embeddings_y_taggrams_VGGish(
                weights_path, audio_path
            )
        elif model == 'lightweight_adapter_v1':
            embeddings, taggrams = embeddings_y_taggrams_LightweightAdapter(
                weights_path, audio_path
            )
        elif model == 'lightweight_adapter_v2':
            embeddings, taggrams = embeddings_y_taggrams_LightweightAdapter(
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
    conv_model_dataset_combinations = [(m, d) for m in config.CONV_MODELS for d in config.DATASETS]
    total_operations = len(tracks) * len(conv_model_dataset_combinations)
    total_operations += len(tracks) * len(config.TRANF_MODELS)
    
    # Single progress bar for all operations
    with tqdm(total=total_operations, desc="Processing tracks", unit="operation") as pbar:
        for track in tracks:
            track_id = track['id']
            
            for model, dataset in conv_model_dataset_combinations:
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
            
            for i, model in enumerate(config.TRANF_MODELS):
                # Check if already processed
                existing = database.get_embedding(track_id, model, config.MODEL_SIZES[i])
                if existing:
                    pbar.update(1)
                    continue
                
                # Extract embeddings and taggrams
                embedding_id, success = extract_embeddings_for_track(
                    track_id, model, config.MODEL_SIZES[i]
                )
            
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
            duration = get_audio_duration(audio_path)
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
    conv_model_dataset_combinations = [(m, d) for m in config.CONV_MODELS for d in config.DATASETS]
    total_count = len(conv_model_dataset_combinations) + len(config.TRANF_MODELS)
    
    # Process convolutional models with datasets
    for model, dataset in conv_model_dataset_combinations:
        print(f"\n{model}/{dataset}:")
        
        # Extract embeddings and taggrams
        embedding_id, success = extract_embeddings_for_track(
            track_id, model, dataset
        )
        
        if success:
            success_count += 1
    
    # Process transformer models with model sizes
    for i, model in enumerate(config.TRANF_MODELS):
        model_size = config.MODEL_SIZES[i]
        print(f"\n{model}/{model_size}:")
        
        # Extract embeddings and taggrams
        embedding_id, success = extract_embeddings_for_track(
            track_id, model, model_size
        )
        
        if success:
            success_count += 1

    # Mark track as processed
    database.mark_track_processed(track_id)
    
    print("\n" + "=" * 60)
    print(f"Completed {success_count}/{total_count} combinations for {filename}")
    print("=" * 60)
    
    return success_count > 0


def extract_and_store_acoustic_features():
    """
    Extract acoustic features (spectral_centroid and tempo) for all tracks
    and store them in the database.
    
    Uses AcousticFeatureExtractor to extract:
    - spectral_centroid (index 0): Mean spectral centroid in Hz
    - tempo (index 5): Estimated tempo in BPM
    
    Returns:
        Dictionary with extraction statistics
    """
    print("\n" + "=" * 60)
    print("EXTRACTING ACOUSTIC FEATURES")
    print("=" * 60)
    
    # Ensure database schema is up to date
    database.migrate_add_acoustic_features()
    
    # Get all tracks
    tracks = database.get_all_tracks()
    
    if not tracks:
        print("No tracks found in database. Run 'flask index-audio' first.")
        return {'tracks_processed': 0, 'tracks_skipped': 0, 'errors': 0}
    
    # Initialize feature extractor
    extractor = AcousticFeatureExtractor(sample_rate=config.SAMPLE_RATE)
    
    stats = {
        'tracks_processed': 0,
        'tracks_skipped': 0,
        'errors': 0
    }
    
    print(f"\nProcessing {len(tracks)} tracks...")
    
    for track in tqdm(tracks, desc="Extracting features", unit="track"):
        track_id = track['id']
        filename = track['filename']
        
        # Skip if features already exist
        if track['spectral_centroid'] is not None and track['tempo'] is not None:
            stats['tracks_skipped'] += 1
            continue
        
        audio_path = Path(config.AUDIO_DIR) / filename
        
        if not audio_path.exists():
            print(f"\n  Warning: Audio file not found: {audio_path}")
            stats['errors'] += 1
            continue
        
        try:
            # Extract features (no normalization needed for storage)
            features = extractor.extract_features(str(audio_path), normalize=False)
            
            # Extract specific features we need
            spectral_centroid = float(features[0])  # Index 0: spectral_centroid
            tempo = float(features[5])  # Index 5: tempo
            
            # Update database
            database.update_track_acoustic_features(track_id, spectral_centroid, tempo)
            
            stats['tracks_processed'] += 1
            
        except Exception as e:
            print(f"\n  Error processing {filename}: {e}")
            stats['errors'] += 1
    
    print("\n" + "=" * 60)
    print("ACOUSTIC FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {stats['tracks_processed']}")
    print(f"Skipped (already exists): {stats['tracks_skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60)
    
    return stats


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
    
    # Build model/dataset combinations for convolutional models
    conv_model_dataset_combinations = [(m, d) for m in config.CONV_MODELS for d in config.DATASETS]
    
    # Process convolutional model/dataset combinations
    for model, dataset in conv_model_dataset_combinations:
        combo_key = f"{model}_{dataset}"
        result = process_model_dataset_combination(model, dataset, genre_map, genre_tags)
        if result is not None:
            results[combo_key] = {
                'aggregate_stats': result['aggregate_stats']
            }
            del result

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process transformer models (whisper, mert) with their model sizes
    for i, model in enumerate(config.TRANF_MODELS):
        model_size = config.MODEL_SIZES[i]
        combo_key = f"{model}_{model_size}"
        result = process_model_dataset_combination(model, model_size, genre_map, genre_tags)
        if result is not None:
            results[combo_key] = {
                'aggregate_stats': result['aggregate_stats']
            }
            del result
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results