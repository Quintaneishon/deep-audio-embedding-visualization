import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import g
import io
import sys
import os
from backend.proyecciones import proyectar_embeddings
import csv
# Add parent directory to path to import root config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

def get_db():
    """Get database connection, creating it if it doesn't exist in flask context."""
    if 'db' not in g:
        # Ensure cache directory exists
        Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(config.DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database schema with BLOB columns for vectors."""
    # Ensure cache directory exists
    Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.DATABASE_PATH)
    
    # Create tracks table
    db.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            duration REAL,
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create embeddings table - storing actual vectors as BLOBs
    db.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            embedding_data BLOB NOT NULL,
            embedding_shape TEXT NOT NULL,
            taggram_data BLOB NOT NULL,
            taggram_shape TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE,
            UNIQUE(track_id, model, dataset)
        )
    ''')
    
    # Create indexes for faster lookups
    db.execute('CREATE INDEX IF NOT EXISTS idx_tracks_filename ON tracks(filename)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_lookup ON embeddings(track_id, model, dataset)')
    
    db.commit()
    db.close()
    print(f"Database initialized at {config.DATABASE_PATH}")


# ============================================================================
# Helper Functions for Numpy <-> BLOB conversion
# ============================================================================

def _numpy_to_blob(array):
    """Convert numpy array to BLOB (bytes) and shape string."""
    # Serialize to bytes
    out = io.BytesIO()
    np.save(out, array, allow_pickle=False)
    out.seek(0)
    blob = out.read()
    
    shape_str = str(array.shape)
    
    return blob, shape_str


def _blob_to_numpy(blob):
    """Convert BLOB (bytes) and shape string back to numpy array."""
    in_bytes = io.BytesIO(blob)
    array = np.load(in_bytes, allow_pickle=False)
    return array


# ============================================================================
# Track Functions
# ============================================================================

def get_track_by_filename(filename):
    """Get track record by filename."""
    db = get_db()
    track = db.execute(
        'SELECT * FROM tracks WHERE filename = ?',
        (filename,)
    ).fetchone()
    return track


def insert_track(filename, duration=None):
    """Insert a new track into the database."""
    db = get_db()
    cursor = db.execute(
        'INSERT OR IGNORE INTO tracks (filename, duration) VALUES (?, ?)',
        (filename, duration)
    )
    db.commit()
    # Handle case where track already exists
    if cursor.lastrowid == 0:
        track = get_track_by_filename(filename)
        return track['id'] if track else None
    return cursor.lastrowid


def get_all_tracks():
    """Get all tracks from the database."""
    db = get_db()
    tracks = db.execute('SELECT * FROM tracks').fetchall()
    return tracks


def mark_track_processed(track_id):
    """Mark a track as processed."""
    db = get_db()
    db.execute(
        'UPDATE tracks SET processed_at = ? WHERE id = ?',
        (datetime.now(), track_id)
    )
    db.commit()


# ============================================================================
# Embedding Functions (with vectors in DB)
# ============================================================================

def get_embedding(track_id, model, dataset):
    """
    Get embedding record for a track (metadata only, no vectors).
    Use get_embedding_vectors() to also load the actual arrays.
    """
    db = get_db()
    embedding = db.execute(
        'SELECT id, track_id, model, dataset, embedding_shape, taggram_shape, created_at FROM embeddings WHERE track_id = ? AND model = ? AND dataset = ?',
        (track_id, model, dataset)
    ).fetchone()
    return embedding


def get_embedding_by_filename(filename, model, dataset):
    """
    Get embedding vectors directly by filename.
    
    Returns:
        dict with 'embedding_id', 'embedding', 'taggram' or None
    """
    db = get_db()
    result = db.execute('''
        SELECT e.id, e.embedding_data, e.embedding_shape, e.taggram_data, e.taggram_shape
        FROM embeddings e
        JOIN tracks t ON e.track_id = t.id
        WHERE t.filename = ? AND e.model = ? AND e.dataset = ?
    ''', (filename, model, dataset)).fetchone()
    
    if result:
        embedding = _blob_to_numpy(result['embedding_data'])
        taggram = _blob_to_numpy(result['taggram_data'])
        
        return {
            'embedding_id': result['id'],
            'embedding': embedding,
            'taggram': taggram
        }
    
    return None


def insert_embedding(track_id, model, dataset, embedding_array, taggram_array):
    """
    Insert a new embedding record with vectors stored in database.
    
    Args:
        track_id: Track ID
        model: Model name
        dataset: Dataset name
        embedding_array: Numpy array of embeddings
        taggram_array: Numpy array of taggrams
    
    Returns:
        embedding_id
    """
    db = get_db()
    
    # Convert numpy arrays to BLOBs
    embedding_blob, embedding_shape = _numpy_to_blob(embedding_array)
    taggram_blob, taggram_shape = _numpy_to_blob(taggram_array)
    
    cursor = db.execute(
        '''INSERT OR REPLACE INTO embeddings 
           (track_id, model, dataset, embedding_data, embedding_shape, taggram_data, taggram_shape) 
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (track_id, model, dataset, embedding_blob, embedding_shape, taggram_blob, taggram_shape)
    )
    db.commit()
    return cursor.lastrowid


def get_embedding_coords(red, dataset, metodo, dimensions):
    """
    Load the actual numpy arrays for an embedding and include genre tags.
    
    Returns:
        list: Array of dictionaries with 'data' (projected coordinates) and 'tag' (genre)
    """
    db = get_db()
    result = db.execute(
        '''SELECT e.embedding_data, t.filename 
           FROM embeddings e 
           JOIN tracks t ON e.track_id = t.id 
           WHERE e.model = ? AND e.dataset = ?''',
        (red, dataset)
    ).fetchall()
    
    if not result:
        return None
    
    # Load genre mappings from CSV
    csv_path = Path(config.CSV_PATH)
    genre_map = {}
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                genre_map[row['filename']] = row['genre']
    
    embeddings_list = []
    filenames = []
    for row in result:
        embedding = _blob_to_numpy(row['embedding_data'])
        embeddings_list.append(embedding)
        filenames.append(row['filename'])
    
    # Stack all embeddings into a single 2D array
    embeddings_array = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
    
    # Project embeddings
    projected = proyectar_embeddings(embeddings_array, metodo, n_components=dimensions)
    
    # Build result array with genre tags
    result_array = []
    for i, filename in enumerate(filenames):
        genre = genre_map.get(filename, 'Unknown')
        result_array.append({
            'data': projected[i].tolist(),
            'tag': genre
        })
    
    return result_array


def get_taggram_coords(red, dataset, metodo, dimensions):
    """
    Load the actual numpy arrays for taggrams and include genre tags.
    
    Returns:
        list: Array of dictionaries with 'data' (projected coordinates) and 'tag' (genre)
    """
    db = get_db()
    result = db.execute(
        '''SELECT e.taggram_data, t.filename 
           FROM embeddings e 
           JOIN tracks t ON e.track_id = t.id 
           WHERE e.model = ? AND e.dataset = ?''',
        (red, dataset)
    ).fetchall()
    
    if not result:
        return None
    
    # Load genre mappings from CSV
    csv_path = Path(config.CSV_PATH)
    genre_map = {}
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                genre_map[row['filename']] = row['genre']
    
    # Extract taggrams and filenames
    taggrams_list = []
    filenames = []
    for row in result:
        taggram = _blob_to_numpy(row['taggram_data'])
        taggrams_list.append(taggram)
        filenames.append(row['filename'])
    
    # Stack all taggrams into a single 2D array
    taggrams_array = np.vstack(taggrams_list) if len(taggrams_list) > 1 else taggrams_list[0]
    
    # Project taggrams
    projected = proyectar_embeddings(taggrams_array, metodo, n_components=dimensions)
    
    # Build result array with genre tags
    result_array = []
    for i, filename in enumerate(filenames):
        genre = genre_map.get(filename, 'Unknown')
        result_array.append({
            'data': projected[i].tolist(),
            'tag': genre
        })
    
    return result_array


# ============================================================================
# Database Maintenance Functions
# ============================================================================

def clean_db(drop_tables=False):
    """
    Clean the database by removing all data.
    
    Args:
        drop_tables: If True, drop and recreate all tables (full reset).
                    If False, only delete all records (preserve schema).
    
    Returns:
        dict: Statistics about deleted records
    """
    # Connect directly (not through Flask context)
    db = sqlite3.connect(config.DATABASE_PATH)
    cursor = db.cursor()
    
    try:
        if drop_tables:
            cursor.execute('DROP TABLE IF EXISTS embeddings')
            cursor.execute('DROP TABLE IF EXISTS tracks')
            db.commit()
            db.close()
            
            return {
                'status': 'success',
                'action': 'full_reset',
                'message': 'Database dropped and reinitialized'
            }
        else:
            # Get counts before deletion
            embeddings_count = cursor.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]
            tracks_count = cursor.execute('SELECT COUNT(*) FROM tracks').fetchone()[0]
            
            # Delete all records (respect foreign key constraints by deleting in order)
            cursor.execute('DELETE FROM embeddings')
            cursor.execute('DELETE FROM tracks')
            
            # Reset autoincrement counters
            cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("embeddings", "tracks")')
            
            db.commit()
            db.close()
            
            return {
                'status': 'success',
                'action': 'clear_data',
                'deleted': {
                    'embeddings': embeddings_count,
                    'tracks': tracks_count
                },
                'message': f'Deleted {tracks_count} tracks, {embeddings_count} embeddings'
            }
    except Exception as e:
        db.rollback()
        db.close()
        return {
            'status': 'error',
            'message': str(e)
        }
