import sqlite3
from datetime import datetime
from pathlib import Path
from flask import g

DATABASE_PATH = './cache/audio_cache.db'


def get_db():
    """Get database connection, creating it if it doesn't exist in flask context."""
    if 'db' not in g:
        # Ensure cache directory exists
        Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database schema."""
    db = sqlite3.connect(DATABASE_PATH)
    
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
    
    # Create embeddings table
    db.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            embedding_path TEXT NOT NULL,
            taggram_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (track_id) REFERENCES tracks (id),
            UNIQUE(track_id, model, dataset)
        )
    ''')
    
    # Create projections table
    db.execute('''
        CREATE TABLE IF NOT EXISTS projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            projection_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (embedding_id) REFERENCES embeddings (id),
            UNIQUE(embedding_id, method)
        )
    ''')
    
    db.commit()
    db.close()
    print(f"Database initialized at {DATABASE_PATH}")


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
    return cursor.lastrowid


def get_embedding(track_id, model, dataset):
    """Get embedding record for a track."""
    db = get_db()
    embedding = db.execute(
        'SELECT * FROM embeddings WHERE track_id = ? AND model = ? AND dataset = ?',
        (track_id, model, dataset)
    ).fetchone()
    return embedding


def insert_embedding(track_id, model, dataset, embedding_path, taggram_path):
    """Insert a new embedding record."""
    db = get_db()
    cursor = db.execute(
        '''INSERT OR REPLACE INTO embeddings 
           (track_id, model, dataset, embedding_path, taggram_path) 
           VALUES (?, ?, ?, ?, ?)''',
        (track_id, model, dataset, embedding_path, taggram_path)
    )
    db.commit()
    return cursor.lastrowid


def get_projection(embedding_id, method):
    """Get projection record for an embedding."""
    db = get_db()
    projection = db.execute(
        'SELECT * FROM projections WHERE embedding_id = ? AND method = ?',
        (embedding_id, method)
    ).fetchone()
    return projection


def insert_projection(embedding_id, method, projection_path):
    """Insert a new projection record."""
    db = get_db()
    cursor = db.execute(
        '''INSERT OR REPLACE INTO projections 
           (embedding_id, method, projection_path) 
           VALUES (?, ?, ?)''',
        (embedding_id, method, projection_path)
    )
    db.commit()
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

