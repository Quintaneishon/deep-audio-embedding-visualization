#!/usr/bin/env python3
"""
Script to randomly select 500 songs from MTG-Jamendo dataset with exactly 1 genre tag.
Creates a CSV with song names and tags, then copies the audio files.
"""

import csv
import random
import shutil
import os
from pathlib import Path
from collections import defaultdict

# Paths configuration
DATASET_PATH = Path('/home/ar/Data/Ajitzi/mtg-jamendo-dataset')
TSV_FILE = DATASET_PATH / 'data' / 'autotagging_genre.tsv'
SONGS_DIR = DATASET_PATH / 'songs'
OUTPUT_DIR = Path('/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/audio')
OUTPUT_CSV = OUTPUT_DIR / 'selected_songs.csv'

# Number of songs to select
NUM_SONGS = 1000

def count_genre_tags(tags_list):
    """Count how many genre tags are in the tags list."""
    genre_count = 0
    for tag in tags_list:
        if tag.startswith('genre---'):
            genre_count += 1
    return genre_count

def get_genre_tag(tags_list):
    """Extract the single genre tag from the tags list."""
    for tag in tags_list:
        if tag.startswith('genre---'):
            return tag
    return None

def find_audio_file(path):
    """
    Find the actual audio file given the path from TSV.
    The TSV references paths like '14/214.mp3' but actual files are '214.low.mp3'
    """
    # Extract the directory and filename
    parts = path.split('/')
    if len(parts) == 2:
        dir_name, filename = parts
        # Remove .mp3 extension and add .low.mp3
        base_name = filename.replace('.mp3', '')
        low_quality_filename = f"{base_name}.low.mp3"
        
        # Try low quality version first
        low_quality_path = SONGS_DIR / dir_name / low_quality_filename
        if low_quality_path.exists():
            return low_quality_path
        
        # Try original filename
        original_path = SONGS_DIR / dir_name / filename
        if original_path.exists():
            return original_path
    
    return None

def main():
    print(f"Reading dataset from: {TSV_FILE}")
    
    songs_with_one_genre = []
    
    with open(TSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            # Split tags by tab character (they are tab-separated in the TSV)
            tags_str = row['TAGS']
            tags = tags_str.split('\t') if '\t' in tags_str else [tags_str]
            
            # Count genre tags
            genre_count = count_genre_tags(tags)
            
            # Only include songs with exactly 1 genre tag
            if genre_count == 1:
                genre_tag = get_genre_tag(tags)
                songs_with_one_genre.append({
                    'track_id': row['TRACK_ID'],
                    'path': row['PATH'],
                    'genre': genre_tag
                })
    
    print(f"Found {len(songs_with_one_genre)} songs with exactly 1 genre tag")
    
    # Check if we have enough songs
    if len(songs_with_one_genre) < NUM_SONGS:
        print(f"Warning: Only {len(songs_with_one_genre)} songs available, selecting all of them")
        selected_songs = songs_with_one_genre
    else:
        # Select songs with a more balanced genre distribution
        genre_to_songs = defaultdict(list)
        for song in songs_with_one_genre:
            genre_to_songs[song['genre']].append(song)

        # Shuffle songs within each genre to keep randomness
        for genre_songs in genre_to_songs.values():
            random.shuffle(genre_songs)

        selected_songs = []

        # Round-robin sampling across genres to balance the distribution
        while len(selected_songs) < NUM_SONGS and any(genre_to_songs.values()):
            for genre, genre_songs in list(genre_to_songs.items()):
                if len(selected_songs) >= NUM_SONGS:
                    break
                if genre_songs:
                    selected_songs.append(genre_songs.pop())
                if not genre_songs:
                    # Remove genres that have been exhausted
                    del genre_to_songs[genre]

        print(f"Selected {len(selected_songs)} songs with a more balanced genre distribution (target = {NUM_SONGS})")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create CSV with selected songs
    print(f"Creating CSV at: {OUTPUT_CSV}")
    successful_copies = []
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'genre'])
        
        # Copy files and write to CSV
        for i, song in enumerate(selected_songs, 1):
            source_file = find_audio_file(song['path'])
            
            if source_file and source_file.exists():
                # Create destination filename (use track_id to ensure uniqueness)
                dest_filename = f"{song['track_id']}.mp3"
                dest_path = OUTPUT_DIR / dest_filename
                
                # Copy the file
                try:
                    shutil.copy2(source_file, dest_path)
                    writer.writerow([dest_filename, song['genre']])
                    successful_copies.append(dest_filename)
                    
                    if i % 50 == 0:
                        print(f"Progress: {i}/{len(selected_songs)} files copied")
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")
            else:
                print(f"Warning: Audio file not found for {song['path']}")
    
    print(f"\nCompleted!")
    print(f"Successfully copied {len(successful_copies)} songs to {OUTPUT_DIR}")
    print(f"CSV file created: {OUTPUT_CSV}")
    
    # Print genre distribution
    genre_counts = {}
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre = row['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print("\nGenre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} songs")

if __name__ == '__main__':
    main()

