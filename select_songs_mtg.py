#!/usr/bin/env python3
"""
Script to extract songs from MTG-Jamendo dataset with instrument and mood/theme tags.
Creates a CSV with song information and their non-genre tags for evaluation.
"""

import csv
import random
import shutil
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

# Paths configuration
MTG_JAMENDO_ROOT = Path('/home/ar/Data/Ajitzi/mtg-jamendo-dataset')
AUDIO_DIR = MTG_JAMENDO_ROOT / 'songs'
SPLITS_DIR = MTG_JAMENDO_ROOT / 'data' / 'splits' / 'split-0'

OUTPUT_DIR = Path('/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/audio_mtg')
OUTPUT_CSV = OUTPUT_DIR / 'selected_songs_mtg.csv'

# Select songs with balanced tag distribution
NUM_SONGS = 1000


def load_tags_from_tsv(tsv_file):
    """
    Load tags from a TSV file.
    
    Returns:
        dict: {track_id: {'path': str, 'tags': [tag1, tag2, ...]}}
    """
    tags_dict = {}
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                track_id = parts[0]
                path = parts[3]
                
                # Get all tags (column 5 onwards)
                tag_str = '\t'.join(parts[5:])
                tag_list = tag_str.split('\t') if tag_str else []
                
                # Clean tags (remove prefix)
                clean_tags = []
                for tag in tag_list:
                    if '---' in tag:
                        clean_tag = tag.split('---', 1)[1]
                        clean_tags.append(clean_tag)
                
                if clean_tags:
                    tags_dict[track_id] = {
                        'path': path,
                        'tags': clean_tags
                    }
    
    return tags_dict


def find_audio_file(relative_path):
    """
    Find the audio file in the MTG-Jamendo dataset.
    Tries both .mp3 and .low.mp3 extensions.
    """
    audio_path = AUDIO_DIR / relative_path
    
    if audio_path.exists():
        return audio_path
    
    # Try .low.mp3 version
    if relative_path.endswith('.mp3'):
        low_path = AUDIO_DIR / relative_path.replace('.mp3', '.low.mp3')
        if low_path.exists():
            return low_path
    
    return None


def main():
    print("="*60)
    print("MTG-Jamendo Song Extraction (Instrument & Mood/Theme Tags)")
    print("="*60)
    
    # Load instrument and mood/theme tags from all splits
    print("\nLoading tags...")
    
    instrument_files = [
        SPLITS_DIR / 'autotagging_instrument-train.tsv',
        SPLITS_DIR / 'autotagging_instrument-validation.tsv',
        SPLITS_DIR / 'autotagging_instrument-test.tsv'
    ]
    
    moodtheme_files = [
        SPLITS_DIR / 'autotagging_moodtheme-train.tsv',
        SPLITS_DIR / 'autotagging_moodtheme-validation.tsv',
        SPLITS_DIR / 'autotagging_moodtheme-test.tsv'
    ]
    
    # Load all tags
    all_tracks = {}
    
    print("Loading instrument tags...")
    for tsv_file in instrument_files:
        if tsv_file.exists():
            tags = load_tags_from_tsv(tsv_file)
            for track_id, data in tags.items():
                if track_id not in all_tracks:
                    all_tracks[track_id] = {
                        'path': data['path'],
                        'instrument': set(data['tags']),
                        'mood': set()
                    }
                else:
                    all_tracks[track_id]['instrument'].update(data['tags'])
    
    print("Loading mood/theme tags...")
    for tsv_file in moodtheme_files:
        if tsv_file.exists():
            tags = load_tags_from_tsv(tsv_file)
            for track_id, data in tags.items():
                if track_id not in all_tracks:
                    all_tracks[track_id] = {
                        'path': data['path'],
                        'instrument': set(),
                        'mood': set(data['tags'])
                    }
                else:
                    all_tracks[track_id]['mood'].update(data['tags'])
    
    print(f"\nTotal tracks with instrument/mood tags: {len(all_tracks)}")
    
    # Count tag frequencies
    instrument_counts = Counter()
    mood_counts = Counter()
    
    for track_id, data in all_tracks.items():
        instrument_counts.update(data['instrument'])
        mood_counts.update(data['mood'])
    
    print(f"\nUnique instrument tags: {len(instrument_counts)}")
    print(f"Unique mood/theme tags: {len(mood_counts)}")
    
    # Show top tags
    print("\nTop 10 instrument tags:")
    for tag, count in instrument_counts.most_common(10):
        print(f"  {tag}: {count} tracks")
    
    print("\nTop 10 mood/theme tags:")
    for tag, count in mood_counts.most_common(10):
        print(f"  {tag}: {count} tracks")
    
    # Step 1: Check which audio files exist
    print("\nStep 1: Scanning available audio files...")
    available_tracks = []
    not_found_count = 0
    
    pbar = tqdm(all_tracks.items(), 
                total=len(all_tracks),
                desc="Scanning files",
                unit="track")
    
    for track_id, data in pbar:
        audio_path = find_audio_file(data['path'])
        
        if audio_path and audio_path.exists():
            # Calculate tag diversity score (prefer tracks with multiple tags)
            tag_diversity = len(data['instrument']) + len(data['mood'])
            
            available_tracks.append({
                'track_id': track_id,
                'path': data['path'],
                'audio_path': audio_path,
                'instrument_tags': data['instrument'],
                'mood_tags': data['mood'],
                'tag_diversity': tag_diversity
            })
        else:
            not_found_count += 1
    
    pbar.close()
    
    print(f"\nAudio files found: {len(available_tracks)}")
    print(f"Audio files not found: {not_found_count}")
    
    # Step 2: Select diverse songs
    print(f"\nStep 2: Selecting {NUM_SONGS} songs with diverse tags...")
    
    if len(available_tracks) <= NUM_SONGS:
        selected_tracks = available_tracks
        print(f"Selected all {len(selected_tracks)} available tracks")
    else:
        # Strategy: Select tracks with good tag coverage
        # Prioritize tracks with both instrument AND mood tags
        
        # Separate tracks by tag types
        both_tags = [t for t in available_tracks if len(t['instrument_tags']) > 0 and len(t['mood_tags']) > 0]
        only_instrument = [t for t in available_tracks if len(t['instrument_tags']) > 0 and len(t['mood_tags']) == 0]
        only_mood = [t for t in available_tracks if len(t['instrument_tags']) == 0 and len(t['mood_tags']) > 0]
        
        print(f"  Tracks with both tags: {len(both_tags)}")
        print(f"  Tracks with only instrument: {len(only_instrument)}")
        print(f"  Tracks with only mood: {len(only_mood)}")
        
        # Allocate songs
        target_both = min(NUM_SONGS // 2, len(both_tags))
        target_instrument = min(NUM_SONGS // 4, len(only_instrument))
        target_mood = NUM_SONGS - target_both - target_instrument
        target_mood = min(target_mood, len(only_mood))
        
        selected_tracks = []
        
        # Select from each category
        if both_tags:
            # Sort by tag diversity and select top ones
            both_tags.sort(key=lambda x: x['tag_diversity'], reverse=True)
            selected_tracks.extend(random.sample(both_tags, target_both))
        
        if only_instrument:
            only_instrument.sort(key=lambda x: len(x['instrument_tags']), reverse=True)
            selected_tracks.extend(random.sample(only_instrument[:len(only_instrument)//2], 
                                                target_instrument))
        
        if only_mood:
            only_mood.sort(key=lambda x: len(x['mood_tags']), reverse=True)
            selected_tracks.extend(random.sample(only_mood[:len(only_mood)//2], 
                                                target_mood))
        
        # If we still need more, add randomly
        if len(selected_tracks) < NUM_SONGS:
            selected_ids = {t['track_id'] for t in selected_tracks}
            remaining = [t for t in available_tracks if t['track_id'] not in selected_ids]
            needed = NUM_SONGS - len(selected_tracks)
            if remaining:
                selected_tracks.extend(random.sample(remaining, min(needed, len(remaining))))
        
        # Shuffle
        random.shuffle(selected_tracks)
        
        print(f"Selected {len(selected_tracks)} songs total")
    
    # Step 3: Copy files and create CSV
    print(f"\nStep 3: Copying selected files to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    successful_copies = []
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'filename', 'instrument_tags', 'mood_tags', 'all_tags'])
        
        pbar = tqdm(selected_tracks, desc="Copying files", unit="file")
        
        for track_data in pbar:
            track_id = track_data['track_id']
            source_file = track_data['audio_path']
            
            # Create a clean filename
            dest_filename = f"{track_id}.mp3"
            dest_path = OUTPUT_DIR / dest_filename
            
            try:
                shutil.copy2(source_file, dest_path)
                
                # Format tags for CSV
                instrument_str = '|'.join(sorted(track_data['instrument_tags']))
                mood_str = '|'.join(sorted(track_data['mood_tags']))
                all_tags_str = '|'.join(sorted(track_data['instrument_tags'] | track_data['mood_tags']))
                
                writer.writerow([
                    track_id,
                    dest_filename,
                    instrument_str,
                    mood_str,
                    all_tags_str
                ])
                
                successful_copies.append(dest_filename)
                pbar.set_postfix({'copied': len(successful_copies)})
                
            except Exception as e:
                tqdm.write(f"Error copying {source_file}: {e}")
        
        pbar.close()
    
    print(f"\n" + "="*60)
    print(f"Completed!")
    print(f"Successfully copied {len(successful_copies)} songs to {OUTPUT_DIR}")
    print(f"CSV file created: {OUTPUT_CSV}")
    print("="*60)
    
    # Print final statistics
    print("\nFinal statistics:")
    
    # Count tag distributions in selected songs
    selected_instrument = Counter()
    selected_mood = Counter()
    tracks_with_both = 0
    tracks_with_instrument_only = 0
    tracks_with_mood_only = 0
    
    for track_data in selected_tracks:
        has_instrument = len(track_data['instrument_tags']) > 0
        has_mood = len(track_data['mood_tags']) > 0
        
        if has_instrument and has_mood:
            tracks_with_both += 1
        elif has_instrument:
            tracks_with_instrument_only += 1
        elif has_mood:
            tracks_with_mood_only += 1
        
        selected_instrument.update(track_data['instrument_tags'])
        selected_mood.update(track_data['mood_tags'])
    
    print(f"\nTag coverage:")
    print(f"  Tracks with both instrument & mood tags: {tracks_with_both}")
    print(f"  Tracks with only instrument tags: {tracks_with_instrument_only}")
    print(f"  Tracks with only mood tags: {tracks_with_mood_only}")
    print(f"  Total instrument tags: {len(selected_instrument)}")
    print(f"  Total mood tags: {len(selected_mood)}")
    
    print(f"\nTop 5 selected instrument tags:")
    for tag, count in selected_instrument.most_common(5):
        print(f"  {tag}: {count} tracks")
    
    print(f"\nTop 5 selected mood/theme tags:")
    for tag, count in selected_mood.most_common(5):
        print(f"  {tag}: {count} tracks")


if __name__ == '__main__':
    main()

