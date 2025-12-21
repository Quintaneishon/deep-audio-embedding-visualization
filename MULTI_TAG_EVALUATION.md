# Multi-Tag Evaluation Workflow

This document describes how to evaluate your audio embeddings using **instrument and mood/theme tags** (not genre) to assess generalization capabilities.

## Overview

The multi-tag evaluation metrics provide a way to evaluate your trained models on semantic dimensions they weren't explicitly trained on:

- **Hubness (Skewness)**: Measures if some points become "hubs" that appear in many nearest-neighbor lists (a problem in high-dimensional spaces)
- **nDCG@10**: Normalized Discounted Cumulative Gain - measures retrieval quality based on tag overlap

## Why Use Instrument/Mood Tags?

Your models are trained on **genre labels**, so evaluating on genres would be circular reasoning. Using **instrument** and **mood/theme tags** tests if the learned embeddings capture broader musical characteristics:

- ✅ Tests generalization beyond training objective
- ✅ Evaluates semantic understanding of timbre, instrumentation, and emotional content
- ✅ More realistic music retrieval scenario

## Workflow

### Step 1: Extract MTG-Jamendo Songs with Tags

Run the selection script to extract songs that have instrument and/or mood/theme tags:

```bash
cd /home/ar/Data/Ajitzi/deep-audio-embedding-visualization
python select_songs_mtg.py
```

This will:
- Scan MTG-Jamendo dataset for tracks with instrument/mood tags
- Select ~1000 diverse songs (prioritizing tracks with both tag types)
- Copy audio files to `audio_mtg/`
- Create `audio_mtg/selected_songs_mtg.csv` with tag information

### Step 2: Index and Preprocess the MTG Songs

Update your audio directory configuration to point to the new directory, or manually index:

```bash
# Option 1: Update config.AUDIO_DIR to point to audio_mtg/ temporarily
# Then run:
flask --app backend/server.py init-db
flask --app backend/server.py index-audio
flask --app backend/server.py preprocess-all

# Option 2: Process files individually (if you already have embeddings)
# The compute-multi-tag-metrics command will look for embeddings in the database
```

### Step 3: Compute Multi-Tag Metrics

Once embeddings are extracted for the MTG songs, compute the metrics:

```bash
cd /home/ar/Data/Ajitzi/deep-audio-embedding-visualization
flask --app backend/server.py compute-multi-tag-metrics
```

This will:
- Load instrument and mood/theme tags from the CSV
- Retrieve embeddings for each model/dataset combination
- Compute Hubness (Skewness) and nDCG@10 metrics
- Generate a report at `reports/multi_tag_metrics_report.txt`

### Step 4: Analyze Results

Review the generated report:

```bash
cat reports/multi_tag_metrics_report.txt
```

**Interpreting Metrics:**

**Hubness (Skewness):**
- `< 0.2`: Excellent - low hubness problem
- `0.2 - 0.5`: Acceptable - moderate hubness
- `> 0.5`: Poor - significant hubness problem

**nDCG@10:**
- `> 0.7`: Excellent retrieval quality
- `0.5 - 0.7`: Good retrieval quality
- `0.3 - 0.5`: Moderate retrieval quality
- `< 0.3`: Poor retrieval quality

## File Structure

```
deep-audio-embedding-visualization/
├── select_songs_mtg.py              # Extract MTG songs with tags
├── evaluation/
│   └── multi_tag_metrics.py         # Metrics implementation
├── audio_mtg/                       # MTG songs (created by script)
│   └── selected_songs_mtg.csv      # Tag mappings
├── reports/
│   └── multi_tag_metrics_report.txt # Evaluation results
├── db/
│   └── preprocessing.py            # Added compute_multi_tag_metrics()
└── backend/
    └── server.py                   # Added Flask command
```

## Dataset Information

**MTG-Jamendo Dataset Tags:**
- **87 Genre tags** (used for training)
- **40 Instrument tags** (used for evaluation)
  - Examples: voice, piano, guitar, synthesizer, drums, bass, strings
- **56 Mood/Theme tags** (used for evaluation)
  - Examples: emotional, relaxing, energetic, documentary, film, corporate

**Coverage:**
- ~25,135 tracks with instrument tags
- ~18,486 tracks with mood/theme tags
- Many tracks have multiple tags (multi-label scenario)

## Notes

- The evaluation uses **Jaccard similarity** for multi-label relevance scoring
- Tracks without any instrument/mood tags are filtered out
- Results are computed separately for:
  - Combined tags (instrument + mood)
  - Instrument-only tags
  - Mood-only tags

## References

- **Hubness**: Radovanović, M., et al. "Hubs in space: Popular nearest neighbors in high-dimensional data." (2010)
- **nDCG**: Järvelin, K., & Kekäläinen, J. "Cumulated gain-based evaluation of IR techniques." (2002)
- **MTG-Jamendo Dataset**: Bogdanov, D., et al. "The MTG-Jamendo Dataset for Automatic Music Tagging." (2019)

