# Deep Audio Embedding Visualization

A system for extracting, visualizing, and analyzing deep audio embeddings using MusiCNN and VGG models trained on MSD (Million Song Dataset) and MTAT (MagnaTagATune) datasets.

## Overview

This project processes audio files to extract embeddings and taggrams, stores them in a SQLite database, and provides visualization through dimensionality reduction techniques (UMAP, t-SNE). It includes genre similarity analysis and a React-based frontend for interactive visualization.

## Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher (for frontend)
- CUDA-compatible GPU (recommended for faster processing)

## Project Structure

```
deep-audio-embedding-visualization/
├── audio/                    # Place your audio files here (.mp3, .wav, etc.)
├── backend/                  # Flask server and processing logic
│   ├── server.py            # Flask API endpoints
│   ├── main.py              # Model inference functions
│   ├── proyecciones.py      # Dimensionality reduction (UMAP, t-SNE)
│   └── utils.py
├── db/                       # Database and preprocessing
│   ├── database.py          # SQLite operations for embeddings
│   └── preprocessing.py     # Audio indexing and batch processing
├── ML/                       # Neural network models
│   ├── MusiCNN.py           # MusiCNN architecture
│   ├── VGG.py               # VGG architecture
│   ├── modules.py           # Shared neural network components
│   └── pesos/               # Model weights directory
│       ├── msd/             # Million Song Dataset weights
│       │   ├── musicnn.pth
│       │   └── vgg.pth
│       └── mtat/            # MagnaTagATune weights
│           ├── musicnn.pth
│           └── vgg.pth
├── ui-embeding-visualization/  # React frontend
├── config.py                # Project configuration
└── requirements.txt         # Python dependencies
```

## Installation

### 1. Create and activate Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify model weights

Ensure the following model weight files exist:

```
ML/pesos/
├── msd/
│   ├── musicnn.pth  # MusiCNN trained on Million Song Dataset
│   └── vgg.pth      # VGG trained on Million Song Dataset
└── mtat/
    ├── musicnn.pth  # MusiCNN trained on MagnaTagATune
    └── vgg.pth      # VGG trained on MagnaTagATune
```

If weights are missing, obtain them from the original model sources or training pipeline.

### 4. Initialize database

Create the SQLite database schema for storing embeddings:

```bash
flask --app backend.server init-db
```

This creates `db/audio_cache.db` with tables for tracks, embeddings, and taggrams.

If you want to run the React frontend:

```bash
cd ui-embeding-visualization
npm install
cd ..
```

## Usage

### Processing Audio Files

#### Step 1: Place audio files

Copy your audio files to the `audio/` directory. Supported formats: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`

#### Step 2: Index audio files

Scan the audio directory and register files in the database:

```bash
flask --app backend.server index-audio
```

This command:
- Scans `audio/` for audio files
- Extracts duration metadata
- Creates database entries without processing

#### Step 3: Extract embeddings and taggrams

Process all indexed tracks to extract embeddings:

```bash
flask --app backend.server preprocess-all
```

This command:
- Loads audio files at 16kHz sample rate
- Runs inference on all model/dataset combinations (4 total: MusiCNN-MSD, MusiCNN-MTAT, VGG-MSD, VGG-MTAT)
- Extracts 200-dimensional embeddings and 50-dimensional taggrams per track
- Stores results as binary BLOBs in SQLite
- Displays progress bar

Processing time depends on:
- Number of audio files
- File durations
- GPU availability (significantly faster with CUDA)

#### Step 4: Compute genre similarity (optional)

If you have genre labels in `audio/selected_songs.csv`:

```bash
flask --app backend.server compute-genre-similarity
```

This analyzes:
- Genre centroids (mean embeddings per genre)
- Cosine similarity between tracks and genre centroids
- Agreement rates (prediction vs. actual genre)
- Statistics for both embeddings and taggrams

### Running the Application

#### Start backend server

```bash
flask --app backend.server run
```

Backend API available at `http://localhost:5000`

Available endpoints:
- `GET /embeddings?red=musicnn&dataset=msd&metodo=umap&dimensiones=2` - Get projected embeddings
- `GET /taggrams?red=vgg&dataset=mtat&metodo=tsne&dimensiones=3` - Get projected taggrams
- `GET /audios` - List all audio files

#### Start frontend (separate terminal)

```bash
cd ui-embeding-visualization
npm start
```

React application opens at `http://localhost:3000`

### Processing Individual Files

To process a single audio file:

```bash
flask --app backend.server preprocess-track "filename.mp3"
```

This is useful when adding new files incrementally.

## Database Schema

### Tables

**tracks**
- `id`: Primary key
- `filename`: Unique audio filename
- `duration`: Track duration in seconds
- `processed_at`: Timestamp of last processing
- `created_at`: Record creation timestamp

**embeddings**
- `id`: Primary key
- `track_id`: Foreign key to tracks
- `model`: Model name ('musicnn' or 'vgg')
- `dataset`: Dataset name ('msd' or 'mtat')
- `embedding_data`: BLOB containing numpy array (200 dimensions)
- `embedding_shape`: String representation of array shape
- `taggram_data`: BLOB containing numpy array (50 dimensions)
- `taggram_shape`: String representation of array shape
- `created_at`: Record creation timestamp

## Model Details

### MusiCNN
- Architecture: Convolutional neural network with vertical/horizontal filters
- Input: Mel-spectrogram (96 mel bands, 16kHz)
- Embedding size: 200 dimensions (MSD) or 200 dimensions (MTAT)
- Taggram size: 50 tags

### VGG
- Architecture: VGG-like CNN with residual connections
- Input: Mel-spectrogram (128 mel bands, 16kHz)
- Embedding size: 512 dimensions
- Taggram size: 50 tags

## Dimensionality Reduction

Embeddings are projected to 2D or 3D using:

**UMAP (Uniform Manifold Approximation and Projection)**
- Parameters: `n_neighbors=15`, `min_dist=0.1`
- Better preservation of global structure
- Faster computation

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Parameters: `perplexity=30`, `n_iter=1000`
- Better preservation of local structure
- Slower computation

## Maintenance Commands

### Clean database

Remove all records but preserve schema:

```bash
flask --app backend.server clean-db
```

Drop and recreate all tables:

```bash
flask --app backend.server clean-db --drop-tables
flask --app backend.server init-db
```

## Performance Notes

**GPU Acceleration**
- Automatically uses CUDA if available
- Significantly faster processing (10-50x speedup)
- Check on startup for: "GPU acceleration enabled: [GPU name]"

**Processing Time Estimates** (approximate, per 3-minute track)
- With GPU: 1-2 seconds per model/dataset combination
- Without GPU: 10-30 seconds per model/dataset combination
- Total per track: 4-8 seconds (GPU) or 40-120 seconds (CPU)

## Configuration

Edit `config.py` to modify:
- `AUDIO_DIR`: Audio files directory
- `DATABASE_PATH`: SQLite database location
- `MODELS`: List of models to use
- `DATASETS`: List of datasets to use
- `MODEL_WEIGHTS`: Paths to model weight files

## Troubleshooting

**Missing model weights**
- Ensure all 4 weight files exist in `ML/pesos/`
- Check file paths in `config.py`

**Database locked errors**
- Close all Flask processes
- Check for stale database connections

**Out of memory errors**
- Reduce batch size in preprocessing
- Process files individually using `preprocess-track`
- Use CPU instead of GPU for very large files

**Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

