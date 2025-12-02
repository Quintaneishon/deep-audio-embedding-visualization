# Deep Audio Embedding Visualization

A comprehensive system for extracting, training, visualizing, and analyzing deep audio embeddings using multiple state-of-the-art models. Supports both CNN and Transformer architectures with pre-trained models (MusiCNN, VGG, Whisper, MERT, VGGish) and custom contrastive learning training (WhisperContrastive).

## Overview

This project provides:
- **Inference**: Extract embeddings from pre-trained models for visualization and analysis
- **Training**: Train custom models using supervised contrastive learning on genre data
- **Visualization**: Interactive 2D/3D projections using UMAP and t-SNE
- **API**: Flask backend with React frontend for exploration
- **Analysis**: Genre similarity metrics and embedding quality evaluation

## Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher (for frontend)
- CUDA-compatible GPU (recommended for training and faster inference)

## Project Structure

```
deep-audio-embedding-visualization/
├── audio/                    # Audio files (.mp3, .wav, etc.)
├── backend/                  # Flask server and API
│   ├── server.py            # Flask endpoints and CLI commands
│   ├── main.py              # Model inference functions
│   ├── proyecciones.py      # UMAP/t-SNE dimensionality reduction
│   └── utils.py
├── db/                       # Database and preprocessing
│   ├── database.py          # SQLite operations
│   ├── preprocessing.py     # Audio processing pipeline
│   └── utils.py
├── ML/                       # Neural network models
│   ├── models/              # Model architectures
│   │   ├── MusiCNN.py
│   │   ├── VGG.py
│   │   ├── Whisper.py
│   │   ├── MERT.py
│   │   ├── VGGish.py
│   │   └── WhisperContrastive.py  # Trainable contrastive model
│   ├── pesos/               # Pre-trained weights
│   │   ├── msd/             # Million Song Dataset
│   │   └── mtat/            # MagnaTagATune
│   ├── checkpoints/         # Training checkpoints
│   ├── logs/                # TensorBoard logs
│   ├── dataset.py           # MTG-Jamendo dataset loader
│   ├── losses.py            # Supervised contrastive loss
│   ├── modules.py           # Shared components
│   └── train_contrastive.py # Training script
├── ui-embeding-visualization/  # React frontend
├── config.py                # Configuration
├── requirements.txt         # Python dependencies
├── select_songs.py          # Dataset preparation utility
└── compare_models.py        # Model comparison tool
```

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify model weights

Ensure pre-trained weights exist:

```
ML/pesos/
├── msd/
│   ├── musicnn.pth
│   └── vgg.pth
└── mtat/
    ├── musicnn.pth
    └── vgg.pth
```

### 4. Initialize database

```bash
flask --app backend.server init-db
```

### 5. Install frontend (optional)

```bash
cd ui-embeding-visualization
npm install
cd ..
```

## Quick Start

```bash
# 1. Place audio files in audio/ directory

# 2. Index audio files
flask --app backend.server index-audio

# 3. Extract embeddings (all models)
flask --app backend.server preprocess-all

# 4. Start backend server
flask --app backend.server run
# Backend available at http://localhost:5000

# 5. Start frontend (separate terminal)
cd ui-embeding-visualization
npm start
# Frontend opens at http://localhost:3000
```

## Usage

### Inference Pipeline

#### Step 1: Place audio files

Copy audio files to `audio/` directory. Supported formats: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`

#### Step 2: Index audio files

```bash
flask --app backend.server index-audio
```

Scans audio directory and registers files in database.

#### Step 3: Extract embeddings

```bash
# Process all indexed tracks
flask --app backend.server preprocess-all

# Process single track
flask --app backend.server preprocess-track "filename.mp3"
```

Extracts embeddings and taggrams for all model/dataset combinations:
- **MusiCNN**: MSD, MTAT (200-dim embeddings, 50-dim taggrams)
- **VGG**: MSD, MTAT (512-dim embeddings, 50-dim taggrams)
- **Whisper**: base, small, tiny (512-768 dim embeddings)
- **MERT**: 95M, 330M (768-1024 dim embeddings)
- **VGGish**: pretrained (128-dim embeddings)
- **WhisperContrastive**: base (128-dim embeddings)

#### Step 4: Compute genre similarity (optional)

If you have genre labels in `audio/selected_songs.csv`:

```bash
flask --app backend.server compute-genre-similarity
```

Analyzes genre centroids, cosine similarity, and agreement rates.

### Running the Application

#### Backend server

```bash
flask --app backend.server run
```

Available at `http://localhost:5000`

#### Frontend (separate terminal)

```bash
cd ui-embeding-visualization
npm start
```

Opens at `http://localhost:3000`

## Implementation

Train custom models using supervised contrastive learning on your own datasets.

### Dataset Preparation

Extract balanced genre samples from FMA dataset:

```bash
python select_songs.py
```

Configures FMA paths in script and creates `audio/selected_songs.csv` with balanced genre distribution.

### Training Pipeline

Training uses the MTG-Jamendo dataset for genre-based contrastive learning.

#### Configuration

Edit `config.py` to set training parameters:

```python
CONTRASTIVE_TRAINING = {
    'model_name': 'base',          # Whisper model: tiny, base, small
    'projection_dim': 128,         # Projection head output dimension
    'batch_size': 16,              # Adjust for GPU memory
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'temperature': 0.07,           # Contrastive loss temperature
    'audio_duration': 30.0,        # Seconds
    'early_stopping_patience': 10,
}

# Dataset paths
MTG_JAMENDO_ROOT = '/path/to/mtg-jamendo-dataset'
MTG_JAMENDO_AUDIO_DIR = '/path/to/mtg-jamendo-dataset/songs'
MTG_JAMENDO_SPLITS_DIR = '/path/to/mtg-jamendo-dataset/data/splits/split-0'
```

#### Start training

```bash
python ML/train_contrastive.py \
    --model-name base \
    --projection-dim 128 \
    --batch-size 16 \
    --num-epochs 50 \
    --learning-rate 1e-3
```

Training outputs:
- **Checkpoints**: `ML/checkpoints/whisper_contrastive_TIMESTAMP/`
- **Logs**: `ML/logs/whisper_contrastive_TIMESTAMP/` (TensorBoard)

Monitor training:

```bash
tensorboard --logdir ML/logs
```

Displays validation loss, model size, convergence speed, and recommendations.

## API Reference

### GET /audios

Returns list of all audio filenames.

**Endpoint:** `/audios`

**Response:**

```json
[
  "track001.mp3",
  "track002.mp3"
]
```

### GET /tags

Returns list of all genre tags.

**Endpoint:** `/tags`

**Response:**

```json
[
  "genre---rock",
  "genre---jazz",
  "genre---ambient"
]
```

### GET /embeddings

Returns projected embedding coordinates for visualization.

**Endpoint:** `/embeddings`

**Query Parameters:**

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `red` | string | `musicnn` | `musicnn`, `vgg`, `whisper`, `mert`, `vggish`, `whisper_contrastive` | Model architecture |
| `dataset` | string | `msd` | `msd`, `mtat`, `base`, `small`, `tiny`, `95m`, `330m`, `pretrained` | Dataset/model size |
| `metodo` | string | `umap` | `umap`, `tsne` | Dimensionality reduction method |
| `dimensions` | integer | `2` | `2`, `3` | Embedding dimensions |

**Example Request:**

```
GET /embeddings?red=whisper&dataset=base&metodo=umap&dimensions=2
```

**Example Response:**

```json
{
  "name": "embeddings_whisper_base_umap_2",
  "data": [
    {
      "audio": "../audio/track001.mp3",
      "coords": [4.82, 5.01],
      "name": "track001.mp3",
      "tag": "genre---ambient"
    },
    {
      "audio": "../audio/track002.mp3",
      "coords": [2.65, 2.69],
      "name": "track002.mp3",
      "tag": "genre---hiphop"
    }
  ]
}
```

### GET /taggrams

Returns projected taggram coordinates (intermediate layer activations).

**Endpoint:** `/taggrams`

**Query Parameters:** Same as `/embeddings`

**Example Request:**

```
GET /taggrams?red=vgg&dataset=mtat&metodo=tsne&dimensions=3
```

**Example Response:**

```json
{
  "name": "taggrams_vgg_mtat_tsne_3",
  "data": [
    {
      "audio": "../audio/track001.mp3",
      "coords": [10.17, 6.43, -2.15],
      "name": "track001.mp3",
      "tag": "genre---ambient"
    }
  ]
}
```

### GET /audio/<filename>

Serves audio files for playback.

**Endpoint:** `/audio/<filename>`

**Example:**

```
GET /audio/track001.mp3
```

**Response:** Binary audio stream (`audio/mpeg`)

## Frontend Development

The React frontend provides interactive visualization of embeddings.

### Location

```
ui-embeding-visualization/
├── src/
│   ├── components/
│   │   ├── Grafica.jsx         # Visualization chart
│   │   ├── SelectorGrafico.jsx # Model/parameter selector
│   │   └── SidePane.jsx        # Controls panel
│   ├── hooks/
│   │   └── useMakeRequest.jsx  # API client
│   └── App.js
└── package.json
```

### Development

```bash
cd ui-embeding-visualization
npm start
```

Frontend runs at `http://localhost:3000` and connects to backend at `http://localhost:5000`.

### Build for production

```bash
npm run build
```

Output in `build/` directory.

## Model Details

### MusiCNN

- **Architecture**: CNN with vertical/horizontal filters
- **Input**: Mel-spectrogram (96 mel bands, 16kHz)
- **Embedding**: 200 dimensions
- **Taggram**: 50 tags
- **Training**: Supervised on MSD/MTAT
- **Best for**: Fast genre classification

### VGG

- **Architecture**: VGG-like CNN with residual connections
- **Input**: Mel-spectrogram (128 mel bands, 16kHz)
- **Embedding**: 512 dimensions
- **Taggram**: 50 tags
- **Training**: Supervised on MSD/MTAT
- **Best for**: Deep convolutional features

### Whisper

- **Architecture**: Transformer encoder (speech-to-text)
- **Input**: Log-mel spectrogram (80 mel bands, 16kHz)
- **Embedding**: 512 (base), 768 (small), 384 (tiny)
- **Taggram**: 50 tags
- **Training**: Self-supervised on 680k hours of speech
- **Best for**: Transfer learning, temporal patterns

### MERT

- **Architecture**: Transformer encoder (Wav2Vec2-style)
- **Input**: Waveform (24kHz)
- **Embedding**: 768 (95M), 1024 (330M)
- **Taggram**: 50 tags
- **Training**: Self-supervised on 160k hours of music
- **Best for**: State-of-the-art music understanding

### VGGish

- **Architecture**: VGG-inspired CNN for audio
- **Input**: Mel-spectrogram
- **Embedding**: 128 dimensions
- **Training**: Pre-trained on AudioSet
- **Best for**: General audio features

### WhisperContrastive

- **Architecture**: Frozen Whisper encoder + trainable projection head
- **Input**: Log-mel spectrogram (80 mel bands, 16kHz)
- **Embedding**: 128 dimensions (from projection)
- **Training**: Supervised contrastive learning on MTG-Jamendo
- **Best for**: Fine-tuned genre discrimination

### Model Comparison

| Model | Type | Domain | Sample Rate | Embedding Dim | Speed | Quality |
|-------|------|--------|-------------|---------------|-------|---------|
| MusiCNN | CNN | Music | 16 kHz | 200 | Fast | Good |
| VGG | CNN | Music | 16 kHz | 512 | Fast | Good |
| Whisper | Transformer | Speech | 16 kHz | 384-768 | Medium | Good |
| MERT | Transformer | Music | 24 kHz | 768-1024 | Slow | Best |
| VGGish | CNN | Audio | 16 kHz | 128 | Fast | Good |
| WhisperContrastive | Transformer | Music | 16 kHz | 128 | Medium | Excellent (fine-tuned) |

## Database Schema

### tracks table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `filename` | TEXT | Unique audio filename |
| `duration` | REAL | Track duration (seconds) |
| `processed_at` | TIMESTAMP | Last processing time |
| `created_at` | TIMESTAMP | Record creation time |

### embeddings table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `track_id` | INTEGER | Foreign key to tracks |
| `model` | TEXT | Model name |
| `dataset` | TEXT | Dataset/size name |
| `embedding_data` | BLOB | Numpy array (dimensions vary) |
| `embedding_shape` | TEXT | Array shape string |
| `taggram_data` | BLOB | Numpy array (50 dimensions) |
| `taggram_shape` | TEXT | Array shape string |
| `created_at` | TIMESTAMP | Record creation time |

## Configuration

Edit `config.py` to customize paths and parameters:

### Core Paths

```python
AUDIO_DIR = 'audio/'                    # Audio files
DATABASE_PATH = 'db/audio_cache.db'     # SQLite database
MODEL_DIR = 'ML/models/'                # Model architectures
```

### Model Configuration

```python
CONV_MODELS = ['musicnn', 'vgg']
DATASETS = ['msd', 'mtat']

TRANF_MODELS = ['whisper', 'mert', 'whisper_contrastive', 'vggish']
MODEL_SIZES = ['base', '95m', 'base', 'pretrained']

MODEL_WEIGHTS = {
    'musicnn': {
        'msd': 'ML/pesos/msd/musicnn.pth',
        'mtat': 'ML/pesos/mtat/musicnn.pth'
    },
    'whisper_contrastive': {
        'base': 'ML/checkpoints/whisper_contrastive_TIMESTAMP/best_model.pth'
    },
    # ... other models
}
```

### Training Configuration

```python
CONTRASTIVE_TRAINING = {
    'model_name': 'base',
    'projection_dim': 128,
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'temperature': 0.07,
    'audio_duration': 30.0,
    'early_stopping_patience': 10,
}

MTG_JAMENDO_ROOT = '/path/to/mtg-jamendo-dataset'
```

### Dimensionality Reduction

```python
ON_DEMAND_PROJECTION_METHODS = ['tsne', 'umap']

# UMAP parameters: n_neighbors=15, min_dist=0.1
# t-SNE parameters: perplexity=30, n_iter=1000
```

## Performance Notes

### GPU Acceleration

- Automatically uses CUDA if available
- 10-50x speedup for inference
- Required for practical training

Check GPU status in logs: `"GPU acceleration enabled: [GPU name]"`

### Processing Time (per 3-minute track)

**Inference:**
- With GPU: 1-2 seconds per model
- Without GPU: 10-30 seconds per model
- Total per track: 4-8 seconds (GPU) or 40-120 seconds (CPU)

**Training:**
- Base model: ~4-6 hours for 50 epochs (GPU)
- Tiny model: ~2-3 hours for 50 epochs (GPU)

## Maintenance Commands

### Clean database

Remove all records:

```bash
flask --app backend.server clean-db
```

Drop and recreate tables:

```bash
flask --app backend.server clean-db --drop-tables
flask --app backend.server init-db
```

## Troubleshooting

### Missing model weights

Ensure all weight files exist in `ML/pesos/` or download pre-trained models.

### Database locked errors

Close all Flask processes and check for stale connections.

### Import errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Training not starting

- Verify MTG-Jamendo dataset paths in `config.py`
- Check audio files exist in `MTG_JAMENDO_AUDIO_DIR`
- Ensure split files exist in `MTG_JAMENDO_SPLITS_DIR`

### Frontend can't connect to backend

- Ensure backend is running on port 5000
- Check CORS is enabled in `backend/server.py`
- Verify API endpoint URLs in frontend code
