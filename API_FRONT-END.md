# API Documentation

This document describes the available API endpoints for the Deep Audio Embedding Visualization backend service.

---

## Table of Contents
1. [GET /audios](#get-audios)
2. [GET /tags](#get-tags)
3. [GET /embeddings](#get-embeddings)
4. [GET /taggrams](#get-taggrams)
5. [GET /audio/<filename>](#get-audiofilename)

---

## GET /audios

Returns a list of all available audio track filenames in the database.

### Request

**Endpoint:** `/audios`

**Method:** `GET`

**Parameters:** None

### Response

**Content-Type:** `application/json`

**Response Body:** Array of strings (filenames)

### Example Response

```json
[
  "track001.mp3",
  "track002.mp3",
  "track003.mp3",
  "track004.mp3"
]
```

---

## GET /tags

Returns a list of all available tags (genres) in the database.

### Request

**Endpoint:** `/tags`

**Method:** `GET`

**Parameters:** None

### Response

**Content-Type:** `application/json`

**Response Body:** Array of strings (tag names)

### Example Response

```json
[
    "genre---easylistening",
    "genre---dance",
    "genre---ambient",
    "genre---rock",
    "genre---bossanova"
]
```

---

## GET /embeddings

Computes and returns embedding coordinates for audio tracks based on specified parameters.

### Request

**Endpoint:** `/embeddings`

**Method:** `GET`

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `red` | string | No | `musicnn` | Neural network architecture to use for embeddings |
| `dataset` | string | No | `msd` | Dataset to use |
| `metodo` | string | No | `umap` | Dimensionality reduction method |
| `dimensions` | integer | No | `2` | Number of dimensions for the embedding space |

### Possible Parameter Values

**`red` (Neural Network):**
- `musicnn` (default)
- `vgg`

**`dataset`:**
- `msd` (Million Song Dataset - default)
- `mtat` (MagnaTagATune)

**`metodo` (Dimensionality Reduction Method):**
- `umap` (default)
- `tsne`

**`dimensions`:**
- `2` (default) - 2D visualization
- `3` - 3D visualization

### Response

**Content-Type:** `application/json`

**Response Body:** Object containing embedding data

```json
{
  "data": [...]
}
```

### Example Request

```
GET /embeddings?red=musicnn&dataset=msd&metodo=umap&dimensions=2
```

### Example Response

```json
{
    "data": [
        {
            "audio": "../audio/track_0353107.mp3",
            "coords": [
                4.814800262451172,
                5.014310359954834
            ],
            "name": "track_0353107.mp3",
            "tag": "genre---ambient"
        },
        {
            "audio": "../audio/track_1054383.mp3",
            "coords": [
                2.652076005935669,
                2.6902248859405518
            ],
            "name": "track_1054383.mp3",
            "tag": "genre---hiphop"
        }
    ]
}
```

---

## GET /taggrams

Computes and returns taggram coordinates based on specified parameters. Taggrams represent aggregated tag information in the embedding space.

### Request

**Endpoint:** `/taggrams`

**Method:** `GET`

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `red` | string | No | `musicnn` | Neural network architecture to use |
| `dataset` | string | No | `msd` | Dataset to use |
| `metodo` | string | No | `umap` | Dimensionality reduction method |
| `dimensions` | integer | No | `2` | Number of dimensions for the taggram space |

### Possible Parameter Values

**`red` (Neural Network):**
- `musicnn` (default)
- `vgg`

**`dataset`:**
- `msd` (Million Song Dataset - default)
- `mtat` (MagnaTagATune)

**`metodo` (Dimensionality Reduction Method):**
- `umap` (default)
- `tsne`

**`dimensions`:**
- `2` (default) - 2D visualization
- `3` - 3D visualization

### Response

**Content-Type:** `application/json`

**Response Body:** Object containing taggram data

```json
{
  "data": [...]
}
```

### Example Request

```
GET /taggrams?red=musicnn&dataset=msd&metodo=umap&dimensions=2
```

### Example Response

```json
{
    "data": [
        {
            "audio": "../audio/track_0353107.mp3",
            "coords": [
                10.169410705566406,
                6.426023960113525
            ],
            "name": "track_0353107.mp3",
            "tag": "genre---ambient"
        },
        {
            "audio": "../audio/track_1054383.mp3",
            "coords": [
                13.114777565002441,
                3.8818304538726807
            ],
            "name": "track_1054383.mp3",
            "tag": "genre---hiphop"
        }
    ]
}
```

---

---

## GET /audio/<filename>

Serves audio files from the server's audio directory.

### Request

**Endpoint:** `/audio/<filename>`

**Method:** `GET`

**URL Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | string | Yes | The name of the audio file to retrieve (e.g., `track_0353107.mp3`) |

### Response

**Content-Type:** `audio/mpeg` (or appropriate audio MIME type)

**Response Body:** Binary audio file data

### Example Request

```
GET /audio/track_0353107.mp3
```

### Example Response

Binary audio data stream

### Usage

This endpoint is used by the frontend to play audio when clicking on points in the visualization. The audio path is included in the data returned by the `/embeddings` and `/taggrams` endpoints in the format:

```json
{
  "audio": "http://localhost:5000/audio/track_0353107.mp3"
}
```

---

## Notes

- All endpoints return JSON responses (except `/audio/<filename>` which returns binary audio)
- Query parameters are case-insensitive (converted to lowercase)
- Default values are applied when parameters are not provided or are empty
- The `dimensions` parameter must be a valid integer
- Embeddings and taggrams are computed on-demand based on the requested parameters
- Audio files are served with CORS enabled for frontend access

