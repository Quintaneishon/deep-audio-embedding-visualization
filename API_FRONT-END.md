# API Documentation

This document describes the available API endpoints for the Deep Audio Embedding Visualization backend service.

---

## Table of Contents
1. [GET /audios](#get-audios)
2. [GET /tags](#get-tags)
3. [GET /embeddings](#get-embeddings)
4. [GET /taggrams](#get-taggrams)

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

## Notes

- All endpoints return JSON responses
- Query parameters are case-insensitive (converted to lowercase)
- Default values are applied when parameters are not provided or are empty
- The `dimensions` parameter must be a valid integer
- Embeddings and taggrams are computed on-demand based on the requested parameters

