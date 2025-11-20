import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap

def proyectar_embeddings(embeddings, metodo='umap', std_normalize=False, random_state=42, n_components=2):
    """
    Project embeddings to 2D or 3D using UMAP or t-SNE.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_dimensions)
        metodo: Projection method ('umap' or 'tsne')
        std_normalize: Whether to apply standard normalization
        random_state: Random seed for reproducibility
        n_components: Number of dimensions for projection (2 or 3)
    
    Returns:
        coords: 2D or 3D coordinates (n_samples, n_components)
    """
    if embeddings is None or len(embeddings.shape) != 2:
        raise ValueError("Embeddings debe ser una matriz 2D (n_muestras, n_dimensiones).")
    
    if n_components not in [2, 3]:
        raise ValueError("n_components debe ser 2 o 3.")

    X = embeddings.copy()
    n_samples = X.shape[0]
    
    # Check if we have enough samples for dimensionality reduction
    min_samples = n_components if n_components == 2 else 3
    if n_samples < min_samples:
        raise ValueError(f"Se requieren al menos {min_samples} muestras para proyección {n_components}D, pero solo hay {n_samples}. "
                        "Procesa más archivos de audio antes de calcular proyecciones.")

    # STD-Normalization (optional)
    if std_normalize:
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-8
        X = X / stds

    if metodo == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=random_state, max_iter=1000)
        coords = reducer.fit_transform(X)

    elif metodo == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)

    else:
        raise ValueError("Método no válido. Usa: 'tsne' o 'umap'.")

    return coords
