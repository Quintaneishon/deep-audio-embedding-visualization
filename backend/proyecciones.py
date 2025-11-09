import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def proyectar_embeddings(embeddings, metodo='umap', std_normalize=False, random_state=42):

    if embeddings is None or len(embeddings.shape) != 2:
        raise ValueError("Embeddings debe ser una matriz 2D (n_muestras, n_dimensiones).")

    X = embeddings.copy()
    n_samples = X.shape[0]
    
    # Check if we have enough samples for dimensionality reduction
    if n_samples < 2:
        raise ValueError(f"Se requieren al menos 2 muestras para proyección 2D, pero solo hay {n_samples}. "
                        "Procesa más archivos de audio antes de calcular proyecciones.")

    # STD-Normalization (como en el paper)
    if metodo == 'std-pca' or std_normalize:
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-8
        X = X / stds

    if metodo == 'pca' or metodo == 'std-pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(X)

    elif metodo == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state, n_iter=1000)
        coords = reducer.fit_transform(X)

    elif metodo == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)

    else:
        raise ValueError("Método no válido. Usa: 'pca', 'std-pca', 'tsne' o 'umap'.")

    return coords
