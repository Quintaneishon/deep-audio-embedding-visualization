import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
import torch
import librosa
from MusiCNN import Musicnn
from VGG import VGG_Res
from proyecciones import proyectar_embeddings

MSD_W_MUSICNN = './pesos/msd/musicnn.pth'
MTAT_W_MUSICNN = './pesos/mtat/musicnn.pth'

MSD_W_VGG = './pesos/msd/vgg.pth'
MTAT_W_VGG = './pesos/mtat/vgg.pth'

N_TAGS = 50      # Número de etiquetas (tags)
EMBEDDING_DIM = 200  # Dimensión de los Embeddings (la clase Musicnn lo define internamente) 
SR_MUSICNN = 16000     # Tasa de muestreo que MusiCNN espera

DC = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def embeddings_y_taggrams_MusiCNN(pesos, audio, dataset_name='msd', segment_size=None, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using MusiCNN.
    
    Args:
        pesos: Path to model weights
        audio: Path to audio file
        dataset_name: Dataset name ('msd' or 'mtat')
        segment_size: Deprecated - kept for backward compatibility
        sr: Sample rate
    
    Returns:
        embeddings: (1, embedding_dim) - single vector per song
        taggrams: (1, n_tags) - single taggram per song
    """
    model = Musicnn(n_class=N_TAGS, dataset=dataset_name) 
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)  # Move model to GPU
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        taggrams_tensor, embeddings_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGG(pesos, audio, dataset_name='msd', segment_size=None, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using VGG.
    
    Args:
        pesos: Path to model weights
        audio: Path to audio file
        dataset_name: Dataset name ('msd' or 'mtat')
        segment_size: Deprecated
        sr: Sample rate
    
    Returns:
        embeddings: (1, embedding_dim)
        taggrams: (1, n_tags)
    """
    use_simple = (dataset_name == 'mtat')
    model = VGG_Res(n_class=N_TAGS, use_simple_res=use_simple)
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
        taggrams_tensor, embeddings_tensor = model(x)
        
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

