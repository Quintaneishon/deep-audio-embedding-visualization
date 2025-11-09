import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))

import torch
import numpy as np
import librosa
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio
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

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']

DC = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_and_preprocess(audio_path, sr=SR_MUSICNN):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except FileNotFoundError:
        print("\n[ERROR] Archivo no encontrado. Asegúrate de que la ruta sea correcto.")
        return None
    except Exception as e:
        print(f"\n[ERROR] Error al cargar el archivo: {e}")
        return None
        
    return torch.from_numpy(y).float()

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
    print(pesos, audio, dataset_name)
    model = Musicnn(n_class=N_TAGS, dataset=dataset_name) 
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0)
        taggrams_tensor, embeddings_tensor = model(x)
        
        # Get single embedding and taggram for the whole song
        embeddings = embeddings_tensor.squeeze(0).numpy()  # (embedding_dim,)
        taggrams = taggrams_tensor.squeeze(0).numpy()      # (n_tags,)
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGG(pesos, audio, segment_size=None, sr=SR_MUSICNN):
    """
    Extract embeddings and taggrams from full audio using VGG.
    
    Args:
        pesos: Path to model weights
        audio: Path to audio file
        segment_size: Deprecated - kept for backward compatibility
        sr: Sample rate
    
    Returns:
        embeddings: (1, embedding_dim) - single vector per song
        taggrams: (1, n_tags) - single taggram per song
    """
    model = VGG_Res(n_class=N_TAGS)
    model.load_state_dict(torch.load(pesos, map_location=DC))
    model.to(DC)
    model.eval()

    # Load full audio
    y, _ = librosa.load(audio, sr=sr)
    
    # Process entire song at once
    with torch.no_grad():
        x = torch.from_numpy(y).float().unsqueeze(0).to(DC)  # (1, n_samples)
        taggrams_tensor, embeddings_tensor = model(x)
        
        # Get single embedding and taggram for the whole song
        embeddings = embeddings_tensor.squeeze(0).cpu().numpy()  # (embedding_dim,)
        taggrams = taggrams_tensor.squeeze(0).cpu().numpy()      # (n_tags,)
    
    # Reshape to (1, dim) for consistency with downstream code
    embeddings = embeddings.reshape(1, -1)  # (1, embedding_dim)
    taggrams = taggrams.reshape(1, -1)      # (1, n_tags)
    
    return embeddings, taggrams


if __name__ == "__main__":
    embeddings, taggrams = embeddings_y_taggrams_VGG(MSD_W_VGG, './audio/1.mp3')
    print("Embeddings:", embeddings.shape)
    print("Taggrams:", taggrams.shape)

    coords = proyectar_embeddings(embeddings, metodo='umap')
    print(coords[:5])
