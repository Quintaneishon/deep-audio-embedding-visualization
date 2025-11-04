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

MSD_W_MUSICNN = './pesos/msd/musicnn.pth'
MTAT_W_MUSICNN = './pesos/mtat/musicnn.pth'

MSD_W_VGG = './pesos/msd/vgg.pth'
#MTAT_W_vgg = './pesos/mtat/vgg.pth'

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

def embeddings_y_taggrams_MusiCNN(pesos, audio):
    
    if 'msd' in pesos.lower():
        dataset_name = 'msd'
    elif 'mtat' in pesos.lower():
        dataset_name = 'mtat'
    else:
        dataset_name = 'mtat'

    model = Musicnn(n_class=N_TAGS, dataset=dataset_name) 
    
    try:
        model.load_state_dict(torch.load(pesos, map_location=DC))
    except RuntimeError as e:
        print(f"\n[ERROR] Error al cargar los pesos. Asegúrate de que Musicnn está definida con los parámetros por defecto de {dataset_name}.")
        print(f"Detalle: {e}")
        return None, None
        
    model.eval()

    input_tensor = load_and_preprocess(audio)

    if input_tensor is None:
        return None, None
        
    input_tensor = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        taggrams_tensor, embeddings_tensor = model(input_tensor)
    
    taggrams = taggrams_tensor.squeeze(0).numpy()
    embeddings = embeddings_tensor.squeeze(0).numpy()
    
    return embeddings, taggrams

def embeddings_y_taggrams_VGG(pesos, audio):
    
    if 'msd' in pesos.lower():
        dataset_name = 'msd'
    elif 'mtat' in pesos.lower():
        dataset_name = 'mtat'
    else:
        dataset_name = 'mtat'

    model = VGG_Res(n_class=N_TAGS) 
    
    try:
        model.load_state_dict(torch.load(pesos, map_location=DC))
    except RuntimeError as e:
        print(f"\n[ERROR] Error al cargar los pesos. Asegúrate de que VGG_Res está definida con los parámetros por defecto de {dataset_name}.")
        print(f"Detalle: {e}")
        return None, None
        
    model.eval()

    input_tensor = load_and_preprocess(audio)

    if input_tensor is None:
        return None, None
        
    input_tensor = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        taggrams_tensor, embeddings_tensor = model(input_tensor)
    
    taggrams = taggrams_tensor.squeeze(0).numpy()
    embeddings = embeddings_tensor.squeeze(0).numpy()
    
    return embeddings, taggrams


if __name__ == "__main__":
    #embeddings, taggrams = embeddings_y_taggrams_VGG(MSD_W_VGG, './audio/1.mp3')

    embeddings, taggrams = embeddings_y_taggrams_MusiCNN(MSD_W_MUSICNN, './audio/1.mp3')
    #embeddings, taggrams = embeddings_y_taggrams_MusiCNN(MTAT_W_MUSICNN, './audio/1.mp3')

    if embeddings is not None:
        print(f"Dimensiones del Embedding (Penúltima Capa): {embeddings.shape}")
        print(f"Dimensiones del Taggram (Predicción de Tags): {taggrams.shape}")