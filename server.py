from flask import Flask, request
from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, MSD_W_MUSICNN, MTAT_W_MUSICNN,MSD_W_VGG
from flask_cors import CORS
import os

AUDIO_ROUTE = './audio/'

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/audios")
def listar_audios():
    carpeta = "./audio"

    archivos = [
        f for f in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, f))
    ]

    return archivos

@app.route('/embedding')
def embedding():
    try:
        red = request.args.get('red', '').lower()
        pista = request.args.get('pista', '')
        dataset = request.args.get('dataset', '').lower()
    except:
        print('No se encontro param de:',red,pista,dataset)
    
    if red == '':
        red = 'MusiCNN'
    if pista == "":
        pista = '1.mp3'
    if dataset == "":
        dataset = 'MSD'

    funcion = embeddings_y_taggrams_MusiCNN
    ds = MSD_W_MUSICNN

    if red == 'VGG':
        funcion = embeddings_y_taggrams_VGG
        if dataset == 'MSD':
            ds = MSD_W_VGG
    else:
        if dataset == 'MTAT':
            ds=  MTAT_W_MUSICNN

    print('Obteniendo embeddings y taggrams...')
    embeddings, taggrams = funcion(ds,AUDIO_ROUTE+pista)
    return {
        'embeddings_'+red+"_"+dataset+'_'+pista: embeddings.tolist(),
        'taggrams_'+red+"_"+dataset+'_'+pista: taggrams.tolist()
    }