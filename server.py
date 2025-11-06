from flask import Flask, request
from flask_cors import CORS
import os
from utils import embeddings_y_taggrams
from main import TAGS, proyectar_embeddings

AUDIO_ROUTE = './audio/'

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/tags")
def listar_tags():
    return TAGS

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
    
    embeddings, taggrams=embeddings_y_taggrams(red,pista,dataset)
    return {
        'embeddings_'+red+"_"+dataset+'_'+pista: embeddings.tolist(),
        'taggrams_'+red+"_"+dataset+'_'+pista: taggrams.tolist()
    }

@app.route('/representacion')
def representacion():
    try:
        red = request.args.get('red', '').lower()
        pista = request.args.get('pista', '')
        dataset = request.args.get('dataset', '').lower()
        metodo = request.args.get('metodo', '').lower()
    except:
        print('No se encontro param de:',red,pista,dataset)
    embeddings, taggrams = embeddings_y_taggrams(red,pista,dataset)
    print('Metodo',metodo)
    if metodo == '':
        metodo = 'umap'
    coords=proyectar_embeddings(embeddings, metodo=metodo)
    return {
            'representacion_'+metodo+'_'+red+"_"+dataset+'_'+pista : coords.tolist(),
            'taggrams_'+metodo+'_'+red+"_"+dataset+'_'+pista : taggrams.tolist()
    } 