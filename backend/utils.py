from main import embeddings_y_taggrams_MusiCNN, embeddings_y_taggrams_VGG, MSD_W_MUSICNN, MTAT_W_MUSICNN,MSD_W_VGG
AUDIO_ROUTE = './audio/'

def embeddings_y_taggrams(red,pista,dataset):
    if red == '':
        red = 'MusiCNN'
    if pista == "":
        pista = '1.mp3'
    if dataset == "":
        dataset = MSD_W_MUSICNN

    ds='msd'
    print('Obteniendo embeddings y taggrams para ',red,pista,dataset,"...")
    if red.upper() == 'VGG':
        if dataset.upper() == 'MSD':
            dataset = MSD_W_VGG
        embeddings, taggrams = embeddings_y_taggrams_VGG(dataset,AUDIO_ROUTE+pista)
    else:
        if dataset.upper() == 'MTAT':
            dataset=MTAT_W_MUSICNN
            ds=  'mtat'
        else:
            dataset=MSD_W_MUSICNN
        embeddings, taggrams = embeddings_y_taggrams_MusiCNN(dataset,AUDIO_ROUTE+pista,dataset_name=ds)

    return embeddings, taggrams