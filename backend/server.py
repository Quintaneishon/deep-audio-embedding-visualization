from flask import Flask, request, g
from flask_cors import CORS
import os
import click
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'db'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
import database
import preprocessing

AUDIO_ROUTE = './audio/'

app = Flask(__name__)
CORS(app)

# Register database teardown
app.teardown_appcontext(database.close_db)

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

@app.route('/embeddings')
def embedding():
    try:
        red = request.args.get('red', '').lower()
        dataset = request.args.get('dataset', '').lower()
        metodo = request.args.get('metodo', '').lower()
        dimensions = request.args.get('dimensiones', '').lower()
    except:
        print('No se encontro param de:',red,dataset)
    
    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if dataset == '':
        dataset = 'msd'
    if metodo == '':
        metodo = 'umap'
    if dimensions == '':
        dimensions = 2
    else:
        dimensions = int(dimensions)
  
    print(f'Computing embeddings on-demand for ({red}/{dataset})')
    embeddings = database.get_embedding_coords(red, dataset, metodo, dimensions)    
    
    return {
        'name': 'embeddings_'+red+"_"+dataset+'_'+metodo+'_'+str(dimensions),
        'data': embeddings,
    }

@app.route('/taggrams')
def taggrams():
    try:
        red = request.args.get('red', '').lower()
        dataset = request.args.get('dataset', '').lower()
        metodo = request.args.get('metodo', '').lower()
        dimensions = request.args.get('dimensions', '').lower()
    except:
        print('No se encontro param de:',red,dataset)
    
    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if dataset == '':
        dataset = 'msd'
    if metodo == '':
        metodo = 'umap'
    if dimensions == '':
        dimensions = 2
    else:
        dimensions = int(dimensions)
  
    print(f'Computing taggrams on-demand for ({red}/{dataset})')
    taggrams = database.get_taggram_coords(red, dataset, metodo, dimensions)    
    
    return {
        'name': 'taggrams_'+red+"_"+dataset+'_'+metodo+'_'+str(dimensions),
        'data': taggrams,
    }


# ============================================================================
# Flask CLI Commands for Preprocessing
# ============================================================================

@app.cli.command('init-db')
def init_db_command():
    """Initialize the database schema."""
    database.init_db()
    click.echo('Database initialized successfully.')


@app.cli.command('index-audio')
def index_audio_command():
    """Scan audio directory and index all audio files."""
    with app.app_context():
        count = preprocessing.index_audio_files()
        click.echo(f'Indexed {count} new audio files.')


@app.cli.command('preprocess-all')
def preprocess_all_command():
    """Extract embeddings and taggrams for all tracks."""
    with app.app_context():
        click.echo('Starting batch preprocessing...')
        click.echo('This may take a while depending on the number of tracks.')
        stats = preprocessing.process_all_tracks()
        click.echo('\nPreprocessing complete!')
        click.echo(f"  Tracks processed: {stats['tracks_processed']}")
        click.echo(f"  Embeddings created: {stats['embeddings_created']}")
        click.echo(f"  Errors: {stats['errors']}")


@app.cli.command('preprocess-track')
@click.argument('filename')
def preprocess_track_command(filename):
    """Process a single audio file."""
    with app.app_context():
        click.echo(f'Processing {filename}...')
        success = preprocessing.process_single_track(filename)
        if success:
            click.echo(f'Successfully processed {filename}')
        else:
            click.echo(f'Failed to process {filename}')


@app.cli.command('compute-genre-similarity')
def compute_genre_similarity_command():
    """Compute genre similarity scores (final preprocessing step)."""
    with app.app_context():
        click.echo('Computing genre similarity scores...')
        results = preprocessing.compute_genre_similarity_scores()
        
        click.echo('\n=== Summary ===')
        for combo_key, data in results.items():
            stats = data['aggregate_stats']
            click.echo(f"\n{combo_key}:")
            click.echo(f"  Songs analyzed: {stats['total_songs']}")
            click.echo(f"  EMBEDDING-BASED:")
            click.echo(f"    Mean similarity to own genre: {stats['emb_mean_similarity_to_own_genre']:.4f}")
            click.echo(f"    Agreement rate: {stats['emb_agreement_rate']:.2%}")
            click.echo(f"  TAGGRAM-BASED:")
            click.echo(f"    Mean similarity to own genre: {stats['tag_mean_similarity_to_own_genre']:.4f}")
            click.echo(f"    Agreement rate: {stats['tag_agreement_rate']:.2%}")

@app.cli.command('clean-db')
@click.option('--drop-tables', is_flag=True, default=False, help='Drop all tables before cleaning')
def clean_db_command(drop_tables):
    """Clean the database."""
    with app.app_context():
        database.clean_db(drop_tables)
        click.echo('Database cleaned successfully.')