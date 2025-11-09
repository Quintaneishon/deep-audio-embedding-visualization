from flask import Flask, request, g
from flask_cors import CORS
import os
import click
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'db'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
from utils import embeddings_y_taggrams
from main import TAGS, proyectar_embeddings
import database
import preprocessing
import cache_manager
import config

AUDIO_ROUTE = './audio/'

app = Flask(__name__)
CORS(app)

# Register database teardown
app.teardown_appcontext(database.close_db)

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
    
    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if pista == '':
        pista = '1.mp3'
    if dataset == '':
        dataset = 'msd'
    
    # Try to get from cache first
    cached_emb = cache_manager.get_cached_embedding(pista, red, dataset)
    cached_tag = cache_manager.get_cached_taggram(pista, red, dataset)
    
    if cached_emb is not None and cached_tag is not None:
        print(f'Using cached embeddings for {pista} ({red}/{dataset})')
        embeddings = cached_emb
        taggrams = cached_tag
    else:
        # Fallback: compute on-demand
        print(f'Computing embeddings on-demand for {pista} ({red}/{dataset})')
        embeddings, taggrams = embeddings_y_taggrams(red, pista, dataset)
        
        # Optionally cache the results for future use
        try:
            cache_manager.save_to_cache(embeddings, 'embedding', pista, red, dataset)
            cache_manager.save_to_cache(taggrams, 'taggram', pista, red, dataset)
            print(f'  Cached embeddings for future use')
        except Exception as e:
            print(f'  Warning: Could not cache results: {e}')
    
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
    
    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if pista == '':
        pista = '1.mp3'
    if dataset == '':
        dataset = 'msd'
    if metodo == '':
        metodo = 'umap'
    
    print(f'Representacion request: {pista} ({red}/{dataset}) - method: {metodo}')
    
    # Try to get embeddings/taggrams from cache first
    cached_emb = cache_manager.get_cached_embedding(pista, red, dataset)
    cached_tag = cache_manager.get_cached_taggram(pista, red, dataset)
    
    if cached_emb is not None and cached_tag is not None:
        print(f'  Using cached embeddings')
        embeddings = cached_emb
        taggrams = cached_tag
    else:
        # Fallback: compute on-demand
        print(f'  Computing embeddings on-demand')
        embeddings, taggrams = embeddings_y_taggrams(red, pista, dataset)
        
        # Cache the results for future use
        try:
            cache_manager.save_to_cache(embeddings, 'embedding', pista, red, dataset)
            cache_manager.save_to_cache(taggrams, 'taggram', pista, red, dataset)
            print(f'  Cached embeddings')
        except Exception as e:
            print(f'  Warning: Could not cache embeddings: {e}')
    
    # Handle projections: PCA from cache, t-SNE/UMAP on-demand
    if metodo == 'pca':
        # Try to get PCA from cache
        cached_proj = cache_manager.get_cached_projection(pista, red, dataset, 'pca')
        if cached_proj is not None:
            print(f'  Using cached PCA projection')
            coords = cached_proj
        else:
            # Compute and cache PCA
            print(f'  Computing PCA projection')
            coords = proyectar_embeddings(embeddings, metodo='pca')
            try:
                cache_manager.save_to_cache(coords, 'projection', pista, red, dataset, method='pca')
                print(f'  Cached PCA projection')
            except Exception as e:
                print(f'  Warning: Could not cache PCA: {e}')
    else:
        # t-SNE and UMAP are always computed on-demand (better for different parameters)
        print(f'  Computing {metodo.upper()} projection on-demand')
        coords = proyectar_embeddings(embeddings, metodo=metodo)
    
    # Filter to only consider genre tags
    genre_indices = [i for i, tag in enumerate(TAGS) if tag.startswith('genre---')]
    
    if genre_indices:
        # Get taggram values only for genre tags
        genre_values = [taggrams[0, i] for i in genre_indices]
        # Find the index of the max value among genre tags
        max_genre_idx = genre_values.index(max(genre_values))
        dominant_tag_idx = genre_indices[max_genre_idx]
        dominant_tag_name = TAGS[dominant_tag_idx]
        dominant_tag_value = float(taggrams[0, dominant_tag_idx])
    else:
        # Fallback if no genre tags found
        dominant_tag_idx = None
        dominant_tag_name = None
        dominant_tag_value = None

    
    return {
            'representacion_'+metodo+'_'+red+"_"+dataset+'_'+pista : coords.tolist(),
            'taggrams_'+metodo+'_'+red+"_"+dataset+'_'+pista : taggrams.tolist(),
            'dominant_tag_name': dominant_tag_name,
            'dominant_tag_idx': dominant_tag_idx,
            'dominant_tag_value': dominant_tag_value
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
    """Extract embeddings, taggrams, and PCA projections for all tracks."""
    with app.app_context():
        click.echo('Starting batch preprocessing...')
        click.echo('This may take a while depending on the number of tracks.')
        stats = preprocessing.process_all_tracks()
        click.echo('\nPreprocessing complete!')
        click.echo(f"  Tracks processed: {stats['tracks_processed']}")
        click.echo(f"  Embeddings created: {stats['embeddings_created']}")
        click.echo(f"  Projections created: {stats['projections_created']}")
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


@app.cli.command('cache-status')
def cache_status_command():
    """Show cache statistics."""
    with app.app_context():
        tracks = database.get_all_tracks()
        click.echo(f'\nTotal tracks in database: {len(tracks)}')
        
        # Count embeddings
        db = database.get_db()
        embeddings = db.execute('SELECT COUNT(*) as count FROM embeddings').fetchone()
        click.echo(f'Total embeddings cached: {embeddings["count"]}')
        
        # Count projections
        projections = db.execute('SELECT COUNT(*) as count FROM projections').fetchone()
        click.echo(f'Total projections cached: {projections["count"]}')
        
        click.echo(f'\nCache directory: {config.CACHE_DIR}')
        click.echo(f'Database: {config.DATABASE_PATH}') 

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
            click.echo(f"  Mean similarity to own genre: {stats['mean_similarity_to_own_genre']:.4f}")
            click.echo(f"  Genre agreement rate: {stats['genre_agreement_rate']*100:.1f}%")