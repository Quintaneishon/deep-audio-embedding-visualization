from flask import Flask, request, g, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import click
import sys
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'db'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
import database
import preprocessing
import config

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
    tracks = database.get_all_tracks()
    return [track['filename'] for track in tracks]

@app.route("/tags")
def listar_tags():
    genre_map = database.get_tags()
    return list(set(genre_map.values()))

@app.route('/embeddings')
def embedding():
    try:
        red = request.args.get('red', '').lower()
        dataset = request.args.get('dataset', '').lower()
        metodo = request.args.get('metodo', '').lower()
        dimensiones = request.args.get('dimensions', '').lower()
    except:
        print('No se encontro param de:',red,dataset)
    
    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if dataset == '':
        dataset = 'msd'
    if metodo == '':
        metodo = 'umap'
    if dimensiones == '':
        dimensions = 2
    else:
        dimensions = int(dimensiones)
  
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

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from the audio directory with CORS headers."""
    try:
        response = send_from_directory(config.AUDIO_DIR, filename)
        # Add CORS headers to allow Web Audio API access
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return response
    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        return {"error": f"Audio file not found: {filename}"}, 404

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Upload and process a new audio file."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return {'success': False, 'error': 'No file provided'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'success': False, 'error': 'No file selected'}, 400
        
        # Validate MP3 format
        if not file.filename.lower().endswith('.mp3'):
            return {'success': False, 'error': 'Only MP3 files are supported'}, 400
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.AUDIO_DIR, filename)
        
        # Save file (overwrite if exists)
        file.save(filepath)
        print(f"Saved uploaded file: {filename}")
        
        # Process the track to extract embeddings
        print(f"Starting embedding extraction for: {filename}")
        success = preprocessing.process_single_track(filename)
        
        if success:
            print(f"Successfully processed: {filename}")
            return {'success': True, 'filename': filename}, 200
        else:
            print(f"Failed to process: {filename}")
            return {'success': False, 'error': 'Failed to process audio file'}, 500
            
    except Exception as e:
        print(f"Error in upload_audio: {e}")
        return {'success': False, 'error': str(e)}, 500

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


@app.cli.command('extract-acoustic-features')
def extract_acoustic_features_command():
    """Extract acoustic features (spectral centroid and tempo) for all tracks."""
    with app.app_context():
        click.echo('Extracting acoustic features for all tracks...')
        click.echo('This extracts spectral centroid and tempo using librosa.')
        stats = preprocessing.extract_and_store_acoustic_features()
        click.echo('\nAcoustic feature extraction complete!')
        click.echo(f"  Tracks processed: {stats['tracks_processed']}")
        click.echo(f"  Tracks skipped (already exists): {stats['tracks_skipped']}")
        click.echo(f"  Errors: {stats['errors']}")


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
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(__file__), '../reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename
        output_file = os.path.join(reports_dir, f'genre_similarity_report.txt')
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write('=== Genre Similarity Analysis Report ===\n')
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('\n=== Summary ===\n')
            
            for combo_key, data in results.items():
                stats = data['aggregate_stats']
                f.write(f"\n{combo_key}:\n")
                f.write(f"  Songs analyzed: {stats['total_songs']}\n")
                f.write(f"  EMBEDDING-BASED:\n")
                f.write(f"    Mean similarity to own genre: {stats['emb_mean_similarity_to_own_genre']:.4f}\n")
                f.write(f"    Agreement rate: {stats['emb_agreement_rate']:.2%}\n")
                f.write(f"  TAGGRAM-BASED:\n")
                f.write(f"    Mean similarity to own genre: {stats['tag_mean_similarity_to_own_genre']:.4f}\n")
                f.write(f"    Agreement rate: {stats['tag_agreement_rate']:.2%}\n")
                
                # Display MAP@K metrics if available
                if 'map_at_k_genre_precision' in stats:
                    k_value = stats.get('map_at_k_k_value', 10)
                    n_songs = stats.get('map_at_k_n_songs_evaluated', 0)
                    f.write(f"  MAP@{k_value} (K-NEAREST NEIGHBORS):\n")
                    f.write(f"    Songs evaluated: {n_songs}\n")
                    f.write(f"    Genre Precision: {stats['map_at_k_genre_precision']:.4f}\n")
                    f.write(f"    Physical Error: {stats['map_at_k_mean_physical_error']:.4f}\n")
        
        click.echo(f'Report saved to: {output_file}')

@app.cli.command('clean-db')
@click.option('--drop-tables', is_flag=True, default=False, help='Drop all tables before cleaning')
def clean_db_command(drop_tables):
    """Clean the database."""
    with app.app_context():
        database.clean_db(drop_tables)
        click.echo('Database cleaned successfully.')
