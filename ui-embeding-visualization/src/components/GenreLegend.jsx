import '../styles/genrelegend.css';

export const GenreLegend = ({ genres, show, genreColors = {} }) => {
    if (!show) return null;

    // Filter only genres that are actually present in the data
    const visibleGenres = genres && genres.length > 0 
        ? genres.filter(g => g !== 'default')
        : Object.keys(genreColors).filter(g => g !== 'default');

    if (visibleGenres.length === 0) return null;

    return (
        <div className="genre-legend">
            <h4>Genre Colors</h4>
            <div className="legend-items">
                {visibleGenres.map(genre => (
                    <div key={genre} className="legend-item">
                        <div 
                            className="legend-color" 
                            style={{ backgroundColor: genreColors[genre] || '#999999' }}
                        />
                        <span className="legend-label">{genre}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

