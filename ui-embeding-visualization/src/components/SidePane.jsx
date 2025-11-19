import Form from 'react-bootstrap/Form';
import '../styles/sidepane.css';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';

export const SidePane = ({ listaTags, tags, setTags, cargarDatos, progreso }) => {

    const handleTagsChange = (e) => {
        const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
        setTags(selectedValues);
        console.log('Tags seleccionados:', selectedValues);
    };

    // Separate genre tags from other tags
    const genreTags = listaTags.filter(tag => tag.startsWith('genre---'));
    const otherTags = listaTags.filter(tag => !tag.startsWith('genre---'));

    const formatTagDisplay = (tag) => {
        if (tag.startsWith('genre---')) {
            return `${tag.replace('genre---', '').toUpperCase()}`;
        }
        return tag;
    };

    return (
        <div className='sidepane'>
            <div className='item'>
                <p>Tags</p>
                {genreTags.length > 0 && tags.some(t => t.startsWith('genre---')) && (
                    <div style={{ 
                        padding: '8px', 
                        backgroundColor: '#e3f2fd', 
                        borderRadius: '4px', 
                        marginBottom: '10px',
                        fontSize: '12px',
                        color: '#1976d2'
                    }}>
                        <strong>💡 Genre Filter Active:</strong> Gray points = other genres
                    </div>
                )}
                <Form.Select
                    multiple
                    aria-label="Tags"
                    value={tags}
                    onChange={handleTagsChange}
                    className="tags-select"
                >
                    {genreTags.length > 0 && (
                        <optgroup label="🎵 Genres (Click to filter)">
                            {genreTags.map((tag, idx) => (
                                <option value={tag} key={`genre-${idx}`}>
                                    {formatTagDisplay(tag)}
                                </option>
                            ))}
                        </optgroup>
                    )}
                    {otherTags.length > 0 && (
                        <optgroup label="🏷️ Other Tags (Activation)">
                            {otherTags.map((tag, idx) => (
                                <option value={tag} key={`other-${idx}`}>
                                    {tag}
                                </option>
                            ))}
                        </optgroup>
                    )}
                </Form.Select>
                <Button style={{ marginTop: '10px' }} onClick={() => setTags([])}>Limpiar</Button>
            </div>

            {progreso !== 0 && progreso !== 100 &&
                <div>
                    <p>Cargando...</p>
                    <ProgressBar now={progreso} />
                </div>
            }
            {(progreso === 0 || progreso === 100) && <Button style={{ marginTop: '10px' }} onClick={cargarDatos}>Aplicar</Button>}
        </div>
    );
};
