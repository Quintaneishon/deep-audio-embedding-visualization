import { useEffect, useState } from "react";
import { SelectorGrafico } from "./SelectorGrafico";
import { GenreLegend } from "./GenreLegend";
import Plot from "react-plotly.js";
import "../styles/grafica.css";
import { extractGenre } from "../utils/colorGenerator";

// Helper function to normalize taggram data formats
const normalizarTaggrams = (taggrams) => {
    if (!taggrams || taggrams.length === 0) return [];

    if (Array.isArray(taggrams[0]) && typeof taggrams[0][0] === "object") {
        return taggrams.map(tg => tg.map(obj => Number(obj.value)));
    }

    if (Array.isArray(taggrams[0]) && typeof taggrams[0][0] === "number") {
        return taggrams;
    }

    if (typeof taggrams[0] === "object") {
        return taggrams.map(obj => Object.values(obj).map(v => Number(v)));
    }

    return taggrams;
};

export const Grafica = ({
    arquitectura,
    setArquitectura,
    dataset,
    setDataset,
    data,
    layout,
    tipoGrafica,
    setTipoGrafica,
    izq,
    tagsSeleccionados,
    taggrams,
    allTagNames,
    vector,
    setVector,
    dimensiones,
    setDimensiones,
    genreColors = {}
}) => {
    const [plotData, setPlotData] = useState([]);
    const [genresPresent, setGenresPresent] = useState([]);
    const [audioPlayer, setAudioPlayer] = useState(null);
    const [currentPlaying, setCurrentPlaying] = useState(null);

    // Initialize audio player
    useEffect(() => {
        const audio = new Audio();
        setAudioPlayer(audio);
        
        // Cleanup on unmount
        return () => {
            audio.pause();
            audio.src = '';
        };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0) return;
        
        // Normalize taggrams inside useEffect to avoid recreating on every render
        const normalizados = normalizarTaggrams(taggrams);

        // If no tags selected, color by genre
        if (!tagsSeleccionados || tagsSeleccionados.length === 0 || normalizados.length === 0) {
            const updated = data.map(trace => {
                // Extract colors from genre tags and collect unique genres
                const genresSet = new Set();
                const colors = trace.taggrams?.map(item => {
                    const genre = extractGenre(item.tag);
                    if (genre !== 'default') {
                        genresSet.add(genre);
                    }
                    // Use dynamic genreColors from props
                    return genreColors[genre] || genreColors['default'] || '#999999';
                }) || [];

                // Update genres present in the data
                setGenresPresent(Array.from(genresSet).sort());

                // Create hover text with song name and genre
                const hoverText = trace.taggrams?.map(item => {
                    const songName = item.name || 'Unknown';
                    const genre = item.tag || 'Unknown';
                    return `Song: ${songName}<br>Genre: ${genre}`;
                }) || [];

                return {
                    ...trace,
                    text: hoverText,
                    hovertemplate: '%{text}<extra></extra>',
                    marker: {
                        color: colors.length > 0 ? colors : '#999999',
                        size: 6,
                        line: {
                            color: 'white',
                            width: 0.5
                        }
                    }
                };
            });
            setPlotData(updated);
            return;
        }

        // Check if any selected tags are genre tags
        const selectedGenreTags = tagsSeleccionados.filter(tag => tag.startsWith('genre---'));
        const selectedGenres = selectedGenreTags.map(tag => extractGenre(tag));

        // If genre tags are selected, filter by genre (gray out non-matching)
        if (selectedGenres.length > 0) {
            const updated = data.map(trace => {
                // Extract colors, but gray out genres not in selection
                const genresSet = new Set();
                const colors = trace.taggrams?.map(item => {
                    const genre = extractGenre(item.tag);
                    if (genre !== 'default') {
                        genresSet.add(genre);
                    }
                    
                    // If this genre is in the selected genres, use its color
                    if (selectedGenres.includes(genre)) {
                        return genreColors[genre] || genreColors['default'] || '#999999';
                    }
                    // Otherwise, make it gray
                    return '#cccccc';
                }) || [];

                // Update genres present in the data
                setGenresPresent(Array.from(genresSet).sort());

                // Create hover text with song name and genre
                const hoverText = trace.taggrams?.map(item => {
                    const songName = item.name || 'Unknown';
                    const genre = item.tag || 'Unknown';
                    return `Song: ${songName}<br>Genre: ${genre}`;
                }) || [];

                return {
                    ...trace,
                    text: hoverText,
                    hovertemplate: '%{text}<extra></extra>',
                    marker: {
                        color: colors.length > 0 ? colors : '#cccccc',
                        size: 6,
                        line: {
                            color: 'white',
                            width: 0.5
                        }
                    }
                };
            });
            setPlotData(updated);
            return;
        }

        // If non-genre tags are selected, use activation-based coloring
        const indices = tagsSeleccionados
            .map(tag => allTagNames.indexOf(tag))
            .filter(idx => idx >= 0);

        const activaciones = normalizados.map(tg => {
            const valores = indices.map(i => tg[i]);
            return valores.reduce((a, b) => a + b, 0) / valores.length;
        });

        const updated = data.map(trace => {
            // Create hover text with song name and genre
            const hoverText = trace.taggrams?.map(item => {
                const songName = item.name || 'Unknown';
                const genre = item.tag || 'Unknown';
                return `Song: ${songName}<br>Genre: ${genre}`;
            }) || [];

            return {
                ...trace,
                text: hoverText,
                hovertemplate: '%{text}<extra></extra>',
                marker: {
                    color: activaciones,
                    colorscale: "Reds",
                    cmin: 0,
                    cmax: 1,
                    showscale: true,
                    size: 6
                }
            };
        });

        setPlotData(updated);
    }, [tagsSeleccionados, data, taggrams, allTagNames, genreColors]);

    return (
        <div
            className="divGrafico"
            style={izq ? { borderRight: "1px solid black" } : {}}
        >
            <SelectorGrafico
                arquitectura={arquitectura}
                setArquitectura={setArquitectura}
                dataset={dataset}
                setDataset={setDataset}
                vector={vector}
                setVector={setVector}
                tipoGrafica={tipoGrafica}
                setTipoGrafica={setTipoGrafica}
                dimensiones={dimensiones}
                setDimensiones={setDimensiones}
            />

            <div className="grafica" style={{ position: 'relative' }}>
                <GenreLegend 
                    genres={genresPresent} 
                    show={
                        !tagsSeleccionados || 
                        tagsSeleccionados.length === 0 || 
                        tagsSeleccionados.some(tag => tag.startsWith('genre---'))
                    }
                    genreColors={genreColors}
                />
                <Plot
                    data={plotData}
                    layout={layout}
                    onClick={(e) => {
                        if (!e.points || e.points.length === 0) return;
                        
                        const punto = e.points[0];
                        const idx = punto.pointIndex;
                        
                        // Get the track data from taggrams (the full data array)
                        const trackData = taggrams?.[idx];
                        
                        if (!trackData || !trackData.audio || !audioPlayer) {
                            console.warn('Cannot play audio: missing data or player not ready');
                            return;
                        }
                        
                        const audioPath = trackData.audio;
                        const songName = trackData.name || 'Unknown';
                        const genre = trackData.tag || 'Unknown';
                        
                        // If clicking the same song that's playing, pause it
                        if (currentPlaying === audioPath && !audioPlayer.paused) {
                            audioPlayer.pause();
                            setCurrentPlaying(null);
                            console.log('⏸️ Paused:', songName);
                            return;
                        }
                        
                        // Stop current audio and play new one
                        audioPlayer.pause();
                        audioPlayer.src = audioPath;
                        audioPlayer.play()
                            .then(() => {
                                setCurrentPlaying(audioPath);
                                console.log(`🎵 Playing: ${songName} (${genre})`);
                            })
                            .catch((error) => {
                                console.error('❌ Error playing audio:', error);
                                alert(`Could not play audio: ${songName}\nError: ${error.message}`);
                            });
                    }}
                />
            </div>
        </div>
    );
};
