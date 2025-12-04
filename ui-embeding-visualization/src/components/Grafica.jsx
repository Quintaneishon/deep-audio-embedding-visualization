import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import { SelectorGrafico } from "./SelectorGrafico";
import DeckGL from '@deck.gl/react';
import { PointCloudLayer } from '@deck.gl/layers';
import { OrbitView, OrthographicView } from '@deck.gl/core';
import "../styles/grafica.css";

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
    embeddings,
    visualizar,
    setVisualizar,
    agruparPor,
    setAgruparPor,
    dimensiones,
    setDimensiones
}) => {
    const audioRef = useRef(null);
    const [currentPlayingPoint, setCurrentPlayingPoint] = useState(null);
    const [graficaCargando, setGraficaCargando] = useState(false);
    const [hoveredPoint, setHoveredPoint] = useState(null);

    // Transform embeddings data to point cloud format
    const pointCloudData = useMemo(() => {
        if (!embeddings || embeddings.length === 0) {
            return [];
        }
        
        setGraficaCargando(true);
        
        return embeddings.map(emb => {
            // Get color based on tag/genre
            const colorMap = data?.find(trace => 
                trace.name === emb.tag || 
                (trace.x && trace.x.some((x, i) => 
                    Math.abs(x - emb.coords[0]) < 0.0001 && 
                    Math.abs(trace.y[i] - emb.coords[1]) < 0.0001
                ))
            );
            
            // Extract RGB from plotly color or use default
            let color = [100, 150, 200, 200]; // default blue
            if (colorMap && colorMap.marker && colorMap.marker.color) {
                const plotlyColor = colorMap.marker.color;
                if (typeof plotlyColor === 'string') {
                    // Parse hex or rgb color
                    const hex = plotlyColor.replace('#', '');
                    if (hex.length === 6) {
                        color = [
                            parseInt(hex.substr(0, 2), 16),
                            parseInt(hex.substr(2, 2), 16),
                            parseInt(hex.substr(4, 2), 16),
                            200
                        ];
                    }
                }
            }
            
            return {
                position: dimensiones === 3 
                    ? [emb.coords[0], emb.coords[1], emb.coords[2] || 0]
                    : [emb.coords[0], emb.coords[1], 0],
                color: color,
                normal: [0, 0, 1],
                embedding: emb
            };
        });
    }, [embeddings, data, dimensiones]);

    // Set initial view state based on dimensions
    const initialViewState = useMemo(() => {
        if (dimensiones === 3) {
            return {
                target: [0, 0, 0],
                rotationX: 30,
                rotationOrbit: 30,
                zoom: 1,
                minZoom: -5,
                maxZoom: 10
            };
        } else {
            return {
                target: [0, 0, 0],
                zoom: 0,
                minZoom: -5,
                maxZoom: 10
            };
        }
    }, [dimensiones]);

    // Handle point click for audio playback
    const handlePointClick = useCallback((info) => {
        if (!info.object || !info.object.embedding) return;
        
        const embedding = info.object.embedding;
        console.log('Clicked:', embedding);

        if (embedding && embedding.audio) {
            const pointId = `${embedding.name}_${embedding.coords.join('_')}`;

            // Check if clicking the same point that's currently playing
            if (currentPlayingPoint === pointId && audioRef.current) {
                audioRef.current.pause();
                audioRef.current.currentTime = 0;
                setCurrentPlayingPoint(null);
                console.log(`Stopped: ${embedding.name}`);
                return;
            }

            // Stop current audio if playing a different track
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current.currentTime = 0;
            }

            // Construct the audio URL
            const audioUrl = `http://localhost:5000/audio/${embedding.audio}`;
            console.log("Playing audio from:", audioUrl);

            // Play the new audio
            audioRef.current = new Audio(audioUrl);

            audioRef.current.addEventListener('ended', () => {
                setCurrentPlayingPoint(null);
            });

            audioRef.current.play().catch(err => {
                console.error("Error playing audio:", err);
                alert(`Error playing: ${embedding.name}`);
                setCurrentPlayingPoint(null);
            });

            setCurrentPlayingPoint(pointId);
            console.log(`Playing: ${embedding.name} (${embedding.tag})`);
        }
    }, [currentPlayingPoint]);

    // Create point cloud layer
    const layers = useMemo(() => {
        if (!pointCloudData || pointCloudData.length === 0) {
            setGraficaCargando(false);
            return [];
        }

        setTimeout(() => setGraficaCargando(false), 500);

        return [
            new PointCloudLayer({
                id: 'point-cloud-layer',
                data: pointCloudData,
                getPosition: d => d.position,
                getNormal: d => d.normal,
                getColor: d => {
                    // Highlight currently playing point
                    if (currentPlayingPoint && 
                        d.embedding && 
                        currentPlayingPoint.includes(d.embedding.name)) {
                        return [255, 255, 0, 255]; // Yellow for playing
                    }
                    return d.color;
                },
                pointSize: 3,
                pickable: true,
                autoHighlight: true,
                highlightColor: [255, 255, 255, 200],
                onClick: (info) => handlePointClick(info),
                onHover: (info) => {
                    if (info.object) {
                        setHoveredPoint(info.object.embedding);
                    } else {
                        setHoveredPoint(null);
                    }
                }
            })
        ];
    }, [pointCloudData, currentPlayingPoint, handlePointClick]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === "Escape") {
                if (audioRef.current) {
                    audioRef.current.pause();
                    audioRef.current.currentTime = 0;
                }
                setCurrentPlayingPoint(null);
                console.log("Audio detenido con ESC");
            }
        };

        window.addEventListener("keydown", handleKeyDown);

        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, []);

    return (
        <div
            className="divGrafico"
            style={izq ? { borderRight: "1px solid black" } : {}}
        >
            <SelectorGrafico
                arquitectura={arquitectura}
                setArquitectura={setArquitectura}
                dataset1={dataset}
                setDataset={setDataset}
                tipoGrafica={tipoGrafica}
                setTipoGrafica={setTipoGrafica}
                visualizar={visualizar}
                setVisualizar={setVisualizar}
                agruparPor={agruparPor}
                setAgruparPor={setAgruparPor}
                dimensiones={dimensiones}
                setDimensiones={setDimensiones}
            />
            {graficaCargando && <p>Cargando grafica...</p>}
            <div className="grafica" style={{ position: 'relative' }}>
                <DeckGL
                    views={dimensiones === 3 ? [new OrbitView()] : [new OrthographicView()]}
                    initialViewState={initialViewState}
                    controller={true}
                    layers={layers}
                    style={{ width: '100%', height: '100%' }}
                />
                {hoveredPoint && (
                    <div style={{
                        position: 'absolute',
                        zIndex: 1,
                        pointerEvents: 'none',
                        left: '10px',
                        top: '10px',
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        color: 'white',
                        padding: '10px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        maxWidth: '200px'
                    }}>
                        <div><strong>Name:</strong> {hoveredPoint.name}</div>
                        <div><strong>Genre:</strong> {hoveredPoint.tag}</div>
                    </div>
                )}
            </div>
        </div>
    );
};
