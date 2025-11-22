import { useEffect, useState, useRef } from "react";
import { SelectorGrafico } from "./SelectorGrafico";
import Plot from "react-plotly.js";
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
    tagsSeleccionados,
    taggrams,
    allTagNames,
    embeddings,
    visualizar,
    setVisualizar,
    agruparPor,
    setAgruparPor,
}) => {
    const [plotData, setPlotData] = useState([]);
    const audioRef = useRef(null);
    const [currentPlayingPoint, setCurrentPlayingPoint] = useState(null);

    useEffect(() => {
        if (!data || data.length === 0 || !embeddings || embeddings.length === 0) {
            setPlotData(data);
            return;
        }

        // Add hover text with name and genre information
        const updatedData = data.map(trace => {
            // For each point in this trace, find the corresponding embedding
            const hoverTexts = trace.x.map((x, i) => {
                const y = trace.y[i];
                const embedding = embeddings.find(e =>
                    Math.abs(e.coords[0] - x) < 0.0001 &&
                    Math.abs(e.coords[1] - y) < 0.0001
                );

                if (embedding) {
                    return `Name: ${embedding.name}<br>Genre: ${embedding.tag}`;
                }
                return `x: ${x.toFixed(2)}<br>y: ${y.toFixed(2)}`;
            });

            return {
                ...trace,
                text: hoverTexts,
                hovertemplate: '%{text}<extra></extra>'
            };
        });

        setPlotData(updatedData);
    }, [data, embeddings]);

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
            />

            <div className="grafica">
                <Plot
                    data={plotData}
                    layout={layout}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                    config={{ responsive: true }}
                    onClick={(e) => {
                        try {
                            const x = e.points[0].x;
                            const y = e.points[0].y;
                            console.log(x, y);

                            // Find the embedding that matches the clicked point
                            const embedding = embeddings.find((emb) =>
                                Math.abs(emb.coords[0] - x) < 0.0001 &&
                                Math.abs(emb.coords[1] - y) < 0.0001
                            );

                            if (embedding && embedding.audio) {
                                // Create a unique identifier for the clicked point
                                const pointId = `${embedding.name}_${x}_${y}`;

                                // Check if clicking the same point that's currently playing
                                if (currentPlayingPoint === pointId && audioRef.current) {
                                    // Stop the audio
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

                                // Add event listener for when audio ends
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
                        } catch (err) {
                            console.error("Error handling click:", err);
                        }
                    }}
                />
            </div>
        </div>
    );
};
