import { useEffect, useState } from "react";
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
    allTagNames
}) => {
    const [plotData, setPlotData] = useState([]);

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

    const normalizados = normalizarTaggrams(taggrams);

    useEffect(() => {
        if (!data || data.length === 0) return;

        if (!tagsSeleccionados || tagsSeleccionados.length === 0 || normalizados.length === 0) {
            setPlotData(data);
            return;
        }

        const indices = tagsSeleccionados
            .map(tag => allTagNames.indexOf(tag))
            .filter(idx => idx >= 0);

        const activaciones = normalizados.map(tg => {
            const valores = indices.map(i => tg[i]);
            return valores.reduce((a, b) => a + b, 0) / valores.length;
        });

        const updated = data.map(trace => ({
            ...trace,
            marker: {
                color: activaciones,
                colorscale: "Reds",
                cmin: 0,
                cmax: 1,
                showscale: true,
                size: 6
            }
        }));

        setPlotData(updated);
    }, [tagsSeleccionados, data, taggrams, allTagNames]);

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
            />

            <div className="grafica">
                <Plot
                    data={plotData}
                    layout={layout}
                    onClick={(e) => {
                        const punto = e.points[0];
                        const idx = punto.pointIndex;
                        if (!normalizados || normalizados.length <= idx) return;

                        const vector = normalizados[idx];

                        const topTags = vector
                            .map((v, i) => ({ i, v }))
                            .sort((a, b) => b.v - a.v)
                            .slice(0, 3)
                            .map(t => `${allTagNames[t.i]}: ${t.v.toFixed(2)}`);
                        alert(`Top tags del punto:\n${topTags.join("\n")}`);
                    }}
                />
            </div>
        </div>
    );
};
