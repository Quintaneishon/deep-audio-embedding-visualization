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
    allTagNames,
    embeddings,
    visualizar,
    setVisualizar
}) => {
    const [plotData, setPlotData] = useState([]);
    useEffect(() => {
        if (!data || data.length === 0) return;


        if (!tagsSeleccionados || tagsSeleccionados.length === 0) {
            setPlotData(data);
            return;
        }

        // const indices = tagsSeleccionados
        //     .map(tag => allTagNames.indexOf(tag))
        //     .filter(idx => idx >= 0);

        // const activaciones = normalizados.map(tg => {
        //     const valores = indices.map(i => tg[i]);
        //     return valores.reduce((a, b) => a + b, 0) / valores.length;
        // });

        // const updated = data.map(trace => ({
        //     ...trace,
        //     marker: {
        //         color: activaciones,
        //         colorscale: "Reds",
        //         cmin: 0,
        //         cmax: 1,
        //         showscale: true,
        //         size: 6
        //     }
        // }));

        // setPlotData(updated);
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
                visualizar={visualizar}
                setVisualizar={setVisualizar}
            />

            <div className="grafica">
                <Plot
                    data={plotData}
                    layout={layout}
                    onClick={(e) => {
                        try {
                            const x = e.points[0].x;
                            const y = e.points[0].y;
                            const embedTag = embeddings.filter((e) => { return e.coords[0] == x && e.coords[1] == y })
                            alert(`Tag: ${embedTag[0].tag}`);
                        } catch (e) { }
                    }}

                />
            </div>
        </div>
    );
};
