import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';
import { useEffect, useState } from 'react';

function App() {
  const [progreso, setProgreso] = useState(0);
  // Selector SidePane
  const [tags, setTags] = useState([]);
  const [listaTags, setListaTags] = useState([]);
  const [canciones, setCanciones] = useState([]);
  const [listaCanciones, setListaCanciones] = useState([]);
  const [visualizar, setVisualizar] = useState('embedding');
  const [agruparPor, setAgruparPor] = useState('tag');
  const [dimensiones, setDimensiones] = useState(2);

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('musicnn');
  const [dataset1, setDataset1] = useState('msd');
  const [tipoGrafica1, setTipoGrafica1] = useState('umap');

  const [graf1, setGraf1] = useState([]);
  const [taggrams1, setTaggrams1] = useState([])
  const [embeddings1, setEmbeddings1] = useState([])
  const [taggramData1, setTaggramData1] = useState([])

  const layout = { autosize: true, title: { text: '' } }

  const { obtenerAudios, obtenerEmbeddings, obtenerTaggrams, obtenerTags } = useMakeRequest();

  const getTracesBy = (nombre, coords) => {
    const grupos = {};
    coords.forEach(p => {
      if (!grupos[p[nombre]]) grupos[p[nombre]] = { x: [], y: [], z: [] };
      grupos[p[nombre]].x.push(p.coords[0]);
      grupos[p[nombre]].y.push(p.coords[1]);
      if (dimensiones === 3) {
        grupos[p[nombre]].z.push(p.coords[2]);
      }
    });

    let type = dimensiones === 3 ? "scatter3d" : 'scatter'

    const traces = Object.entries(grupos).map(([name, g]) => {
      const trace = {
        x: g.x,
        y: g.y,
        type,
        mode: "markers",
        marker: { size: 10 },
        name,
        showlegend: true,
      };
      if (dimensiones === 3) {
        trace.z = g.z;
      }
      return trace;
    });

    return traces;
  }

  const cargarDatos = async () => {
    setGraf1([]);
    setTaggrams1([])
    setProgreso(25);

    // === 1 GRAFICA ===
    const emb1 = await obtenerEmbeddings(arquitectura1, dataset1, tipoGrafica1, dimensiones);
    setProgreso(50);
    const tg1 = await obtenerTaggrams(arquitectura1, dataset1, tipoGrafica1, dimensiones);

    const coordEmbeddings = emb1.data;
    const coordsTaggrams1 = tg1.data;

    setEmbeddings1(coordEmbeddings);
    setTaggramData1(coordsTaggrams1)

    const tracesEmbeddings = getTracesBy(agruparPor, coordEmbeddings)
    setGraf1(tracesEmbeddings);

    setProgreso(75)

    const tracesTaggrams = getTracesBy(agruparPor, coordsTaggrams1);
    setTaggrams1(tracesTaggrams);

    setProgreso(100);
  };

  useEffect(() => {
    obtenerAudios().then(data => {
      setListaCanciones(data)
    })
    obtenerTags().then(data => {
      setListaTags(data);
    })
  }, [])

  useEffect(() => {
    if (embeddings1.length > 0) {
      const tracesEmbeddings = getTracesBy(agruparPor, embeddings1)
      setGraf1(tracesEmbeddings);
    }
    if (taggramData1.length > 0) {
      const tracesTaggrams = getTracesBy(agruparPor, taggramData1);
      setTaggrams1(tracesTaggrams);
    }
  }, [agruparPor])

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane listaTags={listaTags} tags={tags} setTags={setTags} canciones={canciones} setCanciones={setCanciones} listaCanciones={listaCanciones} cargarDatos={cargarDatos} progreso={progreso}></SidePane>
        <div className='graficas'>
          <Grafica dimensiones={dimensiones} setDimensiones={setDimensiones} agruparPor={agruparPor} setAgruparPor={setAgruparPor} visualizar={visualizar} setVisualizar={setVisualizar} embeddings={visualizar === 'embedding' ? embeddings1 : taggramData1} arquitectura={arquitectura1} setArquitectura={setArquitectura1} dataset1={dataset1} setDataset={setDataset1} layout={layout} data={visualizar === 'embedding' ? graf1 : taggrams1} tipoGrafica={tipoGrafica1} setTipoGrafica={setTipoGrafica1}></Grafica>
        </div>
      </div>
    </div>
  );
}

export default App;
