import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';
import { useEffect, useState } from 'react';

function App() {
  const [progreso, setProgreso] = useState(0)
  // Selector SidePane
  const [tags, setTags] = useState([]);
  const [listaTags, setListaTags] = useState([]);
  const [canciones, setCanciones] = useState([]);
  const [listaCanciones, setListaCanciones] = useState([]);
  const [visualizar,setVisualizar] = useState('embedding')

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('musicnn');
  const [dataset1, setDataset1] = useState('msd');
  const [tipoGrafica1, setTipoGrafica1] = useState('umap')
  const [graf1, setGraf1] = useState([]);
  const [taggrams1, setTaggrams1] = useState([])
  const [embeddings1,setEmbeddings1] = useState([])
  const [taggramData1,setTaggramData1] = useState([])

  const layout = { autosize: true, title: { text: '' } }
  // Selector Grafica 2
  const [arquitectura2, setArquitectura2] = useState('MusiCNN');
  const [dataset2, setDataset2] = useState('MSD');
  const [tipoGrafica2, setTipoGrafica2] = useState('tsne')
  const [graf2, setGraf2] = useState([]);
  const [taggrams2, setTaggrams2] = useState([])

  const { obtenerAudios, obtenerEmbeddings, obtenerTaggrams, obtenerTags } = useMakeRequest();


  const cargarDatos = async () => {
    setProgreso(25);

    // === 1 GRAFICA ===
    const emb1 = await obtenerEmbeddings(arquitectura1, dataset1, tipoGrafica1, 2);
    const tg1 = await obtenerTaggrams(arquitectura1, dataset1, tipoGrafica1, 2);

    const coords1 = emb1.data;
    const coordsTaggrams1 = tg1.data;
    const x1 = coords1.map(p => p.coords[0]);
    const y1 = coords1.map(p => p.coords[1]);
    setEmbeddings1(coords1);
    setTaggramData1(coordsTaggrams1)
    const grupos = {};
    coords1.forEach(p => {
      if (!grupos[p.name]) grupos[p.name] = { x: [], y: [] };
      grupos[p.name].x.push(p.coords[0]);
      grupos[p.name].y.push(p.coords[1]);
    });

    const traces = Object.entries(grupos).map(([name, coords]) => ({
      x: coords.x,
      y: coords.y,
      type: "scatter",
      mode: "markers",
      marker: { size: 10 },
      name: name,
      showlegend: true
    }));

    setGraf1(traces);

    const gruposTaggrams = {};
    coordsTaggrams1.forEach(p => {
      if (!gruposTaggrams[p.name]) gruposTaggrams[p.name] = { x: [], y: [] };
      gruposTaggrams[p.name].x.push(p.coords[0]);
      gruposTaggrams[p.name].y.push(p.coords[1]);
    });

    const tracesTaggrams = Object.entries(gruposTaggrams).map(([name, coords]) => ({
      x: coords.x,
      y: coords.y,
      type: "scatter",
      mode: "markers",
      marker: { size: 10 },
      name: name,
      showlegend: true
    }));

    setTaggrams1(tracesTaggrams);
    setProgreso(50);

    // === 2 GRAFICA ===
    const emb2 = await obtenerEmbeddings(arquitectura2, dataset2, tipoGrafica2, 2);
    const tg2 = await obtenerTaggrams(arquitectura2, dataset2, tipoGrafica2, 2);

    const coords2 = emb2.data;
    const x2 = coords2.map(p => p.coords[0]);
    const y2 = coords2.map(p => p.coords[1]);

    setGraf2([
      {
        x: x2,
        y: y2,
        type: "scatter",
        mode: "markers",
        name: "Embeddings",
        taggrams: tg2.data
      }
    ]);

    setTaggrams2(tg2.data);
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

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane listaTags={listaTags} tags={tags} setTags={setTags} canciones={canciones} setCanciones={setCanciones} listaCanciones={listaCanciones} cargarDatos={cargarDatos} progreso={progreso}></SidePane>
        <div className='graficas'>
          <Grafica visualizar={visualizar} setVisualizar={setVisualizar} embeddings={visualizar === 'embedding'?embeddings1:taggramData1} arquitectura={arquitectura1} setArquitectura={setArquitectura1} dataset1={dataset1} setDataset={setDataset1} layout={layout} data={visualizar === 'embedding'? graf1: taggrams1} tipoGrafica={tipoGrafica1} setTipoGrafica={setTipoGrafica1} tagsSeleccionados={tags} taggrams={taggrams1} allTagNames={listaTags}></Grafica>
          {
            //<Grafica arquitectura={arquitectura2} setArquitectura={setArquitectura2} dataset1={dataset2} setDataset={setDataset2} layout={layout} data={graf2} tipoGrafica={tipoGrafica2} setTipoGrafica={setTipoGrafica2} tagsSeleccionados={tags} taggrams={taggrams2} allTagNames={listaTags}></Grafica>
          }
        </div>
      </div>
    </div>
  );
}

export default App;
