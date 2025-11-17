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

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('MusiCNN');
  const [dataset1, setDataset1] = useState('MSD');
  const [tipoGrafica1, setTipoGrafica1] = useState('umap')
  const [graf1, setGraf1] = useState([]);
  const [taggrams1, setTaggrams1] = useState([])

  const layout = { width: 500, height: 500, title: { text: '' } }
  // Selector Grafica 2
  const [arquitectura2, setArquitectura2] = useState('MusiCNN');
  const [dataset2, setDataset2] = useState('MSD');
  const [tipoGrafica2, setTipoGrafica2] = useState('tsne')
  const [graf2, setGraf2] = useState([]);
  const [taggrams2, setTaggrams2] = useState([])

  const { obtenerAudios, obtenerEmbeddings, obtenerTaggrams, obtenerTags } = useMakeRequest();


  const cargarDatos = async () => {
    console.log("Ejecutando request...", arquitectura1, dataset1, tipoGrafica1);
    setProgreso(25);

    // ======== PRIMERA GRÁFICA ========
    const emb1 = await obtenerEmbeddings(arquitectura1, dataset1, tipoGrafica1, 2);
    const tg1 = await obtenerTaggrams(arquitectura1, dataset1, tipoGrafica1, 2);

    const coords1 = emb1.data;      // [[x,y], [x,y], ...]
    const tags1 = tg1.data;         // [[t1..tn], [t1..tn], ...]

    const x1 = coords1.map(p => p[0]);
    const y1 = coords1.map(p => p[1]);

    setGraf1([
      {
        x: x1,
        y: y1,
        type: "scatter",
        mode: "markers",
        name: "Embeddings",
        taggrams: tags1
      }
    ]);

    setTaggrams1(tags1);
    setProgreso(50);

    // ======== SEGUNDA GRÁFICA ========
    console.log("Ejecutando request...", arquitectura2, dataset2, tipoGrafica2);

    const emb2 = await obtenerEmbeddings(arquitectura2, dataset2, tipoGrafica2, 2);
    const tg2 = await obtenerTaggrams(arquitectura2, dataset2, tipoGrafica2, 2);

    const coords2 = emb2.data;
    const tags2 = tg2.data;

    const x2 = coords2.map(p => p[0]);
    const y2 = coords2.map(p => p[1]);

    setGraf2([
      {
        x: x2,
        y: y2,
        type: "scatter",
        mode: "markers",
        name: "Embeddings",
        taggrams: tags2
      }
    ]);

    setTaggrams2(tags2);
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
          <Grafica izq arquitectura={arquitectura1} setArquitectura={setArquitectura1} dataset1={dataset1} setDataset={setDataset1} layout={layout} data={graf1} tipoGrafica={tipoGrafica1} setTipoGrafica={setTipoGrafica1} tagsSeleccionados={tags} taggrams={taggrams1} allTagNames={listaTags}></Grafica>
          <Grafica arquitectura={arquitectura2} setArquitectura={setArquitectura2} dataset1={dataset2} setDataset={setDataset2} layout={layout} data={graf2} tipoGrafica={tipoGrafica2} setTipoGrafica={setTipoGrafica2} tagsSeleccionados={tags} taggrams={taggrams2} allTagNames={listaTags}></Grafica>
        </div>
      </div>
    </div>
  );
}

export default App;
