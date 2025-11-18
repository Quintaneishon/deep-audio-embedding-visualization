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
  const [tipoGrafica1, setTipoGrafica1] = useState('PCA')
  const [graf1, setGraf1] = useState([]);
  const [taggrams1, setTaggrams1] = useState([])

  const layout = { width: 500, height: 500, title: { text: '' } }
  // Selector Grafica 2
  const [arquitectura2, setArquitectura2] = useState('MusiCNN');
  const [dataset2, setDataset2] = useState('MSD');
  const [tipoGrafica2, setTipoGrafica2] = useState('PCA')
  const [graf2, setGraf2] = useState([]);
  const [taggrams2, setTaggrams2] = useState([])

  const { obtenerAudios, obtenerRepresentacion, obtenerTags } = useMakeRequest();


  const cargarDatos = async () => {
    console.log("Ejecutando request...", arquitectura1, dataset1, tipoGrafica1);

    // --- PRIMERA GRÁFICA ---
    let primeraGrafica = [];
    let todosTaggrams1 = [];
setProgreso(25)
    for (let can of canciones) {
      console.log("Obteniendo para", can);
      const resp = await obtenerRepresentacion(arquitectura1, dataset1, can, tipoGrafica1);

      const keys = Object.keys(resp);
      const keyEmb = keys.find(k => k.includes("representacion"));
      const keyTags = keys.find(k => k.includes("taggrams"));

      const coords = resp[keyEmb];
      const tg = resp[keyTags];

      todosTaggrams1 = todosTaggrams1.concat(tg);

      const x = coords.map(p => p[0]);
      const y = coords.map(p => p[1]);

      primeraGrafica.push({
        x,
        y,
        type: "scatter",
        mode: "markers",
        name: can,
        taggrams: tg,
      });
    }

    setGraf1(primeraGrafica);
    setTaggrams1(todosTaggrams1);
    setProgreso(50)

    // --- SEGUNDA GRÁFICA ---
    console.log("Ejecutando request...", arquitectura2, dataset2, tipoGrafica2);

    let segundaGrafica = [];
    let todosTaggrams2 = [];

    for (let can of canciones) {
      console.log("Obteniendo para", can);
      const resp = await obtenerRepresentacion(arquitectura2, dataset2, can, tipoGrafica2);

      const keys = Object.keys(resp);
      const keyEmb = keys.find(k => k.includes("representacion"));
      const keyTags = keys.find(k => k.includes("taggrams"));

      const coords = resp[keyEmb];
      const tg = resp[keyTags];
      todosTaggrams2 = todosTaggrams2.concat(tg);

      const x = coords.map(p => p[0]);
      const y = coords.map(p => p[1]);

      segundaGrafica.push({
        x,
        y,
        type: "scatter",
        mode: "markers",
        name: can,
        taggrams: tg,
      });
    }

    setGraf2(segundaGrafica);
    setTaggrams2(todosTaggrams2);
    setProgreso(100)
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
