import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';

import { useEffect, useState } from 'react';

const ddata = [
                        {

                            x: [1, 2, 3],

                            y: [2, 6, 3],

                            type: 'scatter',

                            mode: 'lines+markers',

                            marker: { color: 'red' },

                        },

                        { type: 'bar', x: [1, 2, 3], y: [2, 5, 3] },

                    ]

function App() {

  // Selector SidePane
  const [tags, setTags] = useState([]);
  const [canciones, setCanciones] = useState([]);
  const [listaCanciones, setListaCanciones] = useState([]);

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('MusiCNN');
  const [dataset1, setDataset1] = useState('MSD');
  const [graf1,setGraf1]= useState(ddata);

  const layout = { width: 500, height: 500, title: { text: '' }}
  // Selector Grafica 2
  const [arquitectura2, setArquitectura2] = useState('MusiCNN');
  const [dataset2, setDataset2] = useState('MSD');
  const [graf2,setGraf2]= useState(ddata);

  const { getEmbeddingsTaggrams,obtenerAudios } = useMakeRequest();

  const [embeddingsyTaggrams, setEmbeddingsyTaggrams] = useState({});

  const cargarDatos = ()=>{

  }

  useEffect(() => {
    obtenerAudios().then(data=>{
      setListaCanciones(data)
    })
  }, [setEmbeddingsyTaggrams])

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane tags={tags} setTags={setTags} canciones={canciones} setCanciones={setCanciones} listaCanciones={listaCanciones} cargarDatos={cargarDatos}></SidePane>
        <div className='graficas'>
          <Grafica arquitectura={arquitectura1} setArquitectura={setArquitectura1} dataset1={dataset1} setDataset={setDataset1}layout={layout} data={graf1}></Grafica>
          <Grafica arquitectura={arquitectura2} setArquitectura={setArquitectura2} dataset1={dataset2} setDataset={setDataset2}layout={layout} data={graf2}></Grafica>
        </div>
      </div>
    </div>
  );
}

export default App;
