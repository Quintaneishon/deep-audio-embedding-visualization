import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';

import { useEffect,useState } from 'react';

function App() {
  const {getEmbeddingsTaggrams} = useMakeRequest();

  const [embeddingsyTaggrams,setEmbeddingsyTaggrams] = useState({});

  useEffect(()=>{
    getEmbeddingsTaggrams().then(response =>{
      console.log(response)
      setEmbeddingsyTaggrams({...embeddingsyTaggrams,...response})
    })
  },[setEmbeddingsyTaggrams])

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane></SidePane>
        <div className='graficas'>
          <Grafica></Grafica>
          <Grafica></Grafica>
        </div>
      </div>
    </div>
  );
}

export default App;
