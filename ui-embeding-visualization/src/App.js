import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import { Grafica } from './components/Grafica';
import { SidePane } from './components/SidePane';
import { useMakeRequest } from './hooks/useMakeRequest';
import { useEffect, useState } from 'react';
import { extractGenresFromData, generateColorPalette, PREDEFINED_COLORS } from './utils/colorGenerator';

function App() {
  const [progreso, setProgreso] = useState(0)
  // Selector SidePane
  const [tags, setTags] = useState([]);
  const [listaTags, setListaTags] = useState([]);

  // Selector Grafica 1
  const [arquitectura1, setArquitectura1] = useState('musicnn');
  const [dataset1, setDataset1] = useState('msd');
  const [tipoGrafica1, setTipoGrafica1] = useState('umap')
  const [vector1, setVector1] = useState('embeddings');
  const [graf1, setGraf1] = useState([]);
  const [taggrams1, setTaggrams1] = useState([])
  const [dimensiones1, setDimensiones1] = useState(2);
  
  // Dynamic color mapping for genres
  const [genreColors, setGenreColors] = useState(PREDEFINED_COLORS);

  const layout = { width: 550, height: 550, title: { text: '' } }

  const { obtenerEmbeddings, obtenerTaggrams, obtenerTags } = useMakeRequest();


  const cargarDatos = async () => {
    setProgreso(25);

    console.log('Loading data with params:', {
      arquitectura: arquitectura1,
      dataset: dataset1,
      vector: vector1,
      tipoGrafica: tipoGrafica1,
      dimensiones: dimensiones1
    });

    // === 1 GRAFICA ===
    let response = {};
    if (vector1 === 'embeddings') {
      response = await obtenerEmbeddings(arquitectura1, dataset1, tipoGrafica1, dimensiones1);
    } else {
      response = await obtenerTaggrams(arquitectura1, dataset1, tipoGrafica1, dimensiones1);
    }

    const data = response.data || [];

    // Extract genres and generate dynamic color palette
    const genres = extractGenresFromData(data);
    if (genres.length > 0) {
      console.log('Detected genres:', genres);
      
      // Generate colors for genres not in predefined list
      const newColors = { ...PREDEFINED_COLORS };
      const predefinedGenres = Object.keys(PREDEFINED_COLORS);
      const unknownGenres = genres.filter(g => !predefinedGenres.includes(g));
      
      if (unknownGenres.length > 0) {
        const generatedColors = generateColorPalette(unknownGenres);
        Object.assign(newColors, generatedColors);
        console.log('Generated colors for new genres:', unknownGenres);
      }
      
      setGenreColors(newColors);
    }

    setGraf1([
      {
        x: data.map(p => p.coords[0]),
        y: data.map(p => p.coords[1]),
        type: "scatter",
        mode: "markers",
        name: "Embeddings",
        taggrams: data
      }
    ]);

    setTaggrams1(data);
    setProgreso(100);
  };

  useEffect(() => {
    obtenerTags().then(data => {
      setListaTags(data);
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className='main'>
      <div className='mainsection'>
        <SidePane listaTags={listaTags} tags={tags} setTags={setTags} cargarDatos={cargarDatos} progreso={progreso}></SidePane>
        <div className='graficas'>
          <Grafica 
            arquitectura={arquitectura1} 
            setArquitectura={setArquitectura1} 
            dataset={dataset1} 
            setDataset={setDataset1} 
            vector={vector1} 
            setVector={setVector1} 
            layout={layout} 
            data={graf1} 
            tipoGrafica={tipoGrafica1} 
            setTipoGrafica={setTipoGrafica1} 
            tagsSeleccionados={tags} 
            taggrams={taggrams1} 
            allTagNames={listaTags} 
            dimensiones={dimensiones1} 
            setDimensiones={setDimensiones1}
            genreColors={genreColors}
          />
          {
          //<Grafica arquitectura={arquitectura2} setArquitectura={setArquitectura2} dataset1={dataset2} setDataset={setDataset2} layout={layout} data={graf2} tipoGrafica={tipoGrafica2} setTipoGrafica={setTipoGrafica2} tagsSeleccionados={tags} taggrams={taggrams2} allTagNames={listaTags}></Grafica>
          }
        </div>
      </div>
    </div>
  );
}

export default App;
