import Form from 'react-bootstrap/Form';
import '../styles/selectorgrafico.css';

export const SelectorGrafico = ({ arquitectura, setArquitectura, dataset, setDataset,tipoGrafica,setTipoGrafica, visualizar,setVisualizar }) => {

    const handleArquitecturaChange = (e) => {
        setArquitectura(e.target.value);
        if (e.target.value === 'VGG') {
            setDataset('MSD')
        }
        console.log('Arquitectura seleccionada:', e.target.value);
    };

    const handleDatasetChange = (e) => {
        setDataset(e.target.value);
        console.log('Dataset seleccionado:', e.target.value);
    };

    return (
        <div className='selectorGrafico'>
            <div>
                <p>Arquitectura</p>
                <Form.Select
                    aria-label="Arquitectura"
                    value={arquitectura}
                    onChange={handleArquitecturaChange}
                >
                    <option value="musicnn">MusiCNN</option>
                    <option value="vgg">VGG</option>
                </Form.Select>
            </div>

            <div>
                <p>Dataset</p>
                <Form.Select
                    aria-label="Dataset"
                    value={dataset}
                    onChange={handleDatasetChange}
                >
                    <option value="MSD">MSD</option>
                    {arquitectura === 'musicnn' && <option value="mtat">MTAT</option>}
                </Form.Select>
            </div>
            <div>
                <p>Tipo Grafica</p>
                <Form.Select
                    aria-label="Grafica"
                    value={tipoGrafica}
                    onChange={(e)=>{
                        console.log('Tipo Grafica',e.target.value)
                        setTipoGrafica(e.target.value)
                    }}
                >
                    {/* <option value="pca">PCA</option>
                    <option value="std-pca">STD-PCA</option> */}
                    <option value="tsne">t-SNE</option>
                    <option value="umap">UMAP</option>
                </Form.Select>
            </div>
             <div>
                <p>Visualizar</p>
                <Form.Select
                    aria-label="Visualizar"
                    value={visualizar}
                    onChange={(e)=>{
                        setVisualizar(e.target.value)
                    }}
                >
                    <option value="embedding">Embeddings</option>
                    <option value="taggram">Taggrams</option>
                </Form.Select>
            </div>
        </div>
    );
};
