import Form from 'react-bootstrap/Form';
import '../styles/selectorgrafico.css';

export const SelectorGrafico = ({ arquitectura, setArquitectura, dataset, setDataset, vector, setVector, tipoGrafica, setTipoGrafica, dimensiones, setDimensiones }) => {
    return (
        <div className='selectorGrafico'>
            <div>
                <p>Arquitectura</p>
                <Form.Select
                    aria-label="Arquitectura"
                    value={arquitectura}
                    onChange={(e) => {
                        console.log('Arquitectura', e.target.value)
                        setArquitectura(e.target.value)
                    }}
                >
                    <option value="musicnn">MusiCNN</option>
                    <option value="vgg">VGG</option>
                    <option value="whisper">Whisper</option>
                </Form.Select>
            </div>

            <div>
                <p>Dataset</p>
                <Form.Select
                    aria-label="Dataset"
                    value={dataset}
                    onChange={(e) => {
                        console.log('Dataset', e.target.value)
                        setDataset(e.target.value)
                    }}
                >
                    <option value="msd">MSD</option>
                    <option value="mtat">MTAT</option>
                    <option value="base">Base</option>
                    <option value="small">Small</option>
                </Form.Select>
            </div>
            <div>
                <p>Vector</p>
                <Form.Select
                    aria-label="Vector"
                    value={vector}
                    onChange={(e) => {
                        console.log('Vector', e.target.value)
                        setVector(e.target.value)
                    }}
                >
                    <option value="embeddings">Embeddings</option>
                    <option value="taggrams">Taggrams</option>
                </Form.Select>
            </div>
            <div>
                <p>Tipo Grafica</p>
                <Form.Select
                    aria-label="Grafica"
                    value={tipoGrafica}
                    onChange={(e) => {
                        console.log('Tipo Grafica', e.target.value)
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
                <p>Dimensiones</p>
                <Form.Select
                    aria-label="Dimensiones"
                    value={dimensiones}
                    onChange={(e) => {
                        console.log('Dimensiones', e.target.value)
                        setDimensiones(Number(e.target.value))
                    }}
                >
                    <option value="2">2</option>
                    <option value="3">3</option>
                </Form.Select>
            </div>
        </div>
    );
};
