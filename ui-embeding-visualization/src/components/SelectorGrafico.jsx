import Form from 'react-bootstrap/Form';
import '../styles/selectorgrafico.css';

export const SelectorGrafico = ({ arquitectura, setArquitectura, dataset, setDataset,tipoGrafica,setTipoGrafica, visualizar,setVisualizar, agruparPor,setAgruparPor, dimensiones,setDimensiones }) => {

    const handleArquitecturaChange = (e) => {
        setArquitectura(e.target.value);
        if (e.target.value === 'whisper' || e.target.value === 'whisper_contrastive') {
            setDataset('base')
        } else if (e.target.value === 'vggish') {
            setDataset('pretrained')
        } else if (e.target.value === 'mert') {
            setDataset('95m')
        }else {
            setDataset('mtat')
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
                    <option value="whisper">Whisper</option>
                    <option value="vggish">VGGish</option>
                    <option value="mert">MERT</option>
                    <option value="whisper_contrastive">Whisper Contrastive</option>
                </Form.Select>
            </div>

            <div>
                <p>Dataset</p>
                <Form.Select
                    aria-label="Dataset"
                    value={dataset}
                    onChange={handleDatasetChange}
                >
                    {(arquitectura === 'musicnn' || arquitectura === 'vgg') && <option value="mtat">MTAT</option>}
                    {(arquitectura === 'musicnn' || arquitectura === 'vgg') && <option value="msd">MSD</option>}
                    {(arquitectura === 'whisper' || arquitectura === 'whisper_contrastive') && <option value="base">Base</option>}
                    {(arquitectura === 'vggish') && <option value="pretrained">Pretrained</option>}
                    {(arquitectura === 'mert') && <option value="95m">95M</option>}
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
            <div>
                <p>Agrupar por</p>
                <Form.Select
                    aria-label="Agrupar por"
                    value={agruparPor}
                    onChange={(e)=>{
                        setAgruparPor(e.target.value)
                    }}
                >
                    <option value="tag">Tag</option>
                    <option value="name">Cancion</option>
                </Form.Select>
            </div>
            <div>
                <p>Dimension</p>
                <Form.Select
                    aria-label="Dimension"
                    value={String(dimensiones)}
                    onChange={(e)=>{
                        setDimensiones(parseInt(e.target.value))
                    }}
                >
                    <option value="2">2D</option>
                    <option value="3">3D</option>
                </Form.Select>
            </div>
        </div>
    );
};
