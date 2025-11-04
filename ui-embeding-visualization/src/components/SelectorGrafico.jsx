import Form from 'react-bootstrap/Form';
import '../styles/selectorgrafico.css'

export const SelectorGrafico = () => {

    return (
        <div className='selectorGrafico'>
            <div>
                <p>Arquitectura</p>
                <Form.Select aria-label="Arquitectura">
                    <option value="MusiCNN">MusiCNN</option>
                    <option value="VGG">VGG</option>
                </Form.Select>
            </div>
            <div>
                <p>Dataset</p>
                <Form.Select aria-label="Dataset">
                    <option value="MSD">MSD</option>
                    <option value="MTAT">MTAT</option>
                </Form.Select>
            </div>
        </div>
    );
}