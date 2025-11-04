import Form from 'react-bootstrap/Form';
import '../styles/sidepane.css'

export const SidePane = () => {

    return (
        <div className='sidepane'>
            <div className='item'>
                <p>Tags</p>
                <Form.Select aria-label="Tags">
                    <option value="MusiCNN">MusiCNN</option>
                    <option value="VGG">VGG</option>
                </Form.Select>
            </div>
            <div className='item'>
                <p>Cancion</p>
                <Form.Select aria-label="Cancion">
                    <option value="MSD">MSD</option>
                    <option value="MTAT">MTAT</option>
                </Form.Select>
            </div>
        </div>
    );
}