import { useState } from 'react';
import Form from 'react-bootstrap/Form';
import '../styles/selectorgrafico.css';

export const SelectorGrafico = ({ arquitectura, setArquitectura, dataset, setDataset }) => {

    const handleArquitecturaChange = (e) => {
        setArquitectura(e.target.value);
        if (e.target.value == 'VGG') {
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
                    <option value="MusiCNN">MusiCNN</option>
                    <option value="VGG">VGG</option>
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
                    {arquitectura == 'MusiCNN' && <option value="MTAT">MTAT</option>}
                </Form.Select>
            </div>
        </div>
    );
};
