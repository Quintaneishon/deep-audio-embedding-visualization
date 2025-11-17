
export const useMakeRequest = () => {
    const URL = "http://localhost:5000";

    const getEmbeddingsTaggrams = async (red, dataset, pista) => {
        let reqRed = '';
        if (red) {
            reqRed = 'red=' + red
        }
        let reqDataset = '';
        if (dataset) {
            reqDataset = 'dataset=' + dataset;
        }
        let reqPista = ''
        if (pista) {
            reqPista = 'pista=' + pista;
        }
        let params = ''
        if (reqRed !== '' && reqDataset !== '' && reqPista !== '') {
            params = '?' + reqRed + '&' + reqDataset + '&' + reqPista
        }

        const data = await fetch(URL + '/embedding' + params);
        const eyt = await data.json()
        console.log(eyt)
        return eyt;
    }

    const obtenerAudios = async () => {
        const data = await fetch(URL + '/audios');
        const audios = await data.json()
        return audios
    }
    const obtenerTags = async () => {
        const data = await fetch(URL + '/tags');
        const tags = await data.json()
        return tags
    }

    const obtenerEmbeddings = async (red, dataset, metodo, dimensiones = '') => {
        const url = URL + `/embeddings?red=${red}&dataset=${dataset}&metodo=${metodo}&dimensiones=${dimensiones}`;
        const resp = await fetch(url);
        return await resp.json(); // {name, data}
    };

    const obtenerTaggrams = async (red, dataset, metodo, dimensiones) => {
        const url = URL + `/taggrams?red=${red}&dataset=${dataset}&metodo=${metodo}&dimensions=${dimensiones}`;
        const resp = await fetch(url);
        return await resp.json(); // {name, data}
    };

    return {
        obtenerEmbeddings,
        obtenerTaggrams,
        obtenerAudios,
        obtenerTags
    }
}