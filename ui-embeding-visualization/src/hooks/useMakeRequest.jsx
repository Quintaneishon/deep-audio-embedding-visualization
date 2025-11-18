export const useMakeRequest = () => {
    const URL = "http://localhost:5000";

    const obtenerAudios = async () => {
        const resp = await fetch(URL + '/audios');
        return await resp.json();
    };

    const obtenerTags = async () => {
        const resp = await fetch(URL + '/tags');
        return await resp.json();
    };

    const obtenerEmbeddings = async (red, dataset, metodo, dimensions = 2) => {
        const url = `${URL}/embeddings?red=${red.toLowerCase()}&dataset=${dataset.toLowerCase()}&metodo=${metodo.toLowerCase()}&dimensions=${dimensions}`;
        const resp = await fetch(url);
        return await resp.json(); // { data: [...] }
    };

    const obtenerTaggrams = async (red, dataset, metodo, dimensions = 2) => {
        const url = `${URL}/taggrams?red=${red.toLowerCase()}&dataset=${dataset.toLowerCase()}&metodo=${metodo.toLowerCase()}&dimensions=${dimensions}`;
        const resp = await fetch(url);
        return await resp.json(); // { data: [...] }
    };

    return {
        obtenerEmbeddings,
        obtenerTaggrams,
        obtenerAudios,
        obtenerTags
    };
};
