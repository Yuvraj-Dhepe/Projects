import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### MultiModal RAG
    - For building a multimodal RAG system following pieces are required:
       - Gather the dataset for text-image pairs.
       - Generate embeddings: Create embeddings for each modality using CLIP for images and transformers for text.
       - Store embeddings in a vector database: Use Qdrant to store these multimodal embeddings.
       - Retrieve relevant data: Query the vector database to retrieve the most relevant information based on a multimodal input.
       - Generate responses: Combine retrieved data to genrate coherent and context-aware responses using an LLM.

    - Basically the query will be text, we will retrieve relevant text for this query by using vector embeddings, along with the relevant text we will also retrieve the relevant multi-modal data, and use modal's image understanding to generate a relevant response for the user query by making use of both the image data and the retrieved text.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The goal is to build a multimodal RAG system that can: 
    - Embed both the images and their corresponding text using CLIP. 
    - Store these embeddings in a vector database (Qdrant) for efficient retrieval. 
    - Query the system using either text, image, or a combination of both. 
    - Generate contex-aware responses using an LLM that integrates the retrieved multi-modal information.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Library Imports""")
    return


@app.cell
def _():
    from glob import glob
    import random
    from PIL import Image
    from IPython.display import display
    from fastembed import TextEmbedding, ImageEmbedding 
    from raggie import vecdb, retriever, utils,raggy
    import numpy as np 
    return (
        Image,
        ImageEmbedding,
        TextEmbedding,
        display,
        glob,
        np,
        random,
        utils,
    )


@app.cell
def _(mo):
    mo.md(r"""### Dataset Preparation""")
    return


@app.cell
def _(glob):
    image_files = glob("./data/social_posts_dailyds/*.jpg")
    text_files = glob("./data/social_posts_dailyds/*.txt")
    return image_files, text_files


@app.cell
def _(image_files, text_files):
    len(image_files), len(text_files)
    return


@app.cell
def _(image_files):
    documents = []

    for i in range(1, len(image_files) + 1):
        text_file = f"./data/social_posts_dailyds/{i}.txt"
        image_file = f"./data/social_posts_dailyds/{i}.jpg"

        text = open(text_file).read()
        doc = {"text": text, "image": image_file}
        documents.append(doc)
    return (documents,)


@app.cell
def _(documents):
    documents[1]
    return


@app.cell
def _(Image, display, documents, random, utils):
    num = random.randint(0, len(documents) - 1)
    utils.pretty_print(documents[num]["text"])
    display(Image.open(documents[num]["image"]))
    return


@app.cell
def _(mo):
    mo.md(r"""### Embed Dataset""")
    return


@app.cell
def _(ImageEmbedding, TextEmbedding):
    class EmbedData:
        def __init__(self, 
                    documents, 
                    text_model_name = 'Qdrant/clip-ViT-B-32-text',
                    image_model_name = "Qdrant/clip-ViT-B-32-vision"):
            # Initialize text embedding model 
            self.documents = documents 
            self.text_model = TextEmbedding(model_name = text_model_name)
            self.text_embed_dim = self.text_model._get_model_description(text_model_name).dim

            # Initialize image embedding model
            self.image_model = ImageEmbedding(model_name = image_model_name)
            self.image_embed_dim = self.image_model._get_model_description(image_model_name).dim

        def embed_texts(self, texts):
            text_embeddings = list(self.text_model.embed(texts))
            return text_embeddings

        def embed_images(self, images):
            image_embeddings = list(self.image_model.embed(images))
            return image_embeddings
    return (EmbedData,)


@app.cell
def _(EmbedData, documents):
    embeddata = EmbedData(documents)
    embeddata.text_embeds = embeddata.embed_texts([doc['text'] for doc in documents])
    embeddata.image_embeds = embeddata.embed_images([doc['image'] for doc in documents])
    return (embeddata,)


@app.cell
def _(embeddata, np):
    np.array(embeddata.text_embeds)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - Change the vecdb `create_collection` implementation so that it can now embed the images and text both.
    - Add upload embeddings function to vecdb, that will take in the embeddata object. 
    - Need to change the retriever implementation & raggy implementation to accomodate the above changes too. 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary 
    To build a rag system, the following components are always required:
    - Embedding Component: The one that will play with the embedding model and yield data embeddings.
    - VecDB Component: The one that will use the embedding instance to generate and store the vector embeddings in as efficient manner as possible. 
    - Retriever Component: This component will use the vec db component to retrieve the embeddings based on the query embedding. 
    - RAG unifier component: This component brings together the above components to build the whole system and integrate the LLM (Multimodal/Modal) to generate responses for user queries, for the context retrieved by the mentioned components.
    """
    )
    return


if __name__ == "__main__":
    app.run()
