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
    from raggie import utils
    import random
    import PIL
    # from Ipython import display
    return PIL, glob, random, utils


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
def _(PIL, display, documents, random, utils):
    num = random.randint(0, len(documents) - 1)
    utils.pretty_print(documents[num]["text"])
    display(PIL.Image.open(documents[num]["image"]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
