import marimo

__generated_with = "0.15.1"
app = marimo.App(width="full", layout_file="layouts/dummy.slides.json")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import base64
    import ollama
    from IPython.display import Image
    from unstructured.partition.pdf import partition_pdf
    import unstructured
    from IPython.display import Markdown, display
    from tqdm import tqdm
    from llama_index.embeddings.ollama import OllamaEmbedding
    return Markdown, base64, display, ollama, partition_pdf, tqdm, unstructured


@app.cell
def _(os):
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return


@app.cell
def _(mo):
    mo.md(r"""### PDF Parsing""")
    return


@app.cell
def _(partition_pdf):
    file_path = "docs/attention.pdf"

    chunks = partition_pdf(
        filename=file_path,
        # enable detection and extraction of tables as structured elements
        infer_table_structure=True,
        # NOTE: hi_res strategy uses detectron2_onnx, for oo, seg, layout analysis in documentation or images. The use is extract meaningful elements like text, tables and images.
        strategy="hi_res",
        # Specifying the types of blocks to be treated as images
        extract_image_block_types=["Image"],
        # Conversion of images to a base64 encoded format
        extract_image_block_to_payload=True,
        # Control how the document is split into chunks for processing. Splitting by titles ensures logical segmentation of the document into coherent sections. Highly useful for research document
        chunking_strategy="by_title",
    )
    return (chunks,)


@app.cell
def _(chunks):
    print(len(chunks))
    return


@app.cell
def _(Markdown, chunks, display):
    display(Markdown(str(chunks[1].text)))
    return


@app.cell
def _(chunks):
    ### We can find the which chunk has what type of elements
    print(chunks[2].metadata.orig_elements)
    return


@app.cell
def _(mo):
    mo.md(r"""## Seperating images, texts & tables""")
    return


@app.cell
def _(chunks, unstructured):
    texts, tables, images = [], [], []

    for chunk in chunks:
        if isinstance(chunk, unstructured.documents.elements.Table):
            tables.append(chunk)
        if isinstance(chunk, unstructured.documents.elements.CompositeElement):
            texts.append(chunk)
            chunk_elements = chunk.metadata.orig_elements
            # iterate over all elements of this chunk
            for element in chunk_elements:
                if isinstance(element, unstructured.documents.elements.Image):
                    images.append(element.metadata.image_base64)
    return images, tables, texts


@app.cell
def _(images, tables, texts):
    print("Total Texts:", len(texts))
    print("Total Images:", len(images))
    print("Total Tables:", len(tables))
    return


@app.cell
def _(base64, images, mo):
    image_data_point = base64.b64decode(images[0])
    mo.image(image_data_point)
    return


@app.cell
def _():
    import os

    os.makedirs("docs/images", exist_ok=True)
    return (os,)


@app.cell
def _(base64, images):
    for idx, image in enumerate(images):
        image_data = base64.b64decode(image)
        path = (
            f"docs/images/attention_paper_image_{idx}.jpeg"  # consistent filename
        )
        with open(path, "wb") as f:
            f.write(image_data)
        print(f"Saved: {path}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Summarizing images, texts, and tables
    - We will use gemma3:12b-it-qat for generating summaries of images
    - We will use gemma3n:e2b for summarizing text chunks and tables
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Summarizing images""")
    return


@app.cell
def _(base64, ollama):
    def get_image_summary(filepath):
        # read the file back into base64
        with open(filepath, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        response = ollama.chat(
            model="gemma3:12b-it-qat",
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the image:",
                    "images": [img_b64],  # send base64 instead of file path
                }
            ],
        )
        return response["message"]["content"]
    return (get_image_summary,)


@app.cell
def _(get_image_summary, images, tqdm):
    image_summaries = [
        get_image_summary(f"docs/images/attention_paper_image_{i}.jpeg")
        for i in tqdm(range(len(images)))
    ]
    return (image_summaries,)


@app.cell
def _(image_summaries):
    image_summaries
    return


@app.cell
def _(Markdown, display, image_summaries):
    display(Markdown(image_summaries[0]))
    return


@app.cell
def _(mo):
    mo.md(r"""#### Summarizing texts""")
    return


@app.cell
def _(ollama):
    def get_text_summary(text):
        response = ollama.chat(
            model="gemma3n:e2b",
            messages=[{"role": "user", "content": f"Summarize this text {text}"}],
        )
        return response.message.content
    return (get_text_summary,)


@app.cell
def _(get_text_summary, texts, tqdm):
    # In the above code, the texts list we are iterating over is a list of CompositeElement objects. To get the actual text, we use the .text attribute of that object and pass that to the get_text_summary() function.
    text_summaries = [
        get_text_summary(texts[i].text) for i in tqdm(range(len(texts)))
    ]
    return (text_summaries,)


@app.cell
def _(Markdown, display, text_summaries):
    display(Markdown(text_summaries[50]))
    return


@app.cell
def _(mo):
    mo.md(r"""#### Summarizing tables""")
    return


@app.cell
def _(ollama):
    def get_table_summary(table_html):
        response = ollama.chat(
            model="gemma3n:e2b",
            messages=[
                {"role": "user", "content": f"Summarize this table: {table_html}"}
            ],
        )
        return response.message.content
    return (get_table_summary,)


@app.cell
def _(get_table_summary, tables, tqdm):
    # In the above code, the tables list we are iterating over is a list of Table objects. To get the actual HTML, we use the .metadata.text_as_html attribute of that object and pass that to the get_table_summary() function.
    table_summaries = [
        get_table_summary(tables[i].metadata.text_as_html)
        for i in tqdm(range(len(tables)))
    ]
    return (table_summaries,)


@app.cell
def _(table_summaries):
    table_summaries
    return


@app.cell
def _(mo):
    mo.md(r"""### Embedding the summaries""")
    return


@app.cell
def _():
    from raggie import embedder, retriever, raggy, vecdb, utils
    return embedder, raggy, retriever, vecdb


@app.cell
def _(embedder, image_summaries, table_summaries, text_summaries):
    batch_size = 32
    embeddata = embedder.EmbedData(batch_size=batch_size)
    embeddata.embed(text_summaries + image_summaries + table_summaries)
    return (embeddata,)


@app.cell
def _(embeddata):
    embeddata.embeddings
    return


@app.cell
def _(embeddata, vecdb):
    database = vecdb.QdrantVDB(
        collection_name="dummy_attention", vector_dim=1024, batch_size=512
    )
    database.define_client()
    database.create_collection()
    database.ingest_data(embeddata)
    return (database,)


@app.cell
def _(database, embeddata, raggy, retriever):
    retriver = retriever.Retriever(database, embeddata)
    rag = raggy.RAG(retriver, llm_name="llama3.2-vision:latest")
    return (rag,)


@app.cell
def _():
    ##### Prompt text context
    return


@app.cell
def _(rag):
    text_query = (
        """What are the types of model architectures proposed in the paper?"""
    )
    text_answer = rag.query(text_query)
    return (text_answer,)


@app.cell
def _(text_answer):
    text_answer
    return


@app.cell
def _():
    ##### Prompt image context
    return


@app.cell
def _(image_summaries):
    image_summaries[0]
    return


@app.cell
def _(rag):
    image_query = """What is depicted in the transformer architecture's overall structure?"""

    image_answer = rag.query(image_query)
    return (image_answer,)


@app.cell
def _(image_answer):
    image_answer
    return


@app.cell
def _():
    ##### Prompt table context
    return


@app.cell
def _(table_summaries):
    table_summaries[0]
    return


@app.cell
def _(rag):
    table_query = """Which models achieve high BLEU scores?"""

    table_answer = rag.query(table_query)
    return (table_answer,)


@app.cell
def _(table_answer):
    table_answer
    return


if __name__ == "__main__":
    app.run()
