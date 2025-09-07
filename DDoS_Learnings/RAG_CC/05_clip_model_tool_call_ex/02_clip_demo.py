import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### General Steps for CLIP Model Training: 
    - Preparing the Text-Image Pair Dataset
    - Passing inputs through the encoders 
    - Contrastive learning with cross-modal similarity with the goals:
        - To maximize the similarity between embeddings of matched (text, image) pairs
        - To minimize the similarity between embeddings of mismatched (text, image) pairs
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Library Imports""")
    return


@app.cell
def _():
    from datasets import load_dataset
    return (load_dataset,)


@app.cell
def _(mo):
    mo.md(r"""### Loading the Dataset""")
    return


@app.cell
def _(load_dataset):
    data = load_dataset(
        "jamescalam/image-text-demo", split="train", cache_dir="./data"
    )
    return (data,)


@app.cell
def _(data):
    data[3]["text"]
    return


@app.cell
def _(data):
    data[3]["image"]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Model Loading
    - For HF, model loading is always, load the components of the models like pre-processors, tokenizers etc and then load the model
    """
    )
    return


@app.cell
def _():
    from transformers import CLIPProcessor, CLIPModel
    import torch

    model_id = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_id, cache_dir="./data")
    model = CLIPModel.from_pretrained(model_id, cache_dir="./data")

    # move model to device if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return device, model, processor, torch


@app.cell
def _(data, device, processor):
    text = data["text"]
    images = data["image"]

    inputs = processor(
        text=text,  # Text descriptions, which will be tokenized and converted to a format expected by the text encoder.
        images=images,  # Specifies the batch of images, which will be padded/ resized to meet the image encoder'input requirements.
        return_tensors="pt",  # Ensures that the outputs are returned as Pytorch tensors compatible with the CLIP model.
        padding=True,  # Add padding to text sequences to ensur they have consistent lengths within the batch
    ).to(device)
    return (inputs,)


@app.cell
def _(inputs):
    dc = dict(inputs)
    return (dc,)


@app.cell
def _(dc):
    dc.keys()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - The resulting inputs object contains the preprocessed data for both the text and image encoders. 
    - It consists of:
        - input_ids: Tokenized representation of the text_inputs 
        - attention_Mask: Places where the actual text tokens lie, 0 representing the padding tokens
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Model Preds""")
    return


@app.cell
def _(inputs, model):
    outputs = model(**inputs)
    return (outputs,)


@app.cell
def _(outputs):
    outputs.keys()
    return


@app.cell
def _(outputs):
    txt_emb = outputs.text_embeds
    img_emb = outputs.image_embeds
    return img_emb, txt_emb


@app.cell
def _(img_emb, torch, txt_emb):
    text_emb = txt_emb / torch.norm(txt_emb, dim=1, keepdim=True)

    image_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)

    cos_sim = torch.mm(text_emb, image_emb.T).cpu().detach().numpy()
    return cos_sim, image_emb


@app.cell
def _(cos_sim):
    import matplotlib.pyplot as plt

    plt.imshow(cos_sim)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""- The plot depicts that the embeddings along the diagonal have the highest similarity, so the text relevant to a particular caption, has highest similarity""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Text to Image search
    - Get a text prompt, embed it via CLIP
    - Match the text embedding across all the image embeddings in vector db & fetch the top results
    """
    )
    return


@app.cell
def _(inputs):
    valid_ips = {
        key: inputs[key]
        for key in inputs.keys()
        if key in ["input_ids", "attention_mask"]
    }
    return


@app.cell
def _(data, device, image_emb, model, processor, torch):
    query_text = "Dog running on grass"
    # preprocess text (tokenize, etc.)
    q_txt = processor(text=[query_text], return_tensors="pt", padding=True).to(
        device
    )

    # generate & normalize text embeddings
    # NOTE: Normalizing is necessary, for cosine similarity calculation, which finds how aligned they are wrt angle, and not the magnitude of the vectors
    q_txt_features = model.get_text_features(**q_txt) / model.get_text_features(
        **q_txt
    ).norm(dim=-1, keepdim=True)

    # Clculate similarity scores across all image embeddings
    q_txt_similarity = torch.mm(q_txt_features, image_emb.T)

    # Get top-k matches
    top_k = 5
    txt_values, txt_indices = q_txt_similarity[0].topk(min(top_k, len(data)))
    return top_k, txt_indices, txt_values


@app.cell
def _(data, txt_indices, txt_values):
    def visualize_top_k(indices, values, data, top_k=5):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, top_k, figsize=(15, 3))
        for i, (idx, score) in enumerate(zip(indices, values)):
            # Print text and score
            print(f"{data['text'][idx]}: {score:.3f}")

            # Display image
            axes[i].imshow(data["image"][idx])
            axes[i].axis("off")
            axes[i].set_title(f"Score: {score:.3f}")

        plt.tight_layout()
        return plt.gca()


    visualize_top_k(txt_indices, txt_values, data)
    return (visualize_top_k,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Image to image search
    - Pass an input image to the model
    - Get the embedding of the image from the model
    - Find similar embeddings from all the image embeddings
    """
    )
    return


@app.cell
def _(data, device, image_emb, model, processor, top_k, torch):
    query_image = data["image"][2]

    # preprocess image
    q_img_ips = processor(
        images=query_image, return_tensors="pt", padding=True
    ).to(device)

    # generate image embeddings
    q_img_features = model.get_image_features(**q_img_ips)

    # normalize image embedding
    q_img_features = q_img_features / q_img_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores across all image embddings
    q_img_similarity = torch.mm(q_img_features, image_emb.T)

    # Get top-k matches
    img_values, img_indices = q_img_similarity[0].topk(min(top_k, len(data)))
    return img_indices, img_values


@app.cell
def _(data, img_indices, img_values, visualize_top_k):
    visualize_top_k(img_indices, img_values, data)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Zero Shot Clasification
    - One of the powerful use cases of CLIP is zero-shot classification, where the model is applied to classes or data it was never explicitly trained on.
    - We can provide CLIP with unseen image–text pairs and ask it to generate captions based on its pre-learned knowledge, or select the most relevant caption from a given list.
     - Conversely, we can start with a text prompt and let CLIP identify the most suitable image from either its pre-learned embeddings or from a new set of candidate images.
    - These capabilities unify two tasks—image-to-text (captioning/classification) and text-to-image (retrieval)—under the same framework.
    """
    )
    return


@app.cell
def _():
    from PIL import Image
    import requests

    url = "https://static.wikia.nocookie.net/beyblade/images/1/1e/Beyblade.png/revision/latest?cb=20110704200847"
    image = Image.open(requests.get(url, stream=True).raw)
    return (image,)


@app.cell
def _(device, image, model, processor):
    zeroshot_ips = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    # Since this is a classification task, we would already know the classes that we want to classify our new data into.

    # Thus, in the text argument, instead of passing "dog" and "cat", we pass the descriptions of our class so that the semantic similarity between text and image remains coherent with how the model was originally trained.

    zeroshot_ops = model(**zeroshot_ips)
    return (zeroshot_ops,)


@app.cell
def _(zeroshot_ops):
    logits_per_image = zeroshot_ops.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return (probs,)


@app.cell
def _(probs):
    probs
    return


if __name__ == "__main__":
    app.run()
