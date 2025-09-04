import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Library Imports""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Techniques to Build Practical Multimodal RAG system 
    - Building a RAG system that can efficiently retrieve and process information across text, images & structured data
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Clip (Contrastive Language-Image Pretraining) Embeddings
    - These allow us to bridge the gap between images and text for efficient cross-modal understanding
    - Clip model is unique because of it's understanding of images, given a text and vice-versa
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Face Unlock System 
    - To understand the working of clip model, let's try to solve an use-case of building a face-unlock system

    ##### Possible Solutions & Their Challenges
    - Training a binary classifier
        - Requirement of user data class: **Can be taken from user**
        - Requirement of other class: **Needs to be gathered** from real-world
        - If another user want's to use the system, we have to retrain and it can happen the previous user's recognition **isn't possible after retraining**

    - Transfer Learning
        - Here we take pretrained model, that will remove the need of **other class in general**, we only need to cater and fine-tune the model via user's data class
        - However the challenge still would be the adaptation of network, when another user comes in, the system still could forget previous user's.

    ##### A Better Solution
    - To tackle above challenges we need a network:
        - which has been pretrained on facial data, as it has to work, with less user data as well. 
        - that handle's multiple user's with ease 

    In simple terms a network, that has the ability to differentiate among different users & identify similar users, without strictly assigning labels to them. Such a neural network is **Siamese network**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Contrastive Learning using Siamese Networks
    - At it's core, **siamese network** determines whether 2 inputs are similar or not 
    - It does this by learning to map both inputs to a shared embedding space. 
        - If the distance between the embeddings is LOW, they are similar. 
        - If the distance between the embeddings is HIGH, they are dissimilar.

    In our face-unlock system it will be: 
    - If an input pair belong to the same person, the true label will be 0.
    - If a pair belongs to differnt people, the true label will be 1.

    One huge benefit of such a network, that simply determines the distance between 2 inputs, is we don't have **to retrain it, just set a threshold, below which we classify 2 inputs as dissimilar and above which similar**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Contrastive Loss 
    - A Siamese network, will be trained via a contrastive loss, that helps the network, overtime to be able to generate embeddings which have less distance, if they have same label else they have large distance.
    - The loss function looks like:
    $$L = (1-y)^2.D^2 + y.max(0,(margin-D))^2$$

    - Where:
        - `y` is the true label
        - `D` is the distance between 2 embeddings
        - `margin` is a hyperparameter, typically greater than 1

    - For our simple use case, y will be 0 when 2 inputs are similar and will be 1 when 2 inputs are dissimilar.
    - When y = 0, the `L` will be at it's minimum when D is near to 0 
    $$L = (1-y)^2.D^2$$
    - When y = 1, the `L` will be at it's minimum when D is as large as possible
    $$L = y.max(0,(margin-D))^2$$

    - So overall, we are making the network learn, to develop the embeddings in such a manner: 
        - that they are farther apart when the inputs are distinct
        - that they are as close as possible when inputs are similar
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Siamese Network in Face Unlock 
    - First we traine the model on several image pairs using contrastive loss. The pair has a label of 0, if they belong to same class and 1 if they belong to different class.
        - **NOTE**: During training we train the network with single input itself, it's just that the loss which goes into the network is of 2 inputs from a pair.
    - After training the model is shipped to user's device, where we ask user to take multiple selfies. We generate the embeddings of the selfies and store them for future comparison or for simplicity we just take average of these embeddings and save this single average.
    - For multiple user's we can have multiple embedding averages.
    - For unlocking a new pic is taken and it's embedding's distance is calculated to the stored users embedding averages. If the distance is above a threshold, for any user, we unlock the phone, else we keep it locked.

    A simple siamese network implementation could be found in [siamese_network_notebook](./con)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###  Multimodal Prompting
    - Beyond text based prompting, the input combines text, images and more.
    - Understand how is this done in a typical RAG system
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tool Calling
    - Enhance the capabilities of the RAG system dynamically calling external tools or APIs when **required**
    """
    )
    return


if __name__ == "__main__":
    app.run()
