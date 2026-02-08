# Understanding the Auto Regressive Models Architecture through a SLM
## Introduction
Before we dive into the details of architecture, what is a small model actually? 
- These days we see a lot of large language models (_LLMs_) being published by tech giant's however, these Language Models (_LMs_) barely are capable to be ran locally. Also creating one LLM requires  terabytes of data of text for training.
- To tackle this problem, there is a niche area optimizing the model architectures so that they perform best on a specific/domain problem and produce text on that. Also they don't need huge amounts of data but niche curated dataset for a specific task.
- These language models are called as Small Language Models (_SLMs_)

However the amount of params a SLM should have to call it small is a bit blurry. For contextual/blog purpose let's say any model having params less than 1B is a SLM.

- Essentially this whole space is optimizing the fact similar to chip space, where Moore's law holds true. Hopefully with a better future, this LLM space also follows the same law with hardware and software optimizations and one day we can even run a GPT-10 on an Android Phone, without any API call to the internet ;)


- The goal of this blog is to gain an understanding of LLMs architecture.
    - Again we will do this briefly to have enough understanding, of LMs internals without going in the depth's of Math's and functions.
    - We will go through the following concepts:
        - How data is prepared for the training purpose
        - What components the LM have, how they contribute to learning the essence of language, and what do the names of components actually come from
        - How do we calculate the loss for the training
        - How do we pre-train the model
        - How do we run inference via the model
---
## Contents
1. Dataset
2. Assembling the Model Architecture
3. Setting up the SLM training.
4. Pre-training the SLM
5. Running Inference.
--- 
## Dataset
    
### Collection of Dataset
- For this tutorial we will work with **Tiny Stories Dataset**. It consists of stories for 3-4 yr old kids.
- This dataset was:
    - Intelligently Curated Dataset for specific task of story generation. 
        - Likewise, if we want to create a specialized model we have to intelligent curate data for that task.

- With such a dataset what thought process do we have:
    - What type of language model do we develop which learns the essence of the dataset's language.
    - Given we are doing the development of SLM, another thought goes into this process is how small can a model be to learn the language essence and produce text coherently.

- For this blog we will focus on what components go into LM, SLM specifically so that it learns the essence of the language. 

- **Learn means:**
    - Learn structure of language: S - V - O
    - Learn meaning of language: LM should not produce less sensible text for eg., "Blue electrons eat fish."

- Now some characteristics of the dataset:
    - Tiny Stories Dataset has roughly 2M stories.
    - We use 2M stories split to in training and validation set.


Once we have our curated dataset, we would start processing this dataset. This is what we dive into our next part. 



### Pre-processing of dataset
#### Tokenization
The first bit of preprocessing the dataset is Tokenization. Computer don't understand words / text, but only numbers.
This process of converting text to numbers, understood by Language models is called as Tokenization. 

Following are the Ideas that form the basis of Tokenization:
- Word based Tokenization:
    - English Vocabulary has roughly 600,000 unique words
    - So encoding them to numbers at a given point of time and feeding to LM's will not be easy.
        - This perspective comes from Deep Learning basically, where i/p size always corresponds to the model parameters in one way or the other. 
    - Also there will be redundancy for example in word kitten and cat are similar, token and tokenization are also very similar. 


- Character Level Tokenization: 
    - English language has 26 ideal characters ignoring the punctuations. 
    - However, if we think of encoding millions and billions of just character encodings, just to maintain, the language modelling computations will explode, so we even drop this idea. 

- Sub-Word Tokenization: 
    - We choose the middle ground, which is also known as sub-word tokenization. In here we traverse through all characters one by one across all the text, and then those characters that occur very frequently are coupled. 
    - This gives us neither full words nor simple characters, something in between thus it's called sub-word tokenization. 
    - One popular algorithm of tokenization used by many researchers from LM space is Byte-Pair Encoding. We can visualize this in the below code snippet. 


- Essentially after tokenization the whole chunk of text broken down to tokens. For simplicity let's consider each word as a token itself, this will ease out the explanations and keep things fairly simple. 

Eg. `One day a little girl` has 5 words in total. After tokenization, we end up with tokens and their id's. These Id's are nothing but simplest numerical representation for the tokens by the tokenization algorithm. The number of tokens collected here on will be referred to as:

> Vocab Size: The unique tokens collected by the tokenizer. These tokens numerical aspect is the token id. 

Now any token of the whole text space is associated with it's IDs. 
```mermaid
graph TD
    A[Raw Text<br/>one day a little girl]
    A --> B[Tokenization<br/>one, day, a, little, girl]
    B --> C[Vocabulary Lookup<br/>one → 1<br/>day → 11<br/>a → 15<br/>little → 24]
    C --> D[Token IDs<br/>1, 11, 15, 24]

    style C fill:#FFBF00,stroke:#000000,stroke-width:3px,color:#000000
```

For our dataset we have 2M Training set and 20K Validation set stories. Using the byte pair encoding tokenizer, we have a vocab size of 50257 and total tokens in the scale of 90 to 100M for training and 10M for validation.

- We store these tokens into .bin file on disk, one for training and one for validation split. This helps us with:
    - Fast data loading, 
    - Avoid RAM overload, 
    - No need to re-tokenize, 
    - This format is easy to use.

- One more optimization that we do during tokenization, is we collect all the tokens batch them and then store them to disk: Tokens $\rightarrow$ Batches $\rightarrow$ Stored to Disk. Batching helps us for faster writes via multiple cores. 

**Important Code Lines**
- `arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))`, refers to creating a space to store the tokens in numpy array format on disk. 
- We first collect all the tokens and then batch them, this allows us for faster writes. This can be seen via `batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')`


## Assembling the Model Architecture
Once we have selected what domain specific data, tokenized the data, we need to now create a specialized small model for a domain. For this we need to define the model architecture that will learn from this data to ideally output coherent language, having similar patterns to the input data. 

> Ideally there is lot's of history in the NLP space, before the transformer block came, I will briefly introduce some history in bit's and pieces to fill in the understanding, but leave the depth to my next blog or readers own learning journey ;)

Let's look at how the model architecture looks like:
<img
  src="https://blogs.mayankpratapsingh.in/_next/image?url=%2Fimages%2Fblog%2FBirds_Eye_View_of_LLM%2FCover.gif&w=1920&q=75"
  alt="Birds eye view of a large language model pipeline by Mayank Pratap Singh"
/>


- There are 3 main blocks in the Model Architecture: 
    - Input Block: This block processes the data before feeding it to the model.
    - Processing Block: The essential part which learns the form and meaning from the dataset.
    - Output Block: This block serves the purpose of doing inference/predictions and penalizing the model via loss function.

Before we dive deep into the architecture's individual components let's have a bird's eye view over the whole architecture: 

Essentially, in machine learning applications we need some ground-truths from which the model has to infer okay given this input, my goal is to learn the features/patterns in the input and produce whatever output/ground truth the user has shown me during the training.

Likewise in the first input block, we figure out what will be the input to the model and what will be the output to the model, from which it can learn both the form and meaning from the huge amounts of text. We will discuss this in the input/output pair creation section. 

> Not ever ML applications require ground truth, some are non-supervised i.e. without any ground truths such as clustering applications. Language modeling is a self-supervised i.e. where the ground-truth is utilized during models learning process and it comes from the i/p itself. For now just knowing that LLMs are mostly trained under self-supervision is enough. 

Once we have the input and output, we pass this through the model in batches through multiple iterations, and do some mathematical operations of predictions, loss calculation by comparing how off the predictions are from the actual ground truth and based on this we update the model.

The process somewhat looks like this:
- Tokenization & Batching: Raw text is converted into integers (Tokens). These are organized into a matrix of shape [Batch Size, Sequence Length].
- Forward Pass: The model processes the batch. For every token, it outputs a vector of "logits" (scores) for every possible word in its vocabulary.
- Loss Calculation: Using Cross-Entropy Loss, we compare models predicted scores against ground truth i.e. the actual next token in the text.
- Backward Pass (Back-propagation): The "error" is sent backward through the Transformer layers to determine how much each weight contributed to the mistake.
- Gradient Clipping: The gradients are checked. If they are too "steep" (large), they are scaled down to prevent the model from becoming unstable.
- Optimizer Update (AdamW): The optimizer adjusts the weights based on the gradients, also accounting for "momentum" from previous batches.
- Weight Update: The parameters are updated, and the memory of the gradients is cleared for the next batch.
- Repeat: This continues until the data is exhausted (End of Epoch).

```mermaid
graph TD
    A[Batches of Raw Text Data] --> B[Tokenization & Embedding]
    B --> C[Forward Pass]
    C --> D[Calculate Cross-Entropy Loss]
    D --> E[Backward Pass: Compute Gradients]
    E --> F[Gradient Clipping]
    F --> G[AdamW Optimizer Step]
    G --> H[Update Model Parameters]
    H --> I{End of All Batches?}
    I -- No --> B
    I -- Yes --> J[Epoch Complete]
    
    style D fill:#FFBF00,stroke:#000000,stroke-width:3px,color:#000000
    style F fill:#FFBF00,stroke:#000000,stroke-width:3px,color:#000000
```

Now let's dive deep into each of the block.

### Input Block
In this block, we process the tokens further, before it get's fed to the Transformer block. 

- Following processes occur in the Input block: 
    - Tokenization: Going from big blobs of text to a unit of text easy to work with, which we have seen above in the pre-processing step.
    - I/P and O/P pairs generation: We generate the ground-truth for language modelling. 
    - Token Embeddings
    - Position Embeddings


#### Input Output Pair Creation
As we discussed earlier as Language Modelling is a Machine Learning problem, we need to have input pairs, for training a language model. 

> How do we go about finding the ground truth in huge texts? Thinking from the first principles what goal we want to achieve with language models?

> As we think about how humans learn to speak a language, we try to speak one word at a time, thoughtfully thinking about whether it makes sense and is inline to what we spoke before. Similar to this itself, our goal with language models is to predict the next token (instead of word) so that is is inline with the previously predicted tokens both syntactically as well as meaningfully. 

As now the task is next token prediction task, the model needs to take in a sequence of tokens process it and produce the next token in that sequence. 

```mermaid
graph LR
    A["Input Sequence<br/>Token IDs or Embeddings"]
    B["LLM<br/>Transformer Layers"]
    C["Next Token<br/>Probability Distribution"]
    D["Predicted Next Token"]

    A --> B
    B --> C
    C --> D
    
    style D fill:#FFBF00,stroke:#000000,stroke-width:3px,color:#000000
```


- To create these pairs we use something called as context size: 
    - Context Length (_CTX_): Max length of tokens language models look at 1 time before predicting the next token.
        > In our examples we use a dummy context size of 4 and a batch size of 4.

For eg.  
`One day a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.` could be broken down as below chunks and batches. 


<div class="mermaid">
graph TD
    Text["Whole Text\nbroken into chunks of \n4 words each"]

    Text --> c1["c1: One day a little"]
    Text --> c2["c2: girl named Lily found"]
    Text --> c3["c3: a needle in her"]
    Text --> c4["c4: room. She knew it"]
    Text --> c5["c5: was difficult to play"]
    Text --> c6["c6: with it because it"]
    Text --> c7["c7: was sharp."]

    subgraph Batch1["Batch 1"]
        c1
        c2
        c3
        c4
    end

    subgraph Batch2["Batch 2"]
        c5
        c6
        c7
    end

    style Text fill:#FFBF00,stroke:#000000,stroke-width:3px,color:#000000
    style Batch1 fill:#F3E8C7,stroke:#B59F3B,stroke-width:2px
    style Batch2 fill:#F3E8C7,stroke:#B59F3B,stroke-width:2px
</div>

Now once we have chunks, we break down each chunk to input and output pairs as follows: 

|i/p|o/p|
|---|---|
|One|day|
|One day|  a|
|One day a|  little|

So the chunk `c1` creates 3 input/output pairs. For the model we use these pairs for the training. 

> To get these pairs, first we map a chunk to the chunk which just comes after skipping the first token of the chunk. 

This breaking down to pairs we get by simply sliding a window over the chunks we created.  happens during training. In Matrix form it looks like this: 

> IMG: I/P output pair

For c1 and y1 we get input output pairs by pairing the _TIDs_,: 
    - 1 maps to 11.
    - 1, 11 maps to 15
    - 1, 11, 15 maps to 24
    - 1, 11, 15, 24 maps to 11

> Hence each chunk produces roughly #_CTX_ prediction tasks. So the training objective is as simple as next token prediction for each of the tasks. 

As we are doing this next token prediction itself, it's similar to regression, so these models are also called as Auto Regressive Models (_ARMs_), trained via a self-supervised process because we are creating the ground truth from the input data itself.

Now let's move onto the next input processing part, we need to have a better format than token Ids for token representation. 


#### Token Embedding Creation
If we think, then the token Ids (_TIDs_) don't capture any meaning about the sentence or for that matter word itself. 24 has no contextual value of `little`, it's simply a number. To tackle this problem itself researches in machine learning space use embeddings. 

- Think of embeddings as nothing but a higher dimensional space, where we will find more meaning about with respect to any item's properties/ features. These features are also called as dimensions. The higher the dimensions, the more qualitative/richer the substance is. This richness depends on what each dimension represent and how many dimensions we have. For us living organisms, mother Nature had filled this richness, via millions of years of evolution steps ;).

On the same context to capture the richness of every single word, we associate them with a higher number of dimensions. These dimensions are nothing but randomly initialized numbers, which will be determined by Language Model. The more the LM learn these features/richness well. The better. 

Once we know how to represent a tokens richly, we end up with something called as a Token Embedding Matrix (_TEM_). It stores embeddings for all the tokens in a vocabulary/dataset. These embeddings are revised via the language model during the training process, to capture the essence of the word. 
> The dimensions is vocab_size x num_embedding_dims. For our examples we use embedding dimension as `768`

Token embedding matrix serves as a lookup table, initialized randomly and are trainable params for the model. 

#### Position Embedding Matrix
Once we have the TEM, we need another matrix to capture the essence of position of words in the sentence. This is done by another high dimensional feature capturing matrix called position matrix. 
- Position Embedding Matrix (_PEM_) has same num_embedding_dims as that of TEM, just it's num_rows are equal to _CTX_

One we have _TIDs_, _TEM_ and _PEM_, we use them to process each token. For every token we add their embedding vector with position vector to get an input vector. 

Mathematically _TEM_ + _PEM_ yields a input embedding matrix (_IEM_). This _IEM_ is utilized by the next block of the model architecture. Let's see that next. 
---

### Processing Block
#### Norm Layer
We now process the _IEM_. by passing it via Norm Layer. 
> IMG: Norm layer transformation image
- Doing this helps us to: 
    - This helps to stabilize training by stopping gradient explosion.
    - Used to stabilize back-propagation during training.
    - Basically back-propagation depends on gradients which depend on the output of prev. layer in N.N.
    - If the O/P are not of uniform scale & fluctuates bet^n large & small values then the gradient can either stagnate or explode.
    - To avoid this unstable training dynamics, we do layer normalization.
    - Also the O/P distr^n from each layer could change so we normalize it by subtracting the mean from I/P & dividing by standard deviation.
    - This shift of distr^n problem is also called **internal Covariant Shift.**
    - Layer normalization improves training performance.

#### Attention Block
The block where our _SLM_ will actually learn how to produce meaningful short stories with the right form. Before we dive into the implementation part of the attention block, let's see why this is block is necessary in the first place. 
Eg. `A dog went to catch a ball, it couldn't catch it.`
- Here first it refers to dog & 2nd one to ball.
- Now if we want to predict next token for `it ...` such a sentence, should `bark` come after it or `flew`. 

For us humans this is fairly simple, as we know the context. By context I mean, we know what the it refers to, from the previous sentence. In NLP, this is deriving attention from the history. This deriving of history is the goal of the attention block. Without this block the _SLM_ would never learn how much attention needs to be paid to chunks of tokens before predicting the next token.

Now let's dive into the working of this block itself. 

Attention block is similar to embeddings. We begin with huge set of matrices filled with random numbers & tune these numbers over whole data hoping it will capture the essence i.e. form & meaning of the language.

Attention block captures how each token relates/attends to or gives attention to tokens which come before that token. This info is captured by the attention mechanism. The attention mechanism is responsible to convert the _IEM_ into a Context Vector (_CTXVec_). _CTXVec_ will be later also used in output block for loss calculation. 

> IMG: Context Vector formation 2 images.

> IMG: Masked Single Head Self Attention 

One crucial note to point out is we haven't changed dimensions  of embeddings from the input block, and it stays as is till the until we reach to the very last layer of the output block.

Mathematically, way we go from I/P vectors to context vectors is by making use of **Queries, Key & Value matrices weight Matrices.**

- Briefly we take I/P vector embeddings. (4 * 768)
    - Then we multiply them with `query weight matrix` i.e. $W_q$ of dim (768 * 768) & get a `query matrix` of dim (4 * 768)
    - Similarly we multiply input vector embeddings by `key weight matrix` i.e. $W_k$ & get `key matrix` of dim (4 * 768).
    - Lastly likewise we have `value matrix` by multiplying input vector embeddings via a `value weight matrix` i.e. $W_v$

- Here $W_Q, W_K$ & $W_V$ weight matrices are trainable.
- Once we have attention scores they are normalized by $sqrt(d_{keys})$, then we apply softmax to get Attention weights.
- The attention weights are multiplied by $V$ to get Context vectors of dim (4 * 768)

- **Another thing to note here is causality.**
    - Basically each token should only know/attend to tokens before it & not after it.
    - So the Attention scores matrix, we need to mask all the scores for tokens that come after a token.

> IMG: Causal attention scores.

- This avoids model peeking into future for next token prediction.
- So we know now why **causal & masked** is a word for the block name. Also the word **self** is used because each token attends to other tokens in its **own sentence**.
- Add of small N.N block after attention head.

> Code Snip: Of the causal attention and so on.

- Regarding masking process, we simply set values above diagonal to $-inf$, so when we take softmax they go to 0.
- The attention scores are divided by $sqrt$ of Key dims to keep the variance occurring in $Q * K^T$'s dot product should stay as low as possible.
- Then we fill elements above diagonal are replaced with $-\infty$.
- Then we take softmax of $Q * K^T$.
    - This masks token future tokens wrt a token.
    - Ensures the sum of prob. for tokens to be predicted is 1.
- After these operations we can quantitatively estimate how much attention a token has to pay to tokens prev. tokens.

- The trainable matrices are added in hope the math does the magic & learns the form & essence of language.

#### Dropout Block.

- We add a dropout block to remove the lazy neuron problem from the N.N. & improve generalization performance.

#### Shortcut Connection alternate.

- It is added to give another path to flow the gradients & prevent the vanishing gradient problem, when multiple layers are chained together.
- It just means I/P before all the layers occurring before shortcut connection is added to to the O/P produced just before by last layer before shortcut connection.
- In our case O/P of dropout is added to I/P before layer norm 1.
  Likewise I/P before layer norm 2 is added to O/P produced by 2nd dropout at the end of transformer block.

#### Feed Forward N.N.

- After shortcut connection 1, we again normalize the vectors.
- Then we have input for feed forward N.N.

> IMG: FF NN. 
- The feed forward N.N. is again a deep learning thing, basically we project the dimension to higher dimension space & hope that it captures the problem & solves it.
- In our dummy example.
  the I/P dim to Feed forward N.N are 4 * 768 $\xrightarrow{768 * 3072}$ 4 * 3072 dims $\rightarrow$ 4 * 768.
  (4 * 768 $\rightarrow$ 3072 dims)

    - This is expansion compression N.N. which retains the size of I/P but captures richer meaning, more non-linearity's.

- It turns out that this layer is crucial in transformer block & without this llm fails to capture context patterns in underlying data, & cannot answer queries that well.
- This block shifts performance to new levels.

* The activation function for this is **GeLU**.
    - It is diff. before 0.
    - Its +ve but not y=x for +ve inputs.
    - Generally gives good results.

---

- After this we again do dropout for stability.
> Code Snip

Now that's the end of 1 transformer block. We put such n transformer block chain sequentially & finally get logits matrix as an output from the transformer block. These logits still has the same dim as the i.e. 768.


### Output Block
Detailing the output block of transformer architecture. 

One [ ][ ][ ][ ] $\leftarrow$ 768 $\rightarrow$
day
a -||-
little
girl

- After we get O/P out of final transformer we again normalize it via layer normalization.
- Then this 4 * 768 matrix is passed through an output head.
- In O/P the output head the single batch is passed.

4 * 768 (batch size) $\xrightarrow[N.N]{768 * 50257}$ 4 * 50257.
Through another N.N. of (dim * vocabsize)
& we get an O/P of B.S * Vocabsize.

---

# Page 20

**How are the losses calculated for intermediate pairs? One code wise or mathematically?**
**There should be one step in code doing this.**
**Date: / /**

This matrix is **logits matrix tensor**, & we apply softmax to it.

| | $\leftarrow$ 50257 $\rightarrow$ | |
| :--- | :--- | :--- |
| One | - - - - - - - | softmax |
| day | | |
| a | | |
| little | | |

- Each element of column in row corresponds to the prob. of that token from vocab to appear in front of the tokens of that the row belongs to.

- The logits matrix in size is:
  num_batches * batch-size * vocab size.

- **Code Snip.**

- Now for each I/P pair model produces a pred i.e. a single token.

| | [ Vocab size ] | pred | Calculate loss $\leftarrow$ True |
| :--- | :--- | :--- | :--- |
| One | [ ][ ][ ][ ][ ] | the | day (13) |
| day | -||- | age | a (226) |
| a | | tree | little (1000) |
| little | | red | girl. (54) |

---

# Page 21

- The loss is then calculated bet^n true O/P's & model preds.
- This is summed up over all the I/P $\leftrightarrow$ O/P pairs formed in the batch & then back propagated through the model to update the params.
- Basically we calculate the cross entropy loss over all the positions in I/P sequence.
  In our eg. $-\frac{1}{4} (log P_{13} + log P_{226} + log P_{1000} + log P_{54})$

- $\frac{\partial L}{\partial p}$: Take gradient & update params.

- Each code has diff. initializations.
  - **Code Snip.**

- Each we can have multiple Attention heads above we saw a single attention head.
    - Multiple Attention head capture multiple perspectives in a transformer block.

- We can have n transformer block.

---