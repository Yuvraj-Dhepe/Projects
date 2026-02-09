# Understanding the Auto Regressive Models Architecture through a SLM

## Introduction

Before we dive into the details of architecture, what is a small model actually?

- These days we see a lot of large language models (_LLMs_) being published by tech giants however, these Language Models (_LMs_) are barely capable of running locally. Also creating one LLM requires  terabytes of data of text for training.
- To tackle this problem, there is a niche area optimizing the model architectures so that they perform best on a specific/domain problem and produce text on that. Also they don't need huge amounts of data but niche curated dataset for a specific task.
- These language models are called as Small Language Models (_SLMs_)

However the amount of params a SLM should have to call it small is a bit blurry. For the purpose of this blog, we’ll define an SLM as any model with fewer than 1B parameters. While the industry standard for "small" is shifting as hardware improves, 1B is a perfect entry point for understanding the architecture.

- The trend in SLMs is more about Data Quality and Algorithmic Efficiency, somewhat similar to Moore's law. It’s worth noting that we are getting better at "distilling" knowledge into smaller brains, not just waiting for faster chips.
- Hopefully with a better future, this LLM space also follows the same law with hardware and software optimizations and one day we can even run a GPT-10 on an Android Phone, without any API call to the internet ;)

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
4. Running Inference.

---

## Dataset

### Collection of Dataset

- For this tutorial we will work with **Tiny Stories Dataset**. It consists of stories for 3-4 yr old kids.
- This dataset was:
  - Intelligently Curated Dataset for specific task of story generation.
    - Likewise, if we want to create a specialized model we have to intelligently curate data for that task.

- With such a dataset what thought process do we have:
  - What type of language model do we develop which learns the essence of the dataset's language.
  - Given we are doing the development of SLM, another thought goes into this process is how small can a model be to learn the language essence and produce text coherently.

- For this blog we will focus on what components go into LM, SLM specifically so that it learns the essence of the language.

- **Learn means:**
  - Learn the structure of language: Subject-Verb-Object (S-V-O) (e.g., "The cat [S] ate [V] the fish [O]").
  - Learn meaning of language: LM should not produce less sensible text for eg., "Blue electrons eat fish."

- Now some characteristics of the dataset:
  - Tiny Stories Dataset has roughly 2M stories.
  - We use 2M stories split to in training and validation set.

```python
from datasets import load_dataset

# Load the TinyStories dataset from HuggingFace (~2M train + ~22K validation stories)
ds = load_dataset("roneneldan/TinyStories")
```

Once we have our curated dataset, we would start processing this dataset. This is what we dive into our next part.

### Pre-processing of dataset

#### Tokenization

The first bit of preprocessing the dataset is Tokenization. Computer don't understand words / text, but only numbers.
This process of converting text to numbers (not literally, but numbers are used in understanding the smallest unit of language for computers called `Tokens`), understood by Language models is called as Tokenization.

Following are the Ideas that form the basis of Tokenization:

- Word based Tokenization:
  - English Vocabulary has roughly 600,000 unique words
  - So encoding them to numbers at a given point of time and feeding to LM's will not be easy.
    - This perspective comes from Deep Learning basically, where input size always corresponds to the model parameters in one way or the other.
  - Also there will be redundancy for example in word kitten and cat are similar, token and tokenization are also very similar.

- Character Level Tokenization:
  - English language has 26 ideal characters ignoring the punctuations.
  - However, if we think of encoding millions and billions of just character encodings, just to maintain, the language modelling computations will explode, so we even drop this idea.

- Sub-Word Tokenization:
  - We choose the middle ground, which is also known as sub-word tokenization. In here we traverse through all characters one by one across all the text, and then those characters that occur very frequently are coupled.
  - This gives us neither full words nor simple characters, something in between thus it's called sub-word tokenization.
  - One popular algorithm of tokenization used by many researchers from LM space is Byte-Pair Encoding. We can visualize this in the below code snippet.

```python
import tiktoken

# GPT-2's BPE tokenizer; vocab_size = 50257 sub-word tokens
enc = tiktoken.get_encoding("gpt2")

def process(example):
    # encode_ordinary ignores special tokens like <|endoftext|>
    ids = enc.encode_ordinary(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out
```

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

### Important Code Lines

- `arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))`, refers to creating a space to store the tokens in numpy array format on disk.
- We first collect all the tokens and then batch them, this allows us for faster writes. This can be seen via `batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')`

```python
import numpy as np

tokenized = ds.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=8,  # parallelize across 8 CPU cores
)

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)  # total token count for this split
    filename = f'{split}.bin'
    dtype = np.uint16  # safe because max token id (50256) < 2^16 = 65536
    # Memory-mapped file: tokens stored on disk, accessed as if in RAM
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Shard the dataset into 1024 contiguous chunks for batched disk writes
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()  # ensure all data is written to disk
```

## Assembling the Model Architecture

Once we have selected what domain specific data, tokenized the data, we need to now create a specialized small model for a domain. For this we need to define the model architecture that will learn from this data to ideally output coherent language, having similar patterns to the input data.

> Ideally there is lot's of history in the NLP space, before the transformer block came, I will briefly introduce some history in bit's and pieces to fill in the understanding, but leave the depth to my next blog or readers own learning journey ;)

Let's look at how the model architecture looks like:

![Birds eye view of a large language model pipeline by Mayank Pratap Singh](https://blogs.mayankpratapsingh.in/_next/image?url=%2Fimages%2Fblog%2FBirds_Eye_View_of_LLM%2FCover.gif&w=1920&q=75)

- There are 3 main blocks in the Model Architecture:
  - Input Block: This block processes the data before feeding it to the model.
  - Processing Block: The essential part which learns the form and meaning from the dataset.
  - Output Block: This block serves the purpose of doing inference/predictions and penalizing the model via loss function.

Here is the complete configuration of our SLM along with a parameter budget breakdown:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int        # context window length (max tokens the model sees at once)
    vocab_size: int        # number of unique tokens in the tokenizer
    n_layer: int           # number of stacked Transformer blocks
    n_head: int            # number of attention heads per block
    n_embd: int            # embedding dimension (size of each token's vector)
    dropout: float = 0.0   # dropout probability (0 = no dropout)
    bias: bool = True      # whether to use bias in Linear layers and LayerNorms

config = GPTConfig(
    vocab_size=50257,  # GPT-2 BPE tokenizer vocabulary size
    block_size=128,    # context window = 128 tokens
    n_layer=6,         # 6 Transformer blocks stacked
    n_head=6,          # 6 attention heads (head_dim = 384/6 = 64)
    n_embd=384,        # embedding dimension
    dropout=0.1,       # 10% dropout for regularization
    bias=True
)
```

> **Our SLM Parameter Budget (~30M parameters)**

| Component | Shape | Parameters |
| --- | --- | --- |
| Token Embedding (`wte`) | 50257 × 384 | 19,298,688 |
| Position Embedding (`wpe`) | 128 × 384 | 49,152 |
| **Per Transformer Block (×6):** | | |
| &nbsp;&nbsp;LayerNorm 1 (`ln1`) | 384 + 384 | 768 |
| &nbsp;&nbsp;Attention QKV (`c_attn`) | 384 × 1152 + 1152 | 443,520 |
| &nbsp;&nbsp;Attention Output Proj (`c_proj`) | 384 × 384 + 384 | 147,840 |
| &nbsp;&nbsp;LayerNorm 2 (`ln2`) | 384 + 384 | 768 |
| &nbsp;&nbsp;FFN Expansion (`c_fc`) | 384 × 1536 + 1536 | 591,360 |
| &nbsp;&nbsp;FFN Contraction (`c_proj`) | 1536 × 384 + 384 | 590,208 |
| **Block subtotal** | | **1,774,464** |
| **6 Blocks total** | | **10,646,784** |
| Final LayerNorm (`ln_f`) | 384 + 384 | 768 |
| LM Head (`lm_head`) | _tied with `wte`_ | 0 (shared) |
| **Total** | | **~29,995,392 (~30M)** |

Before we dive deep into the architecture's individual components let's have a bird's eye view over the whole architecture:

Essentially, in machine learning applications we need some ground-truths from which the model has to infer okay given this input, my goal is to learn the features/patterns in the input and produce whatever output/ground truth the user has shown me during the training.

Likewise in the first input block, we figure out what will be the input to the model and what will be the output to the model, from which it can learn both the form and meaning from the huge amounts of text. We will discuss this in the input/output pair creation section.

> Not ever ML applications require ground truth, some are non-supervised i.e. without any ground truths such as clustering applications. Language modeling is a self-supervised i.e. where the ground-truth is utilized during models learning process and it comes from the input itself. For now just knowing that LLMs are mostly trained under self-supervision is enough.

Once we have the input and output, we pass this through the model in batches through multiple iterations, and do some mathematical operations of predictions, loss calculation by comparing how off the predictions are from the actual ground truth and based on this we update the model.

The process somewhat looks like this:

- Tokenization & Batching: Raw text is converted into integers (Tokens). These are organized into a matrix of shape [Batch Size, Sequence Length].
- Forward Pass: The model processes the batch. For every token, it outputs a vector of "logits"—which are essentially raw "confidence scores" for every possible word in the vocabulary.
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
  - input and output pairs generation: We generate the ground-truth for language modelling.
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

For eg. `One day a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.` could be broken down as below chunks and batches.

```mermaid
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
```

Now once we have chunks, we break down each chunk to input and output pairs as follows:

| input | output |
| --- | --- |
| One | day |
| One day | a |
| One day a | little |

So the chunk `c1` creates 3 input/output pairs. For the model we use these pairs for the training.

> To get these pairs, first we map a chunk to the chunk which just comes after skipping the first token of the chunk.

This breaking down to pairs we get by simply sliding a window over the chunks we created.  happens during training. In Matrix form it looks like this:

> IMG: input output pair

For c1 and y1 we get input output pairs by pairing the _TIDs_,:

- 1 predicts 11 ("One" → "day")
- 1, 11 predicts 15 ("One day" → "a")
- 1, 11, 15 predicts 24 ("One day a" → "little")

> Hence each chunk produces roughly #_CTX_ prediction tasks. So the training objective is as simple as next token prediction for each of the tasks.

As we are doing this next token prediction itself, it's similar to regression, so these models are also called as Auto Regressive Models (_ARMs_), trained via a self-supervised process because we are creating the ground truth from the input data itself.

In the code, this is handled by the `get_batch` function which randomly samples chunks from our `.bin` files:

```python
# block_size = 128 (context window), batch_size = 32
def get_batch(split):
    # Recreate memmap each call to avoid memory leaks with large files
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

    # Pick batch_size random starting indices from the tokenized data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x = input chunks of shape [batch_size, block_size] → [32, 128]
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y = target chunks shifted by 1 token → [32, 128]
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin_memory + non_blocking for async CPU→GPU transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

Now let's move onto the next input processing part, we need to have a better format than token Ids for token representation.

#### Token Embedding Creation

If we think, then the token Ids (_TIDs_) don't capture any meaning about the sentence or for that matter word itself. 24 has no contextual value of `little`, it's simply a number. To tackle this problem itself researches in machine learning space use embeddings.

- Think of embeddings as nothing but a higher dimensional space, where we will find more meaning about with respect to any item's properties/ features. These features are also called as dimensions. The higher the dimensions, the more qualitative/richer the substance is. This richness depends on what each dimension represent and how many dimensions we have. For us living organisms, mother Nature had filled this richness, via millions of years of evolution steps ;).

On the same context to capture the richness of every single word, we associate them with a higher number of dimensions. These dimensions are nothing but randomly initialized numbers, which will be determined by Language Model. The more the LM learn these features/richness well. The better.

Once we have how to represent a tokens richly, we end up with something called as a Token Embedding Matrix (_TEM_). It stores embeddings for all the tokens in a vocabulary/dataset. These embeddings are revised via the language model during the training process, to capture the essence of the word.

> The dimensions are `vocab_size × n_embd`. In our SLM, this is 50257 × 384, giving us **19,298,688 trainable parameters** — the single largest component of the model.

Token embedding matrix serves as a lookup table, initialized randomly and are trainable params for the model.

```python
# Inside the GPT model __init__:
# wte = "word token embedding" — the Token Embedding Matrix
self.transformer = nn.ModuleDict(dict(
    wte=nn.Embedding(config.vocab_size, config.n_embd),  # [50257, 384] → 19.3M params
    ...
))
```

#### Position Embedding Matrix

Once we have the _TEM_, we need another matrix to capture the essence of the position of words in the sentence. This is done by another high-dimensional feature-capturing matrix called the **Position Embedding Matrix** (_PEM_).

- The _PEM_ has the same `n_embd` (384) as the _TEM_, but its `num_rows` are equal to the context length i.e. `block_size` (128).

Once we have the _TIDs_, _TEM_, and _PEM_, we use them to process each token. For every token, we add its embedding vector to its corresponding position vector to get an input vector.

Mathematically, _TEM_ + _PEM_ yields an **Input Embedding Matrix** (_IEM_). This _IEM_ is then passed into the next block of the model architecture.

```python
# Inside GPT.forward():
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()  # b = batch_size (32), t = sequence_length (≤128)

    # Position indices: [0, 1, 2, ..., t-1]
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)   # [32, 128] → [32, 128, 384] via TEM lookup
    pos_emb = self.transformer.wpe(pos)   # [128] → [128, 384] via PEM lookup (broadcast across batch)
    x = self.transformer.drop(tok_emb + pos_emb)  # IEM = TEM + PEM → [32, 128, 384]
    ...
```

> **Param count:** `wpe` = 128 × 384 = **49,152 parameters** (relatively tiny compared to `wte`).

Without this matrix, our SLM is essentially blind to the order of words. Because Transformers process all tokens in a sequence simultaneously (in parallel), the model sees "The cat ate the fish" and "The fish ate the cat" as identical "bags of words."

To fix this, we "stamp" each token with a positional vector:

- The token "cat" gets a vector that says "I mean 'cat' AND I am the 2nd word."
- This allows the Attention mechanism to understand not just what a word is, but where it is relative to others.

> The critical point here, is we use the same positional vector matrix for the whole training.

- Think of the context window as a bus with **128 seats** (`block_size=128`).
  - We add a specific vector (PEM-0) to whatever word sits in Seat #0.
  - We add a different vector (PEM-1) to Seat #1.
  - This injects structural information into the word. The model learns that "Start of sentence" + "The" implies a subject is coming next.

Note: While we use simple addition here, modern models (like Llama 3) use RoPE (Rotary Positional Embeddings), which rotates the vector in space to capture relative distances even better.

---

### Processing Block

Processing block is the famous `Transformer Block` which consists of attention blocks, which is the critical block of LLM itself.

#### Norm Layer

We now process the _IEM_. by passing it via Norm Layer.

> IMG: Norm layer transformation image

- Doing this helps us to:
  - This helps to stabilize training by stopping gradient explosion.
  - Used to stabilize back-propagation during training.
  - Basically back-propagation depends on gradients which depend on the output of prev. layer in N.N.
  - If the outputs are not of uniform scale & fluctuates between large & small values then the gradient can either stagnate or explode.
  - To avoid this unstable training dynamics, we do layer normalization.
  - Also the outputs distribution from each layer could change so we normalize it by subtracting the mean from input & dividing by standard deviation.
  - This shift of distribution problem is also called **internal Covariant Shift.**
  - Layer normalization improves training performance.

Note on Pre-Norm: In modern SLMs (like Llama or GPT-3 variants), we usually apply this Normalization before the Attention and FFN layers (Pre-Norm) rather than after. This simple switch makes the training significantly more stable and allows us to train deeper networks without the gradients vanishing.

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # learnable scale, shape [384]
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # learnable shift

    def forward(self, x):
        # Normalize across the last dimension (n_embd=384) with epsilon=1e-5 for stability
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

> **Param count per LayerNorm:** 384 (weight) + 384 (bias) = **768 parameters**. There are 2 per block × 6 blocks + 1 final = **13 LayerNorm instances** = 9,984 params total.

#### Attention Block

The block where our _SLM_ will actually learn how to produce meaningful short stories with the right form. Before we dive into the implementation part of the attention block, let's see why this block is necessary in the first place.
Eg. `A dog went to catch a ball, it couldn't catch it.`

- Here first it refers to dog & 2nd one to ball.
- Now if we want to predict next token for `it ...` such a sentence, should `bark` come after it or `flew`.

For us humans this is fairly simple, as we know the context. By context I mean, we know what the `it` refers to, from the previous sentence. In NLP, this is called as deriving attention from the history. This deriving of history is the goal of the attention block. Without this block the _SLM_ would never learn how much attention needs to be paid to chunks of tokens before predicting the next token.

Now let's dive into the working of this block itself.

Attention block does introduce something, that serves similar functionality of introducing higher dimensional space via _IEM_ and _PEM_. We begin with huge set of matrices filled with random numbers & tune these numbers over whole data hoping it will capture the essence i.e. form & meaning of the language.

Attention block captures how each token relates/attends to or gives attention to tokens which come before that token. This info is captured by the attention mechanism. The attention mechanism is responsible to convert the _IEM_ into a Context Vector (_CTXVec_). _CTXVec_ will be later also used in output block for loss calculation.

> IMG: Context Vector formation 2 images.
> IMG: Masked Single Head Self Attention

One crucial note to point out is we haven't changed the dimensions of embeddings from the input block, and it stays **384** (`n_embd`) throughout the entire Transformer until we reach the very last layer of the output block.

Mathematically, we go from input vectors to context vectors is by making use of **Queries, Key & Value matrices weight Matrices.**

- Briefly we take input vector embeddings. (4 × 384) _(using our dummy CTX=4 with n_embd=384)_
  - Then we multiply them with `query weight matrix` i.e. $W_q$ of dim (384 × 384) & get a `query matrix` of dim (4 × 384)
  - Similarly we multiply input vector embeddings by `key weight matrix` i.e. $W_k$ & get `key matrix` of dim (4 × 384).
  - Lastly likewise we have `value matrix` by multiplying input vector embeddings via a `value weight matrix` i.e. $W_v$

> In the code, $W_Q, W_K, W_V$ are fused into a single matrix `c_attn` of shape (384, 1152) for efficiency. The output is then split into Q, K, V each of shape (..., 384).

- Here $W_Q, W_K$ & $W_V$ weight matrices are trainable.
- Once we have **attention scores** they are normalized by $\sqrt{d_{keys}}$ (= $\sqrt{64}$ = 8 in our case, since head_dim=64), then we apply softmax to get Attention weights.
  - The normalization by key dimensions is to avoid our math get too big which leads to the Softmax function 'saturation' — it starts to pick one winner and ignores everyone else, making it impossible for the model to learn from its mistakes (vanishing gradients).
- The attention weights are multiplied by $V$ to get Context vectors of dim (4 × 384)

The names of query, key and value have origins in the field of information retrieval.

- You have a query: something you're looking for
- The database has keys: tags or identifiers that help you match your query to relevant data.
- Associated with these keys, you have values: the actual information or content you want to retrieve.

This analogy translates directly to attention mechanisms in neural networks:

- Query (Q): The current word or token has a "question" about its context. It asks: "What should I pay attention to?"
- Key (K): Every token provides a "key," acting as a label or signal to answer queries from other tokens. It indicates: "Here's what information I can provide."
- Value (V): Once a match (query-key) is found, this is the actual information provided. It says: "Here's the meaning or content you will get if you attend to me."

- **Another thing to note here is causality.**
  - Basically each token should only know/attend to tokens before it & not after it.
  - So the Attention scores matrix, we need to mask all the scores for tokens that come after a token.

> IMG: Causal attention scores.

- This avoids model peeking into future for next token prediction.
- So we know now why **causal & masked** is a word for the block name. Also the word **self** is used because each token attends to other tokens in its **own sentence**.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Single linear layer that produces Q, K, V all at once for efficiency
        # Input: [B, T, 384] → Output: [B, T, 3×384=1152], then split into Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: merges multi-head results back to n_embd dims
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)  # dropout on the residual path
        self.n_head = config.n_head  # 6 heads
        self.n_embd = config.n_embd  # 384
        # Use PyTorch's fused Flash Attention if available (much faster)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # Pre-compute the causal mask: lower-triangular matrix of ones
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # B=batch(32), T=seq_len(≤128), C=n_embd(384)

        # Project input to Q, K, V and split: each is [B, T, 384]
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape for multi-head: [B, T, 384] → [B, T, 6, 64] → [B, 6, T, 64]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Fused attention: Q·Kᵀ/√d_k → mask → softmax → ·V, all in one GPU kernel
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            # Manual attention: compute scores, mask future tokens, apply softmax
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B,6,T,T]
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # causal mask
            att = F.softmax(att, dim=-1)  # normalize to probabilities
            att = self.attn_dropout(att)
            y = att @ v  # weighted sum of values → [B, 6, T, 64]

        # Concatenate heads: [B, 6, T, 64] → [B, T, 6, 64] → [B, T, 384]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Final output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
```

> **Param count per CausalSelfAttention:**
> - `c_attn`: 384 × 1152 + 1152 = **443,520** (produces Q, K, V)
> - `c_proj`: 384 × 384 + 384 = **147,840** (output projection)
> - **Subtotal: 591,360 per block**

- Regarding masking process of future tokens, we simply set the values above the diagonal to $-inf$, so when we take softmax they go to 0.
- The attention scores are divided by $sqrt$ of Key dims to keep the variance occurring in $Q * K^T$'s dot product should stay as low as possible.
- Then we fill elements above diagonal are replaced with $-\infty$.
- Then we take softmax of $Q * K^T$.
  - This masks token future tokens wrt a token.
  - Ensures the sum of prob. for tokens to be predicted is 1.
- After these operations, since we have `probabilities for next token to be predicted` we can quantitatively estimate how much attention a token has to pay to it's prev. tokens.

**Multi-Head Attention: The Panel of Experts**
Up to this point, we’ve looked at a single "head" of attention. But a single head might focus heavily on syntax (e.g., relating "cat" to "ate"). What if we also want to understand the emotional tone, or pronoun relationships?

Real SLMs use Multi-Head Attention. Instead of one giant query/key/value operation, we split the embedding dimension into multiple smaller "heads." In our SLM, we use **6 heads processing 64 dimensions each** (384 / 6 = 64).

Think of this like a panel of experts reading the story:

- Head 1 focuses on Grammar (Subject-Verb relationships).
- Head 2 focuses on Context (Is "Bank" a river bank or money bank?).
- Head 3 focuses on Rhyme or Tone.

These heads run in parallel, and their results are concatenated (glued back together) at the end. This allows the model to capture multiple different "perspectives" of the language simultaneously.

#### Dropout

- In the neuron layers we configure another parameter for lazy neuron problem i.e. dropout. Basically we turn off some of the neurons randomly during the training process, the purpose is to utilize all the neurons efficiently and improve generalization performance of the model.

#### Shortcut Connection alternate

After the dropout block we do deep learning trick i.e. addition of shortcut connection:

- A connection added to give another path to flow the gradients & prevent the vanishing gradient problem, that occurs when multiple NN layers are chained together.
- Shortcut connection refers to taking input, just before all the layers occurring before shortcut connection is taken and added to the output produced just before by last layer before shortcut connection in the figure.

#### Feed Forward Network

While the Attention mechanism is the "social" part of the model (tokens talking to each other to gather context), the **Feed Forward Network** (_FFN_) is the "reflective" part. After the first shortcut connection and another layer of normalization, the tokens are passed through this block to refine their individual meanings.

- Many researchers view the _FFN_ as the "knowledge base" where the model stores "facts" or patterns learned from the dataset.
- The Attention layer figures out *which* information is relevant, and the _FFN_ processes that information to deepen the understanding.

This block is often called an **expansion and contraction** layer:

1. **Expansion:** We project the dimensions into a much higher space (384 → 1536, i.e. `4 × n_embd`). This "upscaling" allows the model to capture complex non-linearities and the nuances of sentence structure.

2. **Contraction:** We then project it back down to the original embedding dimension (1536 → 384).

> **Our SLM Dimensions:**
> $(B \times 128 \times 384) \xrightarrow{\text{Expansion}} (B \times 128 \times 1536) \xrightarrow{\text{Contraction}} (B \times 128 \times 384)$

The activation function used in this network is **GeLU** (Gaussian Error Linear Unit). It is preferred over standard functions like ReLU because it is differentiable at all points and generally yields smoother training dynamics and better performance for LLMs.

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expansion: 384 → 1536 (4× increase)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()  # smooth activation, preferred over ReLU for LLMs
        # Contraction: 1536 → 384 (back to n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [B, T, 384] → [B, T, 1536] → GeLU → [B, T, 384] → dropout
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
```

> **Param count per MLP:**
> - `c_fc` (expansion): 384 × 1536 + 1536 = **591,360**
> - `c_proj` (contraction): 1536 × 384 + 384 = **590,208**
> - **Subtotal: 1,181,568 per block** — the FFN is the heaviest component per Transformer block.

---

After the _FFN_, we apply another round of **Dropout** for stability and a final **Shortcut Connection**.

This marks the end of a single **Transformer Block**. In practice, we stack $n$ of these blocks sequentially. Our SLM uses **6 blocks** (`n_layer=6`). By the time the tokens have passed through all blocks, they have been transformed from simple IDs into highly sophisticated, context-aware vectors of 384 dimensions.

```python
class Block(nn.Module):
    """A single Transformer block: Pre-Norm → Attention → Residual → Pre-Norm → FFN → Residual"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)  # Pre-norm before attention
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)  # Pre-norm before FFN
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Shortcut connection around attention
        x = x + self.mlp(self.ln2(x))   # Shortcut connection around FFN
        return x  # shape unchanged: [B, T, 384]
```

> **Param count per Block:** 768 + 591,360 + 768 + 1,181,568 = **1,774,464**
> **6 Blocks total:** 6 × 1,774,464 = **10,646,784 parameters**

The full model stacks these blocks in a `ModuleList`:

```python
# Inside GPT.__init__():
self.transformer = nn.ModuleDict(dict(
    wte=nn.Embedding(config.vocab_size, config.n_embd),    # Token Embedding [50257, 384]
    wpe=nn.Embedding(config.block_size, config.n_embd),    # Position Embedding [128, 384]
    drop=nn.Dropout(config.dropout),                        # Embedding dropout
    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 6 Transformer blocks
    ln_f=LayerNorm(config.n_embd, config.bias),            # Final layer norm
))
```

### Output Block

From the processing block, we take the matrix of 384-dim vectors (`n_embd`), do a final layer normalization and again pass it through a neural network layer technically called the `Language Model Head (LM Head)`, to bring the 384 dims to vocab dimensions i.e. 50257.

This is also a beautiful symmetry in LLM architectures:
The Token Embedding Matrix at the start and the LM Head at the end are essentially opposites. One turns "ID → Vector," the other turns "Vector → ID scores." In many models, these two actually share the same weights known as `Weight Tying`.

```python
# LM Head: projects 384 dims back to vocab_size (50257) to produce logits
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # [384, 50257]

# Weight Tying: wte and lm_head share the SAME weight matrix
# This saves ~19.3M params and acts as a regularizer
self.transformer.wte.weight = self.lm_head.weight
```

The output from LM Head is logits matrix, which we again pass through a softmax function to get final probabilities. This final matrix is used in loss calculation and after training we will utilize this matrix for quantifying the probability of prediction of next token for the current token, and choose the next token prediction.

The loss calculation Steps look something as follows:

- For every input output pairs, we know the ground truths and the predicted tokens.

| input | pred | output (ground_truth) |
| --- | ---- | ------------------ |
| One | a | day (TID = 11) |
| One day | little | a (TID = 15) |
| One day a | girl | little (TID = 24) |

Now as the goal is to predict the next token correctly, from the final probability matrix, we consider the probabilities of the correct ground truth i.e. we collect only the probability at the _TIDs_, same as the ground truth i.e. $p_{11}, p_{15}, p_{24}$ and calculate categorical cross entropy loss with these.
So for the above example the loss is:
    $-\frac {1}{4} (log(p_{11}) + log(p_{15}) + log(p_{24}))$

In the code, the forward pass through the output block and loss calculation look like this:

```python
# Inside GPT.forward(), after all Transformer blocks:
x = self.transformer.ln_f(x)  # Final layer norm: [B, 128, 384]

if targets is not None:
    # During training: compute logits for ALL positions
    logits = self.lm_head(x)  # [B, 128, 384] → [B, 128, 50257]
    # Flatten batch & sequence dims, then compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # [B*128, 50257]
        targets.view(-1),                   # [B*128]
        ignore_index=-1
    )
    return logits, loss
else:
    # During inference: only compute logits for the LAST token (efficiency)
    logits = self.lm_head(x[:, [-1], :])  # [B, 1, 50257]
    return logits, None
```

We also define a helper to estimate loss across multiple batches for stable evaluation:

```python
def estimate_loss(model):
    """Average loss over eval_iters batches for both train and val splits."""
    out = {}
    model.eval()  # switch to eval mode (disables dropout)
    with torch.inference_mode():  # no gradient tracking for speed
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:  # mixed precision context (bfloat16/float16)
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()  # switch back to training mode
    return out
```

## Setting up the SLM training

Now that we understand the blocks, how do they actually learn? We don't just pass one sentence through; we pass millions, over and over again.

The training loop is the engine that drives the learning. We iterate through our dataset in "Epochs" (one full pass over the data). Inside each epoch, we process the data in batches.

Here is the flow of a single training step:

```mermaid
graph TD
    Start((Start Epoch)) --> Batch["Get Batch of Data<br/>(Inputs & Targets)"]
    Batch --> Forward["<b>Forward Pass</b><br/>Input -> Embeddings -> Transformer Layers -> Logits"]
    Forward --> LossCalc["<b>Calculate Loss</b><br/>Compare Logits vs Targets using Cross Entropy"]
    LossCalc --> Backprop["<b>Backward Pass</b><br/>Calculate Gradients based on Error"]
    Backprop --> Clip["<b>Gradient Clipping</b><br/>Prevent explosions!"]
    Clip --> Optim["<b>Optimizer Step</b><br/>Update Weights using AdamW"]
    Optim --> Zero["<b>Zero Gradients</b><br/>Clear memory for next batch"]
    Zero --> Check{More Batches?}
    Check -- Yes --> Batch
    Check -- No --> End((End Epoch))

    style Forward fill:#FFBF00,stroke:#000000,stroke-width:2px,color:#000000
    style Optim fill:#FFBF00,stroke:#000000,stroke-width:2px,color:#000000
```
You might notice the "Optimizer Step" in the chart. We usually use an optimizer called AdamW. Without getting into the math, think of AdamW as a GPS for the model's weights. It tells the parameters exactly which direction to move and how fast to move to reduce the loss (error) most efficiently.

Here is the full training configuration from our codebase:

```python
# --- Hyperparameters ---
learning_rate = 1e-4           # peak learning rate
max_iters = 60000              # total training steps
warmup_steps = 1000            # linearly ramp up LR for the first 1000 steps
min_lr = 5e-4                  # floor for cosine decay
eval_iters = 500               # evaluate every 500 steps
batch_size = 32                # samples per batch
block_size = 128               # context window (must match config.block_size)
gradient_accumulation_steps = 32  # simulate effective batch of 32×32 = 1024

# --- Mixed Precision ---
# bfloat16 halves memory usage vs float32 with minimal accuracy loss
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Optimizer & Scheduler ---
# AdamW with weight decay for regularization; beta2=0.95 (default is 0.999)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                               betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)       # linear warmup
scheduler_decay = CosineAnnealingLR(optimizer,
                                     T_max=max_iters - warmup_steps,
                                     eta_min=min_lr)                    # cosine decay
# Switch from warmup → cosine at step 1000
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay],
                          milestones=[warmup_steps])

# GradScaler only needed for float16 (bfloat16 doesn't need it)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

And the training loop itself:

```python
best_val_loss = float('inf')
model = model.to(device)

for epoch in tqdm(range(max_iters)):
    # --- Periodic Evaluation ---
    if epoch % eval_iters == 0 and epoch != 0:
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), "best_model_params.pt")

    # --- Forward Pass ---
    X, y = get_batch("train")                        # [32, 128] input and target
    with ctx:                                         # mixed precision context
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps     # scale loss for accumulation

    # --- Backward Pass ---
    scaler.scale(loss).backward()                     # compute gradients (scaled for float16)

    # --- Optimizer Step (every 32 micro-batches) ---
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # gradient clipping
        scaler.step(optimizer)     # unscale gradients and update weights
        scaler.update()            # adjust scaler for next iteration
        optimizer.zero_grad(set_to_none=True)  # free gradient memory

    scheduler.step()  # update learning rate
```

## Running Inference

Once our loss is low and the model is trained, how do we actually use it to write a story? This process is called Inference.

Unlike training, where we feed the whole sentence at once, inference is a loop. The model generates one token, and then we feed that token back into the model to generate the next one. This is why it’s called Auto-Regressive.

The Generation Loop:

- Prompt: We give the model a starting text: "Once upon a time"
- Forward Pass: The model calculates probabilities for the next likely token.
- Sampling: We pick a token from those probabilities.
  - Note: We rarely just pick the #1 highest probability (that makes the model robotic). We introduce a parameter called Temperature.
  - High Temperature (e.g., 1.0): The model takes risks. It might pick a less likely word, leading to more creative (but sometimes nonsense) stories.
  - Low Temperature (e.g., 0.2): The model plays it safe. Ideally for factual tasks.
- Append: We add the chosen token to our input sequence.
- Input becomes: "Once upon a time there"
- Repeat: We run the model again with the new longer input.

How does it stop?
During training, we introduce a special token called <EOS> (End of Sequence). When the model predicts this token during inference, it is essentially saying, "I am finished with the story," and the loop terminates.

Here is the `generate` method that implements this auto-regressive loop:

```python
@torch.no_grad()  # disable gradient tracking for inference speed
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Auto-regressive generation: predict one token at a time, append, repeat.
    idx: starting token ids of shape [B, T]
    """
    for _ in range(max_new_tokens):
        # Crop to context window if sequence exceeds block_size (128)
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)         # forward pass → [B, 1, vocab_size]
        logits = logits[:, -1, :] / temperature  # scale logits by temperature

        if top_k is not None:
            # Zero out all logits below the top-k threshold
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)           # convert to probabilities
        idx_next = torch.multinomial(probs, num_samples=1)  # sample next token
        idx = torch.cat((idx, idx_next), dim=1)      # append to sequence
    return idx
```

And this is how we use it to generate a story:

```python
# Load the trained model
model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=torch.device(device)))

# Encode a prompt and generate 200 new tokens
sentence = "Once upon a time there was a pumpkin."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0)  # [1, num_tokens]
y = model.generate(context, 200)

# Decode token IDs back to human-readable text
print(enc.decode(y.squeeze().tolist()))
```

And there you have it! From raw text to a "thinking" small language model running on your local machine.

In the next part, we will see which components of an Auto Regressive Model, should we replace to convert it to a diffusion model. Before I finish, I would like to add the steps we discussed necessary in the diffusion model process in our previous blog:

- Nosing: Creation of the noising input via a noising schedule.
- Finding a model that predicts Noise: By predicting noise, doesn't mean predicting actual noise, but the value covered by noise.
- De-noising & generation of output

We will define these step for text based diffusion models and the component changes in ARM required to achieve this.

Until then Sankyu for reading and will be glad to receive your feedback on your thought process about Auto Regressive Models.

## Extra Rabbit Hole Discussions

### **The Math behind the Magic: Why Softmax and Cross-Entropy?**

- In our output block we convert our logits into probabilities using **Softmax** and calculate error using **Cross-Entropy**. Why these specific functions though?
  - 1.**Why `exp()` in Softmax?**
    - **No Negatives:** The output of a neural network can be negative (e.g., -5). $e^{-5}$ is a small positive number ($0.006$). Probabilities must be positive!
    - **Winner Takes All:** Exponentials exaggerate differences. If "cat" scores slightly higher than "dog" in raw numbers, the exponential function widens that gap, making the model more decisive.
  
  - 2.**Why Cross-Entropy?**
    - It measures "Surprise." If the model predicts "Dog" with 99% confidence but the answer is "Cat," the surprise (loss) is massive.

  - 3.**The Calculus Trick:**
    - The most beautiful reason is the math. When you combine the natural log ($\ln$) from Cross-Entropy with the exponent ($e^x$) from Softmax, their derivatives cancel out perfectly ($P - Y$). This removes messy constants from the gradient calculation, keeping the training fast and numerically stable.

### **From Tokens to Parallel Loss: The "Magic Trick"**

- I used to think the model reads "One," predicts "day," adds it to the list, and then reads "One day." That is how we *generate* text (Inference), but that is too slow for *training*.

In training, we use a magic trick called **Teacher Forcing** with Parallelism. We don't loop; we process everything at once using giant matrices. Here is the 6-step flow:

- 1. **Batching:** We stack 32 sentences of length 128 into a matrix `[32, 128]`.
- 2. **Embedding:** We convert these IDs to vectors `[32, 128, 384]`. "One" and "day" remain separate vectors; we **do not** combine them into a single "One day" token.
- 3. **Positional Encoding:** We stamp each vector with its location (Seat #0, Seat #1).
- 4. **Forward Pass:** The Transformer processes all 4,096 tokens (32 * 128) simultaneously.
- 5. **Masking:** The Attention block uses a triangular mask to ensure Token #5 cannot "see" Token #6, even though they are processed at the same time.
- 6. **Parallel Loss:** We calculate the loss for all 4,096 predictions instantly and **average them over the entire batch**.
- This approach allows the GPU to crush the computation in a single step without waiting for loops.

Code Snippet: The Batch Loss Calculation

```python
 # Inside the model's forward pass
 if targets is not None:
     # flatten the batch (32) and sequence (128) dimensions together
     # logits.view(-1, ...) -> shape becomes [4096, vocab_size]
     logits = logits.view(-1, logits.size(-1))
     targets = targets.view(-1)
     
     # Calculate cross_entropy for all 4096 tokens at once and return the average
     loss = F.cross_entropy(logits, targets)
```

## Resources

- [What is a VLM?](https://www.vizuaranewsletter.com/p/what-exactly-is-a-vlm-vision-language)
- [Journey of a Single Token](https://www.vizuaranewsletter.com/p/the-journey-of-a-single-token)

## Closing Note

The post aims to be both beginner friendly and more than anyone for me to help me catchup in the space of LLMs, putting pen to paper and jotting my understanding. So feedback is always needed for self-improvement in the process ;). Thank you for reading.
