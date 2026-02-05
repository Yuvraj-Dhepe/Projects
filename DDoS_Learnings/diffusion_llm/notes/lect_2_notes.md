# Understanding the Auto Regressive Models Architecture through a SLM
## Introduction
Before we dive into the details of architecture, what is a small model actually? 
- These days we see a lot of large language models (LLMs) being published by tech giant's however, these Language Models (LMs) barely are capable to be ran locally.
- To tackle this problem, there is a niche area optimizing the model architectures so that they perform best on a specific/domain problem and produce text on that. 
- These language models are called as Small Language Models (SLMs)

However the amount of params a SLM should have to call it small is a bit blurry. For contextual/blog purpose let's say any model having params less than 1B is a SLM.

- Essentially this whole space is optimizing the fact similar to chip space, where Moore's law holds true. Hopefully with a better future, this LLM space also follows the same law with hardware and software optimizations and one day we can even run a GPT-10 on an Android Phone, without any API call to the internet ;)


- The goal of this blog is to gain an understanding of LLMs architecture.
    - Again we will do this briefly to have enough understanding, of LMs internals without going in the depth's of Math's and functions.
    - We will go through the following concepts:
        - How data is prepared for the training purpose
        - What components the LM have, how they contribute to learning the essence of language, and what do the names of components actually come from
        - How do we calculate the loss for the training
        - How do we pretrain the model
        - How do we run inference via the model
---
## Contents
1. Dataset
2. Assembling the Model Architecture
3. Setting up the SLM training.
4. Pretraining the SLM
5. Running Inference.
--- 
## Dataset

### Collection of Dataset
- For this tutorial we will work with **Tiny Stories Dataset**. It consists of stories for 3-4 yr old kids.
- This dataset was:
    - Intelligently Curated Dataset for specific task of story generation. 
        - Likewise, if we want to create a speciallized model we have to intelligent curate data for that task.

- With such a dataset what thoughtprocess do we have:
    - What type of language model do we develop which learns the essence of the dataset's language.
    - Given we are doing the development of SLM, another thought goes into this process is how small can a model be to learn the language essence and produce text coherently.

- For this blog we will focus on what components go into LM, SLM specifically so that it learns the essence of the language. 

- **Learn means:**
    - Learn structure of language: S - V - O
    - Learn meaning of language: LM should not produce less sensible text for eg., "Blue electrons eat fish."

- Now some characteristics of the dataset:
    - Tiny Stories Dataset has roughly 2M stories.
    - We use 2M stories splitted in training and validation set.


Once we have our curated dataset, we would start processing this dataset. This is what we dive into our next part. 



### Pre-processing of dataset
The first bit of preprocessing the dataset is Tokenization. Computer don't understand words / text, but only numbers.
This process of converting text to numbers, understood by Language models is called as Tokenization. 

Following are the Ideas that form the basis of Tokenization:
- Word based Tokenization:
    - English Vocabulary has roughly 600,000 unique words
    - So encoding them to numbers at a given point of time and feeding to LM's will not be easy.
        - This perspective comes from Deep Learning basically, where i/p size always corresponds to the model parameters in one way or the other. 
    - Also there will be redundnacy for example in word kitten and cat are similar, token and tokenization are also very similar. 


- Character Level Tokenization: 
    - English language has 26 ideal characters ignoring the punctuations. 
    - However, if we think of encoding millions and billions of just character encodings, just to maintain, the language modelling computations will explode, so we even drop this idea. 

- Sub-Word Tokenization: 
    - We choose the middle ground, which is also known as sub-word tokenization. In here we traverse through all characters one by one across all the text, and then those charactes that occur very frequently are coupled. 
    - This gives us neither full words nor simple characters, something in between thus it's called sub-word tokenization. 
    - One popular algorithm of tokenization used by many researchers from LM space is Byte-Pair Encoding. We can visualize this in the below code snippet. 


- Essentially after tokenization the whole chunk of text brokes down to tokens. For simplicity let's consider each word as a token itself, this will ease out the explanations and keep things fairly simple. 

Eg. `One day a little girl` has 5 words in total. After tokenization, we end up with tokens and their id's. These Id's are nothing but simplest numerical representation for the `x_n_t collected` tokens by the tokenization algorithm.

Now any token of the whole text space is associated with it's IDs. 
- `one day a little girl` $\frac{tokenization}{\rightarrow}$ [1,3,11,27]

However, if we think, then the `TiDs` don't capture any meaning about the sentence or for that matter word itself. 11 has no contextual value of girl, it's simply a number. To tackle this problem itself researches in machine learning space use embeddings. 

- Think of embeddings as nothing but a higher dimensional space, where we will find more meaning about wrt any item. So basically mathematically if the properties/ features of a substance are represented, they are it's dimensions. The higher the dimensions, the more qualitative/richer the substance is. This richerness depends on what each dimension represent and how many dimensions we have. For us living orgs, mother Nature had filled this richness, via millions of years of evolution steps ;).
Likewise to capture the richness of every single word, we associate them with a higher number of dimensions. These dimensions are nothing but randomly initialized numbers at the beginning of the training, which will be determined by Language Model. The more the LM learn these features/richness well. The better. 

Once we know how to represent a tokens richly, we end up with something called as a Token Embedding Matrix. It stores embeddings for all the tokens in a vocabulary/dataset. These embeddings are tuned via the language model over time. 

---

# Page 3

**Know When To Train.**
**Date: / /**

- **Middle ground is:**
    - **Subword Tokenization:** Here we tokenize
        a) Character ??
        b) Words
        c) Subwords : token izaton
    - Cause of subwords no. of unique words don't stay that huge.

    - This is optimal & we use **byte pair encoding (BPE)** algorithm that breaks huge chunk of text into tokenize i.e. subwords.

    - **BPE:** Merges the commonly occurring bytes together until you reach a prescribed vocab size.
    - See Video.

- **BPE is the tokenizer in tutorial & was used by GPT 2.**
    Dataset $\rightarrow$ Tokenizer $\rightarrow$ Tokens having a Token ID.

- **2 million Training Stories.**
  **20K Validation Stories.**

- **After tokenization**
    Story 1 $\rightarrow$ Tokenizer $\rightarrow$ ids: [1, 11, 55 ...] len 3.
    Story n $\rightarrow$ -||- $\rightarrow$ -||-

- **We store tokens into .bin file to disk to use later.**
    - Fast data loading, - Avoids RAM overload, - No need to re-tokenize, - This format is easy to use.

---

# Page 4

**Date: / /**

- **We create a .bin file & a memory mapped array.**
    * - Use `np.memmap` means the file is backed on disk & looks like numpy array.
      - You can write to it chunk by chunk without holding everything in RAM.

- **We split each training set into batches & put it on disk.**
    We have `<file_split_name>.bin`.

    - We do batches for faster iteration.
    - Add more reason ??

    [Tokens]
    $\downarrow$ Batch
    [ ][ ][ ][ ] $\rightarrow$ All token ids are collected here.
    $\downarrow$ Put to disk

- **If 1 word is 1 token**
    Then Train.bin holds 100M tokens
    Val.bin holds 10M tokens.

- **Add Code Snip.**



- Once we have selected what domain specific data we have to use to create a specialized small model for a domain, we need to define the model architecture that will learn from this data to ideally output coherent language, having similar patterns to the input data. 
    > IMG: Add the image of Blocks of Transformer. 

- There are 3 main blocks in the Model Architecture: 
    - Input Block: This block processes the data before feeding it to the model.
    - Processing Block: The essential part which learns the form and meaning from the dataset.
    - Output Block: This block serves the purpose of doing inference/predictions and penalizing the model via loss function.



## Input Block:
- Input block is responsible for the data processing, before it get's feeded to the Transformer block. 
- For large language models (LLMs), we have terrabytes of data of text used in the training process, however for a small language model training process, we rarely need that amount of data. 
- For SLMs we need focused data, i.e. related to one specific thing, this allows the model to be really good at doing one thing.
- Following processes occur in the Input block: 
    - Tokenization: Going from big blobs of text to a unit of text easy to work with
    - I/P and O/P pairs generation: Here we will know, why the LMs are self-supervised training, why they are called auto-regressive, what do we mean by context size of models and how is it used to process data
    - 

The first part of pre-processing the dataset is 


---

# Page 5

**Date: / /**

### I/P, O/P Pairs Creation.

- **Loss function for training language model is based on I/P & O/P pairs.**
    - Same like regression where we have labels.
    - But in language modelling task there is no label so we create our own output.

- **Purpose of Language Models: Next Token Prediction task.**
    (I/P seq) $\rightarrow$ [LM] $\rightarrow$ Next token.
    Fuel $\rightarrow$ Engine $\rightarrow$ Move Forward.

    - Beauty is the LLM magically figure out form & meaning from the training.

- **Language models have 2 hyperparams with other:**
    - **Max context size:** Length of tokens language models look at 1 time before predicting next token.
        - 4 for dummy example.

    - **Divide entire data into chunks based on context size.**

**Eg:** One day a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.

* When we create dataset model never looks at words but the token ids inside the chunks.

    - **Batch size:** We process I/P & O/P pairs into batches. Batches helps to go through data quickly, update params more frequently.

---

# Page 6

**Date: / /**

**Now we get: I/P & O/P matrices.**

$x_1$ [ 1  11  15  24 ]  $\rightarrow$ Batch size = 4
$x_2$ [ 1  13  14  17 ]  $\rightarrow$ I/P Pair i.e. Input tensor
$x_3$ [ 6   8   9  18 ]
$x_4$ [ 1   7   6   7 ]
      (Context size $\rightarrow$ 4)

$y_1$ [ 11  15  24  11 ]  $\rightarrow$ O/P pair i.e. Output tensor
$y_2$ [ 13  14  17   6 ]
$y_3$ [  8   9  18   1 ]
$y_4$ [  7   6   7  59 ]

- **Here O/P is just I/P shifted to right by 1 token.**

**Eg: For here:**
$x_1$ { One, day, a, little, girl } $\rightarrow$ I/P pair.
$y_1$ { day, a, little, girl, named } $\rightarrow$ O/P pair.

- **Now critical part is in an I/P O/P pair.**
  There are 4 prediction Tasks.
  I/P $\rightarrow$ Target
  One $\rightarrow$ day
  One day $\rightarrow$ a
  One day a $\rightarrow$ little
  One day a little $\rightarrow$ girl

---

# Page 7

**Date: / /**

- **As this is prediction of next token the language models are also called auto regressive models.**
- **As the label are curated automatically they are called Self Supervised Trained.**

- **Summary in brief**
    a) Dataset $\rightarrow$ Tokenize
    b) Decide context & batch size to iterate over tokens.
    c) Create I/P chunks by sliding window of context size over tokens.
    d) The O/P for the I/P is another token chunk shifted to right by 1 token.
    e) For every I/P & O/P pair we create model's I/P & pred. ground truth based on next token ideology.
       So for n-context size the model sees n-prediction task for a single I/P $\leftrightarrow$ O/P pair.

    i.e. I/P $\rightarrow$ Target for $x_1 \leftrightarrow y_1$
    One $\rightarrow$ day
    One day $\rightarrow$ a
    One day a $\rightarrow$ little
    One day a little $\rightarrow$ girl

    g) The model iterates over batch size of 4 in our case all such next token prediction task over all I/P - O/P pairs for a given batch size i.e. 4 in our case.

    h) The training objective is as simple as this next token prediction.

- **Add Code Snippet (53:57)**

---

# Page 8

**Date: / /**

### Going through processing blocks: Token Embed

- I/P $x_1 = [ 1, 11, 15, 24 ]$
                $\uparrow$  $\uparrow$  $\uparrow$  $\uparrow$
               One day  a  little.

- As language has some meaning, we need to capture this info some how.
- For simplicity lets take "meaning" as nothing but more aspects / features of something. Think of it as:

**Eg: little girl named Lily.**
- Every single word has meaning.
- Every single word has semantics / features.
- Mathematically these features are represented as **embedding**.
    $\rightarrow$ A set of numbers learnt by language model, so that it maps to form & meaning of each word.

- In our case for language model we represent tokens via embeddings. The model has to figure out what these embeddings could be so it captures meaning & form for every token.
- As devs we tune how many numbers can the model choose to represent a token.

---

# Page 9

**Date: / /**

**i.e. called as embedding size.**

- In our example we use **embedding size of 768**.
- This is also necessary as simply token ids don't capture token essence / features of language.
    Eg. Cat & Kitten even though similar could have very different token Ids.

- We use an **token embedding matrix** to keep the essence / capture the essence of every individual token.

| Token Ids | $\leftarrow$ 768 $\rightarrow$ | |
| :--- | :--- | :--- |
| 1 | [ ] | One |
| 2 | [ ] | day |
| : | [ ] | a |
| 50,000 | [ ] | little |

- **Token Embedding Matrix just serves as a lookup table.**

- We initialize the token embedding matrix randomly. These are trainable params for the model.
    - We have roughly `vocab size * Embed size` added to model because of TEM i.e. token embedding matrix.

### Data Processors Position Embedding.

- The dog chased the ball.
- It could not catch it.

- Now in the above sentence, how does the model know what is the use of 'it'. Is 'it' referencing dog or ball.

---

# Page 10

**Date: / /**

- To capture this positional essence of language we use another feature representation of tokens in **position embedding matrix**.
- For each token we maintain its positional embeddings.
- In all now we have 2 matrices for tokens.

| Tids | TEM | Context size | PEM |
| :--- | :--- | :--- | :--- |
| 1 | [ ][ ][ ][ ] | 1 | [ ][ ][ ][ ] |
| 2 | -||- | 2 | -||- |
| 3 | -||- | 3 | -||- |
| : | : | : | : |
| 50257 | | 1024 | |

**& also the token-id mapping.**

- The goal of **PEM** is to capture at which position my token comes in Input sequence.
- Since the number of position is restricted by context size.
- PEM has size of (1024, 768) when iterating over an I/P, O/P pair.

- **Now for each token we simply add up the TEM token embedding & Positional embedding to get an Input embedding.**
- **Add Code Snip.**

---

# Page 11

**Date: / /**

### Model Architecture.
### Layer Norm.

- Once we have input embedding lets see the model where all magick happens.

**Input Embeddings**
$\leftarrow$ 768 $\rightarrow$
[ ][ ][ ][ ]  One
[ -||- ]  day
[ -||- ]  a
[ : ]  little girl

- **We process the input embeddings through Layer Normalization.**
    - This helps to stabilize training by stopping gradient explosion.
    - Used to stabilize backpropagation during training.
    - Basically backprop depends on gradients which depend on the O/P of prev. layer in N.N.
    - If the O/P are not of uniform scale & fluctuates bet^n large & small values then the gradient can either stagnate or explode.
    - To avoid this unstable training dynamics, we do layer normalization.
    - Also the O/P distr^n from each layer could change so we normalize it by subtracting the mean from I/P & dividing by standard deviation.
    - This shift of distr^n problem is also called **internal Covariant Shift.**

---

# Page 12

**Date: / /**

- Layer normalization improves training performance.

### Attention Block

- This is where the core of magic in LLM lives. This block is the one that captures the form & meaning of language.

- $\rightarrow$ ! The thing with attention block is similar to embeddings. We begin with huge set of matrices filled with random numbers & tune these numbers over whole data hoping it will capture the essence i.e. form & meaning of the language.

- Lets go to the prev. example.
  The dog chased the ball, it could not catch it.
  - Here first it refers to dog & 2nd one to ball.
  - Now if we plan to predict next token for:
    It $\rightarrow$ how what to write here

---

# Page 13

**Date: / /**

**It barked or It flew...**

- This attention, i.e. how each token relates/attends or gives attention to tokens is decided by the attention block.
- The relation information is captured by attention block's attention mechanism.

- Basically this mechanism converts the input embedding vector to **context vector**, which will be used for loss calculation.

IMG $\rightarrow$ It (768) - Context Vector
$\uparrow$
IMG $\rightarrow$ It (768) - I/P embedding Vector.
Scales to context vector.

$\alpha_{21} \leftarrow \alpha_{22} \rightarrow \alpha_{23} \rightarrow \alpha_{24} \rightarrow \alpha_{25}$
**Eg:** The next day is bright.
$i_1$ $i_2$ $i_3$ $i_4$ $i_5$

next $\Rightarrow$ $i^1_2 = \alpha_{21} \times i_1 + \alpha_{22} \times i_2 + \alpha_{23} \times i_3 + \alpha_{24} \times i_4 + \alpha_{25} \times i_5$

$\alpha_{ij}$ = attention scores.
$i_n$ = I/P embeddings.

$i^1_2$ is the context vector.

---

# Page 14

- This attention block is where words start to understand each others form & meaning.
- Without this block LLM doesn't work.

**Girl** $\rightarrow$ [ 1 ][ 1 ][ 1 ][ 1 ] $\rightarrow$ **Single Masked Multi-Head Self Attention**
$\leftarrow$ 768 $\rightarrow$
(I/P embedding)

$\downarrow$

[ ][ ][ ][ ] Context Vector
$\leftarrow$ 768 $\rightarrow$

- Embedding size doesn't change for tokens until we reach to the very last O/P layer.
- The way we go from I/P vectors to context vectors is by making use of **Queries, Key & Value matrices weight Matrices.**

- Briefly we take I/P vector embeddings. (4 * 768)
    - Then we multiply them with Weight query matrix i.e. $W_Q$ of dim (768 * 768) & get a Query Matrix of dim (4 * 768)
    - Similarly we multiply I/P by Key Weight Matrix & get Key matrix of dim (4 * 768).
    - Lastly likewise we have value matrix.

---

# Page 15

**Date: / /**

$i_1 \xrightarrow{W_Q (768 * 768)} Q (4 * 768)$
$i_2 \xrightarrow{W_K} K (4 * 768)$
$i_3 \xrightarrow{W_V} V (4 * 768)$
$i_4$

$\rightarrow Q * K^T (4 * 4)$ (Attention Scores)

- Here $W_Q, W_K$ & $W_V$ weight matrices are trainable.
- Once we have attention scores they are normalized by $sqrt(d_{keys})$, then we apply softmax to get Attention weights.
- The attention weights are multiplied by $V$ to get Context vectors of dim (4 * 768)

- **Another thing to note here is causality.**
    - Basically each token should only know/attend to tokens before it & not after it.
    - So the Attention scores matrix, we need to mask all the scores for tokens that come after a token.

| | One | day | a | little |
| :--- | :--- | :--- | :--- | :--- |
| One | / | | | |
| day | / | / | | |
| a | / | / | / | |
| little | / | / | / | / |

**IMG: Causal attention scores.**
**Masked to ensure causality in attention block. So every token only sees attention scores of tokens that come before it.**

- This avoids model peeking into future for next token prediction.

---

# Page 16

**Why Softmax ?? Why Cross-Entropy Loss?**
**In general any MLE name??**
**Date: / /**

- So we know now why **causal & masked** is a word for the block name. Also the word **self** is used because each token attends to other tokens in its **own sentence**.
- Add of small N.N block after attention head.
- **Code Snip.**

- Regarding masking process, we simply set values above diagonal to $-inf$, so when we take softmax they go to 0.
- The attention scores are divided by $sqrt$ of Key dims to keep the variance occurring in $Q * K^T$'s dot product should stay as low as possible.
- Then we fill elements above diagonal are replaced with $-\infty$.
- Then we take softmax of $Q * K^T$.
    $\rightarrow$ This masks token future tokens wrt a token.
    $\rightarrow$ Ensures the sum of prob. for tokens to be predicted is 1.
- After these operations we can quantitatively estimate how much attention a token has to pay to tokens prev. tokens.

---

# Page 17

- The trainable matrices are added in hope the math does the magic & learns the form & essence of language.

### Dropout Block.

- We add a dropout block to remove the lazy neuron problem from the N.N. & improve generalization performance.

### Shortcut Connection alternate.

- It is added to give another path to flow the gradients & prevent the vanishing gradient problem, when multiple layers are chained together.
- It just means I/P before all the layers occurring before shortcut connection is added to to the O/P produced just before by last layer before shortcut connection.
- In our case O/P of dropout is added to I/P before layer norm 1.
  Likewise I/P before layer norm 2 is added to O/P produced by 2nd dropout at the end of transformer block.

### Feed Forward N.N.

- After shortcut connection 1, we again normalize the vectors.
- Then we have input for feed forward N.N.

---

# Page 18

**Size of F.F. N.N.**
- The feed forward N.N. is again a deep learny thing, basically we project the dimension to higher dimension space & hope that it captures the problem & solves it.
- In our dummy example.
  the I/P dim to Feed forward N.N are 4 * 768 $\xrightarrow{768 * 3072}$ 4 * 3072 dims $\rightarrow$ 4 * 768.
  (4 * 768 $\rightarrow$ 3072 dims)

    - This is expansion compression N.N. which retains the size of I/P but captures richer meaning, more non-linearities.

- It turns out that this layer is crucial in transformer block & without this llm fails to capture context patterns in underlying data, & cannot answer queries that well.
- This block shifts performance to new levels.

* The act^n function for this is **GeLU**.
    - It is diff. before 0.
    - Its +ve but not y=x for +ve inputs.
    - Generally gives good results.

---

# Page 19

**What are logits actually??**
**Date: / /**

- After this we again do dropout for stability.
- **Code Snip.**

** Now that's the end of 1 transformer block. We put such n transformer block chain sequentially & finally get an I/P logits of the O/P.
    - The O/P has the same dim as the I/P.

### Output block of Model.

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