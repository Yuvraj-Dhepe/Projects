# Diffusion vs. Auto-Regressive Models: A First-Principles View

## Contents

- Introduction to types of language models
- Generative AI through a probabilistic lens
- How diffusion models work for image generation
- Motivation behind language diffusion models
- Benefits and trade-offs
- Extra notes (practical LLM usage)
- Open questions
- Resources

## Introduction to Types of Language Models

Broadly, there are two dominant ways modern AI systems generate data:

1. **Autoregressive Models**
   Models like GPT, Gemini, and LLaMA generate content sequentially — one token at a time — where each new token depends on all previous tokens.
2. **Diffusion Models**
   These dominate image and video generation today (Stable Diffusion, Midjourney) and are now beginning to appear in text generation as well (for example, Mercury from Inception Labs).

Although both families often use similar **Transformer-based backbones**, the real difference lies not in architecture but in **how they model and sample probability distributions**.

## Generative AI Through a Probabilistic Lens

What does it actually mean to “generate” an image?

### Pixel Space as a Probability Landscape

An image is fundamentally a very large vector of pixel values. If we *conceptually* imagine plotting every possible image in a high-dimensional **pixel space**, most of this space corresponds to meaningless noise.

Now imagine this pixel space **annotated by a probability density**:

- Most regions have almost zero probability.
- A few regions form **high-probability pockets**.
- Each pocket corresponds to a recognizable class: pandas, cats, dogs, and so on.

In this sense, we can think of **pixel space as a function of probability**. Sampling from a high-probability pocket yields realistic images; sampling near the edges yields distorted or uncanny ones.

So when a generative model produces an image, what it is really doing is **sampling from a learned probability distribution over pixel space**.

### The Rain Example: Two Meanings of a Probability Distribution

Consider a probability distribution over days of the week representing when it rains most:

| Day          | Probability of Rain |
| --- | --- |
| Mon–Tue      | Low                  |
| **Wed–Thu** | **High**            |
| Fri          | Low                  |
| **Sat–Sun** | **High**            |

This distribution carries **two tightly connected meanings**.

**Inherent (semantic) meaning:**
The distribution represents *which days it rains more often*.

**Mathematical meaning:**
When we sample from the distribution, days with higher probability density appear more frequently.

In practice, both meanings collapse into the same outcome: if you sample repeatedly, Wednesday, Thursday, Saturday, and Sunday show up most often.

Generative models work in exactly the same way — except instead of seven days, the distribution spans millions or billions of dimensions.

## How Diffusion Models Work for Image Generation

Let’s return to the **pixel space** idea.

Earlier, we said meaningful images form high-probability pockets in pixel space. Diffusion models are designed to **learn how to move through this space**.

Diffusion is inspired by a physical process: how structure slowly turns into randomness, like ink dispersing in water.

### Forward Process: Controlled Destruction

- Start with a clean image.
- Add a very small amount of noise.
- Repeat this gradually over many steps.
- Eventually, the image becomes nearly indistinguishable from pure noise.

This forward noising process is simple and fixed — we fully control how noise is added.

### Learning the Reverse Process

Although the forward process destroys information, diffusion models are trained to **learn a statistical reverse process**.

At different noise levels, the model is shown a noisy image and trained to answer a question like:

> “Given this corrupted image, what noise is likely present here?”

In practice, the model often learns to **predict the noise component** added at that step. A loss is computed by comparing the predicted noise to the true noise used during corruption.

Importantly, the model does not learn to perfectly undo noise in a single step. Instead, it learns how to **incrementally move a sample toward higher-probability regions of pixel space**.

### Sampling: From Noise Back to Images

Once training is complete:

- We start from pure noise.
- The model predicts the noise present at the current step.
- This prediction is used to partially remove noise and refine the image.
- The process is repeated many times.

Step by step, the sample drifts through pixel space toward the high-probability pockets discussed earlier — cats, dogs, pandas — until a coherent image emerges.

From a probabilistic perspective, the model learns an approximation ( p'(x) ) of the true data distribution ( p(x) ), and generation is simply sampling from that learned distribution.

## Motivation Behind Language Diffusion Models

Most modern language models are **autoregressive**. They generate text strictly left-to-right, which introduces some natural limitations:

- Earlier tokens cannot be revised easily.
- Global sentence structure emerges late.
- Generation is inherently sequential.

Language diffusion models explore an alternative.

Instead of generating the first word and moving forward, they start from a **noisy version of an entire sentence** and refine all tokens together.

One important difference from images is that **text is discrete**, while diffusion relies on continuity. As a result, language diffusion models operate over **continuous representations of text** (such as embeddings or latent spaces), not raw tokens directly.

The backbone is still typically a Transformer — but the **generation strategy is fundamentally different**.

## Benefits and Trade-offs

### Pros

- **Parallel Refinement**
  Unlike autoregressive models, diffusion-based approaches can refine all tokens simultaneously rather than token-by-token.
- **Potential Inference Efficiency**
  Some diffusion setups reduce memory pressure during inference by avoiding long sequential dependency chains.

- **Training Cost**
  Training diffusion models is computationally expensive due to repeated noising and denoising steps.
- **Sampling Complexity**
  Even with parallelism, many refinement steps are required to converge from noise to a clean output.
- **Early Stage for Language**
  Diffusion-based language models are still experimental and do not yet consistently outperform strong autoregressive baselines.

For most real-world applications, **training a large language model from scratch should be the last resort**. In practice, the typical order looks like:

1. **Few-Shot Prompting** — Provide examples directly in the prompt.
2. **RAG** — Add real-world or up-to-date knowledge via retrieval.
3. **Fine-Tuning** — Shape the model’s style, tone, or niche behavior.
4. **Training from Scratch** — Only when massive data and a clear need exist.

## Open Questions / Next Directions

The discussion above intentionally stays at a first-principles level. Naturally, this raises deeper questions:

- What are the detailed steps that take a sample from pure noise to a specific image class?
- How is text incorporated when training image diffusion models?

These topics deserve deeper treatment on their own.

## Resources

**Diffusion Models (Core Concepts)**

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233)
- [Diffusion Models (video)](https://youtu.be/iv-5mZ_9CPY?si=oJiqCYgjHPuFiSKy)
- [Diffusion Principles (playlist)](https://www.youtube.com/playlist?list=PLPTV0NXA_ZShSUHKLhdHAdAw5oM9vM1Yf)

**Language Diffusion Models**

- [Large Language Diffusion Models](https://arxiv.org/pdf/2502.09992)
- [Dream7B](https://hkunlp.github.io/blog/2025/dream/)

**LLMs & Training Practices**

- [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)
- [SLM From Scratch (video)](https://www.youtube.com/watch?v=pOFcwcwtv3k)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759)

### Closing Note

This post aims to explain diffusion and autoregressive models **from first principles**, without heavy math or implementation details. The goal is intuition — understanding *what distributions are being learned* and *how models move through them* — before worrying about optimizations or benchmarks.