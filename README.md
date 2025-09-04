# Large Language Model

Welcome to my hands-on exploration of **Neural Networks, Transformers, and Generative AI trained on Natural Language Processing (NLP)**.  

This repository documents my step-by-step implementation of core deep learning concepts in NLP, from scratch micrograd engines to MLPs, and finally toward transformer-based **Natural Language Models**.  

The focus throughout is on building **language models** that can learn from text data, generate new sequences, and provide insights into how modern Large Language Models (LLMs) evolve from simple statistical methods.  

---

## Notebook Summaries

### Part 1: Micrograd and Backpropagation
**Notebook:** Building_Micrograd.ipynb

- Built a mini deep learning framework from scratch.  
- Implemented automatic differentiation for efficient gradient computation.  
- Trained neurons, layers, and MLPs using gradient descent.  
- Visualized computation graphs to understand forward and backward passes.  

We recreated the essence of frameworks like PyTorch in just a few hundred lines of code, gaining intuition about backpropagation and neural network fundamentals,  the foundation for training modern **language models**.

---

### Part 2: Statistical Modeling and Neural Networks

**1. MakeMore_Part-1.ipynb**  
- Built a Bigram character-level **language model**.  
- Learned probabilities from character pair counts.  
- Measured performance using negative log-likelihood (NLL).  
- Generated new names using sampling.  

This statistical **Natural Language Model** highlights how likelihood maximization relates to probability-based text generation.

**2. MakeMore_Part-2.ipynb**  
- Explored the strengths and weaknesses of bigram models.  
- Found that while they capture local dependencies, they fail at modeling longer context.  

Bigrams provide insights but struggle to generate realistic names due to limited context,  showing the need for more advanced **language models**.

**3. MakeMore_Part-3.ipynb**  
- Built mappings for characters to indices.  
- Constructed bigram probability matrices.  
- Implemented multinomial sampling to generate names.  

This statistical **language model** demonstrates how sequence patterns are captured, forming the foundation of modern **Natural Language Models**.

**4. MakeMore_Part-4_Neural_Network_Approach.ipynb**  
- Re-implemented the bigram model using a single-layer neural network.  
- Compared count-based vs neural net-based approaches.  
- Introduced gradient descent optimization and regularization.  
- Generated names using a single-layer neural network **language model**.  

Both count-based and neural models yield similar results, but neural networks provide flexibility for scaling to deeper architectures that form the basis of Large Language Models.

---

### Part 3: Training Names Using MLP

**1. MLP.ipynb**  
- Extended the bigram model into a **Multi-Layer Perceptron Language Model**.  
- Used embeddings + hidden layers to capture richer context.  
- Trained with cross-entropy loss and gradient descent.  
- Experimented with mini-batching and learning rate schedules.  
- Generated names using the MLP with a context size of 3.  

MLPs improve modeling capacity for **Natural Language Models** but require careful initialization, optimization, and normalization to stabilize training.

**2. MLP Part-B: Activations, Gradients, and BatchNorm**  
- Investigated activation distributions and gradient flow.  
- Studied how poor scaling leads to unstable training.  
- Introduced Batch Normalization (BN) to stabilize deep models.  
- Discussed drawbacks and alternatives like LayerNorm and GroupNorm.  
- Generated more meaningful names after effective optimization.  

Controlling activations and gradients is crucial for stable deep learning, and BN was a breakthrough that made very deep **language models** feasible.

---
### Part 4: WaveNet

**1. WaveNet.ipynb**  
- Implemented a simplified version of **WaveNet**, originally designed for audio generation.  
- Adapted it for **character-level text generation**.  
- Used causal convolutions to ensure autoregressive behavior.  
- Demonstrated how the model can generate sequential text character by character.  
- Visualized the model architecture and data flow.

WaveNet is powerful for sequential data, and its use of dilation and causal structure allows it to model long-range dependencies effectively without attention.

---

### Part 5: GPT (Generative Pre-trained Transformer)

**1. Generative Pre-trained Transformer Part-1.ipynb**  
- Step 1: **Data Preprocessing** and building a **Bigram Model**.  
- Encoded raw text data into token IDs.  
- Implemented a basic bigram model to predict the next character using the current one.  
- Visualized how the model learns to generate character-level text with minimal context.  
- Highlighted the limitations of short context windows.

**2. Generative Pre-trained Transformer Part-2.ipynb**  
- Step 2: Full **Transformer Architecture** implementation.  
- Added token + positional embeddings, multi-head self-attention, feedforward layers, and normalization.  
- Stacked multiple transformer blocks, trained with cross-entropy loss.  
- Generated text using the trained GPT-like model.  
- Achieved better quality and longer coherent text generation due to longer context and attention.

The Transformer architecture marks a major leap in NLP, enabling models to learn complex patterns with scalability and parallelism.

---
## Key Learnings

- **Backpropagation** is just systematic application of the **chain rule**, enabling deep neural networks to learn from data.
- **Language modeling** has evolved from:
  - Count-based models →  
  - Neural networks (MLPs) →  
  - Convolutional models (WaveNet) →  
  - Attention-based models (Transformers and GPTs).
- **MLPs** improve contextual understanding but require careful tuning (initialization, learning rate, normalization).
- **WaveNet** shows how causal convolutions can model sequential data effectively without attention.
- **Transformers** revolutionized NLP by modeling long-range dependencies using **self-attention**, enabling the training of large-scale **LLMs** like GPT-3 and ChatGPT.
- **Normalization techniques** like **BatchNorm** and **LayerNorm** are essential for stable training and scaling deep networks.
- The journey from **bigram models** to **transformers** illustrates how traditional NLP connects with modern **Large Language Models (LLMs)** in a unified deep learning framework.

---

##  References

- Rumelhart, Hinton, and Williams (1986). *Learning representations by back-propagating errors*. [Nature](https://www.nature.com/articles/323533a0)  
- Bengio et al. (2003). *A Neural Probabilistic Language Model*. [JMLR](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  
- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. [arXiv](https://arxiv.org/abs/1502.03167)  
- Vaswani et al. (2017). *Attention Is All You Need* (Transformer Paper). [arXiv](https://arxiv.org/abs/1706.03762)  
- Oord et al. (2016). *WaveNet: A Generative Model for Raw Audio*. [DeepMind Blog](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)  
- Radford et al. (2018–2020). *GPT, GPT-2, GPT-3 Papers*. [OpenAI](https://openai.com/research)
- **Andrej Karpathy (2023)** — *Zero to Hero* series:   
  GitHub: [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)  
  YouTube: [Zero to Hero Playlist](https://www.youtube.com/playlist?list=PLpOqH6AE0tNkxqvZbw-QgU3DLm6TLb1rx)

---

##  Closing Note

This repository is not just a collection of code, but a **language modeling learning journey** — a step-by-step path I’ve taken to understand and implement the building blocks of modern **Large Language Models (LLMs)**.

Throughout this journey, I’ve:

- Learned from foundational research papers (like *Attention Is All You Need*, *WaveNet*, *Neural Probabilistic Language Models*).
- Followed and implemented the concepts from **Andrej Karpathy’s Zero to Hero** tutorials.
- Built everything from scratch using **PyTorch**, including:
  - Bigram models  
  - MLPs  
  - WaveNet  
  - Full Transformer architectures (GPT-style)

Each notebook concludes with a clear summary to reinforce intuition, and every step builds toward understanding how models like **ChatGPT** work under the hood.

---

If you're a **recruiter** or **collaborator**, I’d be happy to connect and discuss how I can bring these hands-on **language modeling** skills into real-world, impactful AI projects.

This project reflects my commitment to deep learning, my ability to learn from theory and implement it from scratch, and my passion for building **intelligent systems**.
