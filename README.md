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

## Key Learnings

- Backpropagation is simply the chain rule applied systematically.  
- **Language models** evolve from simple count-based methods to advanced **Neural and Transformer-based Natural Language Models**.  
- MLPs provide richer context understanding but demand optimization tricks.  
- Normalization layers like BN are essential for scaling deep networks in **language modeling**.  
- The transition from bigram models to neural networks illustrates how traditional NLP connects with modern **Large Language Models (LLMs)**.  

---

## References

- Rumelhart, Hinton, and Williams (1986). *Learning representations by back-propagating errors*. [Nature](https://www.nature.com/articles/323533a0)  
- Bengio et al. (2003). *A Neural Probabilistic Language Model*. [JMLR](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  
- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. [arXiv](https://arxiv.org/abs/1502.03167)  
- Vaswani et al. (2017). *Attention Is All You Need*. [arXiv](https://arxiv.org/abs/1706.03762)  

---

## Next Steps

- Extend MLPs into multi-layer architectures for larger **language models**.  
- Implement self-attention to handle longer dependencies.  
- Progress toward building a **transformer-based Large Language Model** capable of generating more realistic names and text sequences.  

---

## Closing Note

This repository is not just code but a **language modeling journey**  bridging the gap between theory and practice in deep learning.  
Every notebook ends with a clear summary, ensuring that each step builds intuition for the next step toward modern **Large Language Models**.  

If you are a recruiter or collaborator, I’d love to connect and discuss how I can bring these **language modeling** skills into impactful AI projects.  
