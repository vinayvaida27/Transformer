# Transformer Learning Journey

This repository documents my hands-on journey in building foundational deep learning concepts from scratch, starting with basic neural networks and progressing toward more advanced models like Multi-Layer Perceptrons (MLPs). The focus is on character-level language modeling using a dataset of names, inspired by classic papers and tutorials (e.g., Bengio et al., 2003). 

The notebooks demonstrate key techniques such as backpropagation, gradient descent, embeddings, activations, and batch normalization. This project builds intuition for how modern AI frameworks (like PyTorch) work under the hood and sets the stage for understanding transformers.

Key skills showcased:
- Implementing neural networks from scratch using Python and NumPy.
- Statistical modeling, probability distributions, and loss functions.
- Training models with optimization techniques like gradient descent.
- Handling activations, gradients, and normalization for stable training.
- Generating text (e.g., names) using trained models.

This repo is structured progressively across parts, making it easy to follow the evolution from simple bigram models to more complex MLPs.

## Repository Structure

- **Part-1_Micrograd_and_Backpropagation**
  - Building_Micrograd.ipynb

- **Part-2_Stats_Modeling_and_Neural_Network**
  - MakeMore_Part-1.ipynb
  - MakeMore_Part-2.ipynb
  - MakeMore_Part-3.ipynb
  - MakeMore_Part-4_Neural_Network_Approach.ipynb
  - names.txt

- **Part-3_Training_Names_Using_MLP**
  - MLP.ipynb
  - MLP_Part-B_Activations_&_Gradients,_BatchNorm.ipynb
  - names.txt

## Notebooks Overview

Below is a summary of each notebook, including key steps, topics covered, and conclusions. Each notebook includes code implementations, explanations, and visualizations for clarity.

### Part-1_Micrograd_and_Backpropagation / Building_Micrograd.ipynb

**Summary:**  
This notebook implements a miniature deep learning framework from scratch, focusing on automatic differentiation and backpropagation. It starts with basic derivative approximations and builds up to neurons, layers, and full MLP models trained via gradient descent. Computation graphs are visualized to understand forward and backward passes.

**Key Takeaways:**  
- Derivatives guide parameter updates.  
- Backpropagation applies the chain rule systematically through graphs.  
- Neural networks are composed of simple neurons stacked into layers and MLPs.  
- This replicates the core of frameworks like PyTorch in a few hundred lines of code.

**Conclusion:**  
We successfully implemented a miniature deep learning framework from scratch. Starting from basic derivative approximations, we built a system that supports automatic differentiation, enabling us to compute gradients efficiently across complex graphs. Using this foundation, we implemented neurons, layers, and full MLP models, trained them with gradient descent, and verified how parameters update to minimize loss. Visualizing computation graphs gave a deeper understanding of how data flows forward and gradients flow backward. This hands-on implementation builds intuition about how modern AI libraries work, making it easier to debug, optimize, and extend deep learning models in real-world projects.

### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-1.ipynb

**Summary:**  
This notebook trains a bigram character-level language model. It calculates character combination counts, normalizes them into a probability tensor, and uses this for sampling new words (e.g., names). Model quality is evaluated using negative log-likelihood loss.

**Steps Involved:**  
1. Calculated the count of each character combination.  
2. Normalized this count into a tensor to get the probability distribution.  
3. Adjusted the model parameters to perform sampling of new words (e.g., generating new names).  
4. Summarized the model performance/quality in a single number: negative log-likelihood, and calculated the loss for each prediction.  
5. The lower the negative log-likelihood, the better the model is, because it assigns higher probabilities to the actual next characters in all the bigrams of the training set.

**Goal:**  
- Maximize the likelihood of the data with respect to model parameters (statistical modeling).  
- Equivalent to maximizing the log-likelihood (since log is monotonic).  
- Equivalent to minimizing the negative log-likelihood.  
- Equivalent to minimizing the average negative log-likelihood.

**Conclusion:**  
I have trained a successful bigram character-level language model. The model effectively captures basic patterns in names but is limited to short contexts.

### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-2.ipynb

**Summary:**  
This notebook explores the bigram model's probability distributions and visualizations, highlighting patterns in names. It discusses the model's strengths in local dependencies and limitations in longer contexts.

**Conclusion:**  
The bigram model is simple but effective for capturing local character dependencies. It provides insights into which characters are most likely to follow others. The probability distributions can be visualized, giving a clear picture of patterns in names. However, since it only considers two characters at a time, it struggles with longer-term context and generates unrealistic names.

### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-3.ipynb

**Summary:**  
This notebook builds a bigram character-level language model for generating names. It preprocesses a dataset, creates mappings, constructs a count tensor, normalizes to probabilities, samples new names, and evaluates with negative log-likelihood.

**Key Steps:**  
- Read and preprocess a dataset of names.  
- Build a character-to-index (stoi) and index-to-character (itos) mapping.  
- Construct a count tensor that captures how often each bigram appears in the dataset.  
- Normalize this count tensor to form a probability matrix.  
- Use multinomial sampling to generate new names character by character.  
- Calculate the loss using Negative Log-Likelihood (NLL) to evaluate the model on both training data and unseen examples.

**Conclusion:**  
This simple bigram approach provides an introduction to how language models learn sequential patterns.

### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-4_Neural_Network_Approach.ipynb

**Summary:**  
This notebook compares a count-based bigram model with a neural network approach using a single layer trained via negative log-likelihood. It covers one-hot encoding, forward passes, loss calculation, gradient descent, regularization, and sampling.

**Topics Covered:**  
1. Introduction to the bigram language model.  
2. Building the training set of bigrams (xs, ys).  
3. One-hot encoding of inputs.  
4. Neural network forward pass (logits -> exp -> normalize -> probs).  
5. Interpreting probabilities for correct next characters.  
6. Negative Log-Likelihood (NLL) loss calculation.  
7. Gradient descent and weight updates.  
8. Regularization with weight penalties ((W**2).mean()).  
9. Sampling new names from the trained model.  
10. Comparison of count-based vs neural network-based training.

**Conclusion:**  
The bigram language model is a powerful introduction to character-level sequence modeling. By comparing a classical count-based method with a neural network approach, we see that both capture the same statistical patterns. The neural network version provides a more flexible framework, enabling optimization, regularization, and extension to deeper architectures. This exercise demonstrates the continuity between traditional NLP models and modern deep learning approaches. So this is the same model but we approached it differently with a neural network; the output will really look the same because of the input data. We noticed both methods gave the same result. The gradient-based framework is simple; we use only a single layer of neurons (i.e., Xenc * Weights) to calculate the logits. In the future, we will increase the neural network layers, but the normalizing and gradient-based framework remains the same. So we can complexify the neural networks all the way to transformers.

### Part-3_Training_Names_Using_MLP / MLP.ipynb

**Summary:**  
This notebook extends bigram models to MLPs with embeddings and hidden layers for richer context. It uses a context window, tanh activations, cross-entropy loss, mini-batches, and learning rate scheduling.

**Problem with the Bigram Model:**  
- The bigram model captures only 2 characters of context (27 x 27 = 729 combinations).  
- Extending to 3 characters grows to 19,683 combinations, making count-based training difficult.

**Approach:**  
Inspired by Bengio et al., 2003: A Neural Probabilistic Language Model.

**Conclusion:**  
The MLP-based character-level model significantly improves upon bigram models by capturing longer dependencies between characters. We observed that embeddings provide compact representations of characters; the hidden layer learns non-linear transformations, enabling the model to represent complex patterns; using cross-entropy loss ensures stable training compared to manual negative log-likelihood calculations; mini-batches and learning rate tuning are essential for efficient training on larger datasets. However, the model still faces challenges: explosive starting loss, which highlights the need for smaller activation weights at initialization; difficulty in controlling gradients, requiring careful tuning of optimization strategies; potential instability without techniques like batch normalization.

### Part-3_Training_Names_Using_MLP / MLP_Part-B_Activations_&_Gradients,_BatchNorm.ipynb

**Summary:**  
This notebook discusses activations, gradients, weight initialization, and normalization in deep networks. It covers Batch Normalization (BN), its implementation, running statistics, and alternatives.

**Key Points:**  
- Importance of activations and gradients for stable training.  
- Weight initialization for Gaussian activations.  
- Batch Normalization: normalizes batches, adds gain/bias, uses running stats.  
- Drawbacks: couples examples, introduces jitter.  
- Alternatives: Layer Norm, Group Norm, Instance Norm.

**Conclusion:**  
Importance of activations and gradients: in deep networks, it is crucial to understand the statistics of activations and gradients; poorly scaled activations can cause confident mispredictions leading to very high ("hockey stick") losses; controlling activations prevents values from collapsing to zero or exploding to infinity. Weight initialization: at initialization, we want activations to be roughly Gaussian across layers; proper scaling of weights and biases can help maintain stable distributions; this works well for small/medium networks but becomes hard to manage for very deep networks with many types of layers. Normalization layers: to solve scaling problems in deep networks, normalization layers were introduced; Batch Normalization (BN) (introduced ~2015) was the first and most influential; BN normalizes each batch by subtracting its mean and dividing by its standard deviation; it adds trainable gain (γ) and bias (β) parameters so the network can still learn useful transformations. Running statistics: BN also maintains running mean and running standard deviation as buffers; these are not trained by gradients, but updated during training with a running average; at inference, these stored stats are used so single inputs can be processed consistently. Drawbacks of BatchNorm: BN couples training examples within a batch, causing “jitter” and making debugging harder; it has been known to introduce tricky bugs in practice; alternatives like Layer Normalization, Group Normalization, and Instance Normalization are now common. Impact: despite its issues, BN was a breakthrough that made training very deep networks feasible and stable; the key takeaway: controlling activation statistics is essential for good performance in deep learning.

## How to Run

1. Clone the repo: `git clone <repo-url>`.  
2. Install dependencies: `pip install numpy torch` (for advanced parts).  
3. Open notebooks in Jupyter: `jupyter notebook`.  
4. Use `names.txt` as the dataset for training.

## Future Work

This repo lays the groundwork for transformers. Next steps could include adding attention mechanisms and scaling to full transformer architectures.

## Contact

Feel free to reach out for discussions or collaborations! LinkedIn: [Your LinkedIn] | Email: [Your Email]