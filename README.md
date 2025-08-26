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


## Notebooks Overview

Below is a summary of each notebook, including key steps, topics covered, and conclusions. Each notebook includes code implementations, explanations, and visualizations for clarity.

### Part-1_Micrograd_and_Backpropagation / Building_Micrograd.ipynb

 
This notebook implements a miniature deep learning framework from scratch, focusing on automatic differentiation and backpropagation. It starts with basic derivative approximations and builds up to neurons, layers, and full MLP models trained via gradient descent. Computation graphs are visualized to understand forward and backward passes.


### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-1.ipynb

 
This notebook trains a bigram character-level language model. It calculates character combination counts, normalizes them into a probability tensor, and uses this for sampling new words (e.g., names). Model quality is evaluated using negative log-likelihood loss.


### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-2.ipynb

 
This notebook explores the bigram model's probability distributions and visualizations, highlighting patterns in names. It discusses the model's strengths in local dependencies and limitations in longer contexts.

### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-3.ipynb

 
This notebook builds a bigram character-level language model for generating names. It preprocesses a dataset, creates mappings, constructs a count tensor, normalizes to probabilities, samples new names, and evaluates with negative log-likelihood.


### Part-2_Stats_Modeling_and_Neural_Network / MakeMore_Part-4_Neural_Network_Approach.ipynb

 
This notebook compares a count-based bigram model with a neural network approach using a single layer trained via negative log-likelihood. It covers one-hot encoding, forward passes, loss calculation, gradient descent, regularization, and sampling.


### Part-3_Training_Names_Using_MLP / MLP.ipynb

 
This notebook extends bigram models to MLPs with embeddings and hidden layers for richer context. It uses a context window, tanh activations, cross-entropy loss, mini-batches, and learning rate scheduling.



### Part-3_Training_Names_Using_MLP / MLP_Part-B_Activations_&_Gradients,_BatchNorm.ipynb

 
This notebook discusses activations, gradients, weight initialization, and normalization in deep networks. It covers Batch Normalization (BN), its implementation, running statistics, and alternatives.



## Future Work

This repo lays the groundwork for transformers. Next steps could include adding attention mechanisms and scaling to full transformer architectures.

## Contact

Feel free to reach out for discussions or collaborations! LinkedIn: [Your LinkedIn] | Email: [Your Email]