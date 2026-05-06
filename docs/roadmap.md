# Roadmap

This roadmap builds the repository in a steady engineering progression: begin with small mathematical models, then add models with richer optimization, nonlinear structure, deep learning, attention, and multimodal learning.

## Phase 1: Classical Machine Learning

1. Linear Regression
   - gradient descent
   - mean squared error
   - weight and bias interpretation

2. Logistic Regression
   - sigmoid function
   - binary cross-entropy
   - classification thresholding

3. K-Nearest Neighbors
   - distance metrics
   - classification and regression variants

4. Naive Bayes
   - Gaussian likelihoods
   - class priors
   - log-probability stability

5. Decision Trees
   - entropy
   - information gain
   - recursive splitting

6. Random Forest
   - bootstrap sampling
   - feature subsampling
   - majority voting

7. Gradient Boosting
   - residual learning
   - additive trees
   - learning rate effects

## Phase 2: Unsupervised Learning

1. K-Means
   - centroid updates
   - inertia
   - convergence behavior

2. PCA
   - covariance matrix
   - eigenvectors and eigenvalues
   - dimensionality reduction

## Phase 3: Neural Networks

1. Multilayer Perceptron
   - forward pass
   - backpropagation
   - activation functions

2. Convolutional Neural Network
   - convolution operation
   - pooling
   - image classification demo

3. Recurrent Models
   - RNN cell
   - LSTM
   - GRU

## Phase 4: Attention and Transformers

1. Attention
   - query, key, value projections
   - scaled dot-product attention
   - masking

2. Transformer Encoder
   - multi-head attention
   - feed-forward block
   - layer normalization

3. Transformer Decoder
   - causal masking
   - autoregressive generation

4. GPT-Style Model
   - token embeddings
   - positional embeddings
   - next-token prediction

5. BERT-Style Encoder
   - masked token prediction
   - bidirectional context

## Phase 5: Vision and Multimodal Models

1. Vision Transformer
   - patch embeddings
   - class token
   - image classification head

2. CLIP-Style Model
   - image encoder
   - text encoder
   - contrastive learning objective

3. Multimodal Transformer
   - combined text and image tokens
   - cross-modal attention
   - simple image-text retrieval demo

## Documentation Standard

Each model should include:

- a short intuitive explanation
- mathematical objective
- implementation details
- example script or notebook
- tests validating core behavior
- known limitations and future improvements
