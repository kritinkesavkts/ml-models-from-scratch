# ML Models From Scratch

An intuitive, engineering-focused collection of machine learning models implemented from first principles in Python and NumPy.

This repository is designed for a Machine Learning Engineer role portfolio: every model includes readable code, clear math notes, practical examples, and validation tests. The project starts with classical supervised learning and grows toward deep learning, Transformers, Vision Transformers, and multimodal architectures.

## Goals

- Implement core ML models from scratch using minimal dependencies.
- Explain each model in plain language before showing code.
- Provide small, reproducible examples for every implementation.
- Add tests so the repository demonstrates working engineering practice.
- Build a progressive path from linear models to modern Transformer systems.

## Current Contents

```text
ml_from_scratch/
  bayes/
    gaussian_naive_bayes.py
  cluster/
    kmeans.py
  decomposition/
    pca.py
  linear_models/
    linear_regression.py
    logistic_regression.py
  neighbors/
    knn.py
  neural_networks/
    activations.py
    cnn.py
    layers.py
    losses.py
    mlp.py
  sequence_models/
    cells.py
    rnn_classifier.py
  transformers/
    attention.py
    decoder.py
    encoder.py
    positional_encoding.py
  tree_models/
    decision_tree.py
    gradient_boosting.py
    random_forest.py
  utils/
    metrics.py
examples/
  classical_models_demo.py
  linear_and_logistic_demo.py
  cnn_demo.py
  neural_networks_demo.py
  sequence_models_demo.py
  transformers_demo.py
  tree_models_demo.py
docs/
  roadmap.md
tests/
  test_classical_models.py
  test_linear_models.py
  test_cnn.py
  test_neural_networks.py
  test_sequence_models.py
  test_transformers.py
  test_tree_models.py
```

## Implemented Models

| Category | Model | Status |
| --- | --- | --- |
| Supervised Learning | Linear Regression | Implemented |
| Supervised Learning | Logistic Regression | Implemented |
| Supervised Learning | KNN | Implemented |
| Supervised Learning | Naive Bayes | Implemented |
| Tree Models | Decision Tree | Implemented |
| Tree Models | Random Forest | Implemented |
| Boosting | Gradient Boosting | Implemented |
| Unsupervised Learning | K-Means | Implemented |
| Dimensionality Reduction | PCA | Implemented |
| Neural Networks | MLP from Scratch | Implemented |
| Computer Vision | CNN | Implemented |
| Sequence Modeling | RNN, LSTM, GRU | Implemented |
| Attention | Scaled Dot-Product Attention | Implemented |
| Transformers | Encoder, Decoder, GPT-style model | Implemented |
| Vision | Vision Transformer | Planned |
| Multimodal | CLIP-style Image-Text Model | Planned |

## Quick Start

Create a virtual environment, install the package in editable mode, and run tests:

```bash
python -m venv .venv
python -m pip install -e .[dev]
pytest
```

Run the first example:

```bash
python examples/linear_and_logistic_demo.py
python examples/classical_models_demo.py
python examples/tree_models_demo.py
python examples/neural_networks_demo.py
python examples/cnn_demo.py
python examples/sequence_models_demo.py
python examples/transformers_demo.py
```

## Design Philosophy

The code intentionally avoids hidden framework behavior. For example, linear regression is trained with gradient descent instead of calling a closed-form library solver, and logistic regression explicitly computes sigmoid probabilities and binary cross-entropy gradients.

The style favors:

- small classes with predictable APIs
- typed NumPy arrays
- comments only where they clarify math or shape behavior
- tests that verify learning behavior, not only syntax

## Repository Roadmap

The long-term roadmap is documented in [docs/roadmap.md](docs/roadmap.md).

## License

MIT License.
