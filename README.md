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
  linear_models/
    linear_regression.py
    logistic_regression.py
  utils/
    metrics.py
examples/
  linear_and_logistic_demo.py
docs/
  roadmap.md
tests/
  test_linear_models.py
```

## Implemented Models

| Category | Model | Status |
| --- | --- | --- |
| Supervised Learning | Linear Regression | Implemented |
| Supervised Learning | Logistic Regression | Implemented |
| Supervised Learning | KNN | Planned |
| Supervised Learning | Naive Bayes | Planned |
| Tree Models | Decision Tree | Planned |
| Tree Models | Random Forest | Planned |
| Boosting | Gradient Boosting | Planned |
| Unsupervised Learning | K-Means | Planned |
| Dimensionality Reduction | PCA | Planned |
| Neural Networks | MLP from Scratch | Planned |
| Computer Vision | CNN | Planned |
| Sequence Modeling | RNN, LSTM, GRU | Planned |
| Attention | Scaled Dot-Product Attention | Planned |
| Transformers | Encoder, Decoder, GPT-style model | Planned |
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
