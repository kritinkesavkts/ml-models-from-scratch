from __future__ import annotations

import numpy as np

from ml_from_scratch.transformers import (
    GPTStyleDecoder,
    MultiHeadAttention,
    ScaledDotProductAttention,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    causal_mask,
    sinusoidal_positional_encoding,
)


def run_attention_demo() -> None:
    rng = np.random.default_rng(91)
    X = rng.normal(size=(2, 4, 8))

    attention = ScaledDotProductAttention()
    attention_output = attention.forward(X, X, X)

    multi_head = MultiHeadAttention(d_model=8, n_heads=2, random_state=92)
    multi_head_output = multi_head.forward(X)

    print("Attention")
    print(f"  scaled dot-product output shape: {attention_output.shape}")
    print(f"  attention row sums: {attention.last_attention_weights_[0].sum(axis=-1).round(3).tolist()}")
    print(f"  multi-head output shape: {multi_head_output.shape}")


def run_encoder_demo() -> None:
    rng = np.random.default_rng(93)
    X = rng.normal(size=(2, 5, 8))
    X = X + sinusoidal_positional_encoding(sequence_length=5, d_model=8)[None, :, :]

    encoder = TransformerEncoderBlock(d_model=8, n_heads=2, d_ff=16, random_state=94)
    output = encoder.forward(X)

    print("Transformer Encoder")
    print(f"  input shape: {X.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  output mean/std: {output.mean():.3f}, {output.std():.3f}")


def run_decoder_demo() -> None:
    rng = np.random.default_rng(95)
    X = rng.normal(size=(2, 5, 8))

    decoder = TransformerDecoderBlock(d_model=8, n_heads=2, d_ff=16, random_state=96)
    output = decoder.forward(X)

    gpt = GPTStyleDecoder(
        vocab_size=12,
        max_sequence_length=8,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=97,
    )
    prompt = np.array([[1, 2, 3]])
    logits = gpt.forward(prompt)
    generated = gpt.generate(prompt, max_new_tokens=3)

    print("Transformer Decoder / GPT Style")
    print(f"  causal mask:\n{causal_mask(4).astype(int)}")
    print(f"  decoder output shape: {output.shape}")
    print(f"  GPT logits shape: {logits.shape}")
    print(f"  generated token ids: {generated.tolist()}")


if __name__ == "__main__":
    run_attention_demo()
    run_encoder_demo()
    run_decoder_demo()
