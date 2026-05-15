from __future__ import annotations

import numpy as np

from ml_from_scratch.transformers import (
    BERTStyleEncoder,
    GPTStyleDecoder,
    MultiHeadAttention,
    ScaledDotProductAttention,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    VisionTransformerClassifier,
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


def run_bert_demo() -> None:
    bert = BERTStyleEncoder(
        vocab_size=20,
        max_sequence_length=8,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=98,
    )
    token_ids = np.array([[1, 2, 19, 4, 5], [6, 7, 8, 19, 10]])
    segment_ids = np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
    mask_positions = np.array([2, 3])
    target_token_ids = np.array([3, 9])

    hidden = bert.forward(token_ids, segment_ids)
    logits = bert.mlm_logits(token_ids, segment_ids)
    predictions = bert.predict_masked(token_ids, mask_positions, segment_ids)
    loss = bert.masked_language_modeling_loss(
        token_ids, mask_positions, target_token_ids, segment_ids
    )

    print("BERT-Style Encoder")
    print(f"  hidden shape: {hidden.shape}")
    print(f"  MLM logits shape: {logits.shape}")
    print(f"  masked predictions: {predictions.tolist()}")
    print(f"  MLM loss: {loss:.3f}")


def run_vision_transformer_demo() -> None:
    rng = np.random.default_rng(98)
    images = rng.normal(size=(3, 8, 8))
    model = VisionTransformerClassifier(
        image_size=8,
        patch_size=2,
        n_classes=3,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        random_state=99,
    )
    patches = model.extract_patches(images)
    tokens = model.embed_patches(images)
    logits = model.forward(images)

    print("Vision Transformer")
    print(f"  patches shape: {patches.shape}")
    print(f"  token sequence shape: {tokens.shape}")
    print(f"  logits shape: {logits.shape}")
    print(f"  predicted classes: {model.predict(images).tolist()}")


if __name__ == "__main__":
    run_attention_demo()
    run_encoder_demo()
    run_decoder_demo()
    run_bert_demo()
    run_vision_transformer_demo()
