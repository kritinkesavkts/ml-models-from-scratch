from __future__ import annotations

import numpy as np

from ml_from_scratch.multimodal import CLIPStyleModel


def make_toy_pairs() -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros((3, 4, 4))
    images[0, :2, :2] = 1.0
    images[1, :2, 2:] = 1.0
    images[2, 2:, :] = 1.0
    token_ids = np.array(
        [
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 0],
        ]
    )
    return images, token_ids


def run_multimodal_demo() -> None:
    images, token_ids = make_toy_pairs()
    model = CLIPStyleModel(
        image_shape=(4, 4),
        vocab_size=10,
        embed_dim=6,
        random_state=101,
        temperature=0.1,
    )

    before_similarity = model.similarity(images, token_ids)
    before_retrieval = model.retrieve_text(images, token_ids)
    model.fit(images, token_ids, learning_rate=0.08, n_iterations=600)
    after_similarity = model.similarity(images, token_ids)

    print("Trainable CLIP-Style Image-Text Model")
    print(f"  image embedding shape: {model.encode_image(images).shape}")
    print(f"  text embedding shape: {model.encode_text(token_ids).shape}")
    print(f"  before similarity matrix:\n{before_similarity.round(3)}")
    print(f"  after similarity matrix:\n{after_similarity.round(3)}")
    print(f"  before text retrieval indices: {before_retrieval.tolist()}")
    print(f"  after text retrieval indices: {model.retrieve_text(images, token_ids).tolist()}")
    print(f"  loss improvement: {model.losses_[0]:.3f} -> {model.losses_[-1]:.3f}")
    print(f"  contrastive loss: {model.contrastive_loss(images, token_ids):.3f}")


if __name__ == "__main__":
    run_multimodal_demo()
