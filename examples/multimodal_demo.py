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


def configure_aligned_toy_model(model: CLIPStyleModel) -> None:
    model.image_projection_[:] = 0.0
    model.token_embeddings_[:] = 0.0
    model.text_projection_[:] = np.eye(model.embed_dim)

    flat_indices = [0, 2, 8]
    for idx, feature in enumerate(flat_indices):
        model.image_projection_[feature, idx] = 1.0

    model.token_embeddings_[1, 0] = 1.0
    model.token_embeddings_[2, 0] = 1.0
    model.token_embeddings_[3, 1] = 1.0
    model.token_embeddings_[4, 1] = 1.0
    model.token_embeddings_[5, 2] = 1.0
    model.token_embeddings_[6, 2] = 1.0


def run_multimodal_demo() -> None:
    images, token_ids = make_toy_pairs()
    model = CLIPStyleModel(
        image_shape=(4, 4),
        vocab_size=10,
        embed_dim=6,
        random_state=101,
        temperature=0.1,
    )
    configure_aligned_toy_model(model)

    similarity = model.similarity(images, token_ids)

    print("CLIP-Style Image-Text Model")
    print(f"  image embedding shape: {model.encode_image(images).shape}")
    print(f"  text embedding shape: {model.encode_text(token_ids).shape}")
    print(f"  similarity matrix:\n{similarity.round(3)}")
    print(f"  text retrieval indices: {model.retrieve_text(images, token_ids).tolist()}")
    print(f"  contrastive loss: {model.contrastive_loss(images, token_ids):.3f}")


if __name__ == "__main__":
    run_multimodal_demo()
