from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from recommender import load_movies


DATA_DIR = Path(__file__).parent / "data"

EMBEDDING_PATH = DATA_DIR / "movie_embeddings.npy"
INDEX_PATH = DATA_DIR / "movie_embedding_index.csv"


def create_embeddings():
    # Load our cleaned movie data.
    movies = load_movies().reset_index(drop=True)

    print(f"Loaded {len(movies)} movies.")

    # Load a pretrained language model.
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Get all movie plot descriptions.
    plots = movies["overview"].fillna("").tolist()

    print("Creating plot embeddings...")

    embeddings = model.encode(
        plots,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save the numerical embeddings.
    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    # Save which movie corresponds to each embedding row.
    movies[
        ["id", "title"]
    ].to_csv(
        INDEX_PATH,
        index=False
    )

    print("Finished creating embeddings.")
    print(f"Embedding matrix shape: {embeddings.shape}")

    print(
        f"Saved embeddings to: {EMBEDDING_PATH}"
    )

    print(
        f"Saved index to: {INDEX_PATH}"
    )


if __name__ == "__main__":
    create_embeddings()