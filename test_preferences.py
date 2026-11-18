from recommender import (
    load_movies,
    load_embeddings
)

from preference_model import (
    train_preference_model,
    score_movies
)


movies = load_movies()

embeddings, embedding_index = (
    load_embeddings()
)


def get_movie_id(title):
    match = movies[
        movies["title"].str.lower()
        == title.lower()
    ]

    if match.empty:
        raise ValueError(
            f"Movie not found: {title}"
        )

    return int(match.iloc[0]["id"])


# Simulate some user choices.
history = [
    {
        "winner_id": get_movie_id(
            "The Notebook"
        ),
        "loser_id": get_movie_id(
            "The Dark Knight"
        )
    },
    {
        "winner_id": get_movie_id(
            "Iron Man"
        ),
        "loser_id": get_movie_id(
            "Titanic"
        )
    },
    {
        "winner_id": get_movie_id(
            "Inception"
        ),
        "loser_id": get_movie_id(
            "The Holiday"
        )
    }
]


model = train_preference_model(
    history,
    embeddings,
    embedding_index
)


scores = score_movies(
    movies["id"].tolist(),
    model,
    embeddings,
    embedding_index
)


print("\nTop recommendations:\n")


for result in scores[:10]:

    movie = movies[
        movies["id"] == result["id"]
    ].iloc[0]

    print(
        f"{movie['title']} "
        f"- Score: {result['score']:.3f}"
    )