from recommender import (
    load_movies,
    load_embeddings,
    find_similar_movies
)


movies = load_movies()

embeddings, embedding_index = load_embeddings()


movie_title = "Avatar"

results = find_similar_movies(
    movie_title,
    movies,
    embeddings,
    embedding_index,
    num_results=5
)


print(f"\nMovies similar to {movie_title}:\n")


for movie in results:

    print(
        f"{movie['title']} "
        f"- Similarity: "
        f"{movie['similarity']:.3f}"
    )

    print(
        f"Genres: {movie['genres']}"
    )

    print(
        f"Overview: {movie['overview']}"
    )

    print()