import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bodhiq.utils import get_index


def query_memory(query, tags=None, use_tfidf=False):
    index = get_index(wait=True)

    params = {}
    if tags:
        params["filter"] = " AND ".join([f'tags = "{t}"' for t in tags])

    if use_tfidf:
        # Fetch all documents
        #
        docs = index.get_documents({"limit": 1000}, **params).results
        docs = [dict(doc) for doc in docs]
        if not docs:
            click.echo("❌ No memories found.")
            return

        # Use TF-IDF to find best match
        texts = [doc["text"] for doc in docs]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, X).flatten()

        top_n = min(5, len(docs))
        top_indices = sims.argsort()[::-1][:top_n]

        click.echo(f"💡 Top {top_n} matches (TF-IDF):")
        for rank, idx in enumerate(top_indices, start=1):
            doc = docs[idx]
            click.echo(
                f"{rank}. {doc['text']} [ID={doc['id'][:12]}] (score={sims[idx]:.4f})"
            )

    else:
        # Normal Meilisearch token search

        results = index.search(query, params)
        if not results["hits"]:
            click.echo("❌ No memories found.")
            return

        for hit in results["hits"]:
            tag_str = ", ".join(hit.get("tags", []))
            click.echo(f"- [{hit['id'][:12]}] {hit['text']} ({tag_str})")
