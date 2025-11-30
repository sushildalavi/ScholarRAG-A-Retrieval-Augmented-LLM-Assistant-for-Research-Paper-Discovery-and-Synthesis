"""
Demo entrypoint: index some sample papers and run a search.
"""
from indexer import index_from_source
from search import search_papers


def main():
    print("Indexing sample papers...")
    index_from_source("arxiv", "large language models", limit=3)
    index_from_source("openalex", "transformers", limit=3)
    index_from_source("springer", "computer vision", limit=3)

    print("Running search...")
    results = search_papers("transformer models for summarization", k=5)
    for r in results:
        print(f"[{r['source']}] {r['title']} ({r['year']}) - distance {r['distance']:.4f}")


if __name__ == "__main__":
    main()
