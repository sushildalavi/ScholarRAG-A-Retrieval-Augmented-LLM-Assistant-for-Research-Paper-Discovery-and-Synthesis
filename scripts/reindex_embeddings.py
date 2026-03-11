#!/usr/bin/env python3

import argparse

from backend.services.db import execute, fetchall
from backend.services.embeddings import (
    embed_documents,
    get_embedding_model,
    get_embedding_version,
    get_provider,
    get_raw_embedding_dims,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild chunk embeddings for the active embedding provider/model.")
    parser.add_argument("--document-id", type=int, default=None, help="Only reindex one document id")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunk batch size")
    parser.add_argument(
        "--purge-all",
        action="store_true",
        help="Delete all existing chunk embeddings before reindexing",
    )
    args = parser.parse_args()

    provider = get_provider()
    model = get_embedding_model()
    version = get_embedding_version()
    raw_dim = get_raw_embedding_dims()

    if args.purge_all:
        execute("DELETE FROM chunk_embeddings")
    else:
        execute(
            """
            DELETE FROM chunk_embeddings
            WHERE provider=%s AND model=%s AND embedding_version=%s
            """,
            [provider, model, version],
        )

    params = []
    where = ""
    if args.document_id is not None:
        where = "WHERE chunks.document_id = %s"
        params = [args.document_id]

    rows = fetchall(
        f"""
        SELECT chunks.id, chunks.text, chunks.document_id
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        {where}
        ORDER BY chunks.document_id ASC, chunks.chunk_index ASC, chunks.id ASC
        """,
        params,
    )

    total = len(rows)
    for start in range(0, total, args.batch_size):
        batch = rows[start : start + args.batch_size]
        vectors = embed_documents([row["text"] for row in batch])
        for row, vector in zip(batch, vectors):
            execute(
                """
                INSERT INTO chunk_embeddings (chunk_id, provider, model, embedding_version, dim, vector)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                [row["id"], provider, model, version, raw_dim, vector],
            )
        print(
            f"reindexed {min(start + len(batch), total)}/{total} chunks "
            f"for provider={provider} model={model} version={version}"
        )


if __name__ == "__main__":
    main()
