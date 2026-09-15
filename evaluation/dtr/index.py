"""Encode a corpus with the DTR table tower and search it.

DTR embeddings are 256-dimensional, so they cannot reuse the project's pgvector
columns, which are typed VECTOR(1536). The corpus is small enough that an
in-memory matrix and a matmul beat setting up a second vector store. The paper
likewise uses exhaustive search.

Because this is brute force rather than the HNSW index the other methods query,
retrieval timings from here are not comparable with theirs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluation.dtr.serialization import (
    DEFAULT_MAX_QUERY_LENGTH,
    DEFAULT_MAX_TABLE_LENGTH,
    encode_queries,
    encode_tables,
)

DEFAULT_ENCODE_BATCH_SIZE = 32


def _to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def encode_corpus(model, tokenizer, tables, device, batch_size=DEFAULT_ENCODE_BATCH_SIZE,
                  max_length=DEFAULT_MAX_TABLE_LENGTH, show_progress=True):
    """Embed TableRecords with the table tower.

    Args:
        model: A TapasDualEncoder.
        tokenizer: A TapasTokenizer.
        tables: TableRecords to encode, in index order.
        device: Torch device to run on.
        batch_size: Tables encoded per forward pass.
        max_length: Table sequence length cap.
        show_progress: Whether to draw a progress bar.

    Returns:
        A float32 array of shape (len(tables), projection_dim).
    """
    model.eval()
    out = []
    steps = range(0, len(tables), batch_size)
    for start in tqdm(steps, desc="Encoding tables", unit="batch", disable=not show_progress):
        chunk = tables[start : start + batch_size]
        batch = encode_tables(
            tokenizer, [t.title for t in chunk], [t.table for t in chunk], max_length=max_length
        )
        out.append(model.encode_tables(**_to_device(batch, device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_query_batch(model, tokenizer, queries, device, batch_size=DEFAULT_ENCODE_BATCH_SIZE,
                       max_length=DEFAULT_MAX_QUERY_LENGTH, show_progress=False):
    """Embed query strings with the query tower.

    Returns:
        A float32 array of shape (len(queries), projection_dim).
    """
    model.eval()
    out = []
    steps = range(0, len(queries), batch_size)
    for start in tqdm(steps, desc="Encoding queries", unit="batch", disable=not show_progress):
        batch = encode_queries(tokenizer, queries[start : start + batch_size], max_length=max_length)
        out.append(model.encode_queries(**_to_device(batch, device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@dataclass
class DTRIndex:
    """Table embeddings plus the records they came from."""

    embeddings: np.ndarray  # (N, proj_dim)
    table_ids: list[str]
    recall_keys: list[str]
    group_ids: list[str]

    @classmethod
    def build(cls, model, tokenizer, tables, device, **kwargs):
        return cls(
            embeddings=encode_corpus(model, tokenizer, tables, device, **kwargs),
            table_ids=[t.table_id for t in tables],
            recall_keys=[t.recall_key for t in tables],
            group_ids=[t.group_id for t in tables],
        )

    @classmethod
    def build_streaming(cls, model, tokenizer, records, device,
                        batch_size=DEFAULT_ENCODE_BATCH_SIZE, total=None, **kwargs):
        """Build an index without holding the whole corpus in memory.

        build() needs a sequence, so the caller must materialize every
        TableRecord first. At NQ-Tables scale that is 169,898 DataFrames and
        roughly 2.7 GB held for the duration of the encode, purely to hand the
        encoder one batch at a time; it is enough to get the process killed on
        a machine that is otherwise fine. This consumes an iterable instead and
        keeps only the embeddings, which are 174 MB at 256 dimensions.

        Args:
            model: A TapasDualEncoder.
            tokenizer: A TapasTokenizer.
            records: An iterable of TableRecord, consumed once.
            device: Torch device.
            batch_size: Tables per forward pass.
            total: Corpus size, for the progress bar only.
            **kwargs: Passed to encode_corpus for each chunk.

        Returns:
            A DTRIndex.
        """
        chunks, table_ids, recall_keys, group_ids = [], [], [], []
        batch = []

        def flush():
            if not batch:
                return
            chunks.append(
                encode_corpus(model, tokenizer, batch, device,
                              batch_size=batch_size, show_progress=False, **kwargs)
            )
            table_ids.extend(r.table_id for r in batch)
            recall_keys.extend(r.recall_key for r in batch)
            group_ids.extend(r.group_id for r in batch)
            batch.clear()

        bar = tqdm(total=total, desc="Encoding tables", unit="table")
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                bar.update(len(batch))
                flush()
        bar.update(len(batch))
        flush()
        bar.close()

        return cls(
            embeddings=np.concatenate(chunks, axis=0),
            table_ids=table_ids,
            recall_keys=recall_keys,
            group_ids=group_ids,
        )

    def search(self, query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        """Rank tables against each query by inner product.

        DTR scores with a bare dot product, with no normalization, so this is
        the same scoring function used during training.

        Args:
            query_embeddings: Array of shape (Q, proj_dim).
            top_k: Number of tables to return per query.

        Returns:
            Integer array of shape (Q, top_k) holding corpus positions, best
            first.
        """
        scores = query_embeddings @ self.embeddings.T
        k = min(top_k, scores.shape[1])
        # argpartition finds the top k cheaply; argsort then orders just those.
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(top, ordered, axis=1)

    def save(self, path: Path):
        np.savez(
            path,
            embeddings=self.embeddings,
            table_ids=np.array(self.table_ids, dtype=object),
            recall_keys=np.array(self.recall_keys, dtype=object),
            group_ids=np.array(self.group_ids, dtype=object),
        )

    @classmethod
    def load(cls, path: Path):
        blob = np.load(path, allow_pickle=True)
        return cls(
            embeddings=blob["embeddings"],
            table_ids=list(blob["table_ids"]),
            recall_keys=list(blob["recall_keys"]),
            group_ids=list(blob["group_ids"]),
        )
