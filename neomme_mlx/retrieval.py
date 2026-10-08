"""Retrieval scoring independent of neural backend."""
import numpy as np


def mean_maxsim(query, document):
    """Mean of each query token's maximum similarity, excluding padding."""
    query = np.asarray(query, dtype=np.float32)
    document = np.asarray(document, dtype=np.float32)
    if query.ndim != 2 or document.ndim != 2 or not len(query) or not len(document):
        raise ValueError("Nonempty unpadded token embeddings required")
    # Bound temporary memory for long documents.
    maxima = np.full(len(query), -np.inf, dtype=np.float32)
    for start in range(0, len(document), 512):
        maxima = np.maximum(maxima, (query @ document[start:start + 512].T).max(axis=1))
    return float(maxima.mean())


def rank(query, documents, mode="late"):
    if mode == "late":
        scores = [mean_maxsim(query["multi"], d["multi"]) for d in documents]
    elif mode == "dense":
        scores = [float(np.asarray(query["dense"]) @ np.asarray(d["dense"])) for d in documents]
    else:
        raise ValueError("mode must be late or dense")
    return sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
