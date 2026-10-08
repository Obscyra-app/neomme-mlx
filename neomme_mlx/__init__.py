"""Native MLX NeoMME retrieval. Model imports are lazy for CPU-only tooling."""

def load_model(path, dtype="float32"):
    from .model import NeoMME
    return NeoMME(path, dtype=dtype)
