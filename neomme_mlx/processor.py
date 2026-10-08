"""CPU NumPy/tokenizers processor; NeoMME layout adapted under Apache-2.0."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tokenizers import Tokenizer


def resize_bilinear_antialias(image, height, width):
    """Torch CPU uint8 bilinear AA: int16 coefficients and per-pass rounding.

    Arithmetic follows PyTorch UpSampleKernel.cpp (BSD-3-Clause).
    """
    value = np.asarray(image, dtype=np.uint8)
    for axis, target in [(1, width), (0, height)]:
        original = value.shape[axis]
        if original == target:
            continue
        scale = original / target
        support = max(scale, 1.0)
        kernels = []
        for i in range(target):
            center = (i + 0.5) * scale
            start = max(int(center - support + 0.5), 0)
            end = min(int(center + support + 0.5), original)
            indices = np.arange(start, end)
            weights = np.maximum(0, 1 - np.abs((indices - center + 0.5) * (1 / support)))
            weights /= sum(weights)
            kernels.append((indices, weights))
        maxweight = max(float(w.max()) for _, w in kernels)
        precision = 0
        while precision < 22 and int(0.5 + maxweight * (1 << (precision + 1))) < 32768:
            precision += 1
        rows = []
        for indices, weights in kernels:
            weights = (weights * (1 << precision) + 0.5).astype(np.int64)
            v = np.take(value, indices, axis=axis).astype(np.int64)
            shape = [1] * v.ndim
            shape[axis] = len(weights)
            total = np.sum(v * weights.reshape(shape), axis=axis) + (1 << (precision - 1))
            rows.append(np.clip(total >> precision, 0, 255).astype(np.uint8))
        value = np.stack(rows, axis=axis)
    return value



class Processor:
    def __init__(self, model_path, max_side=1024, max_length=4096):
        path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.config = json.loads((path / "config.json").read_text())
        self.patch = self.config["patch_size"]
        if not isinstance(max_side, int) or max_side < 1 or max_side > 2048:
            raise ValueError("max_side must be in 1..2048")
        if not isinstance(max_length, int) or not 1 <= max_length <= self.config["max_position_embeddings"]:
            raise ValueError("Invalid max_length")
        self.max_side, self.max_length = max_side, max_length
        self.special = {name: self.tokenizer.token_to_id(f"<{name}>") for name in ["pad", "query", "doc", "img", "row", "mask"]}

    def image(self, image):
        if not isinstance(image, Image.Image):
            with Image.open(image) as opened:
                image = opened.copy()
        image = image.convert("RGB")
        w, h = image.size
        scale = min(1.0, self.max_side / max(w, h))
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        pixels = np.asarray(image)
        if (nh, nw) != (h, w):
            pixels = resize_bilinear_antialias(pixels, nh, nw)
        p = self.patch
        gh, gw = (nh + p - 1) // p, (nw + p - 1) // p
        padded = np.zeros((gh * p, gw * p, 3), dtype=np.uint8)
        padded[:nh, :nw] = pixels
        pixels = (padded.astype(np.float32) * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
        patches = pixels.reshape(gh, p, gw, p, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, p * p * 3)
        s = self.special
        ids = [s["doc"], s["img"]] + ([s["img"]] * gw + [s["row"]]) * gh
        positions = [[0, 0], [1, 1]] + [[2 + r, 2 + c] for r in range(gh) for c in range(gw + 1)]
        return ids, np.array(positions, dtype=np.int32).T, patches

    def __call__(self, items, task="document"):
        """Items are strings (text) or dicts {'image': path/PIL.Image}."""
        if task not in ("query", "document"):
            raise ValueError("task must be query or document")
        if isinstance(items, (str, dict)):
            items = [items]
        if not items:
            raise ValueError("Empty batch")
        rows, positions, patches = [], [], []
        for item in items:
            if isinstance(item, dict):
                if task != "document" or set(item) != {"image"}:
                    raise ValueError("Images must be standalone documents")
                ids, pos, pix = self.image(item["image"])
                patches.append(pix)
            elif isinstance(item, str):
                if "<img>" in item:
                    raise ValueError("<img> is reserved for image documents")
                prompt = "<query>" + item + "<mask>" * 10 if task == "query" else "<doc>" + item
                ids = self.tokenizer.encode(prompt, add_special_tokens=True).ids
                pos = np.tile(np.arange(len(ids), dtype=np.int32), (2, 1))
            else:
                raise TypeError("Expected text or image document")
            if len(ids) > self.max_length:
                raise ValueError(f"Input has {len(ids)} tokens, limit {self.max_length}; shorten input or resize image")
            rows.append(ids)
            positions.append(pos)
        n = max(map(len, rows))
        ids = np.full((len(rows), n), self.special["pad"], dtype=np.int32)
        mask = np.zeros_like(ids)
        pos = np.zeros((2, len(rows), n), dtype=np.int32)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
            pos[:, i, :len(row)] = positions[i]
        result = {"input_ids": ids, "attention_mask": mask}
        if patches:
            result.update(position_ids=pos, pixel_values=np.concatenate(patches))
        return result
