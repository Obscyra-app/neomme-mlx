"""NeoMME inference in MLX.

Adapted from Hugging Face Transformers NeoMME, Copyright 2026 H Company
and HuggingFace Inc., Apache-2.0. No Torch/Transformers runtime imports.
"""
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def rms(x, eps):
    f = x.astype(mx.float32)
    return (f * mx.power(mx.mean(f * f, axis=-1, keepdims=True) + eps, -0.5)).astype(x.dtype)


def normalize(x):
    f = x.astype(mx.float32)
    return f / mx.maximum(mx.sqrt(mx.sum(f * f, axis=-1, keepdims=True)), 1e-12)


class NeoMME:
    def __init__(self, path, dtype="float32"):
        path = Path(path)
        self.config = c = json.loads((path / "config.json").read_text())
        if c["model_type"] != "neomme" or c["hidden_act"] != "relu2":
            raise ValueError("Unsupported model configuration")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        self.dtype = getattr(mx, dtype)
        self.w = {k: v.astype(self.dtype) for k, v in mx.load(str(path / "model.safetensors")).items()}
        expected = {
            "model.embed_tokens.word_embeddings.weight": (c["vocab_size"], c["embedding_rank"]),
            "model.embed_tokens.embedding_projection.weight": (c["hidden_size"], c["embedding_rank"]),
            "model.value_embeddings.weight": (c["vocab_size"], c["num_key_value_heads"] * c["head_dim"]),
            "model.patch_embeddings.norm.weight": (3 * c["patch_size"] ** 2,),
            "model.patch_embeddings.norm.bias": (3 * c["patch_size"] ** 2,),
            "model.patch_embeddings.up_proj.weight": (2 * c["hidden_size"], 3 * c["patch_size"] ** 2),
            "model.patch_embeddings.down_proj.weight": (c["hidden_size"], 2 * c["hidden_size"]),
            "model.patch_embeddings.down_proj.bias": (c["hidden_size"],),
            "multi_vector_head.proj.weight": (c["embedding_dim"], c["hidden_size"]),
        }
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}"
            expected[p + ".lambdas"] = (2,)
            expected[p + ".self_attn.exclusive_self_attention.alpha"] = (c["num_attention_heads"],)
            for name, heads in [("q_proj", c["num_attention_heads"]), ("k_proj", c["num_key_value_heads"]), ("v_proj", c["num_key_value_heads"])]:
                expected[p + ".self_attn." + name + ".weight"] = (heads * c["head_dim"], c["hidden_size"])
            expected[p + ".self_attn.output_projection.gate_proj.weight"] = (c["num_attention_heads"] * c["head_dim"], c["hidden_size"])
            expected[p + ".self_attn.output_projection.o_proj.weight"] = (c["hidden_size"], c["num_attention_heads"] * c["head_dim"])
            expected[p + ".mlp.up_proj.weight"] = (c["intermediate_size"], c["hidden_size"])
            expected[p + ".mlp.down_proj.weight"] = (c["hidden_size"], c["intermediate_size"])
        actual = {k: v.shape for k, v in self.w.items()}
        if actual != expected:
            raise ValueError("Unexpected weight names or shapes; refusing partial model load")
        mx.eval(self.w)
        self.h = c["num_attention_heads"]
        self.kv = c["num_key_value_heads"]
        self.d = c["head_dim"]
        self.eps = c["norm_eps"]
        globals_ = [i for i, t in enumerate(c["layer_types"]) if t == "full_attention"]
        self.value_layers = {globals_[0], globals_[-1]}

    def linear(self, x, prefix):
        y = x @ self.w[prefix + ".weight"].T
        bias = self.w.get(prefix + ".bias")
        return y if bias is None else y + bias

    def rotary(self, positions, layer_type):
        c = self.config["rope_parameters"][layer_type]
        if c["rope_type"] != "default":
            raise ValueError("Only default RoPE supported")
        dim = int(self.d * c.get("partial_rotary_factor", 1.0))
        inv = mx.power(float(c["rope_theta"]), -mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        f = positions[..., None].astype(mx.float32) * inv
        f = mx.stack([f[0, ..., 0::2], f[1, ..., 1::2]], axis=-1).reshape(*f.shape[1:])
        f = mx.concatenate([f, f], axis=-1)[:, None]
        return mx.cos(f).astype(self.dtype), mx.sin(f).astype(self.dtype)

    @staticmethod
    def rotate(x, cos, sin):
        dim = cos.shape[-1]
        r, rest = x[..., :dim], x[..., dim:]
        half = mx.concatenate([-r[..., dim // 2:], r[..., :dim // 2]], axis=-1)
        return mx.concatenate([r * cos + half * sin, rest], axis=-1)

    def __call__(self, input_ids, attention_mask=None, position_ids=None,
                 pixel_values=None, dense_dim=None, return_hidden=False):
        ids_np = np.asarray(input_ids, dtype=np.int32)
        if ids_np.ndim != 2 or ids_np.shape[1] > self.config["max_position_embeddings"]:
            raise ValueError("Expected bounded batch x sequence input IDs")
        if ids_np.size == 0 or ids_np.min() < 0 or ids_np.max() >= self.config["vocab_size"]:
            raise ValueError("Invalid token IDs")
        mask_np = np.ones_like(ids_np) if attention_mask is None else np.asarray(attention_mask)
        if mask_np.shape != ids_np.shape or not np.all(mask_np.sum(axis=1) > 0):
            raise ValueError("Invalid or empty attention mask")
        ids = mx.array(ids_np)
        mask = mx.array(mask_np.astype(bool))
        b, n = ids.shape
        x = self.linear(self.w["model.embed_tokens.word_embeddings.weight"][ids], "model.embed_tokens.embedding_projection")
        if pixel_values is not None:
            previous = np.pad(ids_np[:, :-1], ((0, 0), (1, 0)), constant_values=0)
            patches_mask = (ids_np == self.config["image_token_id"]) & (previous != self.config["document_token_id"])
            pix = mx.array(pixel_values).astype(self.dtype)
            if pix.ndim != 2 or pix.shape != (int(patches_mask.sum()), 3 * self.config["patch_size"] ** 2):
                raise ValueError("Patch count or width does not match placeholders")
            p = "model.patch_embeddings"
            pix = mx.fast.layer_norm(pix, self.w[p + ".norm.weight"], self.w[p + ".norm.bias"], eps=1e-5)
            pix = self.linear(nn.gelu(self.linear(pix, p + ".up_proj")), p + ".down_proj")
            indices = mx.array(np.flatnonzero(patches_mask).astype(np.int32))
            x = x.reshape(b * n, -1).at[indices].add(pix - x.reshape(b * n, -1)[indices]).reshape(b, n, -1)
        if position_ids is None:
            pos = mx.broadcast_to(mx.arange(n)[None, None], (2, b, n))
        else:
            pos = mx.array(position_ids)
            if pos.ndim == 2:
                pos = mx.broadcast_to(pos[None], (2, b, n))
            if pos.shape != (2, b, n):
                raise ValueError("Invalid two-axis position shape")
        x = initial = rms(x, self.eps)
        values = self.w["model.value_embeddings.weight"][ids].reshape(b, n, self.kv, self.d).transpose(0, 2, 1, 3)
        rotations = {t: self.rotary(pos, t) for t in set(self.config["layer_types"])}
        distance = mx.abs(mx.arange(n)[:, None] - mx.arange(n)[None, :])
        masks = {}
        for i, layer_type in enumerate(self.config["layer_types"]):
            p = f"model.layers.{i}"
            window = self.config.get("per_layer_config", {}).get(f"{i:02d}", {}).get("sliding_window", self.config["sliding_window"])
            if layer_type == "full_attention":
                window = None
            if window not in masks:
                valid = mask[:, None, None, :]
                if window is not None:
                    valid = valid & (distance[None, None] <= window)
                masks[window] = valid
            lambdas = self.w[p + ".lambdas"]
            z = rms(lambdas[0] * x + lambdas[1] * initial, self.eps)
            a = p + ".self_attn"
            q = rms(self.linear(z, a + ".q_proj").reshape(b, n, self.h, self.d).transpose(0, 2, 1, 3), self.eps)
            k = rms(self.linear(z, a + ".k_proj").reshape(b, n, self.kv, self.d).transpose(0, 2, 1, 3), self.eps)
            v = self.linear(z, a + ".v_proj").reshape(b, n, self.kv, self.d).transpose(0, 2, 1, 3)
            cos, sin = rotations[layer_type]
            q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
            if i in self.value_layers:
                v = v + values
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.d ** -0.5, mask=masks[window]).transpose(0, 2, 1, 3)
            vu = normalize(mx.repeat(v, self.h // self.kv, axis=1).transpose(0, 2, 1, 3)).astype(self.dtype)
            alpha = mx.tanh(self.w[a + ".exclusive_self_attention.alpha"])[None, None, :, None]
            out = out - alpha * mx.sum(out * vu, axis=-1, keepdims=True) * vu
            out = out.reshape(b, n, -1) * mx.sigmoid(self.linear(z, a + ".output_projection.gate_proj"))
            x = x + self.config["residual_multiplier"] * self.linear(out, a + ".output_projection.o_proj")
            z = rms(x, self.eps)
            z = mx.maximum(self.linear(z, p + ".mlp.up_proj"), 0)
            x = x + self.config["residual_multiplier"] * self.linear(z * z, p + ".mlp.down_proj")
            mx.eval(x)
        x = rms(x, self.eps)
        multi = mx.where(mask[..., None], normalize(self.linear(x, "multi_vector_head.proj")), 0)
        pooled = mx.sum(x.astype(mx.float32) * mask[..., None], axis=1) / mx.sum(mask, axis=1, keepdims=True)
        if dense_dim is not None:
            if not isinstance(dense_dim, int) or not 0 < dense_dim <= pooled.shape[-1]:
                raise ValueError("Invalid dense dimension")
            pooled = pooled[..., :dense_dim]
        result = {"dense": normalize(pooled), "multi": multi}
        if return_hidden:
            result["hidden"] = x
        mx.eval(result)
        return result
