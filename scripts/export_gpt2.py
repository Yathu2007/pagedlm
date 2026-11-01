#!/usr/bin/env python3
"""Export HF GPT-2 weights to the pagedlm .bin format (see docs/format.md).

Usage:
    python scripts/export_gpt2.py --out models/gpt2.bin [--model openai-community/gpt2] [--dtype f32|f16]
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

MAGIC = b"PGLM"
VERSION = 1
ARCH_GPT2 = 1
DTYPES = {"f32": (0, np.float32), "f16": (1, np.float16)}

HEADER_SIZE = 256
HEADER_FMT = "<4sIIIIIIIIIIIIf"   # magic .. tied_embeddings, norm_eps  (56 B)
INDEX_FMT = "<64sII4IQQ"          # name, dtype, ndim, shape[4], byte_offset, byte_size
INDEX_SIZE = struct.calcsize(INDEX_FMT)
ALIGN = 64

assert struct.calcsize(HEADER_FMT) == 56
assert INDEX_SIZE == 104


def round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def build_tensor_list(hf: dict, cfg: dict):
    """Return [(name, array)] in file order, with Conv1D weights transposed to [out, in]."""
    def get(key):
        # Some checkpoints prefix keys with "transformer."
        return hf[key] if key in hf else hf["transformer." + key]

    def linear_w(key):
        # HF Conv1D stores [in, out]; pagedlm wants [out, in].
        return np.ascontiguousarray(get(key).T)

    tensors = [
        ("tok_emb.weight", get("wte.weight")),
        ("pos_emb.weight", get("wpe.weight")),
    ]
    for i in range(cfg["n_layer"]):
        h, L = f"h.{i}.", f"layers.{i}."
        tensors += [
            (L + "ln1.weight",       get(h + "ln_1.weight")),
            (L + "ln1.bias",         get(h + "ln_1.bias")),
            (L + "attn.qkv.weight",  linear_w(h + "attn.c_attn.weight")),
            (L + "attn.qkv.bias",    get(h + "attn.c_attn.bias")),
            (L + "attn.out.weight",  linear_w(h + "attn.c_proj.weight")),
            (L + "attn.out.bias",    get(h + "attn.c_proj.bias")),
            (L + "ln2.weight",       get(h + "ln_2.weight")),
            (L + "ln2.bias",         get(h + "ln_2.bias")),
            (L + "mlp.fc.weight",    linear_w(h + "mlp.c_fc.weight")),
            (L + "mlp.fc.bias",      get(h + "mlp.c_fc.bias")),
            (L + "mlp.proj.weight",  linear_w(h + "mlp.c_proj.weight")),
            (L + "mlp.proj.bias",    get(h + "mlp.c_proj.bias")),
        ]
    tensors += [
        ("final_ln.weight", get("ln_f.weight")),
        ("final_ln.bias",   get("ln_f.bias")),
    ]
    return tensors


def expected_shapes(cfg: dict) -> dict:
    d, ff, V, T = cfg["n_embd"], 4 * cfg["n_embd"], cfg["vocab_size"], cfg["n_positions"]
    shapes = {"tok_emb.weight": (V, d), "pos_emb.weight": (T, d),
              "final_ln.weight": (d,), "final_ln.bias": (d,)}
    for i in range(cfg["n_layer"]):
        L = f"layers.{i}."
        shapes.update({
            L + "ln1.weight": (d,), L + "ln1.bias": (d,),
            L + "attn.qkv.weight": (3 * d, d), L + "attn.qkv.bias": (3 * d,),
            L + "attn.out.weight": (d, d), L + "attn.out.bias": (d,),
            L + "ln2.weight": (d,), L + "ln2.bias": (d,),
            L + "mlp.fc.weight": (ff, d), L + "mlp.fc.bias": (ff,),
            L + "mlp.proj.weight": (d, ff), L + "mlp.proj.bias": (d,),
        })
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=DTYPES, default="f32")
    args = ap.parse_args()
    dtype_code, np_dtype = DTYPES[args.dtype]

    cfg = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    hf = load_file(hf_hub_download(args.model, "model.safetensors"))

    tensors = build_tensor_list(hf, cfg)
    want = expected_shapes(cfg)
    assert len(tensors) == len(want), (len(tensors), len(want))
    for name, arr in tensors:
        if arr.shape != want[name]:
            raise SystemExit(f"shape mismatch for {name}: got {arr.shape}, expected {want[name]}")
        if len(name.encode()) >= 64:
            raise SystemExit(f"name too long: {name}")
    tensors = [(n, np.ascontiguousarray(a, dtype=np_dtype)) for n, a in tensors]

    # Lay out: header, index, then each tensor at the next 64 B boundary.
    offset = round_up(HEADER_SIZE + len(tensors) * INDEX_SIZE, ALIGN)
    entries = []
    for name, arr in tensors:
        entries.append((name, arr, offset))
        offset = round_up(offset + arr.nbytes, ALIGN)

    header = struct.pack(
        HEADER_FMT, MAGIC, VERSION, ARCH_GPT2, dtype_code, len(tensors),
        cfg["n_layer"], cfg["n_head"], cfg["n_head"], cfg["n_embd"], 4 * cfg["n_embd"],
        cfg["vocab_size"], cfg["n_positions"], 1, cfg["layer_norm_epsilon"],
    ).ljust(HEADER_SIZE, b"\0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header)
        for name, arr, off in entries:
            shape = list(arr.shape) + [0] * (4 - arr.ndim)
            f.write(struct.pack(INDEX_FMT, name.encode(), dtype_code, arr.ndim,
                                *shape, off, arr.nbytes))
        for name, arr, off in entries:
            f.write(b"\0" * (off - f.tell()))   # alignment padding
            f.write(arr.tobytes(order="C"))

    print(f"wrote {args.out}: {len(entries)} tensors, {args.out.stat().st_size:,} bytes, "
          f"dtype={args.dtype}, data starts at {entries[0][2]}")


if __name__ == "__main__":
    main()
