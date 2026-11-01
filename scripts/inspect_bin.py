#!/usr/bin/env python3
"""Validate a pagedlm .bin file independently of the C++ loader.

Checks the header, the index (alignment, bounds, overlaps, sizes) and, with --check-hf,
that every tensor round-trips bit-exactly against the original HF weights.

Usage:
    python scripts/inspect_bin.py models/gpt2.bin [--list] [--check-hf openai-community/gpt2]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_FMT = "<4sIIIIIIIIIIIIf"
INDEX_FMT = "<64sII4IQQ"
INDEX_SIZE = struct.calcsize(INDEX_FMT)
HEADER_FIELDS = ["magic", "version", "arch", "dtype", "n_tensors", "n_layer", "n_head",
                 "n_kv_head", "d_model", "d_ff", "vocab_size", "max_seq_len",
                 "tied_embeddings", "norm_eps"]
NP_DTYPE = {0: np.float32, 1: np.float16}


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def read(path: Path):
    buf = path.read_bytes()
    hdr = dict(zip(HEADER_FIELDS, struct.unpack_from(HEADER_FMT, buf, 0)))
    if hdr["magic"] != b"PGLM":
        fail(f"bad magic {hdr['magic']!r}")
    if hdr["version"] != 1:
        fail(f"unsupported version {hdr['version']}")
    entries = []
    for i in range(hdr["n_tensors"]):
        name, dt, ndim, s0, s1, s2, s3, off, size = struct.unpack_from(
            INDEX_FMT, buf, 256 + i * INDEX_SIZE)
        entries.append(dict(name=name.rstrip(b"\0").decode(), dtype=dt, ndim=ndim,
                            shape=(s0, s1, s2, s3)[:ndim], offset=off, size=size))
    return buf, hdr, entries


def validate(buf, hdr, entries):
    index_end = 256 + len(entries) * INDEX_SIZE
    prev_end, names = index_end, set()
    for e in entries:
        n = e["name"]
        if n in names:
            fail(f"duplicate name {n}")
        names.add(n)
        if e["dtype"] not in NP_DTYPE:
            fail(f"{n}: unknown dtype {e['dtype']}")
        itemsize = np.dtype(NP_DTYPE[e["dtype"]]).itemsize
        if e["size"] != int(np.prod(e["shape"])) * itemsize:
            fail(f"{n}: byte_size {e['size']} != prod{e['shape']} * {itemsize}")
        if e["offset"] % 64:
            fail(f"{n}: offset {e['offset']} not 64 B-aligned")
        if e["offset"] < prev_end:
            fail(f"{n}: overlaps previous data/index (offset {e['offset']} < {prev_end})")
        if e["offset"] + e["size"] > len(buf):
            fail(f"{n}: runs past end of file")
        prev_end = e["offset"] + e["size"]


def tensor(buf, e):
    return np.frombuffer(buf, dtype=NP_DTYPE[e["dtype"]], count=int(np.prod(e["shape"])),
                         offset=e["offset"]).reshape(e["shape"])


def check_hf(buf, entries, model):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from export_gpt2 import build_tensor_list

    cfg = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    ref = dict(build_tensor_list(load_file(hf_hub_download(model, "model.safetensors")), cfg))
    if set(ref) != {e["name"] for e in entries}:
        fail("tensor names differ from HF export list")
    for e in entries:
        got = tensor(buf, e)
        want = ref[e["name"]].astype(got.dtype)
        if not np.array_equal(got, want):
            fail(f"{e['name']}: values differ from HF")
    # Spot-check the transpose directly against the raw HF layout.
    raw = load_file(hf_hub_download(model, "model.safetensors"))
    key = "h.0.attn.c_attn.weight" if "h.0.attn.c_attn.weight" in raw else "transformer.h.0.attn.c_attn.weight"
    qkv = tensor(buf, next(e for e in entries if e["name"] == "layers.0.attn.qkv.weight"))
    if qkv[7, 3] != raw[key][3, 7].astype(qkv.dtype):
        fail("layers.0.attn.qkv.weight is not the transpose of HF c_attn")
    print(f"HF round-trip: all {len(entries)} tensors bit-exact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--list", action="store_true", help="print every tensor")
    ap.add_argument("--check-hf", metavar="MODEL", help="compare values against HF weights")
    args = ap.parse_args()

    buf, hdr, entries = read(args.path)
    print("header:", {k: v for k, v in hdr.items() if k != "magic"})
    validate(buf, hdr, entries)
    print(f"index OK: {len(entries)} tensors, data starts at {entries[0]['offset']}, "
          f"file size {len(buf):,} B")
    if args.list:
        for e in entries:
            print(f"  {e['name']:<28} {str(list(e['shape'])):<14} off={e['offset']:<11} size={e['size']}")
    if args.check_hf:
        check_hf(buf, entries, args.check_hf)
    print("PASS")


if __name__ == "__main__":
    main()
