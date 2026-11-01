# `.bin` weight format (version 1)

All integers are **little-endian**. All offsets are **absolute byte offsets from the start
of the file**.

```
offset 0      [Header]   256 B, fixed
offset 256    [Index]    n_tensors entries × 104 B
              [padding]  zeros up to the next multiple of 64
              [Data]     each tensor starts at a multiple of 64; zero padding between tensors
```

## Header (256 B)

| Offset | Type      | Field             | GPT-2 small value          |
|-------:|-----------|-------------------|----------------------------|
| 0      | char[4]   | `magic`           | `"PGLM"` (no NUL)          |
| 4      | u32       | `version`         | 1                          |
| 8      | u32       | `arch`            | 1 = GPT-2 (2 = Llama, later) |
| 12     | u32       | `dtype`           | 0 = FP32, 1 = FP16         |
| 16     | u32       | `n_tensors`       | 148                        |
| 20     | u32       | `n_layer`         | 12                         |
| 24     | u32       | `n_head`          | 12                         |
| 28     | u32       | `n_kv_head`       | 12 (= `n_head` for GPT-2)  |
| 32     | u32       | `d_model`         | 768                        |
| 36     | u32       | `d_ff`            | 3072                       |
| 40     | u32       | `vocab_size`      | 50257                      |
| 44     | u32       | `max_seq_len`     | 1024                       |
| 48     | u32       | `tied_embeddings` | 1 (LM head reuses `tok_emb.weight`) |
| 52     | f32       | `norm_eps`        | 1e-5                       |
| 56     | —         | reserved          | zeros up to byte 256       |

## Index entry (104 B each, starting at byte 256)

| Offset | Type     | Field         | Notes                                           |
|-------:|----------|---------------|-------------------------------------------------|
| 0      | char[64] | `name`        | NUL-padded                                      |
| 64     | u32      | `dtype`       | same codes as the header                        |
| 68     | u32      | `ndim`        | 1 or 2                                          |
| 72     | u32[4]   | `shape`       | unused dims are 0                               |
| 88     | u64      | `byte_offset` | absolute, multiple of 64                        |
| 96     | u64      | `byte_size`   | = product(shape) × sizeof(dtype)                |

88 is a multiple of 8, so a plain C struct with these fields in this order has no hidden
padding and is exactly 104 B (worth a `static_assert`).

## Layout rules

- Every tensor is **row-major** (C order).
- Every matmul weight is **`[out, in]`** (like PyTorch `nn.Linear`). HF GPT-2 stores its
  `Conv1D` weights as `[in, out]`; the exporter transposes them. The loader never
  transposes; it only checks.
- Tied embeddings: the LM head is **not** stored. Use `tok_emb.weight` (`[vocab, d_model]`).

## Tensor names (GPT-2)

| Name                         | Shape          | HF source (transposed?)      |
|------------------------------|----------------|------------------------------|
| `tok_emb.weight`             | [50257, 768]   | `wte.weight`                 |
| `pos_emb.weight`             | [1024, 768]    | `wpe.weight`                 |
| `layers.{i}.ln1.weight/bias` | [768]          | `h.{i}.ln_1.*`               |
| `layers.{i}.attn.qkv.weight` | [2304, 768]    | `h.{i}.attn.c_attn.weight` (T) |
| `layers.{i}.attn.qkv.bias`   | [2304]         | `h.{i}.attn.c_attn.bias`     |
| `layers.{i}.attn.out.weight` | [768, 768]     | `h.{i}.attn.c_proj.weight` (T) |
| `layers.{i}.attn.out.bias`   | [768]          | `h.{i}.attn.c_proj.bias`     |
| `layers.{i}.ln2.weight/bias` | [768]          | `h.{i}.ln_2.*`               |
| `layers.{i}.mlp.fc.weight`   | [3072, 768]    | `h.{i}.mlp.c_fc.weight` (T)  |
| `layers.{i}.mlp.fc.bias`     | [3072]         | `h.{i}.mlp.c_fc.bias`        |
| `layers.{i}.mlp.proj.weight` | [768, 3072]    | `h.{i}.mlp.c_proj.weight` (T) |
| `layers.{i}.mlp.proj.bias`   | [768]          | `h.{i}.mlp.c_proj.bias`      |
| `final_ln.weight/bias`       | [768]          | `ln_f.*`                     |

Tensors appear in the file in this order (layer by layer). 2 + 12 × 12 + 2 = 148 tensors.
