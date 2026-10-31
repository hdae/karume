> Point-in-time record (2026-10-31) of CPU-only export measurements for the Gated DeltaNet (GDN) recurrence: how `torch.export` treats upstream's two recurrence functions, how large their decomposed graphs are, and what a MOSS-shaped `Qwen3_5GatedDeltaNet` layer needs on the recipe side to reach the exporter with a symbolic T. Measurement material for open points 1 to 4 of [ADR 0135](../decisions/0135-gated-delta-rule-scan.md) (proposed); not a design source of truth. Scratch scripts, raw JSON and guard logs are under `outputs/diag/gdn-export-2026-10-31/` of the main checkout (not tracked). Nothing ran on a GPU.

# GDN export measurements: length-axis specialisation, decomposed size, recipe-side rewrites

Question: does entry condition ① of ADR 0135 decision 2 hold — does exporting upstream's recurrence
lose the length axis and the "this is a delta-rule scan" information? Does a custom op keep T
symbolic through the Karume exporter for a whole GDN layer, and what do the two export gaps outside
the recurrence (the `rsqrt` of the q / k l2-normalisation and the causal conv1d's trailing slice)
and the r = 2 head layout need?

Answer: ① holds. Under `Dim("T")` both upstream functions specialise T to the example value through
their Python loops. At fixed T, after `run_decompositions(curated_decompositions())`, the recurrent
function is 30 nodes per token (122,911 at T = 4096) and the chunked one 67 nodes per 64-token chunk
(5,697 at T = 4096); each graph is valid for one T, and neither converts to IR as it is. A
`torch.library.custom_op` stand-in is one node with a symbolic T and survives the decompositions. A
1-layer `Qwen3_5GatedDeltaNet` at the MOSS shape converts to 47 IR nodes with a symbolic T after
three recipe-side changes, none of which copies `forward` or adds vocabulary: the attribute swap of
the recurrence, the l2-normalisation written as `rms_norm(x, eps/d) · d^-½`, and the instance
attribute `causal_conv1d_fn` set to a left-pad conv. The l2norm rewrite is not bit-identical to
upstream's (≤ 4 ULP) and costs 0.07–0.13 of the o band. The conv rewrite is bit-identical in the
conv. r = 2 exports with a rank-4 undo of the repeat.

## 1. Setup

- CPU only (the GPU was in use by another project's training run). The repository's uv venv
  `tools/.venv`: torch 2.13.0+cpu, transformers 5.14.1 (`models/qwen3_5/modeling_qwen3_5.py`;
  line numbers below are that file's). `kernels`, `flash-linear-attention` (`fla`) and
  `causal-conv1d` are not installed, so upstream binds its torch functions and the stock `forward`
  runs.
- Exporter: `tools/exporter` at main `4a2e0284`, unchanged. `export → run_decompositions(
  curated_decompositions()) → normalize_graph → convert`, the steps of `karume.pipeline.
  export_module`, run one by one so each stage can be read.
- Shape: one MOSS-Image text-encoder GDN layer — hidden 4096, 16 key heads, 32 value heads, key and
  value head dim 128, conv kernel 4 (8,192 conv channels), B = 1. `torch.export.export(...,
  strict=False)` with `Dim("T", min=1, max=4096)` (min 2 for §2), example T = 64 unless stated.
- **Scratch custom op**: `gdnscratch::gated_delta_scan(q, k, v, g, beta) → o`, a
  `torch.library.custom_op` with a fake that returns `v.new_empty(v.shape)`. Its eager body calls
  upstream's `torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=False` after
  `repeat_interleave` when r > 1. It is not the product op (its own namespace, no copied body). For
  §5 it was registered with the exporter in-process — a handler that emits the IR op name
  `gated_delta_scan_scratch` with arity 5, rank 4 / 4 / 4 / 3 / 3 and f32 checks, added to
  `EMITTABLE_OPS`, and the output-shape check run on every other node with the real contracts and on
  the scan node by hand (o shaped as v). No exporter source was edited.
- Layer weights: random init (`torch.manual_seed(20261031)`; upstream's own init, so `A_log` =
  log U(0, 16), `dt_bias` = 1). §6.1 adds **real** data: layer 0 of the MOSS-Image text encoder
  (weights from the local Hugging Face cache, cast to f32) on a 512-token neutral English prompt
  (a scene description, repeated), run through `input_layernorm` and the stock layer, with q / k / v
  / g / β captured at the recurrence call.

## 2. The length axis under `Dim("T")`

One module per upstream function, called as `Qwen3_5GatedDeltaNet` calls them (q / k already
repeated to 32 heads, `use_qk_l2norm_in_kernel=True`), the five inputs sharing one `Dim("T",
min=2, max=4096)`. Guards from `TORCH_LOGS=dynamic`.

| Function                          | Example T | Outcome                                                                                       | Guard from upstream source                        |
| --------------------------------- | --------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `torch_recurrent_gated_delta_rule` | 64        | `Constraints violated (T)`: "your code specialized it to be a constant (64)", fix `T = 64`  | `Eq(T, 64)` at :354 (`for i in range(sequence_length)`) |
| `torch_recurrent_gated_delta_rule` | 100       | same, constant 100                                                                            | `Eq(T, 100)` at :354                              |
| `torch_chunk_gated_delta_rule`     | 64        | same, constant 64                                                                             | `Eq(ceil(T/64), 1)` at :309 (`for i in range(0, total_sequence_length // chunk_size)`) |
| `torch_chunk_gated_delta_rule`     | 100       | 23 guards that not every T in range satisfies, then "solving the guards … resulted in a specialized value of 100" | `Eq(ceil(T/64), 2)` at :309; the others come from the padding arithmetic (`pad_size`, :270) through the chunk reshapes (`_reshape_view_helper`) |

Both functions lose T. The recurrent one unrolls one step per token. The chunked one unrolls one
step per chunk, and every shape after the padding is an expression in `T mod 64`. Its graph would
be valid for one chunk count at best; the exporter's `Dim("T")` makes it one T.

## 3. Node counts at fixed T

Static export at each T, then `run_decompositions(curated_decompositions())` (call_function nodes,
including views). "Non-view" leaves out `view` / `unsqueeze` / `squeeze` / `expand` / `alias` /
`_assert_tensor_metadata` / `sym_size`. Times are wall-clock on this machine (12 cores).

| Function  | T    | Nodes before decomp | Nodes after decomp | Non-view | Export + decomp (s) |
| --------- | ---- | ------------------- | ------------------ | -------- | ------------------- |
| recurrent | 64   | 1,573               | 1,951              | 1,305    | 1.7 + 2.0           |
| recurrent | 512  | 12,325              | 15,391             | 10,265   | 10.3 + 16.5         |
| recurrent | 4096 | 98,341              | 122,911            | 81,945   | 76.8 + 114.5        |
| chunk     | 64   | 934                 | 1,476              | 1,159    | 1.1 + 2.0           |
| chunk     | 512  | 1,186               | 1,945              | 1,397    | 1.4 + 2.6           |
| chunk     | 4096 | 3,202               | 5,697              | 3,301    | 3.5 + 5.9           |

- The recurrent form grows by 30.0 nodes per token ((122,911 − 15,391) / 3,584), 20.0 of them
  non-view. Per token: 8 `unsqueeze`, 6 `select`, 5 `mul`, 2 `sum`, 1 each of `add`, `sub`, `exp`,
  `view`, and **the functionalised indexed write** `core_attn_out[:, :, i] = …` (:365) as `arange`, `eq`,
  `expand`, `where` over the whole output and a `copy`.
- The chunked form grows by 67 nodes per chunk ((5,697 − 1,945) / 56). Its intra-chunk loop
  (`for i in range(1, chunk_size)`, :293) is unrolled 63 times regardless of T (63 `slice_scatter`
  at every T); the per-chunk indexed write (:315) is again an `arange` / `eq` / `where`.
- **Neither decomposed form converts to IR.** Recurrent (all three T): unsupported `aten.copy`
  (T of them), `aten.eq.Scalar` (T − 1) and `aten.rsqrt` (2), all Core ATen. Chunk (all three T):
  `reshape/permute chain without end points of rank ≤ 4` — the chunked tensors are rank 5
  `[B, H, chunks, 64, d]`.

## 4. The scratch custom op alone

`Dim("T", min=1, max=4096)`, example T = 64 and 300, for r = 1 (q / k with 32 heads) and r = 2
(q / k with 16 heads). In all four exports, after `run_decompositions(curated_decompositions())`
the graph has **one** call_function node, `gdnscratch.gated_delta_scan.default`, with output shape
`[1, s, 32, 128]` and the range constraint `s ∈ [1, 4096]`. The decompositions do not enter it
(the namespace is not in the decomposition table), as for `gru_scan` (ADR 0056).

## 5. One layer through the exporter

The stock `Qwen3_5GatedDeltaNet` and recipe-side variants of it. Rewrites are applied as instance
attribute or submodule swaps; `forward` is never copied. "Eager vs stock" compares the patched
layer's eager output with the stock layer's on the same input (N(0, 1), T = 64; the stock output's
largest magnitude is 0.439). "First rejection" is where `convert` stops.

| Variant | Change against the previous row | Eager vs stock (max abs) | `Dim("T")` export | First rejection in the exporter | IR nodes |
| --- | --- | --- | --- | --- | --- |
| stock | — | — | **fails**: T specialised to 64 | — | — |
| swap | `chunk_gated_delta_rule` → wrapper: upstream `l2norm` + scratch op (r = 1) | **bit-identical** | ok, T ∈ [1, 4096] | `aten.rsqrt` ×2 unsupported [Core ATen] (the q / k l2norm) | — |
| l2 | l2norm as `rms_norm(x, (d,), eps/d) · d^-½` | 1.34e-7 | ok | `slice` on the symbolic axis: `slice(conv1d, 2, None, T)` (forward's `[:, :, :T]`, :501) | — |
| conv | `conv1d` module → `F.pad(x, (3, 0))` + `conv1d(padding=0)` | 1.34e-7 | ok | same `slice` (forward still slices the now length-T output) | — |
| **convfn** | instead of the module, instance attribute `causal_conv1d_fn` → pad + conv1d + silu | 1.34e-7 | ok | **none** | **47** |
| convfn, stock l2norm | convfn with upstream's `l2norm` | 2.98e-8 | ok | `aten.rsqrt` ×2 | — |
| convfn, `x·(1/sqrt(·))` | l2norm as `x * (1 / sqrt(Σx² + eps))` | 8.94e-8 | ok | `aten.reciprocal` ×2 unsupported [Core ATen] | — |
| convfn, `x/sqrt(·)` | l2norm as `x / sqrt(Σx² + eps)` | 1.49e-7 | ok | `sum with keepdim=True is not supported` | — |
| r2, select | convfn + undo the repeat: `reshape [B,T,Hk,r,d]` then `[:, :, :, 0]` | 1.34e-7 | ok | `slice of rank 5 cannot run on the strided kernel (rank ≤ 4)` | — |
| r2, slice rank 5 | `reshape [B,T,Hk,r,d]` + `[:, :, :, 0:1]` + reshape | 1.34e-7 | ok | same rank-5 rejection | — |
| r2, slice rank 4 | `reshape [B·T,Hk,r,d]` + `[:, :, 0:1]` + reshape | 1.34e-7 | ok | none | 53 |
| r2, last-axis slice | `reshape [B,T,Hk,r·d]` + `[..., 0:d]` | 1.34e-7 | ok | none | 51 |

- **The swap alone is bit-identical to the stock layer**: an eager body that is upstream's prefill
  function with `use_qk_l2norm_in_kernel=False`, called on q / k normalised by upstream's own
  `l2norm`, reproduces upstream's call with `use_qk_l2norm_in_kernel=True` bit for bit (as ADR 0135
  decision 5 expects for its goldens).
- The 1.34e-7 of the later rows is the l2norm rewrite (§6.1); the 2.98e-8 of "convfn, stock
  l2norm" is torch's CPU silu (§6.2). Decomposing the exported program moves its output by about
  1.8e-7 at most against its own eager run (softplus, SiLU and the gated norm are decomposed); this is
  torch's, not the exporter's.
- **The 47 IR nodes of convfn** (symbol `T`, input `hidden_states [1, T, 4096]`), in graph order:
  `linear` (in_proj_qkv), `permute`, `linear` (in_proj_z), `reshape`, `linear` ×2 (in_proj_b / a),
  `reshape`, **`pad`, `reshape`, `conv1d`, `sigmoid`, `mul`** (the causal conv and SiLU),
  `permute`, `slice` ×3 (q / k / v split), `reshape` ×3, `sigmoid` (β), `exp`, `neg`, `add`, `exp`,
  `log1p`, `gt_scalar`, `where`, `mul` (g, with softplus decomposed), `reshape`, `expand`, `reshape`
  twice (the `repeat_interleave` of q and of k), **`rms_norm`, `mul`** twice (the l2norm rewrite),
  **`gated_delta_scan_scratch`** (inputs `[1,T,32,128]` ×3, `[1,T,32]` ×2, output `[1,T,32,128]`),
  `reshape` ×2, `rms_norm`, `sigmoid`, `mul`, `mul` (the gated RMSNorm, folded by `_fold_rms_norm`),
  `reshape`, `linear` (out_proj). Counts: reshape 13, mul 6, linear 5, sigmoid 3, slice 3, rms_norm
  3, permute 2, exp 2, expand 2, and one each of pad, conv1d, neg, add, log1p, gt_scalar, where and
  the scan.
- With r = 2 the scan node's q / k inputs are `[1,T,16,128]`. The undo adds `reshape` + `slice`
  per tensor (last-axis form) and the `expand` of the repeat stays in the graph.

## 6. The recipe-side rewrites in detail

### 6.1 q / k l2-normalisation without `rsqrt`

Forms, all in f32 on the same inputs: upstream `x·rsqrt(Σx² + 1e-6)` (:244); **rms** =
`rms_norm(x, (d,), eps=1e-6/d) · d^-½`; **mul-recip** = `x·(1/sqrt(Σx² + 1e-6))`; **div** =
`x/sqrt(Σx² + 1e-6)`. Reference: the same formula in f64 on the same f32 inputs. "ULP vs rounded"
is the distance to the f64 value rounded to f32 (0 = correctly rounded). Data: `unit` = N(0, 1),
`spike` = silu(N(0, 1)) (the spike's generator), `real` = MOSS layer 0 (q row norms 0.585–12.8, k
row norms 0.106–27.8). q and k behave alike; the table gives the worse of the two.

| Data, T | Form | Bit-identical to upstream | Max ULP to upstream | rel_l2 vs f64 | Mean / max ULP vs rounded |
| --- | --- | --- | --- | --- | --- |
| unit, 512 | upstream | — | — | 5.00e-8 | 0.424 / 3 |
| unit, 512 | rms | 47.5 % | 4 | 5.89e-8 | 0.522 / 4 |
| unit, 512 | mul-recip | 89.4 % | 3 | 5.12e-8 | 0.436 / 3 |
| unit, 512 | div | 67.2 % | 3 | 4.45e-8 | 0.358 / 3 |
| spike, 512 | upstream | — | — | 5.22e-8 | 0.440 / 3 |
| spike, 512 | rms | 49.0 % | 4 | 5.78e-8 | 0.495 / 3 |
| spike, 512 | mul-recip | 84.0 % | 3 | 5.37e-8 | 0.454 / 3 |
| spike, 512 | div | 61.0 % | 3 | 4.57e-8 | 0.361 / 3 |
| real, 512 | upstream | — | — | 5.44e-8 | 0.469 / 3 |
| real, 512 | rms | 47.6 % | 4 | 6.04e-8 | 0.539 / 4 |
| real, 512 | mul-recip | 86.6 % | 3 | 5.63e-8 | 0.490 / 3 |
| real, 512 | div | 64.9 % | 3 | 4.98e-8 | 0.419 / 3 |

The largest absolute difference between rms and upstream on the normalised vectors is 5.96e-8
(unit), 1.19e-7 (spike) and 1.79e-7 (real). T = 64 gives the same picture (rms 47.8–50.1 %
bit-identical, ≤ 4 ULP).

**Effect on o.** Oracle: the recurrence in f64 on q / k normalised in f64 from the raw f32 values.
"Exact scan" runs the f64 recurrence on each form's f32-normalised q / k, so it isolates the
normalisation's contribution to o. The o band is ADR 0135 decision 6's: 2 × the larger of the two
upstream f32 functions' errors against the oracle, on o.

| Data, T | o band rel_l2 / max_abs | Exact scan, upstream l2norm (rel_l2) | Exact scan, rms (rel_l2) | rms share of the band, exact scan (rel_l2 / max_abs) | Through upstream's chunked function: rms share (rel_l2 / max_abs) |
| --- | --- | --- | --- | --- | --- |
| unit, 64 | 1.29e-6 / 4.60e-7 | 6.82e-8 | 8.50e-8 | 0.066 / 0.013 | 0.500 / 0.493 |
| unit, 512 | 1.01e-6 / 4.12e-7 | 7.13e-8 | 8.55e-8 | 0.085 / 0.021 | 0.502 / 0.500 |
| spike, 64 | 1.14e-6 / 4.90e-7 | 6.66e-8 | 7.45e-8 | 0.065 / 0.016 | 0.503 / 0.496 |
| spike, 512 | 8.60e-7 / 4.18e-7 | 7.01e-8 | 7.05e-8 | 0.082 / 0.024 | 0.501 / 0.500 |
| spike, long memory (g × 0.001), 512 | 5.45e-7 / 1.57e-7 | 6.89e-8 | 7.28e-8 | 0.134 / 0.111 | 0.505 / 0.510 |
| real, 512 | 3.84e-7 / 3.61e-6 | 4.95e-8 | 4.76e-8 | 0.124 / 0.104 | 0.512 / 0.566 |

- The normalisation moves o by about as much as upstream's own normalisation does (rel_l2 4.8e-8
  to 8.6e-8 against 5.0e-8 to 7.1e-8), and by at most 0.134 of the o band.
- Through upstream's chunked function the share is about 0.5 for every form, because the chunked
  function's own error sets the band; the rewrite changes it by at most +0.07 (real, max_abs).
- The division forms are slightly closer to upstream or to f64, but both are rejected by the
  exporter (§5: `aten.reciprocal`; `sum(keepdim=True)`) and would be 5 ops instead of 2.
- The real layer 0's g lies in [−2.81, 0): its decay is much slower than the random init's (down
  to −65).

### 6.2 The causal conv1d

Upstream: `F.silu(self.conv1d(x)[:, :, :T])` with `nn.Conv1d(padding=3, groups=8192)` (:501).
Rewrite: `F.silu(F.conv1d(F.pad(x, (3, 0)), w, padding=0, groups=8192))`, same weight. Input
N(0, 1), `[1, 8192, T]`.

| T | Conv output bit-identical | silu output max abs difference | silu of the same values, contiguous |
| --- | --- | --- | --- |
| 1, 2, 3, 4, 5 | yes | 6.0e-8, 1.2e-7, 1.2e-7, 1.2e-7, 2.4e-7 | bit-identical |
| 64, 512, 4096 | yes | 1.5e-8, 1.5e-8, 3.0e-8 | bit-identical |

The conv is bit-identical at every T. The silu after it is not, only because upstream's slice is
strided and torch's CPU silu takes another code path on a strided input: applied to the same values
made contiguous, it matches the rewrite bit for bit. In the IR, SiLU is `sigmoid` · `mul` in both
cases.

**Where the rewrite goes.** Replacing the `conv1d` submodule (row "conv" of §5) leaves `forward`'s
`[:, :, :T]` in the graph, and `slice` rejects it on the symbolic axis even though it now spans the
whole axis. `forward` has a second branch for the `causal-conv1d` package's function: `if
self.causal_conv1d_fn is not None: mixed_qkv = self.causal_conv1d_fn(x=…, weight=self.conv1d.
weight.squeeze(1), bias=…, activation=…, seq_idx=…)` (:492), with no slice. The attribute is set
in `__init__` (:421, None here), so the recipe can set it per instance as it does for the
recurrence; the replacement asserts `activation == "silu"` and `seq_idx is None`. It exports as
[pad, conv1d, sigmoid, mul] with a symbolic T.

### 6.3 `Qwen3_5RMSNorm` (the decoder layers' `x̂ · (1 + w)`)

Exported alone with `Dim("T")`: IR [`add`, `rms_norm`] — `_fold_rms_norm` folds the normalisation,
and `1 + w` stays a run-time `add` of the weight and a constant. The exported program is
bit-identical to the eager module. (Not part of the GDN layer; the decoder layers around it use it.)

### 6.4 The r = 2 head layout

`forward` repeats q / k with `repeat_interleave(2, dim=2)` (:524-525) before the recurrence call,
so the wrapper sees 32 heads. It exports as `reshape` → `expand` → `reshape` per tensor (the
`expand` makes the copy). Undoing it in the wrapper (rows "r2" of §5):

- `slice` takes only `step == 1`, so the strided form `x[:, :, ::2]` is not an option.
- The natural undo `reshape [B, T, Hk, r, d]` then index or width-1 slice on the r axis is rank 5
  and fails `slice of rank 5 cannot run on the strided kernel (rank ≤ 4)`.
- Two rank-4 undos export: `reshape [B·T, Hk, r, d]` + width-1 slice + reshape (53 nodes) and
  `reshape [B, T, Hk, r·d]` + `[..., 0:d]` (51 nodes, the smaller). Both read the first copy of
  each key head, which is the key head itself.
- The `expand` of the repeat stays in the graph either way. What r = 2 saves is the scan's q / k
  read (16 heads instead of 32) and half the l2norm work (the undo comes first); what it costs is 4
  more IR nodes. The spike measured the same bits for r = 1 and r = 2 (research
  [2026-10-27-gdn-scan-spike](2026-10-27-gdn-scan-spike.md) §5.2), not their speed difference.

## 7. The decomposed form's dispatches and traffic (entry condition ②)

**[estimate]** throughout this section: the counts come from the decomposed FX graphs of §3, not
from an IR graph (neither form converts) and not from a GPU run. Traffic counts, per non-view node,
the bytes of its tensor inputs plus its output (a `select` / `slice` counts its output twice), with
no fusion; a broadcast input counts at its broadcast size, which overstates the indexed write by up
to a third.

| Form, T | Non-view nodes (≈ dispatches) per layer | Traffic per layer | of which the indexed write (`where`) | Rest, per token |
| --- | --- | --- | --- | --- |
| recurrent, 64 | 1,305 | 1.74 GiB | 0.19 GiB | 24.8 MiB |
| recurrent, 512 | 10,265 | 24.41 GiB | 12.00 GiB | 24.8 MiB |
| recurrent, 4096 | 81,945 | 867.48 GiB | 768.02 GiB | 24.9 MiB |
| chunk, 4096 | 3,301 | 32.17 GiB | 18.03 GiB | 3.5 MiB |

- The recurrent form passes over the 2 MiB per-layer state about 12 times per token (24.9 MiB),
  and its functionalised output write touches the whole `[1, 32, T, 128]` output once per token,
  so its traffic grows as T².
- For the 24 GDN layers at T = 4096: about 1.97 million dispatches (24 × 81,945). The host-side
  encode alone, at 0.617 µs per dispatch (measured on the RTX 3080 Ti under Deno — research
  [2026-08-13-host-cost-decomposition](2026-08-13-host-cost-decomposition.md) §4), is about
  50.6 ms per layer and 1.21 s for the 24 layers. At the RTX 3080 Ti's 912 GB/s DRAM bandwidth the
  traffic would take about 1.02 s per layer (0.117 s without the indexed write); the 2 MiB state
  fits in the 6 MiB L2, so this is not a floor.
- The op: one dispatch per layer, no scratch, 2.56 ms per layer and 62.01 ms for the 24 layers at
  T = 4096 (spike §4.1, §4.2). Its own traffic is about 0.25 GiB per layer at r = 1 (q / k / v read
  and o written, 64 MiB each).
- Exporting the recurrent form costs 191 s per layer at T = 4096 (§3), for a graph valid at that T
  only.

## 8. Left for a GPU run

- The IR `rms_norm` kernel against torch for the l2norm rewrite (its reduction order is its own;
  §6.1 measures the formulation in torch only).
- The scan's time at r = 1 against r = 2 (§6.4), and whether the r = 2 undo's 4 extra nodes are
  repaid.
- The 47-node layer graph end to end once the product op, its handler and its CPU reference exist,
  against upstream's layer with the o band.
