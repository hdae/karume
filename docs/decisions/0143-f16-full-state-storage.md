# 0143: f16 storage of the full state slots (`fullStateStorage`)

- Status: proposed (2026-11-07 — implemented and measured on the development machine's RTX 3080 Ti under Node + Dawn; opt-in,
  default f32; in Chrome on the RTX 5070 Ti the check of the integer rounding of decision 3 passes)
- Date: 2026-11-07
- Related: ADR [0066](0066-generation-context-state-slots.md) (the generation context and its state slots) /
  [0110](0110-practical-tier-numerics-contract.md) (decision 2 the contract classes, decision 6 in-device determinism) /
  [0058](0058-numerics-opt-in-contract.md) (numeric changes are named options) /
  [0096](0096-speculative-decoding.md) (the self-tuning gate) / [0083](0083-generation-api-surface.md) (addendum of 2026-11-06,
  `capacity: "fit"`) / [0018](0018-f16-weight-execution.md) (the parity trap of packed f16) / perf-ledger K-100 / research
  [2026-11-07-gemma4-f16-kv](../research/2026-11-07-gemma4-f16-kv.md) (every number below)

## Context

Every Gemma 4 line stores its full-layer KV as f32: 12,288 bytes per position on E2B and 32,768 on E4B and 12B. That cost grows
with the capacity, while the sliding layers' cost is fixed by their window. llama.cpp's default KV type is f16, so per position
Karume holds twice that default.

K-100 assumed this needed f16 in the graph declarations and a re-export. It does not: the graph's states are f32 semantically,
and how a slot is laid out in a GPU buffer is the runtime's choice.

## Decision

1. **An explicit option, on both layers.** `SessionOptions.fullStateStorage: "f32" | "f16"`, default `"f32"`, and the same name
   on Gemma 4's `PipelineOptions`. It is not a quant setting and no manifest stores it, so no re-export and no change to a
   distribution. Only this explicit value turns it on (ADR 0058). No preset declares it.

2. **The form.** A slot in f16 is bound as `array<u32>`, two elements per word, and read with `unpack2x16float`. This is core
   WGSL, so it needs no `shader-f16` and composes with every compute form. The kernels' keys
   gain `:kv16` and their WGSL gains a read function. MUST: for f32 the key part and the function are the empty string, and each
   read expression is what it was, so existing keys and bytes do not move (pinned by the WGSL snapshots).

3. **One rounding point, class C, in integers.** The f16 `state_append` is the only rounding, and unpacking is exact. It rounds to
   nearest with ties to even in integer operations on the f32 bits, and MUST NOT use `pack2x16float`: WGSL leaves that function's
   rounding open, and it rounds toward zero on D3D12 (see below), so the slot would differ between backends. So
   the contract is class C of ADR 0110: atol = 0 against the f32 kernel over a slot rounded to f16 on the host. The current step's
   rows, read from `ins`, stay f32, so a token sees its own run in f32 and earlier runs in f16. The first run therefore matches
   f32 word for word, and prefill chunking moves the numbers deterministically. S, its statistics and the outputs stay f32.

4. **Full slots only.** A sliding slot's cost is fixed by its window, so halving it buys little, and every sliding kernel would
   need a second form.

5. **An even `D`, failing loudly.** One append invocation owns one word. With an odd `D` the last element of a row would share its
   word with the first of the next, and writing half a word is a read-modify-write that loses values silently. An odd `D` fails at
   `createGenerationContext` and in the estimate.

6. **The fused stats + PV stays f32.** `stateAttentionReduce: "parallel-fused"` runs as `"parallel"` on an f16 slot. The fused
   kernel needs a column limit of at most 1024 and a full slot's column limit is its capacity, so the two meet only at capacities
   where halving the slot buys nothing.

7. **A borrower matches its lender.** `createGenerationContext({ borrow })` fails loudly on a different storage: the bytes are
   shared, so a split would bind a packed slot to an f32 kernel with matching shapes and no exception.

8. **The self-tuning speculation gate is refused.** Under f16, verification rows j ≥ 1 read `ins` in f32 while plain decode reads
   the same rows in f16, so the two paths are not bit-identical, and ADR 0110 decision 6 allows a host policy to choose by wall
   clock only between bit-identical paths. A turn MUST say `speculative: "always"` or `false`, which is refused at turn start
   otherwise, and `speculative.gate` is refused before anything is fetched.

9. **Not the default yet.** It changes numbers, and its speed and numbers in Chrome are not measured. Whether a preset should declare it is a later ruling
   that needs at least the Chrome result.

### Why not `pack2x16float`

The first implementation used it. Its check passed on Dawn on Vulkan and failed in Chrome on D3D12 on the RTX 5070 Ti (2026-11-07):
the readers matched and all six append cases mismatched, in 1,393 words. Assuming rounding toward zero predicts the failing word
count of every case (2 / 5 / 11 / 196 / 392 / 787) and the first failing word (research §6).

Rounding in integers also defines what WGSL leaves open: a value from 65,520 up is infinity and NaN stays NaN, with no effect on the
other half of the word, and a subnormal f16 is written like any other. The check runs the append on every rounding boundary of f16,
and it passes in Chrome on that card (2026-11-07).

The attention scores made the opposite choice for the same function (ADR [0031](0031-attention-score-f16-storage.md) addendum of
2026-11-07). The difference is the cost: S is stored in the hot loop of ①QK, where integers cost 3–5% of the attention route, while
an append runs once per token and showed no speed change on the RTX 3080 Ti (research §3). The KV path is also unreleased, so
integers change no number anyone has.

## Acceptance

K-100 had no line, so this ADR sets it. Measured values are in research §2–§4.

| Line | Result |
| ---- | ------ |
| KL against f32 torch no greater than the existing Karume f32 baseline (E2B 7.0e-4, 12B 8.7e-3) | E2B 2.07e-7, 12B 5.61e-4 |
| Top-1 agreement on the compared rows | every row |
| The same answers on the 12B retrieval prompt | the same 85 tokens, 8 of 8 in both |
| Kill: decode about 10% slower on `i4-fast` at long context (a judgement, not a derived number) | E2B +6% at P ≈ 32K and none at P ≈ 127K; 12B −13% at P ≈ 32K and −22% at P ≈ 62K |

The KL lines bound the rounding itself, measured in torch; Karume's implementation is held to the host rounding at atol = 0 by
the GPU tests, and its tokens matched f32 in every measured arm. Those arms ran on Dawn on Vulkan.

## Alternatives

- **`f16` as a declared graph type, with a re-export.** It would put a storage choice into the distribution and need the
  `shader-f16` feature. The packed form needs neither, and a user can turn it on over existing distributions.
- **`pack2x16float`, accepting each backend's rounding.** It would keep the shader short, but it leaves Node and Chrome with
  different numbers and a contract that holds on one backend, and rounding toward zero biases every stored value toward zero.
- **Rounding the current step as well**, so a token reads itself in f16. As far as was read, this is the shape of llama.cpp,
  which writes a step into the cache before attention; its source was not read in full. Karume's order keeps the first run
  word-identical to f32, which makes the check simple.
- **Sliding slots too.** See decision 4.

## Consequences

- 12B saves 16 KiB per position (1 GiB at capacity 65,536), E2B 6 KiB. `estimateSessionMemory` and `fitCapacity` count it: `fit`
  on the 12 GB development machine moves 12B from 32,768 to 65,536.
- 12B decodes faster at long context on that machine, which K-100 had not expected. The cause is not known (research §3).
- Generated text can differ from an f32 run on any turn after the first run. The tokens matched in every arm measured so far, which
  is an observation and not a contract.
- Each f16 attention shape adds a pipeline per key, and each kernel carries the storage parameter.
