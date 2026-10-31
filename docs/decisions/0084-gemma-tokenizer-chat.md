# 0084: Gemma tokenizer, detokenizer and chat template — compile-to-asset and the scope cut

- Status: implemented (accepted 2026-08-31 — the maintainer adopted the recommended option for the
  draft's open questions 6 and 9; stage 1a `a235d539`, stage 4 `088ad297`); partly superseded by
  [0127](0127-node-dawn-toolchain.md) (the CLI demo rules of the 2026-09-12 addendum, 2026-10-16;
  the measurement definitions re-graded 2026-10-18 to SHOULD)
- Date: 2026-08-31
- Scope: `packages/models/src/text/` (family-neutral common layer) / `packages/models/src/gemma/text/`
  (family side) / `tools/export-recipes/` (the compile script). IR, runtime and hub unchanged.
- Related: [0083](0083-generation-api-surface.md) (generation API — token events feed the streaming
  decoder) / [0085](0085-ple-host-gather.md) (PLE, same digest set) /
  [0048](0048-irodori-host-port.md) (decision 2: Unigram lives in `src/text/`; its addendum: **a
  preprocessing helper shared by TS and Python makes parity tautological**) /
  [0065](0065-exporter-core-recipe-split.md) (the compile script is recipe side) /
  [0079](0079-sbv2-two-layer-input.md) (text analysis is the caller's job; one conversion function) /
  [0038](0038-manifest-v1.md) (ownership of `pipelineConfig`)
- Basis: [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  §0.2, §5, §6 ("the draft §n"; measured values and the comparison of candidates live there)

## Context

When this was decided, `packages/models/src/text/` held only four encode-side files (`unigram.ts`,
`added-tokens.ts`, `code-points.ts`, `asset-gates.ts`) and **no detokenizer**. "String in, string
out" for gemma4 needs Gemma's SPM-BPE encode, a streaming decode and the chat format.

Reading the real assets (`inputs/gemma4/gemma-4-E2B-it/`, `inputs/embeddinggemma/google-300m/`)
established three facts that changed the review's assumptions (draft §0.2, §10):

1. **gemma4 and EmbeddingGemma share the BPE core but not the assets.** `.model.merges` is
   **bit-identical by sha256** (514,906 merges), but `.model.vocab` **differs at 6,206 of 262,144
   slots** (reserved-slot names — id 46 is `<|tool>` in G4, `<unused40>` in EG). `added_tokens` has
   **24 vs 6,415** entries; the post_processor adds **nothing in G4, `<bos>` … `<eos>` in EG**. →
   **The implementation can be shared; the assets cannot.**
2. **The Gemma tokenizer needs no Unicode tables.** Normalizer `Replace(" " → "▁")`, pre_tokenizer
   `Split(" ", MergedWithPrevious)`, decoder `Sequence[Replace("▁"→" "), ByteFallback, Fuse]` (same in
   both assets). **None** of Qwen2's three baked tables (`code-ranges`, `caseFold`, `nfcSegments`) is
   needed.
3. **The chat spelling is not Gemma 3's.** It is `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>`
   (source of truth: `sot_token` / `eot_token` / `soc_token` / `think_token` in
   `tokenizer_config.json`). `<start_of_turn>` is in the EG vocab but **not in G4's**.

## Decision

### 1. The tokenizer is compiled to an asset (the recipe checks the real `tokenizer.json` and compiles it to our own schema)

The export recipe (Python) compiles the upstream `tokenizer.json` to a small schema of our own.
**An unknown configuration fails loudly at compile time**, not at run time.

This is existing practice: Qwen2 bakes tables so that JS `\p{L}` / `NFC` / `toLowerCase` are not the
source of truth (`packages/models/src/text/qwen2-tokenizer.ts`); Unigram ports Rust's
`Lattice::viterbi` at code-point positions (`packages/models/src/text/unigram.ts`). **"Port the
implementation, bake the tables" is already the rule.**

The asset holds only **the vocab lines in id order, the merge rank table, 256 byteIds, the added
tokens and the special ids** (Context 2). The upstream `tokenizer.json` is **32.2 MB** (measured);
shipping it raw is waste.

MUST: **byteIds are built explicitly as 256 entries; a missing or duplicate entry fails loudly.**
`<0x00>`…`<0xFF>` are consecutive in both assets (238..493), so `unigram.ts`'s `base + byte` holds,
but that is **a measured fact about these assets, not a guarantee of the schema**; the lookup costs
next to nothing.

MUST: the distribution-corruption checks of `packages/models/src/text/asset-gates.ts` apply
unchanged (`assertUniqueLines`: line number = id, no duplicates; `setUnique`: no later merge
overriding an earlier one); a review fix had already made them family-wide.

### 2. Split — common processing in `src/text/`, acceptance schema and special tokens on the family side

`packages/models/src/text/` (**family-independent**, for the reason of ADR 0048 decision 2): BPE
merge (rank table + tie rule), UTF-8 reconstruction for byte_fallback, the streaming decoder's state
machine. **Decode is split the same way** (byte reconstruction common, special tokens family side).

`packages/models/src/gemma/text/` (family side): the acceptance schema (**exact match** of model
type, flags, normalizer, pre_tokenizer and decoder sequence); the special ids, the `<bos>` policy
and the post_processor equivalent (**G4 adds nothing, EG adds bos…eos** — Context 1).

MUST: **every module is free of side effects** (nothing such as a byte encoder is built at module
scope; the lazy-construction precedent is `qwen2-tokenizer.ts`). It is the precondition for
tree-shaking through the barrel.

### 3. BPE uses a merge queue from the start (the O(n²) full-scan splice is not ported)

Qwen2's `#bpe` scanned every adjacent pair and spliced the lowest rank, O(n²). **Gemma's
pre_tokenizer splits only on spaces**, so **for languages without spaces (Japanese, Chinese), URLs or
base64 one pre-token is the whole input**. The gemma4 goldens already include a Japanese case
(capital-ja, T=10); long Japanese input is a real use.

→ Port Rust's **merge queue and list structure**, and pin **rank and position ties and stale-pair
invalidation** with fixtures. Acceptance: "linear cost on long Japanese text" (stage 1a).

### 4. The detokenizer streams — UTF-8 carry-over is held in a state machine

The shape is `push(id) → committed string`. An incomplete UTF-8 sequence inside a byte_fallback run
is carried over in a buffer; only the committed part is returned.

MUST: **nothing but a byte run is buffered without bound.** A direct test pins this, so a "fake
streaming" decoder that holds every token until `finish()` cannot pass. Otherwise one iteration of
`for await` over ADR 0083 decision 2's `AsyncIterable` would mean nothing to the consumer.

### 5. Chat is one family-specific pure function; the first scope is plain conversation only (open question 6)

`gemma4ChatPrompt(messages, options) → token ids`. No generic template renderer (see Alternatives).

Why narrow: `chat_template.jinja` is **386 lines / 18,569 B** (measured), with tool declaration
formatters (`format_parameters`, `format_function_declaration`), thinking-channel ordering and
tool_call argument serialisation, but **plain conversation needs five spellings and three branches**
(`bos_token`, `'<|turn>system\n'`, `'<turn|>\n'`, `'<|turn>' + role + '\n'` with `assistant` mapped to
`model`, and the generation prompt `'<|turn>model\n'`). Precedent: ADR 0079; the recipe README also
says `<bos>` comes from the chat template, i.e. **the host's job**.

**Rejected input (fails loudly):** `tools`, `reasoning` (thinking), image or audio parts, and any
`role` other than `system|user|assistant|developer`. **Nothing is silently ignored** — "I passed tools
and they were not used" must not pass without an error. The scope widens when a real need appears.

MUST (owner of `<bos>`): **`encode(text)` does not add `<bos>`** (as the measured G4 post_processor).
**Only the chat function adds it**; otherwise chat would produce a double BOS.

MUST (same digest set): the product graph, weight shards, PLE sidecar (ADR 0085), compiled
tokenizer, BOS / special-token policy and prompt / chat format version are **bound into the same
distribution digest set**. The chat format and the EOS set (ADR 0083 decision 8) both come from the
checkpoint's side files (`tokenizer_config.json`, `generation_config.json`); **picked up separately,
one of them goes stale**.

### 6. The EmbeddingGemma asset compile rides along in stage 1a (open question 9)

Stage 1a also compiles the EG assets (the tokenizer part of the backlog item "complete
EmbeddingGemma"). The **implementation** in `src/text/` is shared, **not the assets** (Context 1), so
decision 2's split proves "one implementation, two assets". The EG pipeline, batch>1 export and
runtime attention_mask wiring stayed out of scope, in the backlog.

### 7. Verification (how parity fixtures are taken)

- MUST: **parity fixtures are taken in Python by calling `tokenizers.Tokenizer.from_file()`
  independently.** A shared preprocessing helper makes parity tautological (ADR 0048's addendum: a
  shared step hid a caption defect).
- The edges are **hit directly**: BPE rank and ties, metaspace boundaries, CJK, combining
  characters, non-BMP, adjacent AddedTokens, bytes 0..255, multiple byte fallbacks, invalid or
  incomplete byte runs, special-token decode.
- **Streaming check:** for short sequences, **every chunk partition** of `push()` concatenates to the
  one-shot `decode(ids)`, plus the direct test of the unbounded-buffer MUST.
- **Chat check:** id sequences match fixtures from HF `apply_chat_template`.

## Alternatives considered

- **The runtime parses `tokenizer.json`** (no compile script): ships and parses **32.2 MB** of JSON
  and scatters the handling of unknown configurations and the "JS Unicode / NFC is not the source of
  truth" discipline over run time. Rejected.
- **Extend `unigram.ts`** (smallest diff): Unigram (Viterbi lattice) and BPE (merge rank) are
  different algorithms. Rejected.
- **A generic (Jinja-like) template renderer:** contradicts "runtime dependencies are Web standard
  APIs only" (`CLAUDE.md`) and "no speculative generalisation". Rejected (needs a new ruling).
- **Leave chat to the caller** (hand-written `<|turn>`): consumers trip over the non-Gemma-3
  spelling (Context 3), and syncing with the EOS set leaks to them. Rejected.

## Consequences

- Stage 1a needed no GPU and ran in parallel with stage 1b; `packages/models/src/gemma/` became a
  family directory (tokenizer and chat first, the pipeline in stage 4).
- The compiled tokenizer and chat format version are members of the digest set (decision 5).
- The review's estimate was corrected twice: the encode side is **cheaper** (no baked tables), but BPE
  **needs a merge queue from the start** (decision 3).

## Addendum (2026-08-31) — public-surface review

Ruling records: `.claude/reviews/2026-08-31_182ced7/` (local, untracked).

- **`gemma4ChatTurn(tokenizer, message)` was added**, the incremental counterpart of decision 5's
  chat function. `gemma4ChatPrompt` renders the whole conversation each turn (O(n²) prefill over
  `GenerationSequence`), and a real consumer had guessed "render everything, strip the leading
  `<bos>`", which matched `chat()` at temperature 0 **without any guarantee**. The turn-local contract
  is canonical in the function and its doc. MUST: **the `<turn|>` that closes the previous turn is
  not included** — generation emits it, the sequence holds it as the uncommitted frontier
  (`pendingToken`) and the next `generate` prepends it (ADR 0083 decision 4); rendering it twice
  makes two separators without any error. A check covers `gemma4ChatPrompt(whole) =
  gemma4ChatPrompt(up to the first turn) ⧺ generated text ⧺ [<turn|>] ⧺ gemma4ChatTurn(next) ⧺ …` for
  arbitrary splits. MUST: **`assistant` is rejected** (generation fills model turns, and upstream
  folds consecutive assistant messages into one turn — re-render with `gemma4ChatPrompt` to insert
  one). It is valid only right after a turn closed with `<turn|>`; a turn cut by max-tokens or
  `break` continues by generation (`prompt: []`).
- **`parseGemma4PipelineConfig` was made public**, keeping decision 5's digest set on the consumer
  side. Callers who read `karume.json` themselves for `fromAssets` hold hub's
  `ModelEntry.pipelineConfig` (`Record<string, unknown>`) and would otherwise cast with `as` or
  duplicate three numbers (both diverge silently when the distribution changes). `fromAssets` now
  runs **the same checks** as `fromPretrained` (unknown keys, value ranges,
  `chunkLength ≤ capacity ≤ maxPosition`) before opening any bytes.
- **`Gemma4Pipeline.sampler` became `defaultSampler`**, and `Gemma4PipelineConfig.sampler` narrowed
  from `SamplerSpec` to `Gemma4DefaultSampler` (`temperature`, `topK`, `topP`, **all required**) —
  ADR 0083 decision 7 respelt. The old name read as "the sampler in use", but it is the **default**
  for requests that omit one. `SamplerSpec` would be wider than the accepted set (`logitBias`,
  `seed`, `repetitionPenalty` are not recommendations), exposing fields that type-check but fail
  parsing. Partial declarations are refused so that a combination found **nowhere upstream**
  ("recommended temperature, low-level default top-k") cannot arise.
- **Breaking changes (unreleased surface):** the rename, the narrowed type, and `fromAssets`
  rejecting invalid declarations before loading. All GPU goldens unchanged.

## Addendum (2026-09-12) — token notification and demo measurement

At the maintainer's request for speed comparisons, `Gemma4ChatOptions` and `Gemma4ChatTurnOptions`
gained an optional `onToken(id)`, called synchronously on the generation event before decoding. Stop
tokens are excluded as in the existing events; special tokens and tokens hidden by stop strings are
included. The function is captured when the request is issued, and its exception propagates as a
generation failure, as with `onPrefill`. Return values, the sequence and stop accounting are
unchanged.

The rules for the Deno CLI demos (warm-up by default, `--no-warmup`, the JSON fields `elapsedMs`,
`generatedTokens`, `ttftMs`, `decodeTokensPerSecond`) are superseded by ADR 0127 decision 5 on
2026-10-16: the demos were not ported and are to be rebuilt as browser applications. The rebuilt
applications SHOULD keep the definitions: TTFT runs from the start to the first non-stop token;
decode speed is `(non-stop tokens - 1) / (completion time - time of the first token)`; tokens are
counted as delivered before decoding, without re-tokenizing; TTFT with zero tokens, and speed with
zero or one token or zero elapsed time, are reported as not measured (re-graded 2026-10-18 to SHOULD:
the CLI demos they governed no longer exist).

## Addendum (2026-10-10) — Qwen2 BPE uses the same merge queue

Decision 3's "Qwen2's pre_tokenizer splits finely, so O(n²) does no harm" was wrong: letters without
separators and Japanese without punctuation are one pre-token under Qwen2's classes too, one
Japanese character is three byte-level symbols, and the 512-token truncation acts only per pre-token
(item R5-006 of the external deep review of 2026-10-04).

- Gemma and Qwen2 share `mergeSymbols` in `src/text/bpe.ts` (a priority queue by rank then position,
  invalidating stale entries). Only the byte-level mapping and rule lookup by spelling stay in Qwen2.
- Qwen2 numbers symbols locally within a pre-token; the vocab is needed only for the final
  spellings, and the accepted set equals the former full adjacent scan's.
- On the real vocab the id sequences matched the former implementation (2,036 inputs, 0 mismatches).
