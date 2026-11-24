# Backlog — the source of truth for wave order and work items

> This file alone is the source of truth for the project's **priorities, wave order and unfinished items**.
> Status vocabulary: `now` (current wave) / `next` (the next big wave) / `later` / `release` / `parked` (with a revival condition).
> Operating contract: ① completed items are **deleted** (git and the ADRs / research hold the history) ② measured values and design arguments are
> **not written** here — point to the source (ADR / research / ledger) ③ the source of truth for filing, adopting or rejecting performance candidates and for their kill criteria is
> [perf-ledger](perf-ledger.md); this file only refers to them as waves ④ the source of truth for by-design constraints is
> [limitations](limitations.md) — a constraint appears here only when a ruling turns it into work.

## now — after the 0.23.0 release (released 2026-11-20)

0.23.0 is out: the five JSR packages (`@karume/runtime`, `@karume/hub`, `@karume/safetensors`, `@karume/convert`,
`@karume/models`), every distribution repository published again as `karume/6` and pinned, and the published-smoke
green on Node and on Deno (17 manifests and the sbv2 `fromPretrained`). What it shipped is the CHANGELOG's
`[0.23.0]`. The uploads' fragmentation tables read no FAILED row; of the 568 files of 64 MiB or more one is below 10 MiB
per term, a part of `karume-anima` that this release did not upload (the `later` item "Two anima parts read as
fragmented"). The order of what comes next is the maintainer's; the items put after this release are the first group of
`later`.

- **Regular Gemma 4 `bf16-fast`** (perf-ledger K-110; published with `karume-gemma4` on 2026-11-20): the sha256 rows
  `e2b/bf16-fast` and `e4b/bf16-fast` of an environment other than the RTX 3080 Ti's, which has them (written only on
  the maintainer's OK). E2B's speculative decode is in the first item of `later`.
- **Irodori**: v4.1-small-mf has no self A/B band for its `i8-a8` (ADR [0150](decisions/0150-irodori-meanflow.md)), and
  v4-Large's Playground page has not been run in a browser (ADR [0151](decisions/0151-irodori-v4-large.md)). Their
  order is not ruled.

## next — Gemma 4, in the maintainer's order (2026-11-08)

The items go in this order. The candidates of the other families wait until Gemma 4 has been gone through (the
maintainer's ruling of 2026-11-06).

1. **26B A4B (MoE)** (maintainer 2026-11-08): every expert resident (ADR
   [0145](decisions/0145-gemma4-26b-moe-resident.md), released in 0.19.0) and the exact expert cache with its host tier,
   self-speculation and per-layer slots (ADR [0146](decisions/0146-exact-expert-cache.md) and its addenda, released in
   0.20.0) are built; the measurements are [research 2026-11-08](research/2026-11-08-gemma4-26b-moe-cache.md) and
   [research 2026-11-10](research/2026-11-10-gemma4-26b-fast.md), the grouped `expert_linear` is perf-ledger K-104. The
   cache is built even if slow (the maintainer's ruling of 2026-11-09: a model whose experts do not fit in VRAM becomes
   possible in principle). Left:
   - the drafter and vision of 26B;
   - the grouped `expert_linear` in a chat's decode in Chrome on the RTX 5070 Ti (its `kernel-check` identity passed
     there on 2026-11-19);
   - a depth sweep of self-speculation in Chrome (the Node sweep is research 2026-11-10 §7);
   - drafts chained on the GPU;
   - the load after a cut for its last row alone;
   - paying a miss as a transfer rather than a re-run (the PoC of
     [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md) paid it with one fence; the `later`
     item "MoE page-fault").
2. **Vision on E2B / E4B** (later by the maintainer's ruling of 2026-11-07; ADR
   [0144](decisions/0144-gemma4-12b-vision.md), Context: E2B and E4B have a real vision tower and causal attention, and
   need the product graph's input for image vectors but not the marking of which rows are one image). 12B's vision
   shipped with 0.18.0.

## later

- **After the 0.23.0 release (filed together on 2026-11-19; the order among them is not ruled)**:
  - Put after the release by the maintainer's rulings of 2026-11-19: the GEMV derivation's evidence rules
    (perf-ledger K-113; built 2026-11-20; the registered tables derived again under them 2026-11-21), and the bf16 shapes
    that could take a parallel row of the default profile — both in item 6 of "The move to maintained specs (ADR 0152)" below; the first
    publication of the 2.9B Anima distribution — "Keep bf16 where the upstream is bf16";
    and EmbeddingGemma 2 ("EmbeddingGemma 2" below).
  - Put after the release by the maintainer's ruling of 2026-11-20: `apple-metal-3` keeps matching the M2 alone, and an
    Apple adapter that reports no description (the M5) gets a table derived from its own records — on the M5 the M2's
    parallel rows are faster than the default's but its sequential choices are not, and part of them were not compared
    word for word there ([research 2026-11-19](research/2026-11-19-gemv-tables-across-devices.md) §7). Built
    2026-11-23 in another form (ADR [0156](decisions/0156-shared-geometry-profile-tables.md)): the M5 has a measured
    table, `apple-m5`, matched by its description, from its full geometry sweep; an Apple `metal-3` adapter that
    reports no description gets the shared table `apple-metal-3-shared`, derived from the M5's sweep and three M2
    records of the M5's geometries ([spec](spec/geometry-profiles.md#shared-tables)). `apple-m5` carries the M5's
    own GEMV choices, derived from its two GEMV sweep records of 2026-11-23 (43 sequential exceptions, 66 parallel rows
    changed or added — [spec](spec/geometry-profiles.md#the-gemv-choices-of-linear)); `apple-metal-3-shared` runs the
    default profile's (a shared table carries tile geometries only). Left, in item 6 of "The move to maintained specs
    (ADR 0152)": a rule for a shared table's GEMV choices; an M2 full
    geometry sweep, which re-derives `apple-metal-3`'s tile families and lets the shared table compare every candidate
    on both devices; the M1, M3 and M4 are not measured.
  - Here without a ruling on their timing: the two items below.
  - **E2B's speculative decode under `bf16-fast` against `bf16`** (perf-ledger K-110): not measured, because the
    workbench's `gemma4-bench` entry has no drafter arm, so E2B's sentence on speed says "without the drafter"
    (`GEMMA4_SPEED_NOTE` in `tools/export-recipes/gemma4/distribution.py`); it needs a drafter arm in that entry. The
    maintainer's Playground chats with the drafter on the RTX 5070 Ti (2026-11-19) gave sound replies; they are not an
    A/B.
  - **Small checks found while preparing 0.23.0 (filed 2026-11-19; none approved yet)**:
    - The Wan recipe has no build from published containers pinned by digest, which Anima's and Irodori's have: a
      rebuild needs the series, or series folders laid out from a mirror by hand (0.23.0 did the latter). Filed
      2026-11-20, after the release by the maintainer's ruling.
    - Nothing compares the versions of the five JSR packages with one another or with the `^` pins between them
      (`packages/models/deno.json`, `packages/convert/deno.json`): `verify_package_manifests_test.ts` compares each
      package's `deno.json` with its own `package.json` only.
    - `packages/runtime/tests/verify_test_imports_test.ts` names the packages in a hand-written pattern
      (`runtime|hub|safetensors|convert|models`), so a new package is added there by hand.
    - `tools/published-smoke` imports only `@karume/hub` and `@karume/models`; the other three packages are not imported
      by it directly.
    - The public-surface helper (`packages/runtime/tests/helpers/public-surface.ts`) writes
      `tests/fixtures/public-surface.json` under `KARUME_SURFACE=write` without creating `tests/fixtures/` when it is
      missing.
    - `tools/lang/check-doc-links.ts` accepts an absolute filesystem path as a link target when the file exists on the
      machine that runs it; such a link resolves nowhere else (CI went red on one, 2026-11-19). Proposed: reject
      absolute targets.
  - **Found while adding EmbeddingGemma 2 (filed 2026-11-21; none approved yet)**:
    - Two cases of `tools/export-recipes/irodori/tests/test_tokenizer_ref.py` (`TestTableReproducesSplitter`) end
      as errors, not as explicit skips, in an environment without `tokenizers` that holds the assets: the fixture
      helper `_unigram_tokenizer` imports `tokenizers` without `importorskip`. `uv run --locked --isolated pytest`
      exits 1 on the development machine because of them; CI skips them (it has no assets).
    - The Families table of `tools/export-recipes/README.md` and `docs/assets-layout.md` have no row for the
      `embeddinggemma2` recipe and its upstream snapshot under `inputs/embeddinggemma2/`.
    - The attention hook of the Gemma 4 recipe refuses `attention_mask=None`, so the EmbeddingGemma 2 recipe carries
      a bidirectional hook of its own; one shared helper for the Gemma-line recipes is not built.
  - **Moving the pinned transformers from 5.14.1 to 5.23.0 (filed 2026-11-21; a task of its own by the maintainer's
    ruling of that day, each difference in a result confirmed one by one when it is done)**: the pin stays 5.14.1 for
    every dependency group and every per-run `--with` but the EmbeddingGemma 2 recipe's, which adds 5.23.0 per run. A
    trial move on 2026-11-21 was taken back; it locked with nothing published after 2026-11-14 (transformers 5.14.1 →
    5.23.0, tokenizers 0.22.2 → 0.23.3 and huggingface-hub 1.27.0 → 1.33.0, both required by it; uv has no standing
    cut-off here — a lock made with `--exclude-newer` records the option and a later `--locked` run fails until a
    plain `uv lock` removes it). What the trial found, the cases called red being red under 5.23.0:
    - The Gemma 4 recipe does not run on 5.23.0 (183 red cases): `Gemma4TextConfig` keeps `head_dim` /
      `num_key_value_heads` per layer (`global_head_dim` / `num_global_key_value_heads` are no longer attributes, and
      reading a per-layer attribute on the global config raises), the tiny test configs whose KV-shared layer has
      no earlier layer of its type are refused by the config, and `Gemma4UnifiedVisionEmbedder.forward` returns
      `BaseModelOutputWithPooling` (the projected features are `pooler_output`). Upstream's RoPE tables for E2B /
      E4B are bit-identical to 5.14.1's.
    - The Qwen3.5 text tower recipe does not run (15 red cases): `Qwen3_5GatedDeltaNet` no longer holds the
      instance attributes its rewrites replace (it calls module functions under the kernel-hub decorators), and the
      module no longer has `chunk_gated_delta_rule` or `is_fast_path_available` (`moss.text_embeds` reads the latter).
    - Upstream rewrote `torch_chunk_gated_delta_rule`: the exporter's copy (of 5.14.1) differs from it by up to
      3.7e-9 on under 2 % of the elements, so the bit-for-bit case of `test_gated_delta_scan.py` and the band case of
      `tests/test_gated_delta_fixture.py` are red; the runtime fixture `gated-delta-scan.safetensors` would get
      other bands.
    - umT5 attention now goes through the attention interface (SDPA by default; eager no longer upcasts the
      softmax to f32): the spelled-out encoder and its references are no longer bit-exact with upstream (8 red
      cases), and the published umT5 / Wan tokenizer assets record 5.14.1, so the distribution gates refuse them
      (`test_the_plan_passes_every_gate` is red) until they are regenerated; the Wan text embeddings would change.
    - Re-exported on 5.23.0, Anima's text encoder graph has 56 more nodes (a reshape after SDPA and a permute per
      layer, from `sdpa_attention_forward`) with bit-identical eager outputs; every recipe that exports through
      transformers' `sdpa` would change its graph at its next export.
    - tokenizers 0.23.3 folds the grapheme tables of Anima's T5 normalizer differently (newer Unicode data:
      `extend` gains 6 ranges and widens 2, `prepend` loses U+11A3A); `packages/models/tests/fixtures/anima-text/parity.json` and
      the distributed T5 table would change when regenerated. No check is red.
    - Not re-checked on 5.23.0 (assets absent here): the umT5, Wan and Qwen3.5 tokenizer fixtures, the MOSS stage-0
      records, the real-weight exports of siglip2, birefnet, depth-anything, deberta and
      sbv2.
    - Comments in `packages/models/src` still name transformers 5.14.1 as the source of truth (the T5 / umT5 /
      Qwen2 / Qwen3.5 tokenizers, the image preprocessing, the umT5 relative positions, Gemma 4 vision).

- **The move to maintained specs (ADR 0152) (filed 2026-11-15; ADR
  [0152](decisions/0152-maintained-specs-and-declared-numeric-requirements.md) accepted; the maintainer confirmed this
  order on 2026-11-15)**. Each item is its own commit with its behaviour tests. Item 1 has landed; what has landed of
  items 2 to 7 is described by the specs it cites. Left:
  2. **Records of a run** (the WGSL a test dispatches, `outputs/references/`, the verification cache's records):
     - A mismatch reports the differing bytes of the two artifacts; a distance per artifact type (pixels, samples,
       tensor values) is not built.
  3. **Kernel selection as one table per op and dispatch programs**
     (`packages/runtime/src/runtime/linear-strategies.ts`, `state-attention-strategies.ts`, `linear-program.ts`;
     [research 2026-11-15](research/2026-11-15-linear-splitk.md) for the split-K linear):
     - No op spec object was made: nothing would read it yet.
     - The fused attention was not reshaped: `attentionRoute`, `fusedAttentionViolation` and `attentionHeadMajorCopies`
       are already the single decisions its consumers call. Its kernels are not selected by level; its levels are
       recorded when item 4 selects by them.
     - The linear→static_quantize rule keeps its own conditions and its own builder; the memory estimate and the
       verification cache do not read dispatch programs yet.
     - The split-K linear: f32 storage only; not counted by the memory estimate; no module in `packages/kernel-checks`
       (so no browser check); slower than the reference kernel at K = 1536. A fast form would reuse the GEMM skeleton.
     - WGSL snapshots of strategies other than reference kernels are still byte fixtures.
  4. **Requirement levels** ([spec/numerics.md](spec/numerics.md)):
     - The regression corpus ([spec/verification.md](spec/verification.md#the-regression-corpus)): entries for the f16
       compute paths and the 26B MoE graph (388 MiB per layer); rows for environments other than the development
       machine's; the kernel keys that only other step shapes select. Checking a `reorder` strategy inside its op band
       on the corpus's activations (replaying one op on values taken from an `exact` run) is built when a defect shows
       that the kernel checks' own inputs miss something (the maintainer's ruling of 2026-11-16).
     - The derived applicability check that replaces the `FamilySessionPolicy` tables (ADR 0147 follow-ups, code item
       4).
     - The subgroup GEMV, the subgroup RMS reduction and the byte-plane GEMV are `reorder` strategies with no
       end-to-end gain measured on any adapter, so no geometry profile lists them and no Session selects them; they run
       through the private key `REORDER_STRATEGIES` (the kernel checks and tests). A measured gain lists a strategy in
       a table.
     - `reorder` beside `linearCompute: "a8"` / `"f16"` is accepted and leaves linear on the approximation's kernels.
     - The fusions at `exact` on a backend where a fused kernel does not give the unfused bits: recorded and accepted
       (the Metal ruling of 2026-11-11/12); no run-time check drops a rule.
  5. **Verification, second part** ([spec/verification.md](spec/verification.md#the-verification-cache); the default
     mode stays `record`):
     - Making `reuse` the default, after it has been used on the long end-to-end files (by 2026-11-16, 18 recorded runs
       had judged 2,560 predictions, 544 of them a reuse and none wrong, mostly of short files — counted only here), and
       the release rule "an uncached full run when the environment changed".
     - Leaving the runtime's host code out of the key of the models' e2e files (ADR 0152 decision 6) is put off (the
       maintainer's ruling of 2026-11-16): commits that touch the runtime touch kernels, tuning tables, strategy
       selection and fusion, rarely the executor alone, so it barely moves their re-run time (measured in commit
       `feaed7b6`), and a wrong classification would reuse a stale result.
     - Kernel changes re-running only the files that compile them: the one lever for commits that touch the runtime.
       It needs the WGSL a file compiles derived without running it (an open point of ADR 0152).
     - What the key does not see ([spec](spec/verification.md#what-the-key-does-not-see)); a file whose every case is
       skipped writes no record, so it runs each time.
  6. **The workbench** (ADR 0152 calls it the harness; [spec/workbench.md](spec/workbench.md)). It stays a local
     application for tests and measurements on the maintainer's machines (the maintainer's ruling of 2026-11-19): no
     public build of it is hosted, and its production build keeps working as the spec describes. Left, in this order:
     - Where per-GPU evidence records are kept for good (`outputs/workbench/` is outside git; ADR 0147 follow-ups, code
       item 8, decides it).
     - The GEMV choices of the default profile and of the device tables
       ([spec/geometry-profiles.md](spec/geometry-profiles.md#the-gemv-choices-of-linear); perf-ledger K-110, K-112,
       K-113). Direction (the maintainer's ruling of 2026-11-18): the default keeps a row only where no measured device
       is slower with it than with the sequential GEMV, chosen over the records of every measured device, and a
       device's own best stays in its table. The ruling answered the RTX 5070 Ti, on which 51 of the default's 66
       parallel rows then derived to another choice (a median ×1.79 per dispatch — counted only here). The default's 22
       bf16 sequential exceptions (2026-11-18) and 17 bf16 parallel rows (2026-11-19) follow it by hand, and the three
       device tables are derived from their records against the current default (2026-11-19, and again under K-113's
       rules on 2026-11-21). Open:
       - the rule itself (to be proposed with the three devices' overlap in numbers), the generator deriving the default
         from records of several adapters, and GEMV sweep records of the devices collaborators measure. The rule has to
         say what removes a row of the default from a device's table: the derivation judges a candidate against the
         default's choice for the shape, so a device drops an inherited exception only when its records show the
         general variant ×1.10 faster in every group, and a device on which the row is slower by less keeps it;
       - the derivation's evidence rules are built (perf-ledger K-113, 2026-11-20): a record stands wherever it timed
         the kernels the current default runs, so a row added to the default no longer discards the other devices'
         records of the shape. The three registered tables are derived again under them (2026-11-21, on the
         maintainer's review of their diffs; `nvidia-ampere` from 24 records of the card's turbo mode and its driver of
         2026-11-20 — the records before that are of another driver and are not mixed with them). Left: the
         development machine's sha256 rows that follow its new choices, re-recorded on the maintainer's approval of
         their list, and the bf16 shapes that could take a parallel row of the default (`2560 × 256` and
         `1536 × 2048`; `8960 × 1536` is slower than the GEMM at 8 rows on the RTX 3080 Ti);
       - where a table's choice at `reorder` is slower than its choice at `exact` (the three devices' GEMV sweep records
         since 2026-11-17, per dispatch, read 2026-11-19; rows with packed activations are not measured; perf-ledger
         K-113): on `nvidia-ampere` 37 groups at 4 and 8 rows (×1.01 to ×5.06), shapes the two other tables hold
         row-blocked rows of their own for; on `nvidia-blackwell` only the output heads at 4 and 8 rows (int4 g32
         `262144 × 3840` of 12B ×1.44 and ×1.57, `262144 × 2816` of 26B ×1.48 and ×1.68, and the int8
         `262144 × 2560` no distribution ships), which a speculative decode's verify run reaches; on `apple-metal-3` 11
         groups, ×1.00 to ×1.12;
       - `apple-metal-3` matches only the Apple M2's description, and stays so (the maintainer's ruling of
         2026-11-20, from the Apple M5's records of 2026-11-19 —
         [research](research/2026-11-19-gemv-tables-across-devices.md) §7): an Apple adapter that reports no
         description gets the shared table `apple-metal-3-shared` (2026-11-23, ADR
         [0156](decisions/0156-shared-geometry-profile-tables.md)), which carries no GEMV choices, and the M5 with its
         description `apple-m5`, which carries the M5's own (from its GEMV sweep records of 2026-11-23). On the M2
         the default's 22 exceptions stay although per dispatch its records time the general variant faster on most of the shapes they
         time, because end to end they were faster (perf-ledger K-112); the per-dispatch comparison, the five
         exceptions `apple-metal-3` keeps and the shapes compared word for word on Metal are in
         [limitations](limitations.md) (the section on the linear GEMV family);
       - a table holds the field as a whole, so every table holds a copy of the default's thread target and general
         variant and of the rows its records do not replace: a change of those in the default regenerates every table
         that carries the field (a core-lane test catches the drift);
       - the memory estimate plans its fusions with the default profile's choices, so a table with parallel rows on
         more shapes makes it an approximation of that adapter's plan where a linear of such a shape feeds a
         `static_quantize` ([limitations](limitations.md); no graph of a Gemma 4 distribution has one). An estimate that
         takes the adapter's table, or its GEMV choices, would remove it;
       - the end-to-end decode with a device table since its derivation of 2026-11-19 is measured on E2B only, on
         `nvidia-blackwell` and `apple-metal-3` (perf-ledger K-110); `nvidia-ampere`, E4B, 12B, 26B and a speculative
         decode are measured with no table's choices since then
         ([research 2026-11-18](research/2026-11-18-blackwell-gemv-table-decode.md) is of `nvidia-blackwell` before
         that derivation). What the planner (`--gemv-plan`) still lists against the current default on the records the
         tables were derived from: on `nvidia-blackwell` the int8 `262144 × 1536` case at one row; on `apple-metal-3`
         three cases at one row (int8 `262144 × 1536` and `1024 × 256`, int4 `3840 × 4096`); on `nvidia-ampere` 290 of
         the full set's 345 groups (counted only here);
       - no test runs a table's parallel rows on their own shapes on the development machine (an injected table would
         allow it): their evidence is the device's sweep records;
       - the choice between sequential variants a few percent apart moves with the records given, and the narrowed
         sweeps measured four hand-picked variants (the planner still lists the others).
     - A benchmark application for collaborators, apart from the workbench (the maintainer's direction of 2026-11-19;
       the maintainer has an AMD device in reach). A collaborator opens it, runs one short measurement and sends the
       result from the page itself — to a form (for example a Google Form) whose address the build reads from the
       environment (`.env`), so that no record travels by hand and the repository carries no address. Not designed
       yet. Open: what it measures and for how long (an exploratory run in place of the full sweeps — the analysis of
       the records at hand is `outputs/diag/sweep-exploration-2026-11-19/`, untracked), what the sent result holds
       and its size (a form field does not take a sweep record of megabytes), whether sending from a cross-origin
       isolated page works with the form's endpoint, what it tells about the device, where it is hosted and the
       build's commit named on the page.
     - Per-adapter tables from sweeps beyond what has landed (the GEMV sweep and its derivation, the generator's
       `--gemv-records` and `--gemv-plan`, the K-88 skeleton forms as geometry-sweep candidates —
       [spec/geometry-profiles.md](spec/geometry-profiles.md)). Open:
       - the Apple tables (2026-11-23, ADR [0156](decisions/0156-shared-geometry-profile-tables.md); `apple-m5`'s GEMV
         choices are derived from the M5's GEMV sweep records, done 2026-11-23): an M2 full geometry sweep, which
         re-derives `apple-metal-3`'s tile families (carried from records of `karume-geometry-sweep/3`, below) and the shared table
         `apple-metal-3-shared` with every candidate timed on both devices (the M2's records of 2026-11-23 time only the
         M5's adopted geometries); the M1, M3 and M4 are not measured; a rule for a shared table's GEMV choices and
         attention key axis (a shared table derives neither);
       - the tile families of `nvidia-ampere` and `apple-metal-3` are carried over from records of
         `karume-geometry-sweep/3`, which no longer derive a table: each is derived again from a new sweep on its
         device, the `gemm` and `conv2d` families from one that measures the K-88 forms (on the RTX 3080 Ti a case of
         that sweep heats the card: in chunks);
       - the K-88 shapes (the Gemma 4 prefill linears, the Wan2.2 and anima VAE conv2d) as cases of the sweep's case
         sets, which moves those families' case-set versions;
       - the two drafter shapes with a large gain whose row-block baseline was unsteady (`256 × 2048`, `256 × 5120`),
         and a parallel form of the int4 output heads that is not slower at four and eight rows (perf-ledger K-101);
       - the bf16 output heads, whose general variant ran at two speeds from record to record on the RTX 3080 Ti: under
         the rules of K-113 a candidate there is read against the variant's fastest time, and one that itself ran at
         two speeds is not adopted (bf16 `262144 × 2560` at one row);
       - a rule for the row-block thread target from the `row-blocks` set; the shared sequential keys whose cases
         disagree (the key has no group length); the packed rows (not measured); the integer and byte-plane GEMV
         tables, which are still in the kernel source;
       - the geometry sweep's rounds taken geometry after geometry (interleaving them as the GEMV sweep does needs a
         steadiness rule from evidence to replace the drift ratio its derivation reads); its derivation accepting or
         refusing a record per op family (the record's measured-candidate fingerprint is one value per record); the
         quick look on the geometry sweep's evaluation set `k88`, whose summary compares every candidate with the
         table's (run it at margin 0).
  7. **Weights from outside the exporter** ([spec/single-file.md](spec/single-file.md),
     [spec/convert.md](spec/convert.md), [spec/safetensors.md](spec/safetensors.md); manifest `karume/6` — ADR
     [0153](decisions/0153-manifest-v6-name-map.md)). `fromSingleFile` on the pipelines, the counterpart of diffusers'
     `from_single_file`: a single file is converted while it loads, against the plan of a preset of a distribution's
     manifest, and keeping the result as a `krm` is optional (the maintainer, 2026-11-17). The format readers are their
     own packages (`@karume/safetensors`, later `@karume/gguf`) and the shared part is `@karume/convert`; hub and models
     hold no reader. GGUF is deferred ([research 2026-11-16](research/2026-11-16-gguf-and-quantized-kernel-generation.md):
     what its types need from storage and kernels); the ledger's symmetric-only rule (no zero point) is re-evaluated when
     a format that needs a minimum is taken up (the maintainer, 2026-11-16). Built for Anima. Left: writing the
     converted container as a `krm` from a pipeline (designed next; `retainConverted` keeps it in host memory only);
     `fromSingleFile` on the other families; int4 targets; LoRA merged before quantization (Anima, then Wan2.2); the
     runtime's `parseSafetensors` moved into `@karume/safetensors`, early, so that one parser reads the header (a
     breaking change of the runtime's surface); a GPU run with an F16 file of the base's 28 blocks (none is at hand);
     example apps per model on request.
  - **Specs** are written as each part lands; the existing `ir-v2.md`, `container-v1.md` and `quantization.md` move
    under `docs/spec/`, one move per commit.
  - **To decide**: the open points of ADR 0152 (its section "Open").

- **Keep bf16 where the upstream is bf16 (filed 2026-11-17; the maintainer's direction; for Anima decided 2026-11-18 —
  ADR [0154](decisions/0154-anima-bf16-preset-names.md))**. A component whose upstream checkpoint is BF16 and that a
  preset does not quantize to integers is stored as bf16, the upstream bits, instead of being rounded to f16: what a
  preset does not ask to quantize follows the precision of the source (bf16 execution: ADR
  [0131](decisions/0131-bf16-weight-execution.md)). Anima follows it in the recipe, the TypeScript side, the local mirrors
  and the RTX 3080 Ti's sha256 rows (CHANGELOG, Unreleased); the maintainer approved the default preset's images, the
  notice text and the preset labels and descriptions on 2026-11-19. The numerics check of bf16 storage computed in f16
  (`packages/runtime/tests/gpu_f16_compute_test.ts`) has not run on D3D12 or Metal.
  - Left: the unpublished 2.9B Anima distribution takes the new names before its first publication, which comes after
    the 0.23.0 release (the maintainer's ruling of 2026-11-19; its order among the post-release items is not ruled).
    Anima's two repositories were published as the bf16 builds on 2026-11-20.
  - MOSS-Image (its BF16 DiT and text encoder ship only as int8), the Gemma 4 E2B drafter (int8 only) and the families
    whose upstream is F32 (Irodori, Wan, Lucida — stored as f16): bf16 presets for them are, in the maintainer's position,
    probably not done — they run well quantized, and bf16 for them would be too heavy for a browser.
  - Still to inspect: the families whose upstream dtype was not read (SBV2's front and voice),
    and whether the `:wbf16` GEMM kernels (M > 1) have their geometry per adapter: the geometry sweep's linear cases
    measure f16 storage only (`CaseWeight` in `packages/runtime/src/tune/cases.ts`); the GEMV side has bf16 rows.

- **ADR 0147 follow-ups — numerics and kernel adoption (filed 2026-11-10; ADR [0147](decisions/0147-numerics-and-kernel-adoption.md)
  accepted; the order among them and against the other items is the maintainer's)**. Code items 1 and 2 (the three
  fusion keys out of `SessionOptions` and out of the manifest vocabulary) landed on 2026-11-15; their re-upload, source
  pins and published smoke are part of the 0.23.0 release (`now`). Code item 3, the Session canary, is not built (ADR
  [0152](decisions/0152-maintained-specs-and-declared-numeric-requirements.md) decision 3). The remaining items keep
  their numbers. Each code item is its own commit with its behaviour tests and states its blast radius:
  - **Code** (the decisions name what each changes):
    4. The derived applicability check at admission replaces the twelve `FamilySessionPolicy` tables: one pure runtime
       function shared with the recipe builders' predicates; the structurally impossible combinations of decision 7 move
       from the first run to admission (tested with a fetch spy); the linear rejections become one shared violation function
       used by admission, `buildLinear` and the estimate; `WRITERS` exhaustiveness, the error type by origin and the routing
       to component Sessions stay (decision 7). MOSS-Image's knob table stops gating, and in the same commit a test compares
       the `session` each preset declares in the distribution mirror with the knob values its checks were run on.
    5. The public applicability query API (user story, export diff, surface snapshot, CHANGELOG) and one line on how to query
       in the model card; the fixed override type covering every numerics option of `SessionOptions` replaces
       `SessionOverrides`, and gemma's explicit `fullStateStorage` folds into it (decision 7).
    6. Device fallbacks become warned switches: f16 without `shader-f16` → `"f32"` (the throw is removed); a warning on
       online attention's three-dispatch fallback; a shared GPU warns at admission, a self-acquired device requests only the available features;
       a manifest's `gpuFeatures` is a request; the attention dp4a canary with both arms outside the band warns and runs that
       attention in f32, and its `emu-exact` verdict warns once (decision 8). The subgroup forms are no public option
       since 2026-11-15, so their fallback is decided when a per-adapter table lists one.
    7. Run-time canaries for linear and conv3d a8's dp4a choice and for online attention's `sg32` / `wg` choice on devices
       where the path is requested (decisions 6 and 9).
    8. `/kernel-check` (the workbench's `kernel-check` entry) covers the default kernels; where the per-GPU evidence
       records are kept for good is decided with it (decision 9).
    9. Wiring: gemma forwards `linearCompute` once the derived check admits it; Wan's text stage passes umT5's options
       (wiring only — no umT5 `i8-a8` preset, perf-ledger K-80 / K-81) (decision 7).
    10. `packedStaticQuantize` passes NaN through (so it becomes exact) (decisions 2 and 5).
    11. `packedStaticQuantize` and `integerGemv` assessed against decision 6's test (certainly on / off);
        each that fails becomes a string union at the next breaking change of this area or at 1.0, whichever comes first.
    12. GPU checks: no grid-stride kernel's numbers depend on the workgroup count (a `LIMIT_CAPS` Uint32 test); bit identity
        under an injected geometry profile (decision 8, device-driven choices).
  - **Checks**:
    1. Fault injection on the core ops (matmul, conv, …) for check C (decision 4).
    2. Gemma's E2E self A/B check, held under the grace clause until it exists (decision 2).
    3. The grace clause's missing points: parallel GEMV's ④ and worst-case record; the subgroup32 reductions' ② and ④
       (decision 2).
    4. The checks of decision 4 (A, P, F, B, G, E): before a `KARUME_UPSTREAM_PARITY` file is retired it is mapped to its
       replacement (a file without one keeps its opt-in lane); patch fidelity for the recipes without a `--verify` mode
       (deberta, gemma4, gemma4_qat, irodori, moss, qwen3_5 on 2026-11-09); deviation records per
       family, starting from BiRefNet's post-processing (ADR 0125) and Gemma 4's vision masks (ADR 0144 decision 1).
  - **Records**: Wan2.2's untiled VAE distance (torch tiled against untiled on the same latents — an ADR 0121 addendum and
    ADR 0147 decision 10); the inventory of published presets with no recorded verification (the self A/B check, the
    census-table row, the quality record — human, machine or none; for example the regular Gemma 4 `bf16` default has no
    recorded human review, ADR 0128 addendum 2026-11-04, while decision 3 (2) never skips the default preset's); the full
    inventory of device-driven choices of 2026-11-09 into `docs/research/` (decision 8); the first per-GPU evidence records on
    the two everyday environments (decision 9).
  - **For a later ruling**: whether Anima's sha rows, recorded on its practical preset, move to its reference preset (decision
    11; re-recording needs the GPU); the wording of ADR 0126's re-entry condition 2 and of perf-ledger K-80 / K-81, which name
    the acceptance tables decision 7 removes (changed only with the maintainer's agreement); whether the old rule "a band's
    derivation table sits in the docstring of the test constant" (ADR 0058 decision 4 ②, not restated by ADR 0147) should be
    restated.
  - **Stale statements found on 2026-11-09 (report only, not yet fixed)**: the `DECIDED` comment above the parallel shapes of the default profile (`linearGemv.parallelShapes`, then `PARALLEL_SHAPES`)
    ("only measured shapes are eligible") against its 22 rows marked unmeasured (ADR 0098 decision 4 was re-graded to SHOULD);
    ADR 0111 decision 2's list of tables; the table comments of gemma (`session-options.ts`) and anima (`pipeline.ts`);
    `session_options_test.ts` checking "all 10 families" (moot once the tables go); ADR 0033 decision 4's test
    cite; ADR 0132 decision 5's v1 workgroup-storage figures (the v2 figures are in its later addendum); the unreachable w4a8 `k % group`
    check; the online kernel's `:emu` against `:dp4aEmu` in the other i8a8 pipeline keys.
- **Fusion-rule mechanisms — the rule list, rules from outside the runtime, comparisons in tune (design approved
  2026-11-11: `.claude/reviews/2026-11-11_extension-mechanisms/DESIGN.md`, not tracked by git)**. Built: the rule list
  with its test-only switch, the fusion candidate search, fusion rules from outside the runtime (ADR
  [0148](decisions/0148-external-fusion-rules.md) and its addenda), the session A/B of `@karume/runtime/tune` with the
  workbench's `session-ab` and `gemma4-session-ab` entries, and two rules the pipelines pass by default:
  `models/ew_chain` on the MOSS-Image DiT ([perf-ledger](perf-ledger.md) F-13) and `models/vae_rms_silu` on the Wan
  and anima VAE decoders (V-10). The gemma pipeline's `fusionRules` stays an experiment (gemma only — maintainer
  2026-11-11; the details are decided later). Left:
  - Runs on Metal of the session A/B's other targets: `models/ew_chain` on the MOSS-Image DiT, `models/vae_rms_silu`
    on Wan's VAE, and the Gemma 4 target. On the anima VAE decode the M2 measured `models/vae_rms_silu` on 2026-11-17:
    it does not give the unfused plan's words there, and it runs there by the ruling of 2026-11-12, which the maintainer
    kept on 2026-11-17 ([known issues](known-issues.md)).
  - `models/vae_rms_silu` on Wan2.2's VAE encoder (its sites are planned, not measured).
  - Not decided: whether the runtime should bind borrowed weights for an external rule, which keeps `dit_skip22`'s 42
    gates and the tile-32 pair unfused today.
  - A public way to switch a built-in rule off for a measurement (today only the test-only `DISABLED_FUSION_RULES`).
    ADR 0147 decision 5 says no public switch replaces the removed fusion keys; whether this departs from it is not
    ruled.
- **Gemma 4 mobile QAT decode speed (maintainer 2026-10-30)**: 0.15.1 took K-97, K-48 step 1b and K-99; the integer GEMV
  (K-45, ADR [0140](decisions/0140-integer-gemv.md)) and the byte-plane lm_head (K-52, ADR
  [0141](decisions/0141-byte-plane-gemv.md)) are opt-in and declared by no preset. Left: their speed away from the Apple
  M2 (the RTX under Dawn, D3D12), the decision to declare them in a preset (ADR 0147 decision 3 (2)), K-48 step 1, K-98, and QAT
  E4B's parallel-GEMV rows, which are unmeasured copies (K-59). The filed comparison with regular E2B's int4 presets
  ended when they were removed on 2026-11-04 (ADR 0128), so the baseline has to be chosen again.
- **Regular E4B's MTP drafter (maintainer 2026-11-01)**: `karume-gemma4`'s E4B ships no drafter:
  `google/gemma-4-E4B-it-assistant` is not exported, and `gemma4.export_drafter` accepts E2B only (the q4_0 QAT line's
  E4B has its QAT drafter from 0.16.0). Scoping memo `.claude/reviews/2026-11-02_gemma4-expansion/e4b-drafter.md`
  (local, untracked).
- **Follow-ups of the Gemma 4 lines (filed 2026-11-04)**:
  - `tools/llm-baseline`: the mobile QAT profiles score with their checkpoint's own tokenizer, which may differ from the
    one `gemma4QatMobile.SOURCES` ships (the regular `-it` repositories' fixed tokenizer — 0.16.0 Changed). No scoring run since the schema 4 / 5 reader (local `inputs/`
    lacks some checkpoints).
  - Function calling and thinking (ADR 0084 addendum 2026-11-04): the `Pipeline.chat` one-shot entry takes no `tools` /
    `enableThinking` (only `chatPrompt` and `ChatSession` do); the parser's exact-integer check reads `JSON.parse`'s
    reviver `context.source`, not verified on Safari.
- **Remainder of the speed / memory wave (filed 2026-09-25; maintainer's ruling = split into two tiers, reference tier / practical tier; within-device determinism MUST hold in the practical tier too)**:
  The finished records of iterations 1 to 3 (contract and groundwork, B3 anima DiT residency, B4 irodori context K/V projection, H-35, K-70, E1 stage 1) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Candidates for the next iteration = wave 2 (DiT speed: B1 → the rest of B2 = perf-ledger F-6) / wave 3 (memory: rest of E1 → E2 → E5 → E4) / wave 4 (practical-tier kernels: the conv2d half of C2 = perf-ledger K-2 → D1 → the rest of D3 = perf-ledger K-95, K-57).
  The candidates come from `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md` §9.3 (not tracked by git) and the perf-ledger section of 2026-09-26 (K-59 to K-69, L-20 to L-23).
  Rest of E1 (BiRefNet's f16 series — ADR [0113](decisions/0113-birefnet-weight-series.md)) = derivation of the Lucida f16 2048² synthetic tolerance (`LUCIDA_F16_2048_SYNTHETIC_TOLERANCE` in `packages/e2e/tests/e2e_birefnet_test.ts` is still a placeholder taken from the f32 series; the RTX 3080 Ti can run it) and phase 2 (the i8 plan + i8 series).
- **Remainder of geometry profiles (K-71 — ADR [0115](decisions/0115-geometry-profiles.md) / [0116](decisions/0116-geometry-profile-row-buckets.md) / [0117](decisions/0117-app-geometry-tuning.md)) (filed 2026-09-27 to 10-03)**:
  The finished records (registration of `nvidia-blackwell`, the 7 row-count buckets, integration into applications (ADR 0117, all 7 stages ✅), the Codex review follow-up, 4 small items) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  ② Regenerating the families `apple-metal-3` carries over from its earlier table (gemm, attention, conv2d, i8a8-linear, i8a8-attention — they mix Chrome 153 and 154; its GEMV rows are from Chrome 157) from the sweeps of one Chrome version happens the next time a sweep is taken on M2, with the workbench's `geometry-sweep` entry (ruling 2026-10-02)
  ③ The Pixel 10a (Mali-G715) table is not registered; the re-run on the new version has not been done (research K-70 §15)
  ④ The `i8a8-linear` family of `apple-metal-3` (perf-ledger K-85, ADR [0115](decisions/0115-geometry-profiles.md) addendum decision 14): its buckets ≤ 256 and 257〜512 run the default geometry until a `full` sweep of that family alone is taken on an M2 (about 20,000 s; on hold) and derived with the table as `base`. The Apple adapters whose `description` Chrome leaves empty run the shared table `apple-metal-3-shared` (2026-11-23, ADR [0156](decisions/0156-shared-geometry-profile-tables.md)), and `apple-metal-3`'s match stays the M2's; that sweep also re-derives the shared table on both devices' full records (item 6 of "The move to maintained specs (ADR 0152)")
- **Remainder of the container-format wave (ADR [0108](decisions/0108-container-format.md)) = stages 4 to 6** (upstream intake of the anima transformer f16 / LoRA runtime A/B / new bit widths, Range fetching and so on —
  the staged-decomposition table of ADR 0108 is the source of truth for what is built and how it is accepted). The records of stages 0 to 3 and up to the 0.13.0 publication (2026-09-25) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
- **Remainder of the cross-model follow-up investigation (2026-09-10 to 11)**: the source of truth for the measurements and the unfinished parts is the [follow-up investigation](research/2026-09-10-codex-mtp-optimization.md#追加-llm-の実行と量子化別比較).
  The acceptance of Qwen3 / MiniCPM5 / E4B / gemma4-qat and the adoption records of K-25 to K-44 are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Remaining = the cost study of sharing the gate/up input (perf-ledger K-50, whose step 0 measurement has not run); isolating the cause of the GPU Instance loss in Chrome;
  distribution recipe / source table and long-text and broad quality evaluation for Qwen3-0.6B; transfer reduction for general sampling (with temperature, penalty / bias, speculative) and a follow-up test on M2.
- **Continued verification of the 9/11 review** (investigation 2026-09-10, [follow-up record](research/2026-09-10-codex-mtp-optimization.md)):
  Decide whether `artifacts.staged_publication` allows several writers to the same final, and if needed design locking and recovery from interruption.
  The follow-up tests of the GPU edge-channel protection on M2 / browsers and whole-model performance measurement remain.
  Do not turn review suggestions into new kernels without verification.
- **Move `planBackingBudgetBytes` into the common options** (filed 2026-09-07 — consequence of ADR 0095): every family except gemma4 builds its
  Session options from the manifest's `session`, so there is no way to change the budget (the default applies). Add it the same way `onRetry` was
  unified into `FromPretrainedHubOptions`. At the same time, merge the budget range check duplicated in `estimate.ts` and `session-options-check.ts` into one function.
- **Anima: add `text+transformer` to the residency values (filed 2026-10-01, maintainer)**: today `transformer` keeps only the DiT Session across generate calls, and
  text_encoder / text_conditioner redo "read → upload → build the Session → tear down" every time (the default VRAM assumption is "one largest stage" —
  ADR [0112](decisions/0112-anima-transformer-residency.md)). The de facto practice in PyTorch / ONNX Runtime is "once loaded, resident until explicitly freed", and
  the maintainer's policy is "the API shape is explicit release; the default is resident as far as VRAM allows; demote automatically when it does not fit". Put the two text Sessions on the existing eviction
  state machine (trial allocation for the next stage → evict if it does not fit → retry). The default is unchanged. The gain depends on the environment (the slower the fetch, the larger the gain). Before starting, measure the VRAM peak including the resident part on M2 / Pixel.
- **Remaining test coverage (filed 2026-09-05)**: the copy branch in `#encodeTimedChunk` of `SubmitScheduler`
  (`packages/runtime/src/gpu/submit.ts`) is still untested.
- **Isolation experiment for `gpuTiming: true` on Metal through Deno's wgpu**: an A/B of which dominates, the number of query sets alive at once or the
  backlog of `destroy()`. The `--diagnostics` flag left with `examples/gemma4` in 0.15, so the run needs a small Deno script. The relevant section of [known-issues](known-issues.md) is the source of truth for steps ①② and the fix candidates.
  It needs real hardware, so the user runs it.
- **The remaining medium-priority item of the diff review's deferred table** (source of truth = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — not tracked by git): W-G4-4 record the origin of the chunk limit in the provenance field `sym_max` and have dist compare it (**rides along with the next Gemma 4 re-export**).
- **Peak VRAM column on the model card (filed 2026-09-04)**: `karume dist` calls the TS-side estimate
  (the `prepareContainer(...).estimate()` family) when generating the card and adds a peak VRAM column to the quant table. Today's
  card shows only the stored bytes, so readers cannot judge whether the model runs on their GPU.
- Follow-up test of the RAM peak of the HF path in Chrome (measured on the Deno side —
  [research 2026-09-24](research/2026-09-24-part-length-ram-peak.md)).
- Collecting performance information from users' environments: a feature in the TS packages that collects it, and the
  results of users running a sample collection on several environments (**filed only** — the items collected, the
  location and the opt-in form are not designed; the per-run JSON records of `apps/workbench` and `apps/playground` are
  the nearest existing pieces).
- Re-assessing `err.cause` on Pixel (8 GB class Android Chrome): loading anima turbo `i4` failed there and the true
  cause is not identified. The reported "BodyStreamBuffer was aborted" is the text Chrome substitutes for a collateral
  abort; the hub has surfaced the true cause since 0.7.0, and the distribution has since moved to block-level fetching.
  The leading hypothesis is memory pressure (about 2.56 GiB resident at completion), which a connection drop or an
  app-side abort cannot be told from until `err.cause` is read on the hardware. Left: that one observation.
- **Follow-ups of accepting ADRs 0127 / 0128 / 0129 / 0131 (filed 2026-10-29, the maintainer's ruling 34 a — each ADR's addendum of
  2026-10-29)**.
  - Geometry sweep identity check for a bf16-stored linear at M > 1 (ADR 0131 open point 2, low priority; the GEMV sweep
    measures bf16 already, the GEMM sweep's `linear` cases do not).
  - Move the Node read-strategy default into `@hdae/fetch-cache` (ADR 0127 open point, after 0.15); hub's probe in
    `packages/hub/src/sources/hf.ts` can then go.
  - Deno's resource / op sanitizers have no replacement under Vitest (ADRs 0127 / [0129](decisions/0129-test-layout-and-lanes.md) open
    point 5): record the loss in [limitations](limitations.md); file a GPU resource-leak check only if a leak shows up.
  - The order in which the rest of the removed 0.14.0 tools and examples (v0.14.0 tag) come back as browser apps (ADR
    0127 decision 5; a measurement or check is an `apps/workbench` entry and a model page an `apps/playground` page —
    [spec/workbench.md](spec/workbench.md)): the tools `anima-residency`, `birefnet-ab`, `flag-bench`, `fusion-hints`,
    `llm-speed`, `mtp-bench`, `opbench`, `ram-peak` and `wan-seams`, and the example
    `qwen3`. `fusion-hints`' search itself is back as the fusion candidate search of
    `packages/e2e` (`tests/helpers/fusion-candidates.ts`, and the opt-in report `KARUME_FUSION_CANDIDATES=<dir>` of
    `e2e_fusion_candidates_report_test.ts`); only its entry is still to come.
  - Move the kept optional asset cases (`gemma4_config`, `gemma4_tokenizer`, `gemma4_chat`, `wan_pipeline`, `census_table` in
    `packages/models/tests`) into `packages/e2e` (ADR 0129 decision 1).
  - Small doc edits: `packages/e2e/README.md` names `outputs/series/` but not `outputs/test-assets/` (ADR
    [0139](decisions/0139-tests-on-distributions-and-test-assets.md)), where the goldens now live; ADR 0120's pointer at
    `WAN_DEFAULT_QUANT` in `tools/export-recipes/wan/distribution.py` carries a line number that has drifted again —
    drop the number.
- **Bundle-size remainders after the models barrel fix (filed 2026-11-04 — ADR [0037](decisions/0037-karume-monorepo.md) §4 addendum)**:
  - A bare import of the `@karume/runtime` barrel still keeps a large bundle under Rolldown (not investigated — the size is in the ADR).
  - esbuild still keeps part of the `@karume/models` barrel that Rolldown drops.
  - `KARUME_SOURCES`'s type (`packages/models/src/sources.ts`) is spelled by hand next to the pure-annotated build, so a
    family table added to one and not the other goes unnoticed by the type check — derive one from the other.
  - `@hdae/yomi` is a dependency of `@karume/models` that only tests use.

- **MOSS-Image (the subpath `@karume/models/moss`, published on Hugging Face as `hdae/karume-moss-image` with 0.15.1 — ADRs [0138](decisions/0138-moss-image-dit-vae.md) and [0137](decisions/0137-qwen3_5-text-encoder.md), accepted)**: what remains is other GPUs' sha rows (this family's rows exist for `node-nvidia-geforce-rtx-3080-ti` only — ADR 0106) and the text encoder at T = 4,096 (ADR 0137 open point 2). Its speed candidates are perf-ledger F-10 to F-12.

- **gemma4 prefill in w4a8 (perf-ledger K-60, maintainer 2026-10-17 "in the backlog")**: design of an acceptance table that keeps decode on the M=1 GEMV and puts only prefill (chunks of M=768) on `linearCompute: "a8"` × i4 (switching by M). Today's a8 falls to the GEMM skeleton even at M=1 and is several times slower than the GEMV ([research 2026-09-19](research/2026-09-19-qat-speed-recon.md); perf-ledger K-60), so opening the acceptance table is not enough.

- **Root fix for the rows that collapse under umT5's a8 (perf-ledger K-81, filed 2026-10-18, maintainer: "root fix first, not even opt-in")**: per-token int8 crushes the remaining components in the rows of function words (spaces, commas, articles, suffixes) ([research 2026-10-18](research/2026-10-18-a8-umt5-sbv2.md) §6; the main cause is token-specific outliers in the input of the FFN exit `wo`; mixing a8 per linear or layer does not remove it). Candidate = per-row mixing (only rows whose crest exceeds a threshold take the f32 path — row routing in the runtime's `linear-i8a8`). An equivalent transform on the export side only helps fixed channels. Once this is closed, re-evaluate umt5's `i8-a8` (the adoption line and the work needed are in K-81).

- **Execution form with f16 activations (w16a16) (filed 2026-10-11, maintainer: "will do it next time")**: today the runtime can store weights in f16 or bf16, and `linearCompute` / `attentionCompute: "f16"` drop only the kernels' shared tiles to f16 (ADR 0028); the activations between ops stay f32 (ADR 0006 rejected f16 activations). The value is the intermediate memory, which pays off where intermediate tensors take most of the VRAM, as in BiRefNet's 2048² (ADR 0113). It requires `shader-f16`. Bit-exact agreement with upstream w16a16 is not expected (the kernels differ — handled at the numeric level of the ADR 0005 addendum of 2026-10-11). Preliminary research: [research 2026-08-04](research/2026-08-04-intermediate-f16-design.md).

- **Decouple the test helpers results.ts / reference.ts from GPU discovery (filed 2026-10-11, Codex deep review R8-008, maintainer's ruling = file only)**:
  `ENVIRONMENT` in `packages/runtime/tests/helpers/environment.ts` is fixed by calling `requestAdapter()` at module evaluation.
  `helpers/results.ts` and `helpers/reference.ts` import it as a value. So `results_writer_test.ts` / `reference_fixture_test.ts`, which say
  "no real GPU needed", also search for an adapter at import time. They do not take a device, and on a machine without an adapter
  the value stays null and they proceed, so the tests do not fail. The only real cost is the `requestAdapter` call. Both functions already take `options.environment`.
  What remains is that the value import of `ENVIRONMENT` runs at module evaluation. Lazy acquisition does not fix it — `openReferences`
  reads the environment key synchronously when tests are registered, and `openResults` is also a synchronous API. The fix would split the two into "a body that takes the environment as an argument" and
  "a thin wrapper that binds the default `ENVIRONMENT`", and point only the CPU tests at the body (the GPU tests' callers do not change).
  After the fix, confirm that with `navigator.gpu` undefined, importing the two tests does not call `requestAdapter`.
  A draft of this split (bodies `results-core.ts` / `reference-core.ts` and a gate test) was made on 2026-11-04 outside
  the approved scope and kept as a local patch, not applied: `.claude/reviews/2026-11-04_overnight/decouple-results-reference.patch`.
- **Out-of-scope remainder of the Codex deep-review fix wave (filed 2026-10-11)**: of the observations that the fix wave's implementers and reviewers recorded as out of scope,
  those not in backlog / known-issues and confirmed still present on main on 2026-10-11 (one per line; the IDs in parentheses are
  findings in `.claude/reviews/2026-10-04_de127de1/` (not tracked by git)). The order of work is undecided.
  - Anima's per-stage estimate (review of ruling 6): `stageNeed` in `packages/models/src/anima/pipeline.ts` estimates the text and VAE stages with no attention knobs, and `withStage` opens those Sessions with `{}`; the two agree only because both default to f32 (the preset's knobs apply to the DiT alone). Candidate: one shared constant for the non-DiT stage options, used by both. Passing the preset's `attentionCompute` / `attentionScoreStorage` there would count a form those Sessions do not run.
  - Output extent of unbatched GEMM params (filed 2026-11-04): `linearParams` (and the GEMV / i8a8 params that share `gemmParams`) do not check the output `m·n` against u32 (matmul and bmm do). `codegen_wgsl_test` pins `[65, 268435455, 16]` as accepted for linear, so that case needs a ruling first.
  - Batch-multiplied flat indices of fused attention params (filed 2026-11-04): `attentionQkParams` / `attentionPvParams` and their i8a8 variants do not receive the batch (B·H on `wid.z`), so `B·H·M·K` and the like are not checked against u32.
  - deform_conv2d i32 coordinates (filed 2026-11-04): the WGSL computes `i32(oy) − i32(dims.padding_h)`, and neither the params (padding is checked only as u32, with no `assertConvAxisExtent` equivalent) nor the contract layer (the attr is a safe integer ≥ 0, the shape only requires an output length ≥ 1) bounds it.
  - The post-acquisition GPU check `assertAcquiredGpuCapabilities` (`packages/models/src/session/gpu-features.ts`, filed 2026-11-04) is not used by moss (`src/moss/pipeline.ts`), the umT5 encoder (`src/umt5/encoder.ts`) or Gemma (`src/gemma/pipeline.ts`), nor by the shared-GPU admission sites of the families; they still write the features + limits checks by hand.
  - PLE index schema check in Python (filed 2026-11-04): `read_ple_index` in `tools/exporter/src/karume/migrate.py` tests `schema in (1, 2)`, so `true` and `1.0` / `2.0` pass (Python's `True == 1`); align it with `packages/models/src/gemma/ple-index.ts` (reject bool first, as `_require_version` in `container.py` does).
  - `packages/models/tests/helpers/moss-dit.ts` keeps its own `readBuffer` / `viewOf` / `floatsOf` (the contract of `helpers/safetensors.ts`); fold it in (importers: `e2e_moss_ab_test.ts`, `e2e_moss_dit_test.ts`, `e2e_moss_generate_test.ts`, `moss_sampler_test.ts`, `moss_dit_host_test.ts`, `helpers/moss-text.ts`, `helpers/moss-vae.ts`) (filed 2026-11-04).
  - A vacuous negative assertion (filed 2026-11-04): `packages/runtime/tests/estimate_test.ts` asserts that the estimate's notes do not include a phrase the estimate never prints (already so before the ADR 0130 translation — the notes were translated in 2-9), so it cannot fail. Pin the phrase the real notes would print instead.
  - Backing difference between runtimes: the cause of the different DiT backing between Deno and Chrome for Wan2.2's 121 frames with the preset `f16+dit8-a8-attn8-s16` (ADR 0124, "Alternatives not taken"; material in [research 2026-10-11](research/2026-10-11-row-block-budget-stage-4b.md)) is not investigated, and Node + Dawn, the development runtime since 0.15, has not been compared. The candidate explanations recorded then are the geometry profile (Deno ran on `default`) and `maxBufferSize` (how regions are split).
  - f32 overflow of SBV2's `z_p`: even when `noiseScale` is finite in double precision, `buildZp` in `packages/models/src/sbv2/host/latent.ts` puts products above the f32 maximum (for example `noiseScale = 1e39`) into a `Float32Array`, turning them into ±Inf, and passes them to the voice graph as they are (the same hole as the `z_noise` check; the GPU outputs `m_p` / `logs_p` are mixed in, so a design for which error type is thrown is needed).
  - The copy of the repository name rule in models: `REPO_SEGMENT_RE` / `assertRepoSpelling` in `packages/models/src/hub/repo-ref.ts` hold a copy of the same rule as hub's `isRepoName` (`packages/hub/src/manifest.ts`) — making it one requires exporting the predicate from hub's mod.ts (an addition to the public API).
  - Length 0 in the CPU reference's `allclose`: `packages/runtime/src/reference/allclose.ts` returns `pass: true` for two length-0 inputs — whether to allow empty input or throw has not been ruled on, and `packages/runtime/tests/reference_ops_test.ts` pins only the currently observed value (review L-R3-4).
  - Writes to pad rows in state attention: the variants in `packages/runtime/src/kernels/state-attention.ts` and `state-attention-stats-pv.ts` each write the same guard separately — a candidate for one shared WGSL fragment (today a static test pins the form of all variants, so a recurring mismatch would be detected).
  - Bilinear interpolation weights: unlike torch's `guard_index_and_lambda`, λ in `packages/runtime/src/kernels/upsample-bilinear2d.ts` is not clamped to [0, 1], so only for shapes where one axis exceeds 2^24 and the source coordinate goes past the edge might the end points not be exact (not reproduced; out-of-range index reads are blocked).
  - Type arguments of DataView / ArrayBufferView (outside the lint rule `karume/typed-array-buffer`): since TS 5.7, `DataView` and `ArrayBufferView` also take a buffer type argument like TypedArray, and omitting it gives the default `ArrayBufferLike`. src still has 4 bare `DataView` type annotations (`packages/runtime/src/gpu/attention-dp4a-canary.ts`, `packages/runtime/src/format/container/header.ts` and 2 in `packages/models/src/audio/wav.ts` — `ArrayBufferView` has 0) — whether to add it to the rule's set of names is a separate ruling.
  - hub: decide whether the form where one manifest puts the same path with the same sha256 at different revisions of the same cross-repository target (two references,
    one cache entity — the hub cache inventory section of [limitations](limitations.md)) is reworked to match by entity key
    (recommendation a of Codex deep review R4-001), or rejected at parse.
  - hub: rejecting a cross-repository reference that points at its own repository applies only to the parse in `loadManifest` from HF. The distribution gate
    (`distribution_gate_test`) and `local-dist-server` (now in `packages/test-support`) use the public
    `parseManifest(text)`, so a manifest that passed on the local mirror can first fail after it is uploaded to HF (`karume dist` already rejects it —
    ADR [0038](decisions/0038-manifest-v1.md) §7 addendum 2026-10-28). Open: whether the gate checks with the destination repository name too, for a mirror the exporter did not build.
- **Merge the duplicated helpers of Wan2.2's T2V / I2V end-to-end pipeline e2e tests (filed 2026-10-10, maintainer's ruling = later)**: `e2e_wan_ti2v_pipeline_test.ts` (T2V) and
  `e2e_wan_ti2v_i2v_pipeline_test.ts` (I2V — ADR 0121 stage 9b) each hold the same small utilities — fixture loading (`readFixture`), difference computation and formatting (`difference` /
  `formatDifference`), bit equality (`bitsEqual`), RGB extraction (`rgbBytes`), counting non-finite values (`countNonFinite`), prompt resolution (`textOf`), band derivation
  (`bandOf` / `roundUpTwoDigits`), formatting of observations (`observe` / `formatObserved`). Move them into `helpers/`. After the move, confirm on the GPU that both bands and sha rows are unchanged
  (the same procedure as when stage 7's A/B check was moved into `helpers/wan-ab-gate.ts`).
- **Return Wan2.2's 121-frame self A/B check to the default lane — dealing with the duration of the full verify (filed
  2026-10-09, maintainer's ruling = later)**: ADR 0121 stage 7's self A/B check is held at two points, 33 frames and 121
  frames (the same "default and longest" rule as Wan2.1 — attention's 8-bit error grows with row length). Putting the
  121-frame part in the default lane would lengthen the full verify, so it is opt-in for now (`KARUME_WAN_TI2V_121F=1`),
  and collapse on long rows is caught only when the opt-in is run. Measures (shortening the duration, changing where it
  lives) are to be considered separately.
- **In-flight submit bound for enqueue in browsers (filed 2026-10-07, maintainer's ruling = later)**: ADR 0123 put the bound on run only
  (if enqueue waits, the call order against another Session's enqueue breaks). To bound enqueue as well, first add a mechanism that runs the bodies of the enqueues in a batch one at a time
  in acceptance order (a per-batch FIFO), then wait at the end. The same mechanism also closes the existing case where the first enqueue (a body with a derivation `await`) can be
  overtaken by a later enqueue of another Session. When to start: when running long batches (video VAE, generation loops) in Chrome
  shows congestion in submission or device lost. There is no actual failure so far (the 121-frame VAE, about 20,000 submits, also completed).

- **Two anima parts read as fragmented in the upload check (filed 2026-09-24; ruling 2026-09-24 = publish as is and act after measuring the real download speed)**: in the fragmentation table of `tools/release/hf-upload.zsh`, part 3 of `text_conditioner` in `anima-turbo-v1.1` and `anima-v1.0` (byte-identical to each other, different from the same part of the other 3 variants) and `shared/text_conditioner/model.bf16-00003` stay below the 10 MiB/term guideline (the [research](research/2026-08-09-xet-fragmentation.md)). The 0.15.0 re-upload did not clear them: the cause is sharing between byte-identical or near-identical parts inside the repository, not cross-repository reuse. Repositories with a published pin do not use delete → recreate (runbook §2 MUST NOT), so decide whether to accept it once the real download speed has been measured (not measured yet), or find another remedy.

- **Revisit the structure of the manifest (`karume.json`) — make optimization flags and presets easier to handle (filed by the maintainer 2026-10-06)**: today the bundle of execution knobs
  is held in the single `session` of a quant preset, and the preset name doubles as the bundle's name (ADR [0074](decisions/0074-quant-seat-naming.md) /
  [0147](decisions/0147-numerics-and-kernel-adoption.md) / [0111](decisions/0111-session-options-composition.md)).
  We want a form in which optimization flags and presets (bundles) are easier to handle. What belongs where (storage dtype, execution knobs, bundles, per-component declarations and so on)
  is to be sorted out later. The material is the open points of the inventory of how fast paths are expressed (2026-10-06, `.claude/reviews/2026-10-06_fast-path-inventory/`, not tracked by git)
  — there is no per-component session (the fast paths of the VAE / text encoder cannot be declared), the keys of new kernel variants, and whether changing a bundle
  changes an existing preset or creates a new one. **Work starts later** (maintainer 2026-10-06 — Wan2.2's optimization proceeds within today's `session`).
- **Candidate for bringing HTTP Range fetching (ADR 0108 stage 6) forward (filed 2026-09-24; not decided yet)**: the
  condition for bringing it forward in ADR [0109](decisions/0109-manifest-v5-container.md) decision 7 is "in the RAM
  peak harness of stage 2, the cold peak exceeds 'part length + overlap'". Stage 3e removed the duplicated holding, but
  the cold peak of the scan type (Deno and Node via HF, the fetch layer's `stream` strategy) still exceeds part length +
  the largest block. The condition formally still holds ([research record](research/2026-09-24-part-length-ram-peak.md)
  item 7, ADR 0108 addendum 5). Browsers (Blob = seek) and local sources are not affected. Its motivation differs from
  that of parked "hub Range parallel + prefetch" (a fragmentation measure).
- **With an unverified source (local), host PLE row reads apply sha256 to the whole block every time (filed 2026-09-24)**:
  host PLE reopens the reader for every row read, so with an unverified source one row read costs a whole block read and
  sha256 every time (28 × 32 MiB per run). The longer gemma4 run on M2 in stage 3e has been attributed to this and to the cost of memory allocation
  ([research record](research/2026-09-24-prerelease-gpu-measurements.md) item 1). The fix is either for the opened container to
  remember the verified blocks and do range reads from the second time on (a verified cache), or to digest each block only once.
  Deciding in an ADR how to handle a local file being replaced during reading (TOCTOU) comes first.
- **Fence count of PLE's in-memory container (filed 2026-09-24)**: the PLE of the GPU-resident preset hands one piece = one part to
  the in-memory container, so Session construction raises one fence per piece
  (the part split in `packages/runtime/src/format/container/memory.ts`). Bundling pieces into parts cuts the fences and
  PLE construction a little at a temporary cost in VRAM during construction (QAT E2B / E4B —
  [research record](research/2026-09-24-prerelease-gpu-measurements.md) item 2). **Ruling 2026-09-24 = do not bundle for now**
  (re-confirmed in the pre-release review of 0.13.0; the implementation size is the conclusion of item 2 of the research record).
  Alternative (filed 2026-09-25, not started): without bundling, let the fences run **one ahead** (start supplying the next piece before the previous fence completes).
  Staging then holds two parts alive at once, so running ahead is allowed only when the total bytes are within the limit (ADR 0108 decision 9:
  "limited by both count and total bytes"). With 256 MiB weight parts this would be +256 MiB, more than the +192 MiB of the bundling option, so
  the target is only the PLE preset (32 MiB pieces). Kill: PLE construction improves by less than 0.3 s (the adoption of the candidate is judged together with L-16 in [perf-ledger](perf-ledger.md)).
- **The `gemma4_qat` fixture tests are skipped wholesale in CI (filed 2026-09-25)**: `series_fixture.py` and
  `test_fixed.py` under `tools/export-recipes/gemma4_qat/tests/` have a module-level
  `pytest.importorskip("transformers")`, and `transformers` exists only in the `gemma4-qat` group, so in CI's recipes
  job, which runs with the default sync, `test_audit.py`, `test_fixed.py` and `test_plan.py` are skipped at collection.
  To decide: a) stub `transformers` and keep 1 to 2 upstream comparisons (mind `isinstance(QuantizedEmbedding)` in
  `checkpoint.py`), or b) add `--group gemma4-qat` to CI (a trade between CI time and the strength of the upstream
  comparison).

- **Leftovers of the code-quality wave (filed 2026-09-22 — source: the same wave in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md) (after the 0.12.0 release))**:
  - Where to put the four functions of `packages/models/src/gemma/pipeline.ts` tied to `gemma.PipelineOptions`
    (`resolveGemma4PleResidency` / `buildGemma4Program` take it, `assertSpeculative` takes its `speculative` field;
    `speculativeSetup` no longer depends on the type, so it may not belong in the move) (creating `pipeline-options.ts`
    was postponed by the ruling of 2026-09-21).
  - sbv2's check of the Unicode range tables (`parseRanges` in `sbv2/text/asset.ts`) differs in wording and stays separate from the shared `parseCodeRanges` in `text/code-ranges.ts`.
  - Splitting `reference/ops.ts` and `ops/shapes.ts` (info — no mutable state and no cycles, zero real harm).

- **Remainder of decode speed (moved from now on 2026-09-20)**: small items K-48 stage 1 (rms_norm→SRQ fusion) / K-49
  (slice aliasing) / K-50 (k+v concatenated GEMV), K-46 (int8 KV — a memory item). Revival condition: when decode speed
  becomes the topic again (re-measure the TTFT attribution of research 2026-09-19 §15.2 first — "the prefill arena path"
  is not in hit runs, as corrected in the addendum of 2026-09-25). The adoption of candidates and the kill criteria are
  in [perf-ledger](perf-ledger.md), the attribution in [research 2026-09-19](research/2026-09-19-qat-speed-recon.md).

- **Host side of a Gemma 4 decode token (2026-11-18)**: H-31 and H-32 are built, and H-29 is measured and not adopted
  ([perf-ledger](perf-ledger.md)). Left: the browser measurements (H-31 by the decode bench's `no-signal` switch; H-32
  has no whole-row arm in the bench, because the key that turns the selection off is test-only; H-29's ramp was not
  measured there), E4B and 12B after the changes, the measurement of K-49, and the sampler without a top-k (H-45). The
  numbers are in [research 2026-11-18](research/2026-11-18-decode-host.md).

- **Fusion hit-count checks for the families without one (filed 2026-11-11, with the fusion candidate search)**: a fusion that
  stops matching only moves performance, so a family without a hit-count check (`packages/e2e/tests/e2e_<family>_fusion_counts_test.ts`)
  would not notice. MOSS-Image's DiT stage gained one with the candidate search; still without: sbv2 (DeBERTa included),
  umT5, BiRefNet, Depth Anything, SigLIP2, Qwen3.5 (MOSS-Image's text encoder), regular Gemma 4 E4B,
  Gemma 4 26B and the q4_0 QAT line, and Irodori v4.1-small, v4.1-small-mf and v4-Large. The opt-in candidate report
  already binds and plans those with a local mirror.

- **Ruling on the contracts that block large in-layer fusion (filed 2026-09-19)**: part of the dispatch gap to WebML is
  blocked jointly by `windowTouchesState` MUST (ADR 0067) and the single-output MUST of fused steps
  (`windowIsSingleOutput`, ADR 0068 addendum 1). The third blocker, the atomic last-arriver merge, is now a MUST NOT of
  ADR 0147 decision 11. The individual candidates (attention→SRQ, residual add→SRQ, attention in 1 dispatch, Q/K/V with 3
  outputs) were all closed by refutation, so a design ruling on whether to relax the contracts comes first
  ([research](research/2026-09-19-qat-speed-recon.md) §9.1). Do not file in-layer fusion candidates before the ruling.
- **Remaining items of the video-generation foundation (investigation 2026-09-10; ADRs [0118](decisions/0118-wan21-video-generation.md) /
  [0119](decisions/0119-wan-umt5-gpu-text-encoder.md) / [0120](decisions/0120-wan-dit-w8a8-seat.md))**:
  - MiniMax H3 (the maintainer's final goal, 2026-10-02): H3 stays at a structural study + per-component candidates
    ([research](research/2026-10-02-video-gen-recon.md) — the size, the offload premise and the licence reading are
    there); there is no browser WebGPU case.
  - Wan small items (filed 2026-10-03, diff review of the ADR 0118 wave — the IDs in parentheses are items of `.claude/reviews/2026-10-03_diff-review-wan/` and `2026-10-03_codex-review-wan/` (not tracked by git)):
    ① measuring the GPU time per submit with the submit layout of normal execution (M8) (the "1 submit ≤ 1 s" check and its recorded values come from timing mode (`gpuTiming`, 1 dispatch = 1 pass), and the chunk boundaries after backing can differ from normal execution — ADR 0118 stage 8 result "scope of the submit check"; needs a GPU)
    ③ a fourth case in golden `conv3d_block` that exercises the v4 and m-tile 64 variants (F4) (the current 3 cases are Wout 5 / 5 / 6 and Cout 6 / 3 / 2 and hit only the scalar variant and m 32. v4 and m 64 are exercised only by the GPU parity tests; needs golden regeneration and confirmation on the GPU)
    ⑤ the CPU-only DiT host test reads and copies the whole 2.6 GB reference (TG-11) (`reference.full-{band,accept}-s14040-t0999` are 2,598,638,524 B each; `ArrayBuffer.slice` in `readBuffer` temporarily doubles it; RSS not measured)
    ⑥ the note separator " / " in results.json collides with " / " inside labels (TG-18) (labels in general, such as `formatNormalized`; no consumer splits it today; a fix would go on the separator side)
    ⑦ narrow the entity-candidate check of `prompt_clean` into an exact check that "rejects only when the string actually changes", using the name table (Python's `html.entities.html5`, 2,231 names) (accepting sequences that do not form an entity, such as `R&D` — ADR 0119 ruling 2026-10-03, maintainer: "a workaround if it can be avoided"; once someone asks for it).
- **Remaining small items of the Wan2.2 acceptance (ADR [0121](decisions/0121-wan22-ti2v-5b.md))**:
  **Adjacent small items of stage 9 (filed 2026-10-10, review)**:
  (a) the distribution assembly (`tools/export-recipes/wan/distribution.py`) does not check the input/output contract of the VAE encoder's 3 containers (input names, symbols, the shape chain pre → attn → post) (the decoder's 2 containers have the pair check `assert_vae_chunk_pair` — a series with only one of them rebuilt passes assembly and is first rejected at admission in the user's `fromPretrained`; filed 2026-10-10, review G4-01, a candidate for the fix wave)
  **Adjacent small items of stages 5 / 6 (filed 2026-10-05)**:
  (a) the tile reference reader and helper functions are duplicated across the Wan VAE e2e tests (`readTileFixture` and `planMeta` in `e2e_wan_ti2v_vae_tiles_test.ts` and `e2e_wan_ti2v_vae_tile32_test.ts`; `poisonFrames` in those two and `e2e_wan_ti2v_vae_a8_test.ts`; `roundUpTwoDigits` — see (d)) → into a shared test helper
  (b) helpers such as the recipe's `_peak_rss_gib` are duplicated in `tools/export-recipes/wan/` (`text_embeds.py`, `vae_tiling.py`, `ti2v_few_step_ref.py`, `export_vae.py`, `export_vae_encoder.py`) → into `_shared`
  (c) the existing 5 DiT inputs have no dtype check (only the 2 inputs added for 2.2 check the dtype)
  (d) rounding up to 2 significant digits can round one unit too high because of floating-point error. `e2e_wan_ti2v_pipeline_test.ts` and `e2e_wan_ti2v_i2v_pipeline_test.ts` are fixed by inserting `toPrecision(12)`. Still without it: `roundUpTwoDigits` in `e2e_wan_ti2v_vae_tiles_test.ts`, `e2e_wan_ti2v_vae_tile32_test.ts` and `e2e_wan_ti2v_vae_encoder_test.ts`, and the same expression inline in `bandCandidate` of `e2e_wan_ti2v_dit_test.ts`, in `e2e_wan_umt5_test.ts`, `e2e_moss_generate_test.ts` and `e2e_moss_dit_test.ts` (can be fixed together when moving into the shared test helper of (a))
  (f) the 2.2 T2V and I2V end-to-end e2e tests (`e2e_wan_ti2v_pipeline_test.ts`, `e2e_wan_ti2v_i2v_pipeline_test.ts`) hold copies of the same helper functions (`readFixture`, `difference`)
  (g) there is no host test combining the admission of the gpu path (`admitWanText`) with the 48 ch, 7-input DiT (stage 6 results)
  (i) `planWanRequest` checks only the element count of the injected `latents`, not the shape (stage 6 results)
  **Adjacent small items of stage 4 (filed 2026-10-05, deferred in review)**:
  (a) move `filePresent` into a neutral test helper (today the VAE helper `packages/models/tests/helpers/wan-ti2v-vae.ts` imports it from the DiT helper `wan-ti2v-dit.ts` in the same directory — the VAE tests depend on importing the DiT helper)
  (b) make the drop-cache fault injection of the shared helper `packages/models/tests/helpers/wan-vae-chunk-loop.ts` assert that the named cache exists in the graph (with a missing name it silently injects no fault — run the 2.2 lane)
  (c) a closed form of DupUp3D (put up0 / up1 on the fused upsample2x) — perf-ledger K-77
  **Adjacent small items of stage 3 (filed 2026-10-04, proposals)**:
  (b) add a paragraph "how to swap it into Wan" to the model card of the experimental mirror (the fine-tuned umT5)
  (c) add a check that compares the model card of the local distribution mirror with the card the recipe draws today (to find missed rebuilds)
  (d) make Wan's text stage check the `session` declared by the umT5 preset instead of silently ignoring it (same root as the component replacement item in known-issues)
  **To add later (filed 2026-10-04, maintainer)**: (a) a TI2V-5B preset with a non-quantized DiT — we want the weights
  ready even if it does not run on the development machine. The maintainer's request named f16; under ADR 0128 decision
  4 the original-weights preset of this upstream is F32 (best effort, opt-in), so settle which one is meant first. The
  RAM premise of the original material is gone with the current machine (ADR 0121 decision 2 and ruling 2). Whether
  checks and sha rows fit on the current GPU has not been measured.
  **Related**: the 2 items of ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) "Open points" (the ruling on publishing third-party compatible components,
  agreement of graph descriptors across versions).
  **umT5 / Wan2.2 small items (filed 2026-10-04)**: (a) the original's index (`pytorch_model.bin.index.json`) and `config.json` are not pinned by sha256 (if the index disagrees with the actual files,
  it fails with a bare `KeyError` — there is no path by which values get swapped; ADR 0122 addendum "stage a results") (b) umT5's `write --check` opens a temporary directory inside the series
  (`.check-*`) — it remains if the process is killed (c) loading `ContainerDitWeights` (the layer-sequential entry point that reads the DiT weights from the container) takes 25 s per forward.
- **Quality improvement of anima base i4 (filed 2026-08-24 — the revival lever of the ruling to skip distribution)**:
  what remains is only **the list of possibilities left unverified** for the i4 preset of the old fused turbo (dedicated
  geometry, g16, calibration amount, one more degradation mechanism — all of them "not tried", not "tried and failed"),
  and the source of truth is [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md)
  §8. g16 remains as one revival lever in parked "anima g16 evaluation". Making adaLN i8 was rejected by a ruling after
  a visual sweep (perf-ledger Q-9); for locating the places with high quantization sensitivity, the source of truth is
  item ④ of "3 known issues + anima base i4 sensitivity" in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md), to which the same ruling
  attaches "the anima DiT i4 line is on hold for a while". The calibrated `*-i4-dyn` series are no longer under
  `outputs/` on the development machine, so a revival starts from a new calibrated export (source of truth for the
  measurements = [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5).
- **irodori adaLN i8 A/B on the shipping rig (filed 2026-08-24)**: whether adaLN i8, which helped in the sim, also improves
  the reading on the shipping rig is unverified ([research 2026-08-24](research/2026-08-24-gptq-expansion-quality.md) §1 /
  §2 — the cost and the sim → shipping transfer limit). Revival = dissatisfaction with the quality of the `i8+dit4` preset (formerly `w4`), or real demand for size optimization.
- **Bring measure_quant's distribution estimate up to J-5b** (found during J-3, 2026-08-22): sbv2's
  `project_distribution` still rests on the pre-J-5b premise "only linear weight slots; conv / embedding i4 has neither a storage form
  nor an execution path" (in fact shipped — ADR 0069 addenda 6/7). Harmless for relative
  comparison, but the estimate is too low and the docstring is stale. Update the target set and the description in one item.
- **Continuing model expansion**: Kokoro-82M (waiting for the ruling to add `lstm_scan` (ADR 0056 decision 8 — its
  three outputs are the open conflict there) and an exporter handler for it; the exporter already wires `getitem` to the
  outputs of a multi-output op with a handler (ADR 0068 addendum 11, 2026-11-08), and node-level multiple outputs were
  allowed by ADR 0068), MobileSAM / SAM 2 (conv_transpose2d), DA-V2 variable resolution (upsample_bicubic2d). The latter
  two need ops outside the vocabulary. The point-in-time record of the candidate survey is
  [recon-2](research/2026-08-14-model-expansion-recon-2.md).
- **Performance candidates**: the 🚧 rows of [perf-ledger](perf-ledger.md) are the source of truth for filing, adoption and order.
- **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache-2.0; the text tower only — the maintainer's go-ahead of
  2026-11-21). Built: the
  recipe `tools/export-recipes/embeddinggemma2/` (source pin, tokenizer asset, the export of one graph with a symbolic
  token length up to 8,192 and its bf16 series, the opt-in comparison with sentence-transformers, the distribution
  recipe with its local mirror `models/karume-embeddinggemma-2/` — model `text`, presets `bf16` (default) and `i8`
  since 2026-11-22, nothing published),
  the series and its e2e, and step 1 of the order below: the TypeScript family `embeddinggemma`
  (`@karume/models/embeddinggemma` — `Pipeline.embed` with the manifest's prompts, widths and token limit, no
  `SOURCES` until the first publication) with its pipeline e2e over the mirror, all in the lane
  `models:embeddinggemma2`. Ruled 2026-11-21: one family `embeddinggemma` serves both models; a text holding an image,
  audio or video placeholder id is refused (the upstream composite model replaces such an id with the pad id;
  sentence-transformers refuses the text); an input is a string or an array of content parts, and placeholders are
  never written in text; a later video input is an array of frames, the frame type (`VideoFrame` or `ImageBitmap`)
  to be decided then. Left, in order:
  1. ~~The TypeScript pipeline and the distribution recipe~~ — built (above).
  2. ~~The model's 8K context~~ — built 2026-11-22 (the maintainer's approval of that day). Stage 4a: the optional
     attr `band` of plain attention, checked inside ①QK from positions with no mask tensor, its kernel check
     `attention-band` and the exporter's `karume.band.to_band_form` (ADR [0155](decisions/0155-attention-band.md)).
     Stage 4b: the recipe declares the band 512 on the 20 sliding layers and feeds no mask, the bound is 8,192 (upstream's
     context; golden cases at 3,062, 5,935 and 8,192 tokens, every existing sha256 row unchanged), the family opens its
     Session with a 128 MiB row-block budget for the attention scores (plan backing 944 MiB instead of 1,648 MiB at
     8,192 tokens on the RTX 3080 Ti) and refuses, before the run, an input longer than the device's buffer limit holds
     (docs/limitations.md). Left from it, unordered:
     - skipping the work outside the band (①QK and ③PV still compute every `T × T` score; each row sees at most 1,025
       keys) — filed as [perf-ledger](perf-ledger.md) K-114, not measured;
     - the per-layer-embedding intermediate (`[1, T, 12288]` f32, 49,152 B per token) on devices with 256 MiB limits,
       which stop at 5,461 tokens — accepted as is (2026-11-22), the graph's per-layer embedding not rewritten;
     - batch and key padding, through a valid-key range the score kernel checks beside the band, for both models;
     - measuring what batch would buy (throughput against one text per call) before building it.
  3. Quant presets and their evidence, then the first publication (on the maintainer's go-ahead). The publication
     also gives the family its `SOURCES` table and turns the Playground's `embeddinggemma-2` mount into the object form
     with `hfRepo` from it.
     - Stage A, the torch-CPU screening of i8 / i4 weight forms and i8 + a8: [research 2026-11-22](research/2026-11-22-embeddinggemma2-quant-screen.md).
     - Built 2026-11-22 (the maintainer's choice that day: one int8 preset, the default stays `bf16`): the preset `i8`
       — the token table and every linear (219 tensors) as per-row symmetric int8, f32 compute, empty session — from the
       series `embeddinggemma-2-text-i8` (`export.py --dtype i8`: rounded before the reference and the goldens, the same
       graph, band, cases and bound); the mirror carries `bf16`★ and `i8` (323,958,008-byte container against
       592,932,727; no `requiredLimits` for `i8`, whose int8 table is exactly the WebGPU default 128 MiB); the series
       e2e of the i8 series against its own goldens (maxAbs at most 2.87e-7), the pipeline e2e over both presets with
       sha256 rows per preset, and the GPU check that each golden text's `i8` vector keeps a cosine of at least 0.999 to
       its `bf16` one at every width (measured 0.99940 to 0.99993 at 768, the screening's values to the seventh digit).
       The card states the screening's numbers with their source.
     - Built 2026-11-23, stage 1 of 2 (the maintainer's decision that day: the preset `i8` takes int8 towers, `bf16`
       does not change in any byte, no new preset name): the series `embeddinggemma-2-vision-i8`
       (`export_vision.py --dtype i8`: the 114 linears and the two learned position tables as per-row int8, the RoPE
       tables f32; 169,138,366 bytes against 336,093,696) and `embeddinggemma-2-audio-i8` (`export_audio.py --dtype
       i8`: the 123 linears and the 12 depthwise convolutions as per-row int8, the two subsampling convolutions and the
       window selector kept bf16 — rounded, the two convolutions alone moved single soft tokens of speech to a
       cosine of 0.20; 295,640,914 bytes against 587,590,521), each rounded before its reference and goldens with the
       other series' graph and cases. The exporter's `fake_quant_int8` takes `functional_weights` (a table read by
       `functional.embedding` rounded as an `embedding` weight), and the tower wrappers share every weight with
       upstream's modules, so the goldens are upstream's forward on the rounded weights; the bf16 series exported again
       byte-identical. The distribution maps each preset's towers to its own storage; the mirror carries both forms of
       each tower with every file of the `bf16` preset byte-identical (the `i8` preset downloads 798,296,220 bytes with
       both towers), and the card states the audio exception. The torch-CPU screening was run again on the shipped form
       ([research 2026-11-23](research/2026-11-23-embeddinggemma2-tower-quant-screen.md): `all-i8` lowest cosine
       0.999648 over 32 generated images, 0.999493 over 32 clips), writing its clips at 16 kHz to
       `inputs/embeddinggemma2/quality-set-16k/`.
     - Built 2026-11-23, stage 2 of 2: the TypeScript and GPU side over the new mirror, with no change to the
       pipeline's code (each preset's mapping already picks each tower's storage). The series e2e of the two i8 tower
       series against their own goldens, each image and waveform held at about three times its measured error (vision
       5.15e-5 to 3.56e-4, audio 1.03e-5 to 2.49e-4 max abs — the f32 series' order) with new sha256 rows per series;
       the pipeline e2e checks that each preset fetches its own storage of every role, and the `i8` preset's image and
       audio rows were recorded again (the 11 rows of the maintainer's approval of that day; every other row
       unchanged), its cosine to upstream's vectors now 0.999627 to 0.999881 (images) and 0.999762 to 0.999907 (audio)
       against the unchanged floor 0.999; the int8 towers need no `requiredLimits` (the 128 MiB device). The quality
       set through the real pipeline (`tools/embeddinggemma2-quality-set/compare-presets.ts`,
       [research 2026-11-23](research/2026-11-23-embeddinggemma2-tower-i8-gpu.md)): every item's cosine of `i8` to
       `bf16` equals the screening's to within 4.4e-8 (lowest 0.999648 over the images and 0.999493 over the clips at
       768), the same top-1 hits as `bf16` in both directions, and median times per item within 3% for images and clips.
     - Left unbuilt, by the same choice: the activation-int8 preset `i8-a8` (the screening: lowest cosine 0.99563 and
       2.39% of the ranking triples inverted at 768, against 0.99940 and 0.74% for `i8`), and every 4-bit preset (lowest
       0.960 to 0.966, 8.2 to 8.5% inverted).
     - Left as is: the folded RoPE tables, a fixed 48 MiB of f32 constants in every preset (cos and sin of
       `[8192, 256]` and `[8192, 512]` — what bounds T at 8,192).
     - Left: the two slow patterns the preset A/B measured on the RTX 3080 Ti — at 8,192 tokens one visit per round
       about 1.8× slower, found the same day to be a major GC of the Node process (the Dawn binding's polling while a
       readback waits), not the `bf16` preset (both presets take the same GPU time); at 2,048 tokens periodic
       stretches about 11× slower in both presets, in the GPU time itself (the slow mode's signature, cause not
       investigated) — filed in [known issues](known-issues.md) with the fix candidates. The record now holds the
       host's GCs per timed visit (`karume-embeddinggemma-preset-ab/2`: major GCs and their summed pause, the other
       kinds apart), so a slow visit that held a major GC shows as one; the other candidates are not chosen.
     - Left: the first publication, awaiting the maintainer's go-ahead.
  4. ~~The token count per row~~ — built 2026-11-22 (the maintainer chose `Pipeline.tokenize` that day):
     `Pipeline.tokenize` returns the ids `embed` would run (a sequence over the limit or holding a placeholder id is
     returned, not refused), and the Playground page `/embeddinggemma` shows each row's count against the limit once
     the pipeline is built and records it per row. The page (built 2026-11-21) was checked by the maintainer in Chrome
     on an Apple M2 on 2026-11-22: a run of the three sample rows at 768 and at 256 dimensions, and the refusal of a
     row holding the image placeholder. Not checked in a browser yet: Cancel, the downloads, GPU timing, a row over the
     limit and the token count.
  5. Image input (approved by the maintainer 2026-11-22; the upstream facts in the untracked investigation notes
     `outputs/diag/eg2-multimodal-2026-11-22/`). Stage 1 built 2026-11-22:
     - the text graph takes the soft rows of a tower — `soft_embeds` `[1, T, 512]` f32 and `soft_mask` `[1, T, 1]`
       bool, merged by one `where` after the scaled token lookup (the Gemma 4 12B product graph's names and form;
       the recipe checks the merge's form and, on every long case, that no soft row is the `input_ids` path bit
       for bit). Every series was exported again: all thirty io outputs and every container part but the
       descriptor's byte-identical, and the lane green with every existing sha256 row and golden untouched. The
       pipeline binds zeros and false for a text (no public API change); on the RTX 3080 Ti under Node that costs a
       `bf16` text call nothing measurable at 16 tokens (fastest visit 12.8 → 12.8 / 13.0 ms) and about 2 ms at
       2,048 (114.9 → 117.2 / 117.4 ms; the workbench's `embeddinggemma-preset-ab`, 31 rounds, two runs after);
     - `export_vision.py`: the vision tower with `embed_vision` as one graph `vision` — `pixels` `[1, 9n, 768]`,
       `position_x` / `position_y` `[1, 9n]` → soft tokens `[1, n, 512]`, the real patches only, in block order (the
       io contract in the recipe README), the 2-D RoPE folded into two `[840, 32]` tables gathered by the positions;
       no new op, kernel or runtime change. Series `embeddinggemma-2-vision` (f32) and `-bf16`, goldens from
       upstream on four synthetic images (139 soft tokens is the resize rule's smallest count — a rounding edge),
       the wrapper within 3.9e-5 of upstream in torch, the series e2e on the GPU within atol 1e-3 (measured 6.5e-5
       to 3.35e-4 — the tower is that sensitive in f32: torch f32 is up to 9.8e-5 from f64) with sha256 rows;
     - `multimodal_reference.py`: upstream SentenceTransformer's token ids and vectors of five whole inputs holding an
       image, as the fixture `packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.*`; Karume's two
       graphs composed in torch reproduce them to 1.8e-6.
     Stage 2 built 2026-11-22 (the maintainer's approval of that day):
     - `embeddinggemma.Pipeline` takes image parts (`{ type: "image", image }`, decoded RGB8 — the Gemma chat shape)
       when built with `vision: true` (the Gemma family's option and manifest shape for its image embedder; without it
       an image part is a `ModelInputError` that says how to build one, and a text call is unchanged — its vector keeps
       the text pipeline's sha256 row). The sequence is upstream's sentence-transformers one: `<bos>`, each run of
       adjacent text (the prefix glued to the first) tokenized as one string, each image as `<|image>` + n × 258880 +
       `<image|>`, `<eos>`; the ids of the fixture's five inputs equal upstream's exactly, and `tokenize` returns them
       (an image counts n + 2 toward both limits). The host preprocessing (`vision.ts`: the Gemma family's
       `imageTargetSize`, the torch uint8 bicubic port, the patch rows in block order) equals the vision series' io
       inputs byte for byte on its four images. One tower run per image, read back into `soft_embeds` with the mask set,
       then the text graph once; the tower's Session takes the family's row-block budget (plan 135.1 MiB against
       329.1 MiB at 280 soft tokens on the RTX 3080 Ti). The budget is upstream's default 280, fixed.
     - The distribution carries the tower as the weights role `vision`, bf16 in both presets until 2026-11-23 (step 3) (series
       `embeddinggemma-2-vision-bf16`; no rounded tower was screened — the card says so), and `pipelineConfig.vision`
       (`softTokenBudget`, the three image ids, `hiddenSize`; the TS reader is the Gemma 4 family's); the recipe checks
       the processor settings against what the host implements, the tower graph and its position tables against the
       budget, and the vision series' record. The mirror was assembled again (the text containers byte-identical).
     - The pipeline e2e over the mirror on the RTX 3080 Ti: `bf16` within 5.44e-6 of upstream's vectors (atol 2e-5; the
       text graph fed upstream's own soft tokens is within 9.8e-7 — the rest is the tower's GPU noise), `i8` at a cosine
       of 0.99976 to 0.99991 (floor 0.999), with sha256 rows per preset.
     - The Playground page `/embeddinggemma`: an optional image per row on a model with the tower, the row's count with
       its soft tokens, the record `karume-playground-embeddinggemma/2` (each image's name, size and RGB8 sha256). Not
       checked in a browser yet.
     Review follow-ups (2026-11-22): per-image tolerances and a per-token cosine floor in the series e2e, a fifth
     golden image whose side is the longest the resize rule gives (`gradient-strip`, 840 × 3 → 13,440 × 48 px:
     positions up to 839, the tables' last row), the text graph's soft-row path checked alone against upstream's
     vectors (the text graph fed upstream's soft tokens, atol 3e-6), seven id-only cases of upstream's ids at run
     boundaries (spaces, newlines, tabs, an empty text between images, Japanese, a prompt before an image) checked
     on the host without the mirror, and the distribution's checks that both position axes are gathered and the
     folded RoPE tables are stored f32.
     Left: video (frames through the same tower); unordered and not chosen: the tower's runs in
     `onRunDiagnostics` (reported for the text graph only) and other soft-token budgets (they would need longer position
     tables). The int8 tower is step 3's (built 2026-11-23). Open couplings, not built: (a) a manifest whose
     `pipelineConfig.vision.softTokenBudget` is below the traced range is admitted — the reader takes 70 or 140,
     the tower was traced and checked for n in 139 to 280 only (the IR records no symbol range to check against), so
     such a hand-made manifest runs the tower at counts no golden covers (560 and 1120 are refused by the position
     extent check; the distribution writes 280); (b) the tower's Session takes the text preset's session options
     and GPU features, and the dist recipe folds the tower's demand into every preset's `requiredLimits` — both
     presets have an empty session and the tower's largest tensor is about 15.7 MB, so nothing changes today, but a
     preset with a knob screened for the text tower alone would apply it to the bf16 tower unscreened, and a
     text-only pipeline inherits the tower's limits.
  6. Audio input (the checkpoint's audio tower is Gemma 4 E2B / E4B's, unchanged; the upstream facts in the untracked
     investigation notes `outputs/diag/eg2-multimodal-2026-11-22/`). Stage 1 built 2026-11-22:
     - `export_audio.py`: the audio tower with `embed_audio` as one graph `audio` — `features` `[1, 4n, 128]` (the audio's
       valid log-mel frames zero-padded to `4n`) and `frame_mask` `[1, 4n]` → soft tokens `[1, n, 512]`, `n` = the
       valid frames / 4 rounded up, 1 to 750 (upstream's front end truncates at 30 s); the mask matters only when the
       valid frames are 2 mod 4 (the io contract in the recipe README). The chunked local attention is rewritten as a
       window of 12 keys per query (a one-hot depthwise `conv1d` copies the keys, `bmm` for both score terms, the
       relative keys folded), the clipped linears clamp by scalars; no new op, kernel or runtime change. 1,202 nodes.
       Series `embeddinggemma-2-audio` (f32, 1,173,714,150 bytes) and `-bf16` (587,590,521), goldens from upstream's
       feature extractor and tower on six synthetic waveforms (every residue of the frame count mod 4, 1 s, 10 s and the
       30 s maximum); the wrapper within 1.01e-4 of upstream in torch (in f64 within 1.3e-12 — exact arithmetic; torch's
       own f32 run is up to 3.0e-4 from f64), the series e2e on the GPU within atol 2e-3 (measured 1.41e-5 to 5.82e-4)
       with sha256 rows, the 30 s golden bit-identical on a device capped at the WebGPU default 128 MiB. On the RTX 3080
       Ti under Node a warm run takes 16.5 / 33.3 / 94.2 ms at 1 / 10 / 30 s (bf16; plan backing 4.2 / 42.0 /
       126.0 MiB), the Session builds in about 0.55 s (bf16) or 1.15 s (f32);
     - `multimodal_reference.py`: four whole inputs holding an audio (an audio alone, a text then an audio, an audio with
       the `query` prompt, an image then an audio) as `audioCases` of the same fixture, with the samples and upstream's
       valid log-mel frames of each audio; the five image cases unchanged byte for byte; Karume's graphs composed in
       torch reproduce the audio cases to 1.2e-7.
     Stage 2 built 2026-11-22:
     - `embeddinggemma.Pipeline` takes audio parts `{ type: "audio", audio }` (`MonoAudio`, 16 kHz mono f32) with
       `audio: true` (the weights role `audio`, fetched only then; `vision` and `audio` may be on together). The host
       front end (`src/embeddinggemma/audio.ts`) computes the log-mel frames with the FFT moved to `src/audio/fft.ts`
       (first written for the vowel detector): the windowed product in f32 as upstream's (since 2026-11-23 — the
       review found that the 5.12e-4 gap of the first build was upstream's f32 product, not its f32 FFT), the FFT and
       the rest in f64; its frames are within one ulp (4.77e-7) of upstream's f32 frames, and their bits equal
       upstream's arithmetic with an f64 FFT (the recipe's `audio-front-end.json`). Refused, never adjusted:
       another rate, over 30 s (upstream truncates silently), at most 160 samples, a non-finite sample.
     - The distribution carries the tower as the weights role `audio`, bf16 in both presets until 2026-11-23 (step 3) (series
       `embeddinggemma-2-audio-bf16`), and `pipelineConfig.audio` (the three audio ids, `hiddenSize`, `maxSoftTokens`
       750, `sampleRate` 16000); the recipe holds the feature extractor's settings in `preprocessor_config.json` and
       `processor_config.json` to the host's, the tower graph and its bound, and the audio series' record. The mirror
       was assembled again (the text and vision containers byte-identical).
     - The pipeline e2e over the mirror on the RTX 3080 Ti: upstream's ids for the four audio cases, `bf16` within
       2.44e-7 of upstream's vectors (atol 7.5e-7; 2.45e-5 and atol 7.5e-5 with the product in f64), the text graph
       fed upstream's soft tokens within 1.12e-7 (atol 3.5e-7), `i8` at a cosine of 0.99982 to 0.99991 (floor 0.999),
       a 30 s audio bit-identical on capped devices, ten sha256 rows (re-recorded on 2026-11-23 for the f32 product).
     - The Playground page `/embeddinggemma`: an optional audio per row on a model with the tower (decoded, resampled
       to 16 kHz and mixed to mono by the browser), its soft tokens in the row's count, the library's refusal of the
       row shown under it, the record `karume-playground-embeddinggemma/3`. Not checked in a browser yet.
     Left, unordered and not chosen: a resampler or decoder in the library (the caller resamples today), audio longer
     than 30 s (upstream truncates; splitting is the caller's), the tower's runs in `onRunDiagnostics` and an
     upstream-bit front end (numpy's complex64 FFT is not copied — one ulp). The int8 audio tower is step 3's (built
     2026-11-23).
- **Practice of benchmarking against ORT Web** (maintainer ruling 2026-08-16): measure models supported by both
  regularly, and investigate whether we are too slow and where the bottleneck is. **It has run for gemma4 only**, and
  the record is [research 2026-09-12](research/2026-09-12-webml-browser-speed.md). Its harness was removed in 0.15 and
  no ORT comparison harness exists today (the workbench's `gemma4-bench` entry compares Karume's own arms), so resuming needs
  one as a browser application (ADR 0127). Remaining target =
  **KokoroTTS** (corrected 2026-08-16 — the original Irodori was a typo. Kokoro runs on the Transformers stack, so
  it is easy to compare; requires Kokoro-82M support on the karume side = the model expansion candidate above, waiting
  for LSTM multi-output). In future, widen it into a comparison matrix that includes browser ONNX + native PyTorch
  (ruled unnecessary for now). The norms for measurement conditions (graph capture ON / freeDimensionOverrides / IO
  binding / checking EP partitioning, recording native EP or JSEP) have their source of truth in
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md).
- **Rolling generation events out to other families (waiting for demand)**: stage events for sbv2 / birefnet / depth /
  siglip2 (no step loop, so only stage transitions can be offered), and an AbortSignal cancellation slot in the
  **generation loop** of anima / irodori (today a throw from onEvent is the step-granular means of cancellation — the
  slot is kept in reserve). MOSS-Image's generation entry has neither events nor cancellation. **AbortSignal on the
  construction path is implemented for anima / irodori / Wan / MOSS** (`anima.PipelineOptions.signal` — checked at stage
  boundaries, passed through to the fetch layer, `signal.reason` propagated as is), so that is the precedent for the
  style (the construction `signal` of sbv2 / birefnet / depth / siglip2 / gemma / umT5 is only passed through to
  the fetch layer, with no checks at stage boundaries). **The generation-loop slot is implemented on the LLM surface**
  (ADR [0083](decisions/0083-generation-api-surface.md) decision 5) **and on Wan** (`wan.GenerateRequest.signal`) —
  rolling it out to other families still waits for demand.
- **RAM-saving sequential tile assembly for large anima resolutions (filed by the ruling of 2026-08-24)**: if VAE decode tiles
  can be composited sequentially without accumulating them, the peak RAM drops (`decodeTiled` still accumulates all tiles in an array before
  compositing). The original motivation, restoring acceptance of the 8 resolutions, was achieved separately by the evenly spaced snap placement of the ADR
  [0033](decisions/0033-vae-fixed-tile-decode.md) addendum.
- **RAM-saving sequential tile assembly for Wan's VAE (filed 2026-10-11, Codex deep review R6A-016)**: the Wan version of the anima
  candidate above. `decodeWanVaeTiles` (`packages/models/src/wan/vae-tiles.ts`) accumulates all tiles in an array, then allocates the final video and
  composites, so (computed when filed, on Wan2.1's 832×480 × 81 frames) the 12 tiles' 764,411,904 B and the output's 388,177,920 B are alive at the same time
  (computed from the shapes; RSS not measured). Candidate: decode row by row, paste into the output, and keep only the row above and the tile to the left. Conditions: keep the blend
  order (vertical → horizontal), and decide adoption after confirming bit identity with today's output by a Uint32 comparison. To measure: RSS for 33 / 81
  frames, readback wait, GPU time (for Wan2.2, measure the same quantities at 1280×704 and at the exit of patchify space).
- **Avoiding duplicate downloads of sha256-identical files across repositories in hub (external feedback proposal ⑥, low priority)**:
  the fetch layer's cache key is already a content key (`['hf', kind, repo, path, sha256]` — ADR
  [0080](decisions/0080-hub-fetch-cache-050.md)), which removed duplication across revisions, but `repo`
  remains a key element, so across repositories the file is fetched again. Proposal: drop `repo` (or add an alias keyed by sha256 alone).
- **cache-less streaming mode** (a prerequisite for the 26B A4B class — the CacheStorage quota is the first wall): phase 1
  (streaming prefetch) cannot fall back to plain fetch without CacheStorage and stays fail-loud
  ([limitations](limitations.md)). Revival = real demand at the 26B A4B class. The first unit is one design memo on "whether sequential reads of phase 2 alone
  work with a source that has no CacheStorage".
- Pinning down the cause of the Metal numeric difference (known-issues); lifting the diagnostics / measurement constraints of the resident path.
- **Spike, if needed: rounding barriers in the states-form attention for Metal** (known-issues, the Metal kernel-check
  section; ruling 2026-11-13: record first, spike later). Put the XOR rounding barrier on `q · scale` / `k · scale`
  in ①′ and the folded ①′, then on `p` in the unrolled PV forms and the read-only reader. Check that the NVIDIA words
  and sha rows do not change, measure the Gemma 4 decode speed (drop it if it costs speed), and run the workbench's `kernel-check` entry
  on the M2 and the RTX 5070 Ti. The folded side's Metal form is unknown, so it may still differ. If D3D12's F = 4
  difference (perf-ledger K-95) has the same cause, E4B could take the folded form.
- **MoE page-fault**: PoC done in a spike outside the repository (2026-09-01) — the readback of a miss can ride on the
  round trip that decode already pays, with zero extra synchronization, and the output is bit-identical to direct
  binding; the PoC's miss cost was dominated by one re-execution fence of about 11 ms from Deno's 10 ms sleep. Since then
  the opt-in exact expert cache (ADR [0146](decisions/0146-exact-expert-cache.md)) pays a miss with a re-run, and
  Chrome's fence and upload costs are measured there ([research 2026-11-08](research/2026-11-08-gemma4-26b-moe-cache.md)
  §6.1). Whether the spike's mechanism still adds anything next to it is not ruled. The start condition is
  subordinate to parked "value-dependent execution selection in the IR". The first unit stops at transcribing the spike's measurements into
  [research](research/) as a point-in-time snapshot.
- The MoE seam (fixed-k routing can be expressed with static shapes — do not bake the absence of experts into the dense API).

- **Remaining items of GEMV row blocks (filed 2026-09-07, consequence of K-21)**: ① a sweep of the parallelism target
  (16384, the saturation point of the RTX 3080 Ti — limitations; the M2 kept the default) on integrated GPUs; the
  override is the static Session option `linearGemvRowsThreadTarget` ② the shader analysis cost on Tint (Chrome, and
  Dawn under Node) is unmeasured; the only measurement is on naga ([research 2026-09-07](research/2026-09-07-gemv-rows-k21.md))
  ③ the Metal u32 check is also unmeasured for the 13 row-block shapes (known-issues).

- **Anima: iteration residency inside the DiT stage only** (filed 2026-09-07 — Codex performance investigation 04 §Anima): keeping the current design that creates and returns a Session per stage
  (a VRAM invariant), patchify the initial latent once inside the DiT stage, reuse the RoPE / cond and uncond embeddings /
  timestep material across iterations, run DiT output → CFG → Euler / DPM++2M update in the same token layout, and unpatchify only at the end.
  The responsiveness of `copyLatents` / onEvent / abort is a contract of the public surface, so it is not cut (also measure a boundary that finishes every few steps).
  **Making only the fixed inputs resident was measured — no gain** (2026-09-10, 1024px / CFG1 / Euler8, all steps / PNG identical).
  Iteration residency including latent / scheduler, and CFG>1, are unverified ([measurements](research/2026-09-10-codex-mtp-optimization.md#ホスト待ちと既存融合の追加測定)).

- **Whether to retire the slot backing before the bucket run** (filed 2026-09-07, limitations "prefill buckets"): the bucket run of the last chunk
  (the miss run) allocates temporaries in the arena while the chunkLength-shaped backing is still loaded, and the uncounted window carries
  "the miss run's arena temporaries (at most one bucket shape)" (with the default ladder, 768 shape + 256 shape ≈ 1.33 times).
  Retiring the activation backing first in generation's miss run removes the window, but the next 768 chunk then needs one more
  rebuild. Rule after seeing whether it helps on 4 GB class devices.

- **A summarizing overflow policy for ChatSession**: `onOverflow` is replaceable, so it goes in as one policy
  implementation. The reconsideration condition "after widening the window" was met by ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md) (capacity is a runtime knob)
  — this is where the deferral statement in the ADR [0083](decisions/0083-generation-api-surface.md) addendum points.

- **Measuring HF CDN concurrency**: once the per-connection limit is measured, re-file raising the download parallelism constant or splitting the parts
  toward the end more finely (the download slot improvement itself is killed —
  [research 2026-09-02](research/2026-09-02-cold-load-dl-timeline.md)).

- **Re-evaluating the GPTQ sweep**: the default stays as is (settled); the opt-in implementation is kept (source of truth =
  [research 2026-08-31](research/2026-08-31-gptq-axes-sweep.md)). Revival condition = re-evaluation with **multiple models ×
  16× calibration data** (a new gemma4 calibration rig is also on hold until then).

- **Moving norm's 1/dim to the host is on hold** (a real-GPU probe comes first; low cost-effectiveness. Making the reduce identity's
  params −inf stays as is = ruled as a set in the same frame as W-2/W-3) / **tolerance option B** (adds a
  reduction-scale term to `allclose` and the `Tolerance` type in `packages/runtime/src/reference/allclose.ts`, which are
  internal but shared by every test tolerance, the e2e ones included — option A's per-op table is implemented.
  [research 2026-08-31](research/2026-08-31-op-tolerance-measurement.md) §8.2).

- **Adjacent filings of the memory management wave** (source of truth = ADR [0089](decisions/0089-memory-limits-preflight.md)
  Consequences / ADR [0090](decisions/0090-shard-spec-v3-tensor-pieces.md)):
  allocation in `GpuContext.createResident` still relies on errorScope (run-time transients fail before allocation in the planning-time
  preflight) / the positioning of `fromAssets` / the reference-first general rule for
  large assets.

- **Handle the N % 4 remainder of fused attention's a8 / s16 on the declared path (filed 2026-10-10)**: today, for shapes where the key length N
  is not a multiple of 4, ③ PV and the S storage fall back to f32 (docs/limitations.md; decided by `attentionRoute`). Key points:
  ③ PV's i8a8 needs the remainder in the A tile fill (n ≥ N as 0) and a `quantize_rows` variant that zero-pads the row width to a multiple of 4.
  The f16 storage of S needs the write race on u32 words that span rows removed structurally, passing S's row width as a uniform separate from N
  to ①②③ (both dense / i8a8). Both touch kernels shared with linear (`quantize_rows`, the GEMM skeleton), so
  keep the output for existing keys with N % 4 == 0 byte-unchanged (the snapshots are the detector). Start condition: when there is a request to use Anima resolutions that are not
  multiples of 32 px in practice, or when a family that accepts N % 4 != 0 shapes has an a8 / s16 preset.
  MOSS-Image accepts such shapes with an a8 preset, but its preset runs them on the online
  kernel, which pads the key axis (ADR 0132 addendum of 2026-10-27); it meets the condition only on a device with
  neither online form, where the practical preset runs the three dispatches with S in f32 and, where S % 4 != 0,
  ③PV in f32 too (ADR [0138](decisions/0138-moss-image-dit-vae.md) decision 10).
  Kill criterion: measured VRAM and speed on those shapes show no difference against today's path that falls back to f32.

## release — standing release gates and gates of a first publication

- Real-asset CI gate (GitHub CI does not exercise local assets; the all-skip guard covers the empty case): decide on promoting goldens to fixtures / fetching assets in the release gate
- Writing up the repository root README; verifying sideEffects of the JSR npm compatibility layer. **Out of scope for 0.8.0
  (ruling 2026-09-03 — replaced with a form whose Status line alone says it is a stub). Revival condition = 1.0 or
  a public announcement**, and starting requires re-measuring the bundle size (the gzip measurement of 2026-08-16
  is outdated by the additions of the gemma4 generation API, the GEMV family and the tokenizer)
- Licence interview (human confirmation of export-recipes' per-family provenance per upstream revision; it runs at every publication — release-runbook). What the interview of 2026-09-25 left open (the per-family ledgers,
  `tools/export-recipes/<family>/THIRD_PARTY_NOTICES.md`, hold more `Unverified` rows than this list — for example
  Irodori v4-Large's):
  rows left as Unrecorded = the revision of modernbert-ja-310m (no local record), the revision of Semantic-DACVAE-Japanese (parent) (not fetched).
  Rows left as Unverified = whether the Civitai permission field of anima-wai-v1.0 can widen the upstream licence, the Weights license of depth_anything (before revision matching).
  The Code license of the code-dependency block is still Unverified in the deberta ledger
  (its Attribution too).
  The machine-readable place for the upstream revision is the container's `provenance.upstreamRevision`
  ([container-v1](container-v1.md) §2.3). The recipes of anima and irodori (main and dacvae), and
  gemma4's `export.py`, still declare a `PROVENANCE` without it, so they **start filling it at their next re-export**
  (through `tools/export-recipes/_shared/upstream.py`; on the same occasion as the `sym_max` field of W-G4-4, which
  `Provenance` does not have yet)
- **Gate for the first PyPI release of `karume` (tools/exporter): no upstream copy in the core wheel** (ruling 57 a,
  2026-11-01). `custom_ops.py` holds a verbatim copy of transformers 5.14.1's `l2norm` / `torch_chunk_gated_delta_rule`
  as the eager body of `karume::gated_delta_scan` (ADR 0135 decision 5), with a third-party notice comment. That
  conflicts with ADR 0065 decision 7 (the core wheel holds no copy of an upstream model implementation; the boundary
  test catches imports, not copies), and the wheel's `license-files` carry only MIT. Fix before the first upload: core
  keeps the op's schema and fake only, and the qwen3_5 recipe (which already depends on transformers) registers the
  eager kernel by importing upstream's function — no copy anywhere, and the "verbatim copy" MUST disappears with it.
  ADR 0065 itself stays: core does not take a transformers dependency
- Re-ruling on the policy of "separating the semantic surface from the implemented subset" for the ops that predate it
  (attention / deform_conv2d / gather / conv_transpose1d / upsample_bilinear2d — a unified rule that does not turn the
  observed subset into op semantics; filed 2026-08-15, ADR [0064](decisions/0064-entry-door-generality-axes.md) decision 1
  and its withdrawal of 2026-10-18). Order not ruled

## parked (with revival conditions)

- **Quantized weight delta as a way to ship or store a LoRA** (on hold, the maintainer 2026-11-15). The integer difference
  between `Q(W + ΔW)` and the distributed `Q(W)`; the measurement is
  [research 2026-11-15](research/2026-11-15-quantized-weight-delta.md). Too heavy as a replacement for a LoRA file.
  Revival = a need to keep a converted int8 container per LoRA. Not measured then: a real LoRA file, `int4`.

- **Gemma 4 MTP (speculative decoding with the official drafter)**: the source of truth for the current state and the remaining items is the K-20 row of [perf-ledger](perf-ledger.md)
  (remaining = M2 warm re-measurement of exploreMax 512, ⑥ k=7 + 3-value gate (awaiting a ruling), attribution of ⑩, 4-C docs). The history of stages 1 to 4-B and K-21 / H-15 is in
  the item "performance wave K-21 → H-15 done" of the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).

- **Value-dependent execution selection in the IR (prerequisite for dynamic MoE expert residency)** (ruling 2026-08-31 — not adopted).
  Per-expert load / eviction is a triple conflict among ① the container's merge layer (`bindDeclarations` — the krm entry `bindGraphs` also delegates to it; the binding table's
  full comparison is `validateAgainstGraph`), a check over all entries that matches the bindings of every initializer before the weights are fetched, ② the immutable Map of resident weights, and ③ the absence of value-dependent execution selection in the IR, so it is not a feature addition but a redesign across 3 modules (measurement record =
  [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)). The default keeps every expert resident
  in VRAM and budgets by total parameters unless the opt-in exact expert cache is chosen ([limitations](limitations.md)):
  since then `expert_linear` (ADR [0145](decisions/0145-gemma4-26b-moe-resident.md) decision 2) removed ③ for fixed-k
  compute, and the exact expert cache (ADR [0146](decisions/0146-exact-expert-cache.md), the maintainer's ruling of
  2026-11-09) removed ② for experts only, without touching ①.
  Adding a "not-yet-arrived initializer" slot is also deferred as subordinate to this item (same ruling). Revival = real shipping demand for an MoE
  that does not fit in VRAM. The first homework then = a close first-hand reading of predictive offloading (SiDA arXiv:2310.18859 / HOBBIT
  arXiv:2411.01433) (the current reading: the readback can be removed, but the wall of keeping the host-side pool resident remains).
  The page-fault mechanism is **PoC'd in a spike outside the repository; practical viability uncertain** (on revival, transcribe the spike's measurements into
  research before entering a design spike — later section "MoE page-fault").

- **Port of SBV2 `adjust_word2ph`** (ruling not to adopt, 2026-08-21 — ADR 0072 decision 8). The word2ph redistribution for accepting edits that change
  the phoneme count (replacing readings that do not align with word boundaries). The reference is the **upstream**
  Style-Bert-VITS2 semantics (LCS diff + 1..6 clamp, the residue raises), not the fork that AivisSpeech
  pins, which "forces consistency by even increments and decrements" (no silent approximation).
  Revival = real demand for reading edits that cannot be expressed with overlays.
- **anima g16 evaluation** (deferred 2026-08-23 — maintainer ruling to prioritize extending GPTQ. SBV2's
  g-axis ruling does not generalize across model lines (research 2026-08-22), so the evaluation itself keeps its value.
  Revival = quality dissatisfaction in the visual check of base i4. Bringing it forward as a background run that does not collide with the main work is allowed
  (maintainer ruling of the same day))
- hub Range parallel + prefetch — revival = fragmented repositories coming back (perf L-3)
- Making GPU timestamps an estimation source, full f16 (option γ), SBV2's engine-dependent NFC
  (`packages/models/src/sbv2/text/tokenizer.ts` uses plain `normalize("NFC")`; first check that DeBERTa's normalizer
  needs NFC, then take the segment-table method of `packages/models/src/text/qwen2-tokenizer.ts` —
  [research 2026-08-03](research/2026-08-03-demo-w8-perf-recon.md)), regenerating the f32 anima series
- The by-design constraints (rank≤4 / OOB NaN / non-finite input / zero-element dimensions / cancellation granularity and others) have their source of truth in
  limitations.md — only items with real demand are promoted here
