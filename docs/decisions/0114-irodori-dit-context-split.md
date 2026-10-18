# 0114: Split the Irodori DiT's conditioning K/V projection into a separate graph `dit_context`

- Status: accepted (the maintainer's ruling of 2026-09-27, option A; voice-clone −12.7 % in the B570
  ABBA beat the −5 % kill criterion — see "Acceptance"); decision 8's "do not rename" re-graded
  2026-10-18 to MAY
- Date: 2026-09-27
- Related: [0047](0047-irodori-dit-execution.md) (replaces its decision 3 "fold G4 into G5 for now") /
  [0054](0054-resident-loop-and-fence.md) (ResidentTensor, batch enqueue, `copyOutputs` — the three
  parts used here) / [0050](0050-irodori-quant-series.md) (irodori weight series) /
  [0069](0069-packed-w4-storage.md) (i4 storage) / [0074](0074-quant-seat-naming.md) (preset name
  grammar) / perf-ledger H-30, H-7, K-54. Code: `tools/export-recipes/irodori/export.py`
  (`DitContextGraph`, `DitGraph`, `name_boundary`), `irodori/distribution.py`,
  `packages/models/src/irodori/dit-loop.ts`, `admission.ts`.

## Context

Each block of the irodori DiT projects context K / V for cross-attention from three conditions (text
256 + speaker 751 + caption 512 = 1519 tokens; upstream `project_context_kv`). This conditioning side
depends on none of the variable inputs (`x_t` / `t_embed` / `mask`): 244 nodes (72 linear) giving the
same values in every forward of a generation. ADR 0047 decision 3 folded it into `dit` and recomputed it
per forward, because a separate graph would upload its 178 MiB output on every run.

ADR 0054 removed that reason (ADR 0047 addendum of 2026-09-25): `copyOutputs` copies a separate
graph's outputs on the GPU into resident tensors passed to `dit` as resident inputs. On the B570
(i8-a8) the conditioning side took **24 %** of one `dit` forward's GPU time at S=750 (14.5 of 60.9 ms).

## Decision

1. **Boundary = the 24 concatenated context K / V tensors, one pair per block.** A new role
   `dit_context` (series directory `dit-context`) holds each block's conditioning projection and the
   first concatenation (text / speaker / caption → 1519 rows).
   - Inputs: `text_state [1,256,512]` / `speaker_state [1,751,768]` / `caption_state [1,512,512]` (the
     projectors' raw outputs, right-padded to Tmax). `text_norm` / `caption_norm` run inside this graph.
   - Outputs: `context_k_0, context_v_0, …, context_k_11, context_v_11` (each `[1,1519,20,64]` f32; K
     after `k_norm`), 178.0 MiB in total.
   - `dit` takes `x_t` / `t_embed` / `mask` + these 24; the three condition states leave `dit`.
   - Not the 72 projections: including the concatenation halves the boundary and removes it from `dit`.
2. **Boundary names are spelled in one place**: `irodori.distribution.irodori_context_kv_names(blocks)`
   (`context_k_<b>` → `context_v_<b>` per block b, ascending). The export writer (renaming IR
   boundaries) and the dist reader build from the same function. **The block count is not baked** —
   the exporter uses `len(model.blocks)`, `models` derives it from the `dit_context` declaration (its
   output count).
3. **Order of the resident path** (`projectContextOnGpu` in `dit-loop.ts`): ① allocate the 24 resident
   tensors; ② open the `dit_context` Session, enqueue it once with `copyOutputs` in a preceding batch,
   finish; ③ **dispose the Session**; ④ open `dit` and the combine / Euler graphs and run the loop
   batch.
   - Not in the loop batch: a Session cannot be disposed inside a batch (ADR 0054), so the 178 MiB
     output slots of `dit_context` would live until finish and double the VRAM increase. The separate
     batch costs one more fence and keeps the increase at the resident K / V's 178 MiB.
4. **Host path** (a device with `gpuTiming` / an `onEvent` subscriber — ADR 0054's existing branch):
   `run` `dit_context` → receive the outputs on the host and `write` them into the resident tensors →
   pass them to `dit`'s `run` as resident inputs. No 178 MiB upload per forward. The Session is folded
   before the resident tensors are allocated. The two paths differ only in how they copy (GPU copy or
   host `write`); `dit` reads the same K / V bytes.
5. **Same Session options**: `dit_context` also receives `ditSessionOptions` (the preset's
   `linearCompute`). Before the split the conditioning linears ran inside `dit` with that execution
   form; another form would move the numerics.
6. **Observation name**: `IrodoriRunComponent` and the `stage` event gain `"dit-context"` (the existing
   `_` → `-` rule for role names). `dit_context` runs once per generation.
7. **Admission** (`admitDitContext` in `admission.ts`, before any weight is fetched) checks four points:
   - `dit_context` has no symbolic dimension (resident tensors need their size at allocation);
   - its outputs are an even number of at least two (K / V pairs per block);
   - the set of `dit` input names = {`x_t`, `t_embed`, `mask`} ∪ the `dit_context` output names;
   - each output has a static shape equal in shape and dtype to `dit`'s input of the same name.
   The runtime checks only the size of resident inputs, so without this a shape or dtype mismatch
   would run silently on K / V read in another layout. The static-dimension checks of the three
   condition states (`maxTextLen` and others) move from `dit` to `dit_context`.
8. **The i4 preset takes both DiT graphs**: `IRODORI_DTYPE_ROLES["i4"] = ("dit", "dit_context")`, so
   `i8+dit4` takes both from the i4 series. Before the split the 72 conditioning weights were stored i4
   inside `dit`; dropping them to i8 would move the preset's numerics. The token `dit` of `i8+dit4`
   denotes both DiT graphs; the name was kept because renaming a published preset breaks
   distributions. A rename MAY be made in a breaking release (re-graded 2026-10-18 to MAY: keeping the
   name protected the then-published presets; 0.15 is a breaking release, and nothing requires the
   rename). The preset description "the other seven graphs" stays correct with the DiT counted as two.
9. **The runtime is unchanged** (resident tensors, `copyOutputs` and resident inputs of `run`).

### Options not adopted

- **B: split the graph in `models` at load time** (TS dependency analysis of the loaded declaration,
  synthesising two graphs) — not adopted. No re-export, but a general graph splitter enters `models`
  (H-7's mechanism in another place); the synthesised declaration bypasses the validation of the join
  layer (`packages/runtime/src/format/container/bind.ts`), and the graph that runs no longer matches
  the one validated at export (eager equivalence, goldens).
- **C: runtime staged execution (perf-ledger H-7)** — not adopted: H-7 was rejected as too complex
  (condition-cache keys, extended arena lifetimes), it would serve irodori alone, and A uses existing
  parts.

## Consequences

- **Numerics are bit-identical**: the split pair equals upstream eager with atol 0 (all golden cases,
  f32 / f16 / i8 / i4); stored bytes of old `dit` (430) = new `dit` (356) ∪ `dit_context` (88), 430 /
  430 byte-identical (overlap: 14 `k_norm`, equal on both sides); GPTQ re-run with the same calibration
  input reproduced the 72 conditioning weights bit for bit; the full-loop goldens and the WAV sha256
  reference values did not move (2026-09-27). The w8a8 census is `dit` 245 / `dit_context` 72 (317
  before). Swapping `context_k_0` and `context_v_0` changes the WAV sha256 (wiring is detected). The
  runtime series e2e covers `dit-context` (one case, atol 5e-4 / rtol 1e-6).
- **Speed**: on the B570 i8-a8, voice-clone −1.17 s (−12.7 %), 30 s utterance −0.67 s (−5.0 %).
- **VRAM**: the resident K / V take 178.0 MiB. The observed GPU buffer peak is unchanged (the DiT
  stage is not where the generation peaks).
- **Breaking distribution change**: manifest weights gain the `dit_context` role (all four series); a
  distribution without it fails as a missing component before any weight is fetched. Published in
  0.14.0 (CHANGELOG 0.14.0, both Irodori repositories republished).
- The `dit` io golden takes the 24 K / V tensors as inputs instead of the three condition states.
- ADR 0054 decision 4's "the three conditions are resident" reads, after this ADR, "the 24 conditioning
  K / V tensors are resident".
- Outside this ADR: sharing adaLN's timestep path across CFG variants (tentatively H-30b) and K-54 (a
  small-M path for a8 M=1 linears; the 144 adaLN M=1 linears take ≈ 19 % of `dit` GPU time).

## Acceptance (2026-09-27, Arc B570 / Linux xe / Deno 2.9.6)

ABBA (i8-a8, A = before / B = after the split, n = 6 per side), identical WAV sha256 on both sides:
voice-clone (S=170, 100 forwards) 9.18 → 8.01 s (−12.7 %), representative (S=750, 60 forwards, 30 s)
13.25 → 12.58 s (−5.0 %); observed GPU buffer peak unchanged (811 / 864 MiB). The kill criterion
(below −5 % on voice-clone → drop) was beaten → adopted. Raw data: `outputs/bench/karume-irodori/2026-09-27_h30/`
and `2026-09-27_op-single*` (untracked).
