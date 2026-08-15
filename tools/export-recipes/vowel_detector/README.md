# Vowel-detector export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Scripts are started as
modules from `tools/export-recipes/` (`uv run python -m vowel_detector.export`); no dependency
group is needed. The generic emit, verify and dist contracts live in
[`../../exporter/README.md`](../../exporter/README.md).

## Vowel-detector CRNN export (one graph, symbolic length)

A small CRNN that turns 10 ms speech features into 8-class lip-sync logits: two Conv1d layers
(the first with stride 2, which halves the time axis), a 2-layer bidirectional GRU with hidden
size 128, and a linear head — 664,744 parameters in total. Feature extraction (80 log-mel bins
plus 3 DSP dimensions) and post-processing (log-softmax, penalised Viterbi, short-segment merge,
`.lab`) both stay on the host; only the network is a graph.

```sh
# one-time input: inputs/vowel-detector/crnn_epoch3.pt (upstream training checkpoint)
uv run python -m vowel_detector.export             # → outputs/series/vowel-detector-crnn-epoch3/
uv run python -m vowel_detector.export --verify    # gru_scan rewrite vs nn.GRU (bit-exact, 5 lengths)
```

No dependency group is needed — `torch` is a base dependency, and the 20-line model definition is
**transcribed verbatim** into the script instead of importing the upstream `vowel_detector`
package (whose import chain pulls in pyopenjtalk and librosa for G2P and feature extraction, none
of which the export touches). `load_state_dict(strict=True)` is what keeps the transcription
honest; `vowel_detector/tests/test_export.py` pins the 22 parameter names and shapes so a drifting
transcription fails without the real weights present.

### The length is symbolic — the GRU is rewritten into scan nodes

`aten.gru.input` survives `torch.export` as a single node, but `run_decompositions` unrolls it
**along time**, which specialises the graph on T — asking for a dynamic `Dim("T")` fails with
`Specializations unexpectedly required (T)`, and T10 = 200 produced 8,434 nodes. So the script
rewrites `nn.GRU` into `karume::gru_scan` / `gru_scan_reverse` calls before tracing
(`vowel_detector/patch.py`, ADR 0056): the input-side GEMM stays a plain `linear` over the
whole time axis, and each layer and direction becomes **one scan node**, joined by `cat`.
Everything else is the upstream `forward`.

The symbol is declared on the **output** grid, as `2*Dim("T")` on the input axis. A bare `Dim("T")`
would make the first Conv1d (kernel 5, stride 2, padding 2) produce `((T−1)//2)+1` — a floor
division, which the dimension language (`coeff·sym+offset`) cannot express. With `2*Dim("T")` the
conv output is exactly `T`. The consequence is a runtime contract: **the number of 10 ms input
frames is always even** (callers drop the odd tail frame, which the 20 ms output grid has no room
for anyway). Binding `T` from the derived `2T` axis is ADR 0057.

| what                | before (T10 = 200, unrolled) | now (symbolic)  |
| ------------------- | ---------------------------- | --------------- |
| IR nodes            | 8,434                        | **18**          |
| depends on length   | yes (38 nodes per frame)     | **no**          |
| `model.safetensors` | 3,892,256 B                  | **2,668,608 B** |
| export time         | 8.1 s                        | **0.6 s**       |
| initializers        | 23                           | 23              |

(Measured 2026-08-14, torch 2.13.0+cpu, on the real checkpoint. The 23 initializers are the 22
checkpoint tensors plus one folded constant — the zero initial hidden state. The weights are
2,658,976 B, so the graph JSON is now under 10 KB instead of growing by ~5.27 KB per input frame.)

The rewrite is **bit-exact against `nn.GRU`**, which is the whole basis for keeping the golden
values: `--verify` compares the two eager paths at five lengths, and every emit compares them again
per golden case before writing the expected outputs (see the gates below).

**Right zero-padding a short utterance into a longer graph does not work** — which is why no part
of this pipeline pads any more. The backward GRU carries state home from the padding: padding a
true length of 137 frames into a 500-frame graph measures a max abs diff of 5.91 and an argmax
agreement of 0.971, with the error concentrated at the tail (0.138 at the head, 5.915 at the end).
A unidirectional model would only see the conv window edge (5.1e-03).

### Gates that run on every emit

- **Weight conversion**: every one of the 22 checkpoint tensors is read back out of the emitted
  container with an independent reader and compared **byte for byte** (`assert_checkpoint_bytes`);
  any initializer that is neither a checkpoint tensor nor a folded `const.` constant fails loudly,
  so a weight cannot come back under another name.
- **Sanity** (orderings only, no thresholds): the silence-like case has the highest mean P(pau)
  of the four, and the voiced-like case the highest mean vowel mass — plus all four outputs must
  differ from one another.
- **Rewrite equivalence**: the expected outputs written into `io.<case>.safetensors` are taken from
  the **reference path** (`nn.GRU` itself), and the rewritten path is compared against them with
  `torch.equal` for every case. Taking the expected values from the rewritten path instead would
  let an exporter bug and a runtime bug agree with each other and both stay green.
- **`--verify`**: the same comparison at five lengths (T10 = 4 / 6 / 18 / 274 plus `--length`),
  measured 0.0 on the real checkpoint. Length-dependent mistakes (the backward scan's boundary, the
  order of the per-layer `cat`) do not all show up at a single length.

### Distribution form (`dist.py --pipeline vowel-detector`)

```sh
uv run python -m vowel_detector.export                 # one graph, any length
cp <upstream>/assets/feature_config.json ../../inputs/vowel-detector/
uv run python dist.py --pipeline vowel-detector        # → models/karume-vowel-detector/
```

The repository ships **one graph** (`crnn`, 2,668,608 B) plus the mel filterbank as an `assets`
entry, because feature extraction happens off-graph and cannot be reproduced without the exact
80 × 257 matrix the model was trained on. **2,759,461 B in total** — the four length buckets it
replaces were 34,088,454 B (−91.9%), since each bucket was a whole container, weights included.

The clip runs at its own length: no padding, no bucket, and therefore no length-dependent numbers.
That also removes a discrepancy the bucketed form had no gate for — the padded path disagreed with
the exact-length path on 3 of the 4 real utterances (a 20 ms boundary shift, a 40 ms `pau`
appearing at the tail, a 40 ms `pau` appearing mid-utterance), because the backward GRU reads state
back out of the padding. `packages/models/tests/e2e_vowel_detector_lab_test.ts` now pins the two
paths to be byte-identical.

`pipelineConfig` comes from two independent places and they are checked against each other before
anything is placed: the feature contract (`sampleRate` / `featureDim` / `classes`) is read verbatim
from the upstream `feature_config.json`, while `maxFrames` is the symbolic upper bound the export
script baked (`SYM_MAX`, in 20 ms frames, doubled). The IR carries symbol _names_ but no ranges, so
the limit exists only in the manifest — the assembly gate checks instead that the graph really is
symbolic (`2T` in, `T` out); a graph baked for one fixed length would otherwise assemble fine and
fail in the user's hands for every clip but one.
