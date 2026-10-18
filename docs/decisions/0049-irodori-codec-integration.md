# 0049: Irodori codec integration — tiling, watermark bypass, LUFS on the host, bundled codec

- Status: accepted (2026-08-12 — the maintainer approved the four axes of the phase-4 plan and
  passed the listening check); partly superseded by [0129](0129-test-layout-and-lanes.md)
  (decision 5: the latent check is opt-in, 2026-10-17); decisions 2 and 3 re-graded 2026-10-18
- Date: 2026-08-12
- Related: [0048](0048-irodori-host-port.md) (phase-3 host contract) /
  [0047](0047-irodori-dit-execution.md) (DiT execution shape) / [0041](0041-manifest-v2.md) /
  recon = [research/2026-08-12-dacvae-codec-recon.md](../research/2026-08-12-dacvae-codec-recon.md)

## Context

Phase 4 adds the DACVAE codec (G6 decoder / G7 encoder). Constraints: (1) decoder activations
reach 527 MiB at S=750 (30 s), so with WebGPU's default `maxStorageBufferBindingSize` (128 MiB)
**allocation fails from S ≥ 183**; (2) upstream's watermark branch (wm_model, 9.3M parameters with
an LSTM) does not fit the IR; (3) reference preprocessing is BS.1770-4 LUFS normalisation (host).

## Decision

### 1. Decoder tiling on the host with a discarded 8-frame halo — **bit-identical**

The decoder main path has no causal layer (symmetric pads, convT of exactly `L·stride`): it is
translation-equivariant, with a one-sided receptive field of 13,793 samples (7.19 frames). With an
8-frame halo on both sides the kept span equals the full-length decode **down to the reduction
order** (Karume's conv kernels gather in a fixed order per output element);
`packages/e2e/tests/e2e_irodori_codec_test.ts` asserts **exact Uint32 equality** on a real z. Tile
length is a **knob that does not change the result** (`codecTileFrames`, default 182 —
`irodori/codec.ts:40`, sized for the default limit); only the halo is a model constant
(`pipelineConfig.codecHaloFrames`). The outer ends of the first and last tiles need no halo.

- Rejected: slice / cat inside the graph — the graph still touches the huge buffers.
- **The encoder is not tiled for now**: short references fit in one pass; 120 s needs 1.47 GB × 2
  and fails at the default limit (`docs/limitations.md`; halo 6 frames per the recon).

### 2. Watermark bypass with form (2) of the upstream README (the 4 waveform-head layers stay)

Both `decoder.alpha = 0.0` **and** `decoder.watermark = wm_model.encoder_block.forward_no_conv`
are applied; the first alone returns 96 channels (the head Snake(96) → conv1d(96→1, k7) → Tanh
doubles as the front of the wm branch). A standing `torch.equal` check at export
(`_main_path_evidence`, `irodori/dacvae/export.py`) shows on every emit that the other 25
watermark-branch modules contribute nothing. **No watermark is applied to generated audio.** An
API option for one MAY be added when needed. (Re-graded 2026-10-18 to MAY, ruling 8a: the deferral
"until the pre-release phase" expired with publication on 2026-08-24, ADR 0050 decision 14; an
added option is non-breaking.)

### 3. The codec ships inside the distribution

Upstream keeps DACVAE in its own repository; Karume bundles it (+365 MB in f32) so that text →
audio completes from one repository (9 graphs since [0114](0114-irodori-dit-context-split.md)).

- A separate `karume-dacvae` repository MAY be split out when a second model uses DACVAE, through
  the existing cross-repository references ([0109](0109-manifest-v5-container.md) decision 3;
  precedent Wan → umT5, [0119](0119-wan-umt5-gpu-text-encoder.md) addendum 2026-10-03) with a pin
  update and a migration notice. (Re-graded 2026-10-18 to MAY: the original reasons — no
  multi-repository references in the hub, an unreleased project — no longer hold.)
- `sampleRate` / `hopLength` come from the codec's `metadata.json`; `sampleRate` equal to
  `frameRate × hopLength` is checked at TS parse (`irodori/config.ts`) and dist assembly
  (`irodori/distribution.py`): a DACVAE with another hop keeps the shapes and changes the voice.

### 4. LUFS normalisation on the host; the tail trim is decided on z

- Reference preprocessing (`decodeWav` → truncate to 120 s → **LUFS −16** → peak limit → reflect
  pad) is host pure functions. LUFS is a full BS.1770-4 port and **runs in f64**
  (`irodori/host/loudness.ts`); an f32 port of upstream's `lfilter` would not be bit-exact either,
  so the difference (≈ 2.2e-5 LU, lost in a WAV) sits in the parity threshold.
- The tail trim (`find_flattening_point`) **picks the position on z and cuts the waveform by
  sample**; cutting the latent first changes the boundary padding. Decoding is always full-length.
- WAV read **/32768** and write **×32767** is an **intentional** asymmetry matching upstream
  (`packages/models/src/audio/wav.ts`).

### 5. E2E keeps both the latent check and the WAV sha256 check

The phase-3 latent check isolates the host path (Euler + CFG + choice of S) from the codec; the WAV
sha256 check is the port's final regression check. (Superseded by ADR 0129 decision 4 on
2026-10-17: the latent comparison with the torch golden runs only with `KARUME_UPSTREAM_PARITY=1`
(`e2e_irodori_latent_test.ts:306-309`); by default the WAV sha256 rows and the self A/B check
(`e2e_irodori_ab_test.ts`) catch regressions. Reference values are per-environment rows since
[0106](0106-device-keyed-references.md).)

### 6. Reference audio is 48 kHz only (no resampling)

A `sampleRate` mismatch fails loudly: a port of torchaudio's `sinc_interp_hann` FIR is a separate
implementation to verify, and upstream skips it for 48 kHz too. If needed, it is a host function.

## Consequences

- Text → (reference wav →) WAV runs in TS (2026-08-12: 12.5 s for 6.44 s of audio).
- GPU `sin` keeps f32 accuracy (errors ≈ 5e-6) for arguments up to ≈22.7π; record in the recon
  (its test `e2e_dacvae_test.ts` was removed in the 0.15 test selection, `4991d3c9`).
- Speaker check tolerance from real latents: atol 7e-4 (`e2e_irodori_test.ts:211`).
- Remaining constraints (`docs/limitations.md`): no encoder tiling / 48 kHz only /
  `WAVE_FORMAT_EXTENSIBLE` rejected / no watermark. The DiT `scores` limit (from ADR 0047) was
  lifted by row-block attention ([0060](0060-row-block-attention.md)).
