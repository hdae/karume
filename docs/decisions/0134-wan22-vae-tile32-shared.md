# 0134: Wan2.2 VAE decoder at a 32-latent tile — a second chunk-graph pair that borrows the 16-tile pair's weights

- Status: accepted (2026-10-23 — the maintainer's ruling 20 a; the gates of decision 7 are in Results. The kill line of
  decision 7.4 was ruled on the 17-frame case and is missed there by 1–3% (0.81–0.83×); the 33-frame clip meets it
  (0.70–0.83×). Kept as the opt-in knob; the maintainer confirmed that reading on 2026-10-24 after viewing the clips
  (the default stays 16 — decision 2))
- Date: 2026-10-23
- Related: ADR [0121](0121-wan22-ti2v-5b.md) decision 6 (tile 16 / overlap 4 — unchanged as the default) /
  [0096](0096-speculative-decoding.md) decision 2 (shared initializers — the mechanism this reuses) / [0133](0133-conv3d-w8a8.md)
  (the int8 decoder whose stage this shortens further) / [0070](0070-shard-loading-admission.md) decision 5 (no guessing of free VRAM —
  why the tile is explicit) / [0010](0010-symbolic-constant-folding.md) (symbolic dimensions — why one symbolic graph is not
  possible) / perf-ledger V-2 / research [2026-10-23-wan22-vae-recon](../research/2026-10-23-wan22-vae-recon.md) §3.

## Context

The Wan2.2 decoder runs in 16-latent tiles with a 4-latent overlap, so at 1280×704 it computes 2.04× the output pixels
(4×7 tiles over 44×80 latents). A 32-latent tile with the same overlap covers the frame in 2×3 tiles at 1.33× — 35% less
decoder work — but ADR 0121 kept tile 16 because the causal caches grow fourfold (about 1.9 GiB of residents at tile 32,
plus the fourfold plan backings), which the 10 GB class cannot hold.

The tile edge is baked into the exported chunk graphs: the latent input is `[48, 1, 16, 16]`, every cache `[C, 2, 16k, 16k]`
and the frames `[12, T, 128, 128]`. One graph with symbolic `h` / `w` is not possible: the mid-block attention reshapes
`[T, C, H, W]` into `[T, 1, 3C, H·W]` (`vae_patch.py`), and an IR dimension cannot carry a product of symbols (ADR 0010); the
VAE encoder split into three graphs for this reason, and splitting the decoder's chunk graphs around the attention would
cut the cache hand-over in two (the maintainer rejected that cost — ruling 20 a). A second fixed-shape pair at tile 32
duplicates the decoder's weights unless they are borrowed: ADR 0096 decision 2 lets a container declare initializers as
shared (no bytes; storage declared as the lender's) and a Session borrow the lender Session's resident weights
(`createSession({ sharedWeights })`, six fail-loudly checks).

## Decision

### 1. A second chunk-graph pair at tile 32 that holds no weight bytes

`wan.export_vae --tile 32 --shared` exports `vae_decoder_first_32` / `vae_decoder_next_32`: the same module traced at a
32-latent tile, with **every initializer declared shared** (`IrShared(tensor = the lender's tensor key)`, the lifted
constants included) and no weight bytes — its binding table is empty (a shared declaration carries no storage in IR v2;
the borrowing Session derives the residency class from the lender at construction, ADR 0096). The lender is the 16-tile
container of the same series and storage (`vae_decoder_first` / `vae_decoder_next`); the pair is written once per series
root (f16 and i8 — the two files come out byte-identical, about 0.1 MB each, and each series carries its own under its
own label). The writer gates that the two graphs are fully shared, that each shared initializer names a tensor of the
lender with the same shape and dtype, that the lender's tile is the series tile and its storage the requested one, and
that the shared tile differs from the lender's; `assert_chunk_graph` runs at tile 32 as it does at 16. The pair writes no
chunk fixtures (the 16-tile fixtures are the chunk-sequence reference; the 32-tile tile references are decision 6).

### 2. The tile is a request knob, default 16, never chosen automatically

`WanGenerateRequest.vaeTile?: 16 | 32` (default 16 — every existing output and sha256 row is unchanged). `32` is accepted
only when the preset's weights carry the `_32` roles (Wan2.2 presets do; Wan2.1 has none and rejects it) and is rejected
otherwise at planning with the reason. There is no automatic choice: WebGPU exposes neither total nor free VRAM (ADR 0070
decision 5), and the 10 GB class cannot hold tile 32 — the limitation names it.

### 3. Session topology: the 16-tile pair lends, the 32-tile pair borrows

For `vaeTile: 32` the decoder stage builds the 16-tile Sessions with the stage's options (the conv knob of ADR 0133
included) as **lenders**, then the 32-tile Sessions as **borrowers** with `sharedWeights` mapping every initializer to the
lender's `exportWeight` of the same name; the lenders outlive the borrowers (the runtime refuses to dispose a lender while a
borrower is alive). The cache and frame residents are sized by the 32-tile layout. VRAM: one copy of the weights
(borrowed), the fourfold backings and caches — about 9.5 GiB on the int8 decoder at 1280×704, 10.5 GiB on the f16 decoder
(measured in decision 7).

### 4. The plan keeps the overlap; the layout comes from the chosen pair

The tile plan (`planWanGenerationTiles`) takes the layout of the chosen pair; the minimum overlap stays 64 px = 4 latents
(ADR 0121 decision 6), so tile 32 steps 28 latents. Blending and unpatchify are unchanged.

### 5. Distribution

Two weights keys (`vae_decoder_first_32` / `vae_decoder_next_32`) with the decoder's labels (f16 / i8), one bytes-less
container per label, in the same series roots as their lenders; the presets' weights maps name them with the decoder's
label. The storage gates of a `_32` role run on its lender's container (the borrower's binding table is empty); the plan
gates the 32 pair like the 16 pair and adds `assert_shared_decoder` (no bytes outside the descriptor, every initializer
shared and present in the lender with the same shape and dtype). Wan2.1's distribution is unchanged (no `_32` roles).

### 6. References and rows for tile 32

`vae_tiling.py --tile 32` writes the tile references as `vae_tiles.<case>.t32.safetensors` (a tile suffix — the 16-tile
names stay); the band is derived for tile 32 as for 16 (band × 5). The pipeline's sha256 rows carry the tile in the case id
(`…-vae32-…`); the degenerate gate holds at tile 32 (one tile ≡ the untiled chunk sequence, Uint32).

### 7. Acceptance

1. The 16-tile rows and the 16-tile containers are byte-identical (the second pair adds files; nothing is rewritten).
2. The runtime's six shared-weight checks pass for both storage labels; a borrower built without a lender fails loudly.
3. Tile 32: the degenerate gate, the band against the Python tile reference, 0 non-finite values, the conv8 census.
4. Kill line: the VAE stage at 1280×704×17 with `i8-a8-attn8-online-conv8` and `vaeTile: 32` not under 0.8× of tile 16
   (−20%) → rejected with the number. Expected −30%. VRAM peak under 12 GiB on the int8 decoder.

## Results (2026-10-23, RTX 3080 Ti, Node + Dawn)

- Recipe (decisions 1, 5, 6): `wan.export_vae --tile 32 --shared` wrote the pair into both series roots (83 KB per
  container, the f16 and i8 borrowers byte-identical; 109 / 113 initializers of first / next declared shared, the three
  lifted constants included; empty binding table); the distribution lists `vae_decoder_first_32` / `next_32` under
  every Wan2.2 preset and gates them on their lenders. `vae_tiling.py --model ti2v-5b --tile 32` writes
  `vae_tiles.wide.t32.safetensors` (2×3 tiles, starts 0,12 / 0,24,48, blends 160 / 64,64; the `band` and `accept`
  cases are 30 latents on one axis and cannot feed a 32-latent graph — skipped and named in the summary); the 16-tile
  references are byte-identical before and after.
- Runtime / models (decisions 2–4): `vaeTile` on the request; admission opens the pair only when a preset carries both
  roles (one alone fails before any fetch), checks the pair's layout (tile 32, the lenders' compression and channels,
  every accepted size) and that every initializer is borrowed by name from the lender of its role; the stage builds the
  16-tile Sessions as lenders and the 32-tile Sessions as borrowers (`sharedWeights` from `exportWeight`), disposes
  borrowers → lenders → caches. The 16-tile path builds no lender and is unchanged (its sha256 rows and containers are
  byte-identical; the reference file gained one row).
- Gates (decision 7, `e2e_wan_ti2v_vae_tile32_test.ts` + the pipeline e2e): the runtime's shared-weight checks pass for
  f16 `{}`, i8 `{}` and i8 `{ convCompute: "a8" }` (borrower weights 0 bytes; lenders 1.01 / 1.03 GiB f16, 0.51 / 0.52
  GiB i8); a borrower without `sharedWeights` fails loudly; the degenerate gate (one 32×32 latent through the tiled path
  ≡ the untiled chunk sequence) is Uint32-identical with 0 non-finite values from NaN-poisoned frames in all three runs;
  the a8 census at tile 32 is w8a8 33 / 35 with no f32 conv3d key; the TS plan equals the Python fixture's meta; the
  band against the tile reference derived on `wide` (maxAbs 1.001e-5 → 5.1e-5 — `wide` is the only case that fits,
  so it both derives and meets the band; opt-in like the 16-tile band). The conv8 `vae32` sha256 row is written for
  this machine (its output differs from every other practical row).
- Timing of the VAE stage with `i8-a8-attn8-online-conv8` at 1280×704 (the kill line of decision 7.4; GPU-wide peak by
  nvidia-smi): 17 frames (the pipeline e2e, two runs) 50.9 → 41.4 s and 50.7 → 42.1 s (**0.81× / 0.83× — not under
  0.8×**); 33 frames (2 steps, two samples each, alternating) tile 16 92.1 / 93.1 s, tile 32 64.5 / 76.8 s (**0.70× on
  the minimums, 0.76× on the means**); peak 9.4 GiB during the stage at tile 32 (4.2 GiB at tile 16), under 12 GiB.
  The expected −30% is reached only on the best sample; the samples spread by up to 19% on this display-shared GPU.
- 121 frames: not measured. A 2-step 1280×704×121 request with `i8-a8-attn8-online-conv8` runs out of device
  memory (`VK_ERROR_OUT_OF_DEVICE_MEMORY` on `CreateBuffer`) under Node + Dawn on the 12 GiB RTX 3080 Ti at tile 16
  as well as at tile 32, in a fresh process each (nvidia-smi 9.3 GiB sampled at the failure) — not a property of
  the tile: the backing-build overlap fixed on 2026-10-24 (ADR 0095 addendum of that day).

## Consequences

- The 10 GB class keeps tile 16; the application chooses 32 explicitly on 12 GiB and above.
- Two more weights keys in the Wan2.2 manifest; about 0.8 MB more to download in total.
- The decoder stage holds four Sessions for tile 32 (two lend, two run).

## Alternatives considered

- One symbolic-shape pair (ruled 18 a first) — not expressible: the attention's `H·W` reshape (Context).
- A weight-carrying tile-32 pair — +0.5 GB (int8) / +1.0 GB (f16) download for the same graphs (ruling 20 a).
- A three-graph split around the attention with symbolic `h` / `w` — the only form that gives any tile; the cache hand-over
  would have to be split, and only 16 and 32 are used.

## Open points

- A free-VRAM signal would allow an automatic choice; WebGPU has none.
- Whether a 64-latent tile pays on 16 GiB+ devices (the same mechanism, another pair).
