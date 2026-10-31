"""Helpers shared by several families that cannot be promoted into the core (PyPI `karume`) (ADR
0065 decision 2).

What lives here is what the per-model recipes need in common but is not the generic exporter's
job — the repository's path spellings ({@link _shared.paths}), the gate the decode-series scripts
share ({@link _shared.decode_series}), the judgement of an i4 series' calibration conditions
({@link _shared.calib_provenance}), the stored-form read-back of a published container
({@link _shared.container_read}), deriving and cross-checking the provenance of an upstream
checkpoint ({@link _shared.upstream}), the compile of the Gemma SPM-BPE tokenizers
({@link _shared.gemma_tokenizer}), the Qwen2 byte-level BPE tables and their reference
implementation ({@link _shared.qwen2_tokenizer}), the pure leaf functions of the VAE decoder's
rank 4 rewrite ({@link _shared.vae_rank4}), and the license texts shipped in the distribution
repositories ({@link _shared.licenses}). Whatever turns out to be promotable into the core (= it
depends neither on the repo topology nor on running the model scripts) moves to `karume`.
"""
