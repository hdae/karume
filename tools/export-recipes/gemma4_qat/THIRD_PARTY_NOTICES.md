# Third-party notices — Gemma 4 mobile QAT

The recipe imports Transformers 5.14.1 and reuses this repository's Gemma wrapper. No upstream
implementation source is copied into this directory or bundled in the exporter wheel.

The source checkpoints are `google/gemma-4-E2B-it-qat-mobile-transformers` and
`google/gemma-4-E4B-it-qat-mobile-transformers`. Their downloaded model cards declare Apache 2.0;
source revisions and download hashes are recorded in the QAT research artifacts. Each export also
records the weight, config, and tokenizer fingerprints in `reference.json`.

The converted text weights retain the original packed integers and scales. The graph is rewritten
into states form, PLE is stored in a host-read sidecar, and RoPE inputs are generated on the host.
The model card states the numerical and validation limits. Distribution assembly includes the
verbatim Apache 2.0 license from `../_shared/licenses/apache_license_2_0.txt` and a modification
notice. It does not publish or upload anything.

See [ADR 0097](../../../docs/decisions/0097-gemma4-qat-integration.md) and the
[research record](../../../docs/research/2026-09-10-codex-mtp-optimization.md) for the source audit
and measurements. Release validation remains separate from the local experimental distribution.
