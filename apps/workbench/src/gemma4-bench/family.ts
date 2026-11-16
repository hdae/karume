/**
 * Op families of the runtime's pipeline keys, for the split tap's attribution table. The families
 * follow the decode attribution of docs/research/2026-10-27-gemma4-decode-attribution.md (§3), read
 * from the key alone: a key is `<kernel name>:<variant…>` (`attention_state_qk:v1:f32:wg16x16:par:gqa`,
 * `linear_gemv_parallel:wi4g32:l32`, `ew:v3:mul:f32>f32:r3:wg256`), and the kernel name decides the
 * family. The research's PLE and lm_head rows are told apart by position in the run, which a key
 * does not carry: their kernels land in linear / norm / elementwise / embedding here. Pure, so
 * `vp test` covers it.
 */

export type OpFamily =
  | "attention-core"
  | "attention-other"
  | "linear"
  | "norm"
  | "elementwise"
  | "static-quantize"
  | "embedding"
  | "other";

/** Display names, in the order the tables list families when their times tie. */
export const OP_FAMILY_LABELS: Readonly<Record<OpFamily, string>> = {
  "attention-core": "attention core",
  "attention-other": "attention other",
  linear: "linear",
  norm: "norm",
  elementwise: "elementwise",
  "static-quantize": "static quantize",
  embedding: "embedding",
  other: "other",
};

/** Kernel names (the key's first field) with a family of their own; the rest go by prefix. */
const NAMED: Readonly<Record<string, OpFamily>> = {
  // KV writes, separate RoPE kernels, strided copies (head layouts, slices).
  state_append: "attention-other",
  rope: "attention-other",
  strided: "attention-other",
  strided_write: "attention-other",
  ew: "elementwise",
  // SRQ: the fixed-scale activation quantize, alone or fused after gelu × mul.
  static_quantize: "static-quantize",
  gelu_mul_static_quantize: "static-quantize",
  embedding: "embedding",
  gather: "embedding",
};

/**
 * The op family of a pipeline key. Attention core = every `attention*` kernel (the states form's
 * QK, row statistics, PV and fused statistics + PV, and the other attention forms' counterparts);
 * linear = `linear` and every `linear_*` kernel (GEMV variants included, also the ones that fuse the
 * SRQ of their input); norm = every `rms_norm*` kernel (`rms_norm_add`, and `rms_norm_rope`, which
 * applies RoPE after the norm). Unknown kernels are `"other"`.
 */
export const familyOfKey = (key: string): OpFamily => {
  const name = key.split(":", 1)[0];
  if (Object.hasOwn(NAMED, name)) return NAMED[name];
  if (name.startsWith("attention")) return "attention-core";
  if (name === "linear" || name.startsWith("linear_")) return "linear";
  if (name.startsWith("rms_norm")) return "norm";
  return "other";
};
