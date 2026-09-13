export const versions = { transformers: "4.2.0", onnxruntime: "1.29.0" } as const;
export const onnxModels = {
  normal: {
    repo: "onnx-community/gemma-4-E2B-it-ONNX",
    revision: "9f4bef82ea6e296bc69f8a2f5939f73af81b07a6",
    dtype: "q4f16",
  },
  qat: {
    repo: "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX",
    revision: "5cd5514efd375abf2801c856a3936b259cc00133",
    dtype: "q2f16",
  },
} as const;
export type ModelKind = keyof typeof onnxModels;
export type Engine = "karume" | "transformers";
export type PrefillBuckets = "default" | "sparse" | "dense";

export type NormalizationMode = "reference" | "submit768" | "fused" | "subgroup32";
