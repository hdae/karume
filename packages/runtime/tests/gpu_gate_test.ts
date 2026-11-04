// The all-skip guard of ADR 0005 ("every case skipped is an explicit FAIL").
//
// Real-GPU tests are skipped one by one with `ignore: !GPU_AVAILABLE`, so on a machine without an
// adapter stage 3 (real-GPU numeric checks) disappears entirely and verify still goes green. That
// is a silent apparent success that makes "not checked" read as "checked", so exactly one test
// fails here.
//
// On a machine with a GPU the guard itself **passes (it shows up as one green case)**. Making it
// ignored would make "the guard works" indistinguishable from "the guard is gone too".

import { test } from "vite-plus/test";
import { assert } from "@std/assert";
import {
  ALLOW_NO_GPU,
  ALLOW_NO_SHADER_F16,
  ALLOW_NO_TIMESTAMP_QUERY,
  GPU_AVAILABLE,
  SHADER_F16_AVAILABLE,
  TIMESTAMP_QUERY_AVAILABLE,
} from "./helpers/gpu.ts";

test.skipIf(ALLOW_NO_GPU)(
  "GPU all-skip guard: every case skipped without an adapter is an explicit FAIL (ADR 0005)",
  () => {
    assert(
      GPU_AVAILABLE,
      "No GPUAdapter could be acquired, so every real-GPU test was skipped. ADR 0005 treats this " +
        "as a FAIL (a release requires a green real-GPU run). To pass on purpose on a machine " +
        "without a GPU, set KARUME_ALLOW_NO_GPU=1.",
    );
  },
);

// A machine that has an adapter but lacks a feature also skips several tests completely within
// that feature's scope, and the only trace is one console.warn line from `helpers/gpu.ts` (verify
// stays green). That is the same silent apparent success as a missing adapter, so each feature
// gets a guard of the same shape with an opt-out.
// On a machine without an adapter the guard above already fails, so this one does not fire twice.
test.skipIf(ALLOW_NO_SHADER_F16 || !GPU_AVAILABLE)(
  "GPU all-skip guard: skipping for a missing shader-f16 is an explicit FAIL (ADR 0005)",
  () => {
    assert(
      SHADER_F16_AVAILABLE,
      "The adapter does not list 'shader-f16', so every real-GPU test of the f16 compute " +
        "variants (ADR 0028) was skipped. ADR 0005 treats this as a FAIL (the write guard of " +
        "the f16 compute variants would never have been checked). To pass on purpose on this " +
        "machine, set KARUME_ALLOW_NO_SHADER_F16=1.",
    );
  },
);

test.skipIf(ALLOW_NO_TIMESTAMP_QUERY || !GPU_AVAILABLE)(
  "GPU all-skip guard: skipping for a missing timestamp-query is an explicit FAIL (ADR 0005)",
  () => {
    assert(
      TIMESTAMP_QUERY_AVAILABLE,
      "The adapter does not list 'timestamp-query', so every real-GPU test of the GPU timing " +
        "diagnostics (ADR 0021) was skipped. ADR 0005 treats this as a FAIL (the GPU timing " +
        "diagnostics themselves would never have been checked). To pass on purpose on this " +
        "machine, set KARUME_ALLOW_NO_TIMESTAMP_QUERY=1.",
    );
  },
);
