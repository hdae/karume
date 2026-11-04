/**
 * Shared helper that reads a golden's `io.safetensors` as public `Tensor`s.
 *
 * Two paths read the same storage convention: the tiny goldens (tests/e2e_golden_test.ts) and the
 * real-weight DeBERTa (tests/e2e_deberta_test.ts). **The convention lives in one place only** —
 * with two copies, fixing just one of them makes them drift apart silently: one path passes, and
 * the other passes by reinterpreting the bits. The naming convention's source of truth is
 * tools/exporter/README.md "Golden layout".
 */

import { assertEquals } from "@std/assert";
import type { Tensor } from "../../mod.ts";
import type { IrDtype } from "../../src/format/ir.ts";
import type {
  SafetensorsDtype,
  SafetensorsFile,
  TensorView,
} from "../../src/format/safetensors.ts";

/**
 * Semantic dtype → storage dtype of the golden io. The exporter normalizes at the boundary and
 * writes i64 → I32 and bool → U32 (u32 0/1) (ADR 0009 / tools/exporter README).
 */
export const IO_ENCODING: Readonly<Record<IrDtype, SafetensorsDtype>> = {
  f32: "F32",
  i32: "I32",
  bool: "U32",
};

/**
 * Takes a tensor of io.safetensors as a view of the declared dtype (no copy).
 * MUST: fail when the storage dtype disagrees with the declaration — every element type is 4
 * bytes, so a silent reread would let a reinterpretation of the bits "pass".
 */
export const ioTensor = (file: SafetensorsFile, view: TensorView, dtype: IrDtype): Tensor => {
  assertEquals(view.dtype, IO_ENCODING[dtype], `dtype of golden tensor '${view.name}'`);
  const count = view.byteLength / 4;
  switch (dtype) {
    case "f32":
      return {
        dtype,
        shape: view.shape,
        data: new Float32Array(file.buffer, view.byteOffset, count),
      };
    case "i32":
      return {
        dtype,
        shape: view.shape,
        data: new Int32Array(file.buffer, view.byteOffset, count),
      };
    case "bool":
      return {
        dtype,
        shape: view.shape,
        data: new Uint32Array(file.buffer, view.byteOffset, count),
      };
  }
};
