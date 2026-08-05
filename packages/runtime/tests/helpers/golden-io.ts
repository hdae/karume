/**
 * golden の `io.safetensors` を公開 `Tensor` として読むための共有ヘルパ。
 *
 * tiny golden（tests/e2e_golden_test.ts）と実重み DeBERTa（tests/e2e_deberta_test.ts）の
 * 2 経路が同じ格納規約を読む。**規約は 1 箇所にしか置かない** — 2 箇所に写すと、片方だけ
 * 直したときに「片方の経路では通り、もう片方ではビット列を再解釈して通ってしまう」形で
 * 静かに食い違う。命名規約の正本は tools/exporter/README.md「golden レイアウト」。
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
 * 意味論 dtype → golden io の格納 dtype。エクスポータは境界で i64 → I32・bool → U32
 * （u32 の 0/1）へ正規化して書く（ADR 0009 / tools/exporter README）。
 */
export const IO_ENCODING: Readonly<Record<IrDtype, SafetensorsDtype>> = {
  f32: "F32",
  i32: "I32",
  bool: "U32",
};

/**
 * io.safetensors のテンソルを宣言 dtype の view で取る（コピーしない）。
 * MUST: 格納 dtype が宣言と食い違ったら落とす — 要素は全型 4 バイトなので、黙って
 * 読み替えるとビット列の再解釈が「通ってしまう」。
 */
export const ioTensor = (
  file: SafetensorsFile,
  view: TensorView,
  dtype: IrDtype,
): Tensor => {
  assertEquals(view.dtype, IO_ENCODING[dtype], `golden テンソル '${view.name}' の dtype`);
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
