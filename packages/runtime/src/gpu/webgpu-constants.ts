/**
 * WebGPU のビットフラグ定数（`GPUBufferUsage` / `GPUMapMode`）の自前定義。
 *
 * MUST: プラットフォームの名前空間（裸の `GPUBufferUsage` / `GPUMapMode`）を**読まない**。
 * これらは WebGPU 非対応環境に存在しない**グローバル束縛**なので、モジュールスコープで読むと
 * `import` しただけで `ReferenceError` になる（`navigator.gpu` のような property 参照と違い、
 * 未定義の識別子は undefined にならない）。利用者は「WebGPU 非対応です」の案内を出すために
 * まず import するのだから、import が落ちる時点で案内の手前で白画面になる。全モジュール
 * 副作用ゼロ（import 時実行の禁止）の隣にある、同じ性格の不変条件。
 *
 * MUST: runtime 内の usage / mapAsync のフラグは**全てこのモジュール経由**にする。関数スコープ
 * なら裸参照でも落ちないが、混在させると「モジュールスコープの裸参照だけが違反」という機械
 * 検出できない規律になる。統一しておけば `rg 'GPUBufferUsage|GPUMapMode' src/` の生き残りが
 * このファイル（とそのテスト）だけになり、違反が grep 1 本で見つかる。
 *
 * 値は WebGPU 仕様（https://www.w3.org/TR/webgpu/）で凍結されたビット値の転記。転記ミスは
 * 沈黙誤動作（usage 不足はバッファ作成が validation エラーで落ち、余剰は静かに通る）になる
 * ため、名前空間がある環境では tests/webgpu_constants_test.ts が全ビットを実物と突合する。
 */

/** `GPUBufferUsage` のビット値（仕様 §buffer-usage）。 */
export const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

/** `GPUMapMode` のビット値（仕様 §buffer-mapping）。 */
export const MAP_MODE = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;
