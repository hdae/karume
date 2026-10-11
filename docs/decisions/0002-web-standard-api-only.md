# 0002 — ランタイム本体の依存は Web 標準 API のみ

- Status: accepted（2026-08-01）
- 根拠資料: recon §0/§1（先行実験プロジェクト（以下プロトタイプ）決定 0003 の
  「Deno 移行可能性の維持」が実際に機能し、`deno check` 残差 7 件まで移植コストを
  縮めた実績の再宣言）

## 決定

ライブラリ本体のランタイム依存は **Web 標準 API のみ**とする。外部パッケージ依存 0。
使用を許すのは `navigator.gpu` / `performance.now` / `TextDecoder` / `ArrayBuffer` 系のみ。
ファイル・ネットワーク I/O は本体に持たず、呼び出し側が `ArrayBuffer`（または
チャンク列）を渡す。

実装規約（プロトタイプの deno-port-recon で移植コストの源と特定された項目の予防）:

- 相対 import は常に `.ts` 拡張子付き。Vite 固有構文（`?raw` / `?url` /
  `import.meta.env` / glob import）は禁止。WGSL はテンプレートリテラルで TS 内に持つ。
- TypedArray は `Float32Array<ArrayBuffer>` のように ArrayBuffer 明示で締める
  （プロトタイプの deno check 残差 6/7 件の根治形）。
- Dawn / wgpu の実装差がある API 面（`wgslLanguageFeatures` 等）へ直接依存しない。
  features / limits / 言語機能は**能力検出層 1 箇所**で正規化し、欠落時は空集合に縮退する。

## 帰結

- Deno / ブラウザ両対応が構造的に成立する（プロトタイプで実証済みの経路）。
- transport（Range 取得・Cache API・並列度制御）はライブラリ外（examples / 補助面）。
  プロトタイプ決定 0011 の分離を継承する。

## 改訂

- 2026-09-25（現況との照合）: 決定の「使用を許すのは `navigator.gpu` / `performance.now` /
  `TextDecoder` / `ArrayBuffer` 系のみ」は**閉じた列挙ではなく例示**として読む。判定基準は
  「Web 標準 API か否か」である。runtime 本体は現在、上の 4 系統に加えて `crypto.subtle.digest`
  （コンテナの block の sha256 照合 — `src/format/container/descriptor.ts`）・`TextEncoder`
  （`src/format/container/header.ts` / `json.ts`）・`queueMicrotask`（`src/gpu/context.ts`）を
  使う。どれも Web 標準で、`Deno.` と `fetch(` は本体に現れない（決定の核は不変）。
- 同日: 帰結の「transport はライブラリ外（examples / 補助面）」は、monorepo 化で
  `@karume/hub` パッケージへ移った（ADR [0037](0037-karume-monorepo.md)）。runtime 本体に
  I/O を持たない点は変わらない。依存規則は「Web 標準 API のみで構成された依存パッケージは可」
  へ拡張された（ADR [0038](0038-manifest-v1.md) §5 の「依存の扱い」）。
- 2026-10-11（`<ArrayBuffer>` 明示の機械検査と例外表）: 実装規約の「TypedArray は
  `Float32Array<ArrayBuffer>` のように ArrayBuffer 明示で締める」を、リポ内の deno lint
  プラグイン `tools/lint/plugin.ts`（`deno.json` の `lint.plugins` に登録 — 依存の追加は無い）の
  規則 `karume/typed-array-buffer` で検査する。走るのは `deno lint`（= `deno task verify` の lint 段）で、
  規則自体の検証は `tools/lint/plugin_test.ts`（`test:core` のレーン）。
  - **判定**: `packages/<pkg>/src/` の下で、型の位置（型注釈・型引数・戻り型・型エイリアス・欄・
    `satisfies`・interface の `extends`・class の `implements`）に書いた TypedArray 12 種の参照が
    型引数を持たないものを報告する。修飾名（`globalThis.Float32Array`）は右端の名前で判定する。値の位置
    （`new Float32Array(n)`・`instanceof`・`typeof` の型クエリ）は構文上型参照ではないので拾わない。
    tests / tools / examples は対象外。
  - **取りこぼす形**: TypedArray の名前を型参照として書かずに既定の `ArrayBufferLike` を作る形
    （`InstanceType<typeof Float32Array>`・`ReturnType<typeof Uint8Array.from>` など）は拾わない。
    判定は構文だけで、型の中身を解かないため。本体には 2026-10-11 時点で 0 件。
    `tools/lint/plugin_test.ts` がこの 2 形を「拾わない」と固定しているので、拾えるようにしたら
    この項とテストを一緒に直す。
  - **理由**: 型引数を省くと TS 5.7 以降は既定の `ArrayBufferLike`（`SharedArrayBuffer` を含む）になり、
    `GPUQueue.writeBuffer` や `CompressionStream` へ渡す値の型が黙って緩む。導入時点で本体に 148 か所
    あった（hub 6・models 139・runtime 3）。
  - **例外の書き方**: 意図して任意のバッファを受ける境界は、lint の ignore ではなく型引数を
    `<ArrayBufferLike>` と**明示**する（暗黙の既定に頼らない）。こうすると例外は
    `rg 'ArrayBufferLike' packages/*/src` で全部列挙でき、無印の書き漏れと見分けがつく。
    例外を足すときは下の表に 1 行足す。

  | 境界                                         | 箇所                                                                                                                                                                                                                                                                                                                                                                                             | 理由                                                                                                                                                                                                                                                                                                                                                                                                         |
  | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | hub の取得の入口                             | `source.ts` の `PinnedSource.readFile` の戻り値・`ManifestReadOptions.parse` の引数 / `fetch.ts` の `isTightView`・`assertTightView`・`decodeManifest` の引数と `fetchAssetsWithBudget` の `bytes`                                                                                                                                                                                               | 外部の取得層（`@hdae/fetch-cache`）は戻り値と検証フックの引数を `Uint8Array<ArrayBufferLike>` で型付けしている。共通層の `assertTightView` が実行時に ArrayBuffer 背面かつ buffer 全体を占めることを検査して `Uint8Array<ArrayBuffer>` へ絞る唯一の点なので、その手前は広い型のまま受ける（`decodeManifest` は `TextDecoder` で読むだけ）。公開の `DirectoryAdapter` は従来どおり `<ArrayBuffer>` を要求する |
  | models の公開 API の入力（CPU で読むだけ）   | `audio/wav.ts` の `encodeWav(samples)`・`decodeWav(bytes)` / `image/png.ts` の `encodePng(rgba)` / `image/preprocess.ts` の `Rgb8Image.data`（birefnet・depth-anything・siglip2・wan の画像入力） / `anima/latents.ts` の `denormalizeLatents(latents, mean, std)` / `anima/preview.ts` の `approximatePreview(latents)` / `vowel-detector/pipeline.ts` の `VowelDetectorPipeline.detect(audio)` | 利用者が渡す値で、本体は添字・`DataView`・`TextDecoder` で読むだけ（GPU や `CompressionStream` へ渡るのは本体が新しく作った配列）。`SharedArrayBuffer` 背面の配列（AudioWorklet のリングバッファなど）や、型引数を持たない他ライブラリの戻り値を写さずに受けるため広い型にする。導入前の公開型（無印 = `ArrayBufferLike`）と同じ型なので、公開面は変わらない                                                 |
  | 上の公開入力をそのまま運ぶ内部の読み取り経路 | `vowel-detector/pipeline.ts` の `detectAudio(state, audio)` / `vowel-detector/features.ts` の `extractFeatures(audio, …)` の `audio`                                                                                                                                                                                                                                                             | `detect` の入力を写さずに特徴抽出まで運ぶ（読むだけ）。ここだけ `<ArrayBuffer>` にすると、型を合わせるためだけの全量の写しが要る                                                                                                                                                                                                                                                                             |
