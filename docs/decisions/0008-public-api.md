# 0008 — 公開 API 面と SemVer

- Status: accepted（2026-08-01）
- 部分改訂: ADR [0009](0009-dtype-i32-bool.md) 決定 — 公開 `Tensor` 型を dtype 判別ユニオンへ
  改訂（f32 = Float32Array / i32 = Int32Array / bool = Uint32Array）。
- 根拠資料: recon §9-1（先行実験プロジェクト（以下プロトタイプ）は index.ts が内部 ≈180
  シンボルの素通し再輸出で実装と乖離、テストが src 直 import のため乖離が検出されない
  — JSR + SemVer と両立しない）

## 決定

- 公開面は **`packages/runtime/mod.ts` に明示的に設計した薄い面のみ**。内部モジュールの素通し再輸出は
  禁止。エクスポートは「利用者ストーリーに現れる型と関数」に限る
  （目安: モデルを開く / 実行する / 資源を解放する / 能力を照会する / 診断を得る）。
- **公開面経由のみで書いた統合テストを常設**し、面の乖離・破壊を機械検出する
  （内部直 import のテストと分離する）。
- サブ面が必要になったら `deno.json` の exports マップで追加する
  （例: `./format`）。安易に増やさない。
- SemVer を守る。公開面の変更はエクスポート差分をレビューで明示してから行う。
  `deno publish --dry-run` を CI に含め、JSR 公開可能状態（型の明示性含む）を常時保つ。
- 公開 API には明示的な戻り値型を付ける（JSR の slow types 回避 + グローバル TS 規約）。

## 帰結

- プロトタイプからの移植時、「何を公開するか」は移植対象の選別基準になる（全部持ってこない）。
- v0.x の間は breaking 可だが、その場合も CHANGELOG で明示する。

## 追記

- 2026-08-05: `parseSafetensors` / `tensorBytes`（+ 型 3 つ）を公開面へ追加。IR コンテナでは
  ない付帯資産（Anima の rope 素表 — ADR 0038 §2 の extras）を models 側が読むための面で、
  models の最小 safetensors 再実装（約 80 行）を削除して 1 実装へ集約した（二重実装の解消）。
  汎用ローダの提供が目的ではない — 厳格検査（被覆・整列・dtype）ごと共有するのが趣旨。
- 2026-09-03: **リポ内の道具（`tools/`）は `packages/runtime/src/` を直接 import してよい**。
  薄い面（`mod.ts`）の制約は「JSR で配る面と実装の乖離を作らない」ためのもので、道具は配布物
  でも消費者でもないので射程の外（`tools/fusion-hints` は `runtime/fusion.ts` の非公開述語を
  唯一の判定点として使う — ADR 0040 決定 1）。逆に道具のために内部 API を公開面へ出すのは、
  面を利用者ストーリー以外の理由で太らせるので禁止（この決定の本文どおり）。
  併せて **テストの置き場は検証対象と同じ層** とする — 道具の面（資産の発見・束縛・集計）を
  見るテストは `tools/<道具>/*_test.ts` に置く。root の `deno test -A` は `tools/` 配下の
  `*_test.ts` も収集する（root `deno.json` の `exclude` は `inputs/ outputs/ models/` のみ・
  2026-09-03 のフル verify ログで実測）。`packages/*/tests/` から `tools/` を import する形
  （層の逆流）は採らない。
