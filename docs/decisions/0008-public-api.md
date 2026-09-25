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
- 2026-09-05: 上の射程に **`packages/runtime/tests/helpers/`** を含める — 道具は
  `shard-files.ts` の解決規則（Python `karume.shards.resolve_shards` の鏡像）を掴んでよい
  （`tools/_shared/assets.ts`）。この規則は `Deno` API（ファイルシステム走査）に依るため
  `src/` へは昇格できない（横断不変条件「ランタイム依存は Web 標準 API のみ」）。道具側へ
  写すと**同じ規則の 3 実装目**（Python / runtime のテスト補助 / 道具）ができ、ずれても
  どこも赤くならないので、掴む形を追認する（2026-09-05 裁定）。
- 2026-09-07: `parseSafetensorsHeader` / `safetensorsHeaderLength`（+ 型 `SafetensorsHeader`）を公開面へ追加。
  2026-08-05 に出した `parseSafetensors` の**部分適用**（buffer 全量を要求せず、先頭 8 + N バイトとファイル全長から
  同じ検査で同じ表を組む）で、models の PLE 行読み（ADR 0085 追記 2026-09-07）が「ヘッダを 1 度小読みして行の
  位置を持つ」ために要る。検査の実装は 1 本（`parseSafetensors` がこれを呼ぶ）。hub 側は `openAsset` と型
  `AssetRangeReader`（ADR 0086 追記 ⑧）、`DirectoryAdapter.readFileRange?` を追加した。
- 2026-09-20: **公開面のスナップショット門を 3 パッケージに常設する**（`packages/<pkg>/tests/public_surface_test.ts`
  と追跡 fixture `tests/fixtures/public-surface.json`）。それまでの門は
  `packages/models/tests/models_barrel_surface_test.ts` の名指しリストだけで、リストに無い綴りの増減は素通りし、
  hub / runtime には門が無かった。2 つの門は役割が違う —— **名指し = 意図の宣言**（「この綴りは出す / 出さない」
  を人が書く。この決定の「薄い面」を守る側）、**スナップショット = 増減の検出**（何が正しいかは言わず、前回との
  差だけを見る）。名指し門はそのまま残す。面の真実源は各 `deno.json` の `exports`（runtime 1 面 / hub 2 面 /
  models 10 面）で、entry ごとに `deno doc --json` を回して `{ name, kind }` の集合を採る。**型 export も採る**
  ので、`Object.keys` では観測できない `export type` の再輸出もここで縛れる（名指し門が取りこぼす面）。GPU も
  実資産も要らず 3 本で合計 0.9 秒、レーン分割の core レーンに入る。更新手順は失敗メッセージが持つ ——
  `KARUME_SURFACE=write` で同じテストを回して fixture を書き直し、差分を CHANGELOG に明示する（この決定の本文
  「公開面の変更はエクスポート差分をレビューで明示」を、レビュー前に機械が差分を出す形にしたもの）。門の実体は
  `packages/runtime/tests/helpers/public-surface.ts` 1 本。**射程は `deno doc` が出す集合**なので、
  `@ignore` を付けた export は doc の出力に載らず、この門も見ない（消える側は `- 名前` の差分で落ちるが、
  `@ignore` 付きで増える面は素通りする）。名指し門と併せて、意図の宣言は人が書く側が持つ。
- 2026-09-25: **上の 2026-08-05 / 09-05 / 09-07 の追記が理由にした配布形は、コンテナ形式（ADR 0108 / 0109）の
  波で退役した**。決定の生死は 3 件で分かれるので、現況をここに記録する（各追記の本文は当時の記録として残す）。
  - 2026-08-05（`parseSafetensors` / `SafetensorsError`）: **決定は生きていて、理由の文言だけが古い**。
    「ADR 0038 §2 の extras」は ADR 0109 決定 4 で退役し、Anima の rope 素表は容器の資産 `rope_base`（役割
    `rope-base`）になった。今の公開理由は「容器の中身として届く safetensors 形式の payload を、models が同じ
    厳格検査で読む」ことで、消費者は sbv2 のスタイル表（`models/src/sbv2/style.ts`）・vowel-detector
    （`models/src/vowel-detector/pipeline.ts`）・rope 素表（`models/src/anima/rope-base.ts`）である。
    同じ追記で出した `tensorBytes` は repo 内の消費者がテストだけになったので、0.13.0 で公開面から外す
    （2026-09-25 裁定 — 退役した配布形の残骸の整理。既リリースの面なので Breaking）。
  - 2026-09-05（道具が runtime のテスト補助を掴む）: **構造は同じで、固有名が全部古い**。`shard-files.ts` と
    Python `karume.shards.resolve_shards` はこの波で削除された。今は `tools/_shared/assets.ts` が
    `packages/runtime/tests/helpers/container-files.ts` の `resolveParts`（Python
    `karume.container.resolve_sequence` / `container_parts` の鏡像）を掴む。`Deno` API に依るので `src/` へ
    昇格できない点と、3 実装目を作らない理由はそのまま成り立つ。
  - 2026-09-07（`parseSafetensorsHeader` / `safetensorsHeaderLength` / 型 `SafetensorsHeader`）: **公開理由が
    消えた**。PLE は safetensors の shard から `model` 容器の資産へ移り、ヘッダを小読みする呼び手が repo 内に
    無い（残る参照は自身の単体テストだけ）。この 3 つは v0.12.0 より後に足した未リリースの面なので、0.13.0 で
    公開面から外す（2026-09-25 裁定）。`parseSafetensors` の内部では部分適用のまま使う。
