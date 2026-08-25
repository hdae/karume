# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-25

## Now

- **0.5.1 リリース段（サンプラー再裁定 — ADR
  [0078](../docs/decisions/0078-anima-sampler-selection.md)）**: anima の既定サンプラーは
  **Euler 維持**へ再裁定（0.5.0 の dpmpp-2m 宣言は同日戻し）。HF は上げ直し済み（anima
  `2682441a` / turbo `88357344` — main 追従の利用者は復旧済み）。0.5.1 = `sampler` 席 +
  pin 更新の 2 点で、残りは push → CI → Release v0.5.1 → JSR。0.5.0 自体は同日クローズ済み
  （リリース記録 = [backlog](../docs/backlog.md) 消化済み節）。**0.5.1 後の次波はユーザー
  裁定待ち**（候補 = backlog now 節）。
- **0.5.0 で変わった面**（消費側の doc はここが索引）:
  - **quant 席名が全て改名された**（ADR [0074](../docs/decisions/0074-quant-seat-naming.md)
    決定 6 の移行表が正本 — 例 `w8a8-s16` → `f16+dit8-a8-attn8-s16` / sbv2 `w8-bert4` →
    `i8+bert4` / irodori `w4` → `i8+dit4`）。`linearCompute` / `attentionCompute` の**値**も
    `"i8a8"` → `"a8"`（カーネル内部識別子・ファイル名・WGSL は実行変種の名前なので不変）。
  - **manifest は `karume/4`**（ADR [0075](../docs/decisions/0075-quant-presentation.md)）—
    quant の `label` / `description`、`requiredLimits`、ファイル参照の越境 `repo` / `revision`
    （ADR [0038](../docs/decisions/0038-manifest-v1.md) 追記 2026-08-25）。旧 format は読めない。
  - **`fromPretrained` の `ref` は必須**（既定ソースの廃止 — ADR
    [0073](../docs/decisions/0073-models-source-pin.md) 追記 2026-08-25）。pin 定数は
    `<FAMILY>[_<VARIANT>]_CURRENT` の 4 本で、位置づけは「**パッケージ版に合わせて自動追従したい
    場合のオプトイン**」= bump のたびに pin 更新 + 動作確認の義務つき。hub は revision 未指定の
    暗黙 `main` 解決に 1 回だけ warn（解決 SHA 印字 + pin / `*_CURRENT` の 2 択案内）。
  - anima に `pipelineConfig.scheduler.type` 席（`euler` / `dpmpp-2m`・省略時 euler）。
    **配布既定は euler**（再裁定 2026-08-25 — ADR 0078。0.5.0 期の公開 revision
    `ebb27bc4` / `6215f965` だけが dpmpp-2m 宣言）。DPM++ 2M は 0.5.1 の
    `AnimaGenerateRequest.sampler` で選ぶ。
  - 0.4.3 で配られた面（消費側 doc の注意点）: `animaLatents()`（途中 latent の逆正規化素材
    — プレビューには要らない）/ `approximatePreview()`（途中 latent → RGB の線形近似。係数は
    **正規化空間**で較正済みなので `copyLatents()` の返り値をそのまま渡す — 逆正規化した値を
    渡すと白飛びする）/ `AssetProgress` の per-file 欄 `fileLoaded` / `fileTotal`（必須欄・
    `verifying` / `complete` では常に等しい）。
- **anima の受理解像度を 8 通り縮小した（E-2）**: VAE タイル本数の上限を入口の受理集合へ足し、
  1456/1488/1584/1648/1680/1776/1840/1936px を名指しで拒否するようにした。**形式上は破壊的だが、
  対象はホスト RAM 破裂で実行不能だった値のみ**。省 RAM の逐次組み立てで受理へ戻す案は
  backlog later。
- **波 J / 波 L はどちらも 2026-08-24 に全クローズ**（波順の正本 =
  [backlog](../docs/backlog.md)）。**教訓 2 件**は現役: ①sim の A/B は同一リグ内でのみ有効 —
  **出荷リグでは GPTQ の丸め解が変わり発話実現が再抽選される**（繊細な性質は転移しない・
  最終裁定は必ず出荷バイトで）②adaLN（modulation の scale/shift/gate）は量子化感度が高い
  （irodori 実測 — 他 DiT へは未実測の仮説）。実測の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](../docs/research/2026-08-24-gptq-expansion-quality.md)。
- PyPI `karume` は未リリース。クローズ済みの波の履歴は
  [backlog](../docs/backlog.md) と各 ADR / research が正本。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。

## Pitfalls（現役のみ）

- **フル走行の verify は VRAM 圧で稀にフレークする**（毎回別のテストが落ち、単独再走は常に緑
  — known-issues）。落ちたファイルの単独再走で切り分ける。
- **eval-images は turbo 配布形以外を指さない**（出力名がソースリポを区別せず siglip2
  実画像門の入力を上書きする — known-issues）。
- **`linearCompute: "a8"` は i8 常駐と i4 常駐で数値契約が別**（i8 = full-k 厳密 / i4 = group
  部分縮約 — ADR [0076](../docs/decisions/0076-w4a8-linear-execution.md)）。取り違えると atol=0 の
  主張が意味を失う。経路の識別はパイプラインキーの `:wi4g32` サフィックスと診断が担う。
- **Metal**: threadgroup `vec4` への動的インデックス書きは黙って捨てられる（`gemm.ts` の
  `storeBTransposed` の switch 展開を新しい箇所で崩さない）。attention i8a8 / conv2d の
  Metal 数値差は known-issues・Metal は gpuTiming 不可（limitations）。
- **融合 matcher は実測形 exact-match** — exporter の発行順・形が変わると黙って外れ、値は
  正しいまま性能だけ落ちる。観測 = `Diagnostics.lastRunFusions` +
  `assets_fusion_counts_test.ts`。**row-block だけは外れ方が性能でなく資源** — 128MiB 級
  device で resource-limit failure に戻る（**分解経路の matcher だけの話** — 保存 attention の
  states 形は行ブロックを op 内蔵で持つ・ADR 0067 決定 7・波 D 済）。
- **RoPE / SiLU 融合の丸め障壁（workgroup memory 往復）は実測依存** — バックエンド更新で
  PNG 門が割れたらまずここを疑う。
- **`deno task verify` はリポ内に worktree を置くと worktree 側まで test を拾う** — worktree は
  リポ外に作る（CLAUDE.md 検証コマンド節。deno.json に exclude は設けない — 2026-08-16 裁定）。
- **Session 構築の重みアップロード後 submit 1 回は瞬間ピーク +2.7GiB を抑えている** — 消さない。
- **資産の置き場**: `models/` = HF へそのまま上げる配布形のみ・系列出力は `outputs/series/`・
  入力素材は `inputs/<ファミリ>/<名前>/` — 綴りの正本は
  `tools/export-recipes/_shared/paths.py` と [assets-layout](../docs/assets-layout.md)。
  格納 dtype はヘッダが正（dist の門が検査）。旧識別子以前の資産は開けない（互換シム無し）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提。JSR npm 互換層の
  `sideEffects: false` 出力は未検証（backlog release）。
