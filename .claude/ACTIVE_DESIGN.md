# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-24

## Now

- **次 = 0.5.0 breaking 波の準備（2026-08-24 ユーザー裁定で着手）**: 破壊的変更を 1 回に
  まとめる束 — ①quant 席名の規則化（ADR
  [0074](../docs/decisions/0074-quant-seat-naming.md) — 格納語彙を `f32/f16/i8/i4` の 1 本へ・
  移行表は ADR が正本〈irodori `w4` → `i8+dit4` の行を 2026-08-24 に追記済み〉）
  ②`linearCompute` / `attentionCompute` の値 `"i8a8"` → `"a8"` 改名 ③表示欄 + `karume/4`
  繰り上げ（ADR [0075](../docs/decisions/0075-quant-presentation.md)）④SBV2 の yomi 依存分離。
  公開リポの再アップロード + pin 更新（ADR 0073）を 1 度で済ませる。骨子 =
  [backlog](../docs/backlog.md) later 節。**現行の席名・ノブ名は改名予定**。
- **波 J（量子化探索・第 2 段）は J-4 で全クローズ（2026-08-24）**: irodori `w4` 席
  （DiT のみ i4 混成の GPTQ・品質 3 ラウンド → 聴感裁定「配布可」→ **HF 公開済み
  `67e9584c`・pin 据え置き** = `w4` は `revision: "main"` 明示が要る）+ anima 素版 3 モデルの
  校正条件モデル別化 + i4 export（**視認裁定で配布スキップ** — 系列は
  `outputs/series-archive/2026-08-23-anima-base-i4/` へ退避・改善候補は backlog later）。
  設計の正本 = ADR [0050](../docs/decisions/0050-irodori-quant-series.md) 追記 2・実測 =
  [research/2026-08-24-gptq-expansion-quality.md](../docs/research/2026-08-24-gptq-expansion-quality.md)。
  **教訓 2 件**: ①sim の A/B は同一リグ内でのみ有効 — **出荷リグでは GPTQ の丸め解が変わり
  発話実現が再抽選される**（繊細な性質は転移しない・最終裁定は必ず出荷バイトで）②adaLN
  （modulation の scale/shift/gate）は量子化感度が高い（irodori 実測 — 他 DiT へは未実測の仮説）。
- **波 L（anima 素版 + バリアント同梱）もクローズ**: L-1〜L-4 消化（`hdae/karume-anima`
  公開・pin 焼き込み・0.4.2 lockstep）+ i4 席の保留は J-4 ②で解消（配布はスキップ）。
  残置 = サンプラー Euler 固定（backlog）。
- **0.4.2 まで JSR リリース済み**（2026-08-24 ユーザー確認。PyPI `karume` は未リリース）。
  直近クローズ済みの波の履歴は [backlog](../docs/backlog.md) と各 ADR / research が正本
  （autoregressive A〜H・全体レビュー・波 I / J / K / L・SBV2 注入席）。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。

## Pitfalls（現役のみ）

- **local の `models/karume-anima/` には配布スキップ裁定（2026-08-24）の `w4` / `w4-a8-s16`
  席が組み込まれたまま**（i4 は dist の宣言必須格納で外せない）— karume-anima を上げ直す時は
  この裁定を先に思い出す。
- **eval-images は turbo 配布形以外を指さない**（出力名がソースリポを区別せず siglip2
  実画像門の入力を上書きする — known-issues）。
- **`linearCompute: "i8a8"` は i8 常駐と i4 常駐で数値契約が別**（i8 = full-k 厳密 / i4 = group
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
