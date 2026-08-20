# 0072: SBV2 テキスト層の外部注入席 — overlay 辞書と given_tone

- Status: accepted（2026-08-20 — ユーザー裁定「両方でお願いします」。利用実装側の
  フィードバックで細部を再調整する前提の初版）
- Date: 2026-08-20
- 関連: ADR [0008](0008-public-api.md)（薄い公開面 — 本 ADR が 1 メソッド + 1 型を
  追加）/ [0039](0039-sbv2-distribution.md)（決定 6 — 辞書取得の暫定形・yomi 依存の緊張）
- 実装: `packages/models/src/sbv2/`（pipeline.ts / text/analyze.ts）

## Context

SBV2 の合成は読み・アクセントを利用者から一切修正できなかった。`@hdae/yomi` の
`analyzeWithWords(dict, text, overlay?)` は修正辞書オーバーレイを受けられるのに、karume は
2 引数で呼んでいて貫通していない。トーン列は g2p 結果から算出され `front` グラフの `tone`
入力にだけ流れる（BERT はトーンを受けない）ため、**長さを保ったトーン差し替えは既存の
整合性検査（`sum(word2ph) === P` / `inputIds.length === word2ph.length`）を壊さない**。
逆に音素列の直接上書きは word2ph の不変条件を壊す。

## Decision

### 1. 修正は 2 軸を別々の正しい席で受ける

- **読み・アクセント型の語彙単位修正 = overlay 辞書**: `Sbv2PipelineOptions.overlay?:
  readonly OverlayEntry[]` と `Sbv2GenerateRequest.overlay?: readonly OverlayEntry[]`。
  request 側があればその 1 回は request 側を**そのまま使う**（合成しない）。検証は yomi 側の
  fail-loudly（surface 正規形・モーラ分割・accentType 範囲）に委ねる。
- **発話単位のアクセント上書き = given_tone**: `Sbv2GenerateRequest.givenTone?:
  readonly number[]`。値域は **0/1 の生値のみ**・長さは解析の phones（add_blank 前・両端
  PAD 込み）と同長 MUST。不一致・値域外は期待/実際を添えて throw。適用はトーン算出後・
  モデル ID 化前の差し替え 1 点。
- **音素列の直接上書きは席にしない**（word2ph 不変条件を壊す）。読みの変更は必ず overlay
  経由で「解析からやり直す」。

### 2. 下書き API `analyzeProsody`

`analyzeProsody(text, { overlay? }) → Promise<Sbv2ProsodyDraft>`、
`Sbv2ProsodyDraft = { phones, tones }`（add_blank 前・PAD 込み・そのまま `givenTone` に
渡せる形）。GPU 不要・決定的。公開する中間表現は**この 2 欄のみ** — word2ph / bertText 等の
内部契約は公開面に固定しない（ADR 0008）。

### 3. yomi 結合は素通しに留める（将来席）

yomi 型は `import type` の素通しで公開し、変換層・抽象は作らない。**将来、SBV2 の入力を
「yomi の解析結果だけ受ける」形にして yomi を models の依存から外す方向をユーザーが表明
（2026-08-20・breaking・時期未定 — backlog later）**。本 ADR の席はその際に overlay が
呼び手側へ移る前提で、結合を最小にしておく。

## Consequences

- 追加は全て optional で非 breaking — 0.4.0（minor）に同乗。
- 「効いたか」を見る口は `analyzeProsody`（+ 既存 dump 経路）が兼ねる。専用の診断欄は
  設けない。
- `givenTone` の要求長は **overlay に依存する**（overlay が読みを変えれば音素数も変わる）—
  下書きと合成は同じ text・同じ overlay で対にする。ずれは長さ検査が拾う（沈黙誤値なし）。
- examples / CLI への入口追加は席のみ（実需が言われたら）。dump 経路（torch 突合）には
  givenTone を入れない — 参照側と食い違うため。
