# 0061: gather / embedding の範囲外添字は quiet NaN 汚染で表面化させる

- Status: accepted（2026-08-15 — 既存裁定の正本化。挙動は 2026-08-02〜03 の実装時から現行で、
  これまで正本が `docs/limitations.md` とカーネル doc に分散していた）
- 関連: ADR [0020](0020-nan-propagation-bitwise.md)（NaN 伝播 — 「gather / embedding の範囲外
  NaN 汚染は別裁定」と明示的に切り離した元）/ [0010](0010-symbolic-constant-folding.md)
  （実運用の添字が export 時 clamp 済み定数由来である根拠）/ 実装内根拠 =
  `packages/runtime/src/kernels/gather.ts` の doc コメント

## Context

- 契約は「添字は範囲内」。しかし違反したときに WebGPU の境界付きアクセス
  （bounds-checked access）は「0 または別の正常値」を**静かに**返すため、無検査だと違反の
  痕跡が結果から消える — fail loudly の不変条件（CLAUDE.md）と正面衝突する。
- カーネルからホストへの例外化は「run 単位のフォールト旗 + readback」という新しい診断
  チャネルの新設が要り、全 op の dispatch 経路に恒常コストを足す。

## Decision

- **GPU カーネルは範囲外添字の該当要素（embedding は該当行）にだけ quiet NaN を書き、実行は
  継続する**。NaN 伝播（ADR 0020）で必ず出力まで表面化する — 例外にはならないが黙りもしない。
- **CPU 参照実装は範囲外で throw する**（意図的な非対称 — 参照側は診断の場で、実行継続の
  要求が無い）。
- **「フォールト旗 + readback」の診断チャネルは導入しない**。必要になった時点で独立に設計する
  （範囲外が実運用で出る経路は現状無い: 添字は export 時に clamp 済みの定数由来 — ADR 0010。
  違反はモデル側の誤りに限られる）。

## Consequences

- 範囲外の症状は「例外」ではなく「出力 NaN」— デバッグ時はまず添字テンソルを疑う。
- LLM 波で動的添字（KV 位置・ルーティング等）が入るときは、この契約のまま成立するかを
  当該 ADR で再確認する（実行時添字が「モデル側の誤りに限られる」前提が変わるため）。
- `docs/limitations.md` の該当節は本 ADR を指す要約になる。

## 追記

- 2026-09-25（Consequences の「動的添字が入るときの再確認」の記録）: LLM 波（ADR
  [0066](0066-generation-context-state-slots.md) / [0083](0083-generation-api-surface.md)）で、利用者の実行時入力
  （prompt の token id）が embedding の添字になった。「違反はモデル側の誤りに限られる」前提は
  ここで崩れたが、**GPU 側の契約（範囲外 = 該当行の quiet NaN・フォールト旗は無し）は変えない**。
  代わりに**利用者由来の添字は models の入口（ホスト境界）で `0..vocabSize−1` の範囲検査を
  MUST とし、範囲外は `ModelInputError`（ADR [0107](0107-model-input-error.md)）で落とす** —
  実体は `packages/models/src/generation/sequence.ts` の prompt 検査と `stopTokens` 検査。
  内部で生成した添字（speculation の draft 列 `generation/speculation.ts`・greedy の選択結果）は
  同じ範囲検査を素の `Error` で掛ける（利用者の入力ではなく実装の不変条件の破れなので
  `ModelInputError` にしない）。Gemma 4 の PLE（`gemma/ple.ts` の `gather`）はホスト側で索引の
  行数に対して同じ検査を掛ける（範囲外は OOB ではなく別 token の有効な行になるので、ここが
  唯一の fail loudly の位置）。**以後、新しい動的添字（MoE ルーティング等）を入れる ADR は同じ
  ホスト境界の範囲検査を MUST とし、GPU の NaN 汚染を利用者入力の検出手段にしない**。
