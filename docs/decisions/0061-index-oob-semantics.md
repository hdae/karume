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
