# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

## eval-images の出力名がソースリポを区別せず、siglip2 実画像門の入力を上書きできる

`examples/anima/eval-images.ts` の出力 PNG 名は quant / steps / 解像度 / seed を綴るが
**`--source`（どの配布形から焼いたか）を綴らない**。siglip2 の実画像 e2e は
`outputs/demo/` のこの PNG を名前で読んで golden（`io.photo-*.safetensors` 内の sha256）と
突合するため、**turbo 以外の配布形で eval-images を実行すると同名で上書きされ、門が
10 本赤になる**（2026-08-23 実発生 — 波 L 後は素版 `models/karume-anima` を指しても
コマンドが成立するため踏みやすくなった。turbo 配布形から焼き直して復旧・golden 不変・
パリティ門が同一内容の復元を証明）。

最小の埋め方はファイル名（または置き場）へソース識別子を入れて golden 側の読み口を追随
させること。それまでの運用: **eval-images は turbo 配布形（`models/karume-anima-turbo`）
以外を指さない**。

## Metal（Apple GPU）で attention i8a8 と conv2d の 2 経路一致が崩れる

実機 **Apple M2 / Deno 2.9.4** で `deno test -A packages/runtime/tests/` が 6 本赤になる
（Linux / Vulkan は全緑）。GEMM の共有タイル書き込みを静的成分へ直した後も残ったもので、
**その機序では説明できない**別の Metal 差。調査の全体は
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)。

- **attention i8a8 系 4 本**（`gpu_attention_i8a8_test.ts` / `gpu_attention_pv_i8a8_test.ts`）。
  `attention-i8a8.ts` の共有配列は `array<u32>` のスカラで動的成分書き込みを持たず、しかも
  **同じ `dot4I8Packed` を使う linear i8a8（`gpu_i8a8_test.ts`）は全通過**する。それでも
  `dot4I8Packed 版とエミュ版が atol=0 で一致する` が落ちるので、整数演算に丸め差が無い以上
  QK / PV では実際に違う値が出ている。
- **conv2d parity 2 本**（`gpu_conv2d_parity_test.ts`）。implicit GEMM ↔ 直接カーネルの
  ビット一致。conv2d の B タイルは `sb[bk * 16u + bcq] = bv4` と vec4 を丸ごと書く形で、
  上記の修正対象ではない。ただし `conv2d_block` golden（atol 1e-6 / rtol 1e-5）は通るので、
  値そのものは概ね正しく**ビット一致だけが崩れている**。

実運用への影響は未確定（この状態でも Mac で正常な画像が生成できている）。ブラウザ実行は
Dawn / Tint 系で naga を通らないため、同じ症状が出るとは限らない（未検証）。

## EmbeddingGemma の batch>1 export が変換段で通らない

`karume export-embeddinggemma --batch N`（N>1）は `karume/convert.py` で fail loudly する
（B=1 は従来どおり成功）。機序は 2 段:

1. transformers（5.14 系）の `masking_utils.find_packed_sequence_indices` が、
   `Gemma3TextModel` 内部の `position_ids [1,T]` と `batch_size=N` の不一致で trace 中に
   packed-sequence 分岐へ入り、`aten.eq.Tensor` / `aten.index.Tensor` / `aten.ne.Scalar` が
   IR まで生き残る（B=1 ではこの不一致が起きず、既存の Tmax 定数 + `sym_prefix_slice`
   畳み込みに吸収される）。
2. この分岐を monkeypatch で外すと、今度は帯マスクの `aten.bitwise_or.Tensor` が
   **bool 定数を IR v1 の initializer にできない**制約（f32 / i32 のみ）に当たる。しかも
   同 patch は **B=1 でも同じエラーを誘発する** — packed-sequence 分岐の存在自体が現行の
   帯マスク定数畳み込みパターン成立の前提になっており、eager 同値のつもりの patch でも
   安全ではない（実測 A/B・2026-08-11）。

根治は convert/normalize 側の一般化（bool 定数の f32/i32 化 or initializer dtype の拡張 +
batch>1 のマスク畳み込み対応）で、コア変換基盤への設計判断が要る。`--batch` フラグ自体は
一般化が入ればそのまま使える形で維持している。

## 融合 attention ② の regcache 変種（epc ≥ 2）に実 GPU 門が無い

`attention_stats` の regcache 変種は `dim > 256`（epc ≥ 2）で初めて 2 スロット以上の静的展開に
なるが、現行の GPU テストで融合 attention を N > 256 で踏むものが 1 本も無い
（`gpu_attention_parity_test.ts` の SHAPES は最大 N=64 → epc は常に 1）。
スナップショットは `attention_stats_rc16.wgsl`（dim 4096 相当）を凍結しているので生成物の
固定はあるが、**実 GPU での値の一致は epc=1 の形でしか確認されていない**。

2026-08-16 に `attentionStatsParams` へ `dim ≤ regCache · 256` の門を入れた際、故障注入で
この空白が判明した（`SHAPES` に N=512 を一時追加してはじめて門が発火した）。
parity テストへ N > 256 の形を 1 本足すのが最小の埋め方。

2026-08-19 追記: 波 H の `e2e_gemma4_test.ts`（`context-en`・T=598）が epc=3
（ceil(598/256)）の regcache 変種を実 GPU で踏み、tolerance 判定（atol 1e-2・実測
maxAbs 2.23e-3）を通過している（同日再実行で確認）。「N > 256 を踏むテストが 1 本も無い」
状態は解消したが、これは**弱い数値検証**であり、ビット単位の恒久 parity 門は依然無い —
最小の埋め方（parity テストへ N > 256 を 1 本）は変わらず有効。
