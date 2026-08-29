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

**同型が `examples/irodori/eval-audio.ts` にもある**（出力 WAV 名 `vowel-<name>[-48k].wav` が
`--source` を綴らない一方、`--source` は任意の Irodori 配布形を受ける）。こちらは母音検出の
chain e2e が**実行前の sha256 検査**で全件赤になるため沈黙誤値ではなく、復旧も canonical
source で焼き直すだけ — 埋め方は eval-images と同じ。

## フル走行の `deno task verify` が GPU VRAM 圧で稀にフレークする

12GiB の GPU に GB 級モデルを連続投入するため、**フル走行では稀に `GpuOutOfMemoryError` /
`GpuDeviceLostError` でどれか 1 本が落ちる**。落ちるテストは毎回違い（特定の 1 本に固有の
欠陥ではない）、**失敗したファイルを単独で再走すると常に緑**になる。2026-08-25 に
verify を並行させた走りで 2 回観測（単独走行でも過去に観測あり）。

運用の回避 = **失敗したファイルを単独で再走して確認する**（緑ならフレーク）。

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

再検証は **Deno 2.9.6**（現行ピン）で行うが期待値は低い: 2.9.5 のリリースノートで WebGPU は
mapped range の修正 1 件のみ（denoland/deno#36257）・2.9.6 は WebGPU 関連の項目ゼロで、
Metal / naga / wgpu の更新の形跡が無い。

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

## Pixel（8GB 級 Android Chrome）で anima turbo i4 のロードが失敗する — 真因未特定

実機報告のエラー文言 "BodyStreamBuffer was aborted" は Chrome が巻き添え中断の reason を
差し替えた固定文言で、真因ではない（hub が巻き添え側を表面化させていた診断バグは
2026-08-25 に修正済み — 真因復元 + バイト予算 + 検証直列化。未リリース）。最有力仮説は
メモリ逼迫（turbo i4 でも完走時常駐 ~2.56GiB + 検証一時。i8 ではブラウザ強制終了の報告
あり）だが、回線切断・アプリ側 abort と見え方が同一のため、修正版で `err.cause` を実機
観測するまで確定できない。常駐そのものの削減は shard 配布 + streamAssets 接続
（backlog next の R1）まで残る。

## HF: base リポの shared/text_encoder が中程度の断片化（4.1 MiB/term — 2026-08-29）

`hdae/karume-anima` の `shared/text_encoder/model.safetensors` の Xet reconstruction が
259〜280 terms（4.1〜4.6 MiB/term・目安は ≥10）。原因は分割 shard 時代のアップロードで
リポ履歴に入った chunk 群への部分 dedup（継承断片化）。runbook §2 の対策 env は
**hf_xet 1.4.3 で退行**しており（`HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED` が消滅・
後継 `HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES` も本件の repo 内 dedup には無効）、
delete→再 up の 2 コミット法でも治らないことを実測済み。実害 = 当該 1 ファイル（1.1GiB）の
DL が数倍遅い。恒久対処候補 = hf_xet の版固定での再検証 / `HF_HUB_DISABLE_XET=1`（素 LFS）
での上げ直し検証 / 履歴整理（越境 pin を巻き込むため turbo と同時に計画）。
