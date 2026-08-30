# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

## フル走行の `deno task verify` が GPU VRAM 圧で稀にフレークする

12GiB の GPU に GB 級モデルを連続投入するため、**フル走行では稀に `GpuOutOfMemoryError` /
`GpuDeviceLostError` でどれか 1 本が落ちる**。落ちるテストは毎回違い（特定の 1 本に固有の
欠陥ではない）、**失敗したファイルを単独で再走すると常に緑**になる。2026-08-25 に
verify を並行させた走りで 2 回観測（単独走行でも過去に観測あり）。

運用の回避 = **失敗したファイルを単独で再走して確認する**（緑ならフレーク）。

## Metal（Apple GPU）で attention i8a8 の GPU 出力が TS 参照と 1 ULP ずれる（+ conv2d parity 2 本）

実機 **Apple M2**（初出 Deno 2.9.4・2026-08-29 に 2.9.6 で再検証）で attention i8a8 系 4 本 +
conv2d parity 2 本が赤（Linux / Vulkan は全緑）。**2026-08-29 のカナリア実機検証で機序の理解が
更新された**:

- **dp4a とエミュの両変種は M2 でもビット同一**（カナリア両腕が同値・PV の相互一致テスト緑・
  qP 整数段 62,088 要素で不一致 0%）。旧記述「整数演算に丸め差が無い以上 QK / PV では実際に
  違う値が出ている」（変種間不一致という推論）は**誤りだった** — 落ちていた比較は
  **GPU vs TS 参照**で、旧観測も同じ形だった可能性が高い（撤回 2026-08-29）。
- 実態は**両変種が共有する f32 エピローグ（scale 適用）の出力が TS 参照とちょうど 1 ULP
  ずれる**（QK 28.4 近傍で 1.9e-6・12.0 近傍で 9.5e-7・PV 0.2 近傍で 1.5e-8）。整数段は厳密
  一致。仮説（未確定）: naga → MSL の FMA 契約差で乗算連鎖の丸めが変わる。
- 実害は **atol=0 のクロスプラットフォーム parity 門が Metal で立たない**ことのみ。品質影響は
  1 ULP で無視できる（Mac で正常な画像が生成できている実績どおり）。ブラウザ実行は
  Dawn / Tint 系で naga を通らないため同じ症状とは限らない（未検証）。
- 変種選択は実走カナリア（ADR [0058](decisions/0058-numerics-opt-in-contract.md) 追記）の
  **判定則 v2** が扱う — 「両腕ビット同一・参照とは帯内（rtol 1e-5）の差」は dp4a を選び
  警告 1 回で実行継続（a8 は動く）。帯外だけが `GpuFeatureError`。
- **カナリア①QK の固定入力は f16 格子へ載せ直した**（2026-08-30・RC1-1）: 倍率を 2 の冪に閉じ
  `acc` の有効桁を f16 仮数に収めることで、既知解 8,192 要素が全て f16 ちょうどになり、
  エピローグに**丸めが 1 度も起きない**。よって①QK は健全な device でも M2 でも既知解と厳密
  一致する見込みで、**この 1 ULP 差が今後観測されるのは③PV 側だけ**になる（③PV は
  `qP = round(127·exp(S−m))` を GPU が作るため丸めが残る）。**M2 実機での再確認は未了** —
  実機の判定分岐が「両腕帯内かつビット同一」から「dp4a 厳密一致」へ移るかは、次に M2 で
  `deno test packages/runtime/tests/gpu_attention_dp4a_canary_test.ts` を回して確かめる。
- **conv2d parity 2 本**（implicit GEMM ↔ 直接カーネルのビット一致・golden の tolerance 判定は
  緑）は従来どおり原因未特定 — 同種のエピローグ丸め差の可能性が高いが未検証。

Deno 2.9.5 / 2.9.6 に Metal / naga / wgpu の更新は無い（denoland/deno#36257 = mapped range の
み）。根治候補 = TS 参照の FMA 許容化 or WGSL 側で丸めを固定する手段の調査（未着手）。記録 =
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)
（時点）と
[research/2026-08-29-chatgpt-review-verification.md](research/2026-08-29-chatgpt-review-verification.md)
（M2 再検証）。

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
