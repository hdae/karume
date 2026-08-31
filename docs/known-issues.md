# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

## フル走行の `deno task verify` が GPU VRAM 圧で稀にフレークする

12GiB の GPU に GB 級モデルを連続投入するため、**フル走行では稀に `GpuOutOfMemoryError` /
`GpuDeviceLostError` でどれか 1 本が落ちる**。落ちるテストは毎回違い（特定の 1 本に固有の
欠陥ではない）、**失敗したファイルを単独で再走すると常に緑**になる。2026-08-25 に
verify を並行させた走りで 2 回観測（単独走行でも過去に観測あり）。

運用の回避 = **失敗したファイルを単独で再走して確認する**（緑ならフレーク）。

## Metal（Apple GPU）で attention i8a8 の GPU 出力が TS 参照と 1 ULP ずれる（+ conv1d/conv2d parity 4 本 + gru_scan parity 2 本）

実機 **Apple M2**（初出 Deno 2.9.4・2026-08-29 に 2.9.6 で再検証・2026-08-31 フル verify で
節の対象を棚卸し）で attention i8a8 系 4 本 + **conv1d parity 2 本 + conv2d parity 2 本**
（旧記述は conv2d のみ — conv1d の 2 本が記載から漏れていた・症状は同型）+
**gru_scan / gru_scan_reverse の分解 parity 2 本**（下の節）が赤（Linux / Vulkan は全緑）。
**2026-08-29 のカナリア実機検証で機序の理解が更新された**:

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
  `qP = round(127·exp(S−m))` を GPU が作るため丸めが残る）。**M2 実機で再確認済み
  （2026-09-01）**: カナリア 16 本すべて緑 — 素の判定が dp4a を選び分岐と厳密一致フラグが
  整合・故障注入系も想定どおり。軸 reduce パリティ（`gpu_reduce_axis_parity_test.ts` 2 本）も
  M2 緑。
- **conv1d / conv2d parity 4 本**（implicit GEMM ↔ 直接カーネルのビット一致・golden の
  tolerance 判定は緑）は従来どおり原因未特定 — 同種のエピローグ丸め差の可能性が高いが未検証。
- **gru_scan / gru_scan_reverse の分解 parity 2 本が M2 で赤（2026-08-31 実測）**:
  `gpu_gru_scan_parity_test.ts` の T4 N1 H128 で 158/512・71/512 要素が 1〜64 ULP 不一致
  （NaN なし・小さい H のケースは緑・Linux / Vulkan は全緑）。**tanh_stable 化（gru_scan v2）
  の regression ではない** — 変更前 HEAD `b35cf5c` で同一署名（同ケース・同件数・同ビット列）の
  赤を実測 = 既存かつ v2 が Metal でも in-band ビット不変であることの実証。機序の見立て =
  本節冒頭と同一クラス: fused 側の縮約 / 更新式を MSL の contraction が跨ぎ、workgroup memory
  往復の**丸め障壁が Metal では障壁として機能していない**（src/kernels/gru-scan.ts の doc が
  予告していた形 — 「丸め障壁は WGSL 仕様の保証ではない」）。H=128 = 縮約長が伸びると顕在化。
  実害は他と同じく「クロス経路の atol=0 門が Metal で立たない」ことのみ — 実品質は
  vowel-detector golden（tolerance 門）が M2 緑で担保。根治候補 = 丸め障壁の式形強化
  （bitcast 往復は無効と実測済みなので別手段 — 下の根治候補と同席・M2 実機ループ要・未着手）。
- **linear の GEMV 族（M=1 × i4 — ADR [0082](decisions/0082-linear-gemv-decode.md)）の u32
  完全一致門が M2 で 1 ULP 赤（2026-09-01 実測）**: `gpu_linear_gemv_test.ts` の門キー検査は
  緑・既定経路との u32 突合が整除形 k128 n64 g32 の列 1 で `0x414b3249` vs `0x414b3248`
  （12.699776… vs 12.699775… = 1 ULP）。Linux / Vulkan は緑。見立て = 本節冒頭と同一クラス
  （naga → MSL の FMA 契約差で乗算連鎖の丸めが変わる — 未確定）。**撤回（2026-09-01）**: 旧記述
  「chat e2e が M2 で golden 完走 = 動作・品質は健全」は**実走の裏付けが無いまま書かれていた**
  （M2 で実測されたのはカナリア / reduce parity / skinny / gemv 門のみ — ユーザー指摘で発覚）。
  当時 chat e2e は M2 で gemma4 prefill NaN（gemv とは独立・`gelu_tanh` の Metal fast-math
  オーバーフロー — `tanh_stable` で解消済み・対話実走は M2 確認済み）により走り切れなかった。
  margin 門が 1 ULP を吸収するという命題は **M2 での温度 0 golden e2e 実測が未取得のまま**
  （NaN 解消で実測可能になった — 立て直しは ADR
  [0082](decisions/0082-linear-gemv-decode.md) 追記 2）。既知の実害は attention i8a8 と
  同じく「クロス経路の atol=0 門が Metal で立たない」こと。**切り分け済み（2026-09-01）**: `gpu_gemm_skinny_test.ts` の
  バケット跨ぎ u32 門は M2 で緑 — **GEMV 固有**（一般則説は棄却）。**裁定 = 既定経路維持**
  （ADR [0082](decisions/0082-linear-gemv-decode.md) 追記 1 — 機序の見立て: GEMM は逆量子化を
  共有タイルへ格納してから読む = 丸め障壁あり / GEMV は 1 式インライン = MSL の contraction が
  跨げる）。根治候補 = GEMV に明示の丸め点を入れる式形の探索（M2 実機ループ要・下の根治候補と
  同席・未着手）。

Deno 2.9.5 / 2.9.6 に Metal / naga / wgpu の更新は無い（denoland/deno#36257 = mapped range の
み）。根治候補 = TS 参照の FMA 許容化 or WGSL 側で丸めを固定する手段の調査（未着手）。記録 =
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)
（時点）と
[research/2026-08-29-chatgpt-review-verification.md](research/2026-08-29-chatgpt-review-verification.md)
（M2 再検証）。

## Metal で out-of-memory errorScope が沈黙する — fail loudly 門が不発（独立バグ）

実機 **M2（24GB / maxBufferSize 14.3GB）**で `gpu_generation_context_test.ts` の
「state 確保の失敗は out-of-memory errorScope で fail loudly」が
**Expected function to reject** で赤（2026-09-01）= **64GiB の state スロット確保が黙って成功
する**（Metal の遅延確保 — wgpu の Metal backend は総量予算を持たず、`newBufferWithLength:` が
物理超過でも nil を返さない形）。実害: runtime の重みアップロード経路は size 門を持たず
errorScope に全面依存しているため（ADR 0070 決定 4）、**errorScope が沈黙する環境では
「確保失敗 = ゴミを読む」が例外なしで通り得る**。バッファ層自体は M2 実測で健全
（Metal NaN 調査時のプローブ実測 — 単一 64〜640MiB・実プロファイル 835 本累積とも全一致。
プローブ群は役目を終え 2026-08-31 に削除済み — 復元は git 履歴から）。
修正候補 = 重み経路への明示サイズ門 +
`karume.json` の `requiredLimits`（hub が parse するのみで現状誰も読んでいない）のロード時実効化
（backlog「数値危険クラス監査波」に同席）。

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
