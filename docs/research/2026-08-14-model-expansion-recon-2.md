# モデル拡充 recon 第 2 次 — 12 候補の判定と、5 本を実装して分かった差分

> 時点スナップショット（2026-08-13〜14）。Web 掃引 6 レッグ + コードベース棚卸し 2 レッグ +
> 群別の適合判定 7 レッグ + 敵対検証 2 レッグの統合に、**同じ波で実際に移植した 5 本の実測**を
> 突き合わせたもの。§1 が「recon が当たったか外れたか」の記録、§2 が**まだ生きているキュー
> 情報**。出典 URL の取得日は 2026-08-13〜14。前次は
> [2026-08-11-model-expansion-recon.md](2026-08-11-model-expansion-recon.md)。

## 0. この波で着地したもの（結論の先出し）

SigLIP2 vision（base + so400m）/ BiRefNet_HR / Lucida / vowel-detector /
Depth Anything V2 Small の 5 本。支える op は 3 本
（`upsample_bilinear2d` = 第 1 層 / `deform_conv2d` = 第 1' 層 ADR 0055 /
`gru_scan` + `gru_scan_reverse` = 第 2 層 ADR 0056、+ 派生次元束縛 ADR 0057）。
横断層として画像前処理（`packages/models/src/image/preprocess.ts`）を新設した。

**op 語彙の増え方は 3 本で、モデル 5 本ぶんの要求を満たした** — BiRefNet と
Depth Anything は新規 op ゼロ（全て第 0 層のパッチで吸収）、SigLIP2 も新規 op ゼロ。

## 1. recon の精度 — 見立てが現物と食い違った 6 件

**次回の recon で同じ誤りを繰り返さないための記録**。いずれも「実装に入って初めて分かった」
のではなく、**着手前のスパイク 1 本で分かる類**だった。

| # | recon の見立て                                                                | 現物                                                                                                                                                                                                                                                   | 教訓                                                                                                                                      |
| - | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | BiRefNet に `aten.rsqrt` が出る（BatchNorm の分解形）                         | **出ない**。curated decomp 後も `_native_batch_norm_legit_no_training` のまま残る                                                                                                                                                                      | 「推論時 BN は定数に畳まれる」は decomp 表に依存する。FOLDABLE 追加では届かず、モジュール差し替えが要った                                 |
| 2 | `aten.roll` が出る（Swin の shifted window）                                  | **そのままでは出ない**。torch が `arange.start_step` + `fmod.Scalar` + `index_select` へ分解し、`index_select` が語彙外で落ちる                                                                                                                        | 落ちる op 名は「元の関数名」とは限らない。**未対応 op の全件列挙を先に採る**のが唯一の確実な手順                                          |
| 3 | BiRefNet の blocker は deform / bilinear / roll / 空間 pad / rsqrt / mean.dim | **実際の正面 blocker は recon に無かった 4 系統** — rank≥5 の reshape/permute 連鎖（`window_partition` / qkv / `image2patches`）・`PatchMerging` の step-2 slice・相対位置バイアスの `index.Tensor` と i64 initializer・窓マスク生成の in-place 代入列 | Web 情報からの op 推定は**モデルの「特徴的な部品」に引っ張られる**。実際に効くのは形状操作の細部で、そこは推定できない                    |
| 4 | SigLIP2 の `resample=2` は bicubic                                            | **bilinear**（PIL 定数は 2=BILINEAR / 3=BICUBIC）                                                                                                                                                                                                      | 設定値の意味は定数表を引く。`AutoImageProcessor` の出力と `tvF.resize` の突合で 3.8e-6 一致・bicubic では最大 47/255 ずれることで確定した |
| 5 | Depth Anything V2 は Small/Base/Large とも Apache-2.0                         | **Apache-2.0 は Small のみ**。Base / Large は CC BY-NC 4.0（2026-08-14 実測）                                                                                                                                                                          | ファミリ単位でライセンスを見ない。**variant ごとに HF API を叩く**。なお前次 recon の「DA3 の Apache 側」は V3 の話で、V2 とは別          |
| 6 | GRU のゲートは `h' = (1−z)⊙n + z⊙h`                                           | 数学的には同値だが torch の分解形は **`h' = (h − n)·z + n`** で**丸めが違う**（10 万要素中 44,345 件・maxdiff 4.77e-07）                                                                                                                               | ビット同一を狙うなら「数式」ではなく**torch が実際に出す演算列**を写す                                                                    |

### 1'. 逆に recon が正しく当てたもの

- `deform_conv2d` の仕様（offset の偶数=y/奇数=x・mask は補間の後・境界外ゼロ埋め）と
  BiRefNet の実引数（k∈{1,1,3,7}・offset_groups 1・1 forward 20 回）。
- bilinear の第 0 層分解が非成立（FLOP 184 倍 / index 805MB〜3.22GB）。
- Lucida は BiRefNet_HR の fine-tune で `birefnet.py` が **diff 0 行**（重みのみ差）。
- Lucida `m35-comfy` の Normalize 焼き込み（実測: 入力チャネル別 RMS 比が 1/ImageNet std と
  0.3% 一致・bias 復元が 0.2% 一致 → **配布対象外**に確定）。
- BiGRU の右 pad が数値的に成立しないこと（§3 で詳述）。

## 2. 未着手候補の判定（生きているキュー情報）

verdict / effort は recon 時点の判定。**この波で語彙に入った 3 op を反映済み**。

| 候補                     | verdict           | effort             | 主 blocker                                                                                                                                                                       | ライセンス                |
| ------------------------ | ----------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| **MobileSAM**            | feasible_with_adr | L                  | `conv_transpose2d`（mask decoder の k=s=2 ×2）。**kernel==stride なので DA-V2 と同じ第 0 層分解が効く可能性が高い**（下記 §4）                                                   | MIT                       |
| **SAM 2.1**              | feasible_with_adr | L                  | 同上 + Hiera の pooling attention（`max_pool2d`）。window partition が rank>4 を出すかは要スパイク                                                                               | Apache-2.0・ungated       |
| **Kokoro-82M**           | feasible_with_adr | XL（G2P 抜きで L） | **双方向 LSTM 6 本**（`gru_scan` の拡張先）+ iSTFT の `conv_transpose1d(K=20,s=5)` が契約外（K−s=15 が奇数で `2·padding == K−stride` を満たせない）+ G2P（misaki）が Python 依存 | Apache-2.0                |
| **MiniCPM5-1B**          | feasible_with_adr | L→XL               | KV cache 実行モデル（ADR 0004 未決筆頭）+ tokenizer 新規実装。**i8 で 1GB を切る唯一の候補**                                                                                     | Apache-2.0・ungated       |
| **Gemma 4 E2B / E4B**    | feasible_with_adr | XL / L             | 同上 + PLE（262144×35×256）の層分割 + **int4 不在**（f16 10GB / i8 5GB）                                                                                                         | **素の Apache-2.0**（§3） |
| **SigLIP2 NaFlex**       | feasible_with_adr | L                  | パッチ系列長がアスペクト比依存 + 位置埋め込みの bilinear resize。**FixRes を複数解像度で持つ方が器に素直**                                                                       | Apache-2.0                |
| **BiRefNet base / lite** | feasible_with_adr | S                  | HR と同一コード → **重み差し替え相当**（この波の資産がそのまま効く）                                                                                                             | MIT                       |
| **YOLO26n**              | feasible_with_adr | L                  | **AGPL-3.0**（配布に copyleft が及ぶ）+ `max_pool2d`（SPPF）+ `topk`（end2end postprocess）                                                                                      | AGPL-3.0                  |
| **DINOv3 ViT-S/16**      | feasible_with_adr | M                  | **技術は新規 op ゼロ**。gated + Meta 独自条項（同条項下でしか再配布できない = MIT リポと非互換）                                                                                 | 独自・gated               |
| **SAM 3 / 3.1**          | blocked           | XL                 | gated（匿名 401）+ **出力インスタンス数がデータ依存**                                                                                                                            | 独自・gated               |
| **LocateAnything-3B**    | blocked           | XL                 | **NVIDIA 非商用** + 自己回帰 grounding（KV cache）+ EOS でデータ依存長                                                                                                           | 非商用                    |
| **Depth Anything V3**    | uncertain         | XL                 | **タスクが V2 と別物**（多視点 + カメラ姿勢の DualDPT）。単眼版 `DA3MONO-LARGE`（Apache-2.0）の I/O 仕様は未確認                                                                 | 混在（Large/Giant は NC） |

**ユーザー方針（2026-08-13）**: 非 MIT/Apache は当面対象外。該当は YOLO26n / DINOv3 /
SAM 3 / LocateAnything / DA3 の NC variant。**後日まとめて扱いを裁定する**。

## 3. vowel-detector — BiGRU と可変長の実測（この波で最も情報量が多かった調査）

### 右ゼロ pad は成立しない、しかも pad 量に依らない

- 逆方向 GRU が pad 側から状態を持ち帰るため、実長 137 vs pad 500 で **max abs diff 5.91 /
  argmax 一致率 0.971**。単方向なら 5.1e-03（conv 窓端のみ）。
- **pad 掃引 120 通り**（4 音声 × pad {2..1024} × 埋め方 3 種）で、劣化は pad 量に対して
  **単調でも比例でもなく pad 2 フレーム（40ms）で既に飽和**することが確定。
  → **バケットの刻みを詰めても品質は改善せず、配布サイズだけ線形に増える**。
- 末尾トリムは対症療法にならない（`|差| > 1e-2` が全フレームに届き、Viterbi が大域最適な
  ので判断は発話のどこでも反転しうる。実測で末尾 80 フレーム捨てても一致せず）。
- 固定長窓 + overlap-discard は**バケットより悪い**（窓の内側でも maxAbs 0.88〜1.45）。
- 埋め方は zero 一択。silence（波形へ無音を継ぎ足す）は**発話内 z 化の統計が pad 側へ
  引っ張られて先頭まで壊す**（argmax 一致率 0.796 まで低下）。

### 帰結 = 第 2 層 RNN op（ADR 0056）

`torch.export` は `aten.gru` を**時間方向へ完全展開**し（38 ノード / 入力フレーム）、
かつ **`export` の段階で T を specialize する**（分解表から外して保存しても T は固定）。
記号 T を得るには `torch.library.custom_op` + `register_fake` で単一ノードにする必要があり、
これは実測で成立した。結果:

|           | バケット版（旧）     | `gru_scan` 版          |
| --------- | -------------------- | ---------------------- |
| IR ノード | 8,434（T10=200）     | **18**（T 非依存）     |
| 配布      | 33.9MB（4 バケット） | **2.67MB**（1 グラフ） |
| `.lab`    | pad 由来の差が残る   | **torch と完全一致**   |

## 4. 共通投資の回収範囲（次に効く順）

1. **画像前処理層** — この波で新設済み。detect / vision-enc / segment / dense の全群が使う。
2. **`upsample_bilinear2d`** — この波で実装済み。BiRefNet 系 + Depth Anything + SAM 系 +
   RMBG-1.4 / IS-Net / MODNet の **5 アーキ以上**に効く。
3. **`conv_transpose2d`** — **未実装だが、DA-V2 で `kernel == stride` の形は第 0 層分解
   （1×1 conv + pixel shuffle）で吸収できることが実証された**。MobileSAM / SAM 2 の mask
   decoder も k=s=2 なので**同じ手が効く公算が高い** — 語彙追加が要らない可能性がある。
   一般の `conv_transpose2d`（kernel ≠ stride）が要るモデルが出てから層判定する。
4. **KV cache 実行モデル** — MiniCPM5 / Gemma 4 / LocateAnything + 将来の Moonshine。
   現状の器の欠け 5 点は [ADR 0004](../decisions/0004-execution-model.md) の未決節と
   ADR 0056 の Consequences を参照。
5. **RNN 逐次構造** — `gru_scan` としてこの波で実装済み。**Kokoro-82M の双方向 LSTM 6 本**が
   次の利用者。LSTM 拡張の唯一の分岐は「`h_n`/`c_n` を消費するか」で、消費するなら IR v1 の
   単一出力制約と正面衝突する（ADR 0056 決定 8）。**着手前に消費形を確認すること。**

## 5. 敵対検証が覆した事実（5 件）

掃引レッグの結論を一次情報で取り直した結果:

- **Gemma 4 は素の Apache-2.0**。`license_link` の先が Apache 全文ページで、Gemma 3 までの
  Terms of Use 型ではない。前次 recon §4 の読みが正しく、今回の掃引側が誤っていた。
- **DINOv3 に "Built with DINOv3" 表示義務は無い**（LICENSE.md 全文 grep でヒット 0）。
  実際の義務は ①Agreement 同梱と同条項での再配布 ②研究公表時の acknowledgment ③用途制限。
  **真の争点は「同条項下でしか再配布できない」= MIT リポとの非互換**。
- SigLIP2 NaFlex も Apache-2.0（推定ではなく HF API のタグで確定）。
- Gemma 4 **E4B にも ONNX 前例がある**（`onnx-community/gemma-4-E4B-it-ONNX`）。
- DA3 の Apache 側は Small / Base / Metric-Large に加え **Mono-Large** も。

## 6. 記録の所在

- 実測の生データ: 各コミット本文（`dc2031d`〜`661762c`）と ADR
  [0055](../decisions/0055-deform-conv2d.md) / [0056](../decisions/0056-gru-scan.md) /
  [0057](../decisions/0057-derived-dim-binding.md)。recon レッグの生出力は揮発。
- 層分類の穴と暫定運用: [known-issues.md](../known-issues.md) の
  「op 追加の層分類が一部の op で自明にならない」節。
- 評価素材の作り方: 画像は `deno task demo:eval-images`（Anima・seed 固定で**バイト同一**を
  実測済み）、日本語音声は `deno task demo:eval-audio`（Irodori）。どちらも
  `outputs/demo/` 配下で `rm -rf` しても台本から再生成できる。
