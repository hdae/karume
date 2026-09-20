# 0098: 量子化 GEMV の K 並列加算を任意指定する

- Status: accepted（2026-09-13）。runtimeは任意指定を維持し、Gemma通常/QAT E2Bの既定quantに明示する。
- 関連: [0058](0058-numerics-opt-in-contract.md)、[0082](0082-linear-gemv-decode.md)、[0096](0096-speculative-decoding.md)、[0097](0097-gemma4-qat-integration.md)。

## 背景

逐次 GEMV は 1 出力の K 方向を 1 thread で計算する。Chrome の実測では長い K の待ちが大きく、
16 byte の圧縮語を複数 lane に配ると改善した。一律適用は遅くなる形があり、実形の掃引とモデル全体の A/B で選別する。
加算順が変わり、QAT は再量子化境界を通して生成列にも差が出る。既定を黙って変更してはならない。

## 決定

1. `SessionOptions.linearGemvReduce` を追加する。値は `"sequential" | "parallel"`、省略時は `"sequential"`。
   Gemma 通常/QAT の pipeline は指定を target / drafter の両 Session へ渡す。家族の既定も変更しない。
2. `parallel` は f32 演算、INT2/INT4/INT8 格納、実測済み形状、物理 M=1..8 に限定する。
   f16/a8 演算との組合せは未実装として構築時に拒否する。f16/f32 格納や対象外形状は従来経路を使い、診断キーで区別する。
   M>8 は既存 prefill 経路。M=1 だけの切替は MTP の verify 行0との一致を壊すため採らない。
3. 128 thread の workgroup 内で、1 出力あたり K の圧縮語を 2/4/16/32 lane に巡回配分する。
   各 lane は語内の順序・逆量子化の f32 丸めを維持し、最後に固定順の木で部分和を足して bias を一度加える。
   subgroup・跨 workgroup の同期・f16 の追加丸めは使わない。workgroup 内の全 thread が barrier を通る。
   通常の誤差帯と参照経路との bit 一致は区別する。同一形状の M=1/4/8 は同じ shader と加算順を使う。
4. 幾何は実測形状の静的な選択表に限る。GPU 名や実行時計から自動調整しない。
   キーは `linear_gemv_parallel` と格納・group・lane 数を含み、同一キーの WGSL は byte 同一。
   dispatch 上限は既存の検査で拒否する。重み・中間バッファの追加確保は無く、共有メモリは 512 byte / workgroup。
5. CLI と Chrome 比較画面から明示的に試せるようにし、結果へ設定を記録する。
   QAT の文章差、M2 未検収、対象外形状の扱いを説明する。公開 API は省略可能属性の追加のみで、IR・資産・既存 golden は変更しない。

## 検収と段階

- 第1段: 隔離実験の速度・FP64参照・固定小規模品質評価・MTP比較を保存する。
- 第2段: 本カーネルを実装し、実験版との比較、端数列・短いK・scale添字・M1/4/8の一致、A/B誤差帯とcensusを検査する。
- 第3段: CLI / Chromeの操作を通じて設定とJSONを検査し、既定の `deno task verify` を通す。
- 第4段: M2で利用者が追試する。既定への昇格や他形状・他モデルへの拡張は、この記録だけでは行わない。

数値の正本は[調査・検収記録](../research/2026-09-12-chrome-gemv-parallel.md)とそこから辿る保存 JSON。品質の良否を速度だけで判断しない。

## quant定義での採用（2026-09-13）

利用者のM2/Chrome測定で通常E2Bは約25％、QAT E2Bは約18〜21％改善し、TTFTはほぼ不変。
保存された英語・日本語の出力を読み、問いへの関連性と文の連続性を確認した。通常のA/Bは64 token全一致。
QATは語句が変わるが、今回の短い出力に崩壊は見られない。RTXの固定64問は45→44正答、NLLは約0.63％増。
大幅劣化を示す証拠は無く、利用者が委任した速度/品質判断の範囲で採用する。広い品質を保証する判断ではない。

- 通常/QAT E2Bの配布recipeに `i4-gemvpar` を追加し、`defaultQuant` に指定する。
  重み写像は従来の `i4` と同じで、`session: { linearGemvReduce: "parallel" }` だけを加える。
  従来の `i4` の意味と参照goldenは変更しない。E4Bの既定は検証前なので `i4` を維持する。
- hubの `SessionSpec` に `linearGemvReduce: "sequential" | "parallel"` を追加する。
  未知値を拒否し、modelsの共通写像も対応する。manifest形式は `karume/4` の省略可能欄の追加で、
  既存の空sessionは従来どおり。旧readerは新欄を未知キーとして拒否し、黙って無視しない。
- GemmaのfromPretrainedは選択quantの宣言をtarget/drafterへ渡す。
  優先順位は呼び手の明示指定 → quant.session → runtimeの参照既定。
  fromAssetsにはquant選択が無いので明示したオプションのみを使う。モデル名・層数・GPU名で自動選択しない。
- Gemmaは今回 `linearGemvReduce` だけをquant.sessionから受理する。
  未実装の他のsession欄を宣言したquantは、重みのダウンロード前に拒否する。値を捨てて続行しない。
- CLIでquantを選べるようにし、Chrome画面の既定はmanifestの `defaultQuant` に従う。
  比較時は同じ選択quantに加算方式を明示してA/Bする。結果には選択quantと有効な加算方式を記録する。
- 既存のローカル配布形・公開pinを自動で書き換えない。recipeから新しい配布形を作ったときに既定が変わる。
  重み自体の再量子化は不要。参照用 `quant: "i4"` と、実行指定 `linearGemvReduce: "sequential"` を残す。

数値と採否の根拠は[M2の採用判断](../research/2026-09-13-m2-gemv-adoption.md)。

## 追記（2026-09-20）— 並列族の積和は明示 `fma()` で綴る

M2（Metal）で同じ数式の 2 カーネル（f32 と packed 活性）が u32 で割れた件の帰結。Metal のコンパイラは
式形ごとに fma 縮約の入れ方を変えるため、`acc + x * d` の綴りでは変種同士が揃わない。並列族
（f32 / packed × linear→SRQ 融合なし / あり）の積和を `acc = fma(x, d, acc)` に固定した。RTX / Vulkan では
数値不変（掃引 540 組で不一致 0・QAT E2B の生成文が同一）、Metal では並列経路の出力が変更前から変わる
（変種同士は揃う）。逐次 GEMV・行ブロック・subgroup 変種は従来の綴りのまま。経緯と実測は
[0105 追記 4](0105-packed-static-quantize-activations.md#追記-42026-09-20-追記-3-の撤回と並列-gemv-族の積和を明示-fma-で綴る決定)。
