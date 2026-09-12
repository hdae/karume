# 0098: 量子化 GEMV の K 並列加算を任意指定する

- Status: experimental（2026-09-12、実装・RTX自動検収済み）。利用者の Chrome 最適化依頼の範囲で追加。M2と人による品質判断は残り、既定には昇格しない。
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
