# 0101: 並列GEMVの加算木を32レーン内の値交換で実行する

- Status: accepted（2026-09-13）。任意指定のみ。既定quantは変更しない。
- 関連: [0098](0098-linear-gemv-parallel.md)、[0100](0100-rms-subgroup-reduction.md)。

## 背景と代替案

RMS subgroup32のM2追試は約0.5%の差で、既定採用を裏付ける改善ではなかった。
更新後のChromeプロファイルでも行列計算が大きい。既存parallelは部分和を共有メモリへ置き、
各段でworkgroup全体を同期する。1出力に使う2〜32レーンは32レーンのsubgroup内に収まる。
入力配分と加算木を保って値を交換すれば、重みコピーや共有メモリを追加せず同期を減らせる。
数値の正本は[調査記録](../research/2026-09-13-gemv-subgroup.md)。

## 決定

1. `SessionOptions.linearGemvReduce`へ`"parallel-subgroup32"`を追加する。
   省略時の`sequential`と既存`parallel`の意味・適用形状・WGSL・キーは維持する。
   新方式もf32演算・INT2/4/8格納・実測済み25形状・M=1..8に限定する。
   対象外の格納・形状・prefillは従来経路。f16/a8演算との組合せは構築時に拒否する。
2. 新方式は`acquireGpu({ subgroups: true })`を要求する。ADR 0100の機能・既知解検査を再利用し、
   Sessionでも必要なdevice/WGSL機能の不足を重み転送前に拒否する。自動fallbackはしない。
   GPU能力を取得しただけでは計算方式を切り替えない。
3. 128スレッド・固定32レーン。出力列はsubgroup IDとsubgroup内IDから求め、local IDとの配置対応を仮定しない。
   各レーンの圧縮語走査・逆量子化・積和はparallelと同一。幅L/2から1の順で`subgroupShuffleXor`を呼び、
   下位レーンだけが対応部分和を加算する。端の出力列も値交換へ参加する。biasは最後に一度加える。
   共有部分和とworkgroup barrierは不要。キーに`subgroup32`を加え、参照カーネルと区別する。
4. parallelとのビット一致は実測の範囲で確認する。未検証GPUの丸めまで一律に保証しない。
   既存golden・許容誤差は維持し、M=1/4/8の先頭行一致を新経路でも検査する。
   MTP全体・E4B・長文・他GPUの品質を、この単体検査だけで検収済みとはしない。
5. Gemma通常/QATの既存オプションからtarget/drafterへ同じ指定を渡す。
   内部GPU取得では指定時だけ必要な機能を要求する。外部GPUはSessionの門で検査する。
   quantの保存語彙・recipe・既定quant・配布形式・公開revisionは変更しない。
   Denoの必要な機能は未提供なので、今回はChromeで明示指定する提供範囲とする。
6. 導入時の比較画面は通常/QAT E2B・Karume・細分化chunk64・RMS融合・投入768を初期選択にし、
   各モデルでparallel→parallel-subgroup32→parallel-subgroup32→parallelを測る（8設定80生成）。
   RMSのsubgroup32とは独立した選択にする。以前の参照・RMS・Transformers.js比較も残す。

## 検証と段階

- 隔離した単体実験: 短K・端列・全レーン数・M1/4/8のFP64許容帯、parallelとのu32一致、実形状のGPU timestamp。
- 隔離した全体比較: 通常/QATの英語・日本語、ABBAを反復し、速度とtoken/stop/textを保存。
- 製品の数値・適用箇所検査: 元のsnapshotを維持し、新しいキーのsnapshot、Sessionの拒否と診断キーを検査。
- Chromeの実UIで数値検査・生成・初期選択・保存JSONを検収し、Denoの全体verifyを通す。
  Denoでは新機能の実走だけを明示SKIPし、Chromeで同じ検査を実行する。
- M2の追試で採用範囲を判断する。RTXの利得をM2の実測として扱わない。

値交換は[WGSLのsubgroupShuffleXor仕様](https://www.w3.org/TR/WGSL/#subgroupshufflexor-builtin)に従う。
マスクはsubgroup内で一様な16以下の2冪とし、全レーンが参加する。外部実装のソースコードは複製していない。

## M2追試後の採否と画面の初期選択（2026-09-13）

[利用者の80生成](../research/2026-09-13-m2-gemv-subgroup-adoption.md)は既存parallelとtoken/stop/textが一致した。
通常版は約9.3%遅く、QATにも高速化の根拠はない。QATは走行中の変動が大きいため、固定した低下率とは扱わない。
M2向けの既定採用を見送る。RTXのQATでは利得が再現しているため、任意指定は維持する。
共有メモリ同期を減らせば他GPUでも速くなる、という仮定は採用根拠にしない。

画面の初期選択だけを既存parallelへ戻す。両E2B・dense・RMS融合・投入768の2設定20生成となり、
GEMV/RMSのsubgroup比較は引き続き選択できる。runtime・モデル・quant・CLIの既定と保存形式は変えない。
既に得られたM2の80生成を同じ設定で取り直す必要はない。

## 追記（2026-09-25）: 決定 3 の積和を明示 `fma()` に揃える

[ADR 0105 追記 4](0105-packed-static-quantize-activations.md#追記-42026-09-20-追記-3-の撤回と並列-gemv-族の積和を明示-fma-で綴る決定)が
並列族の積和を `acc = fma(x, d, acc)` に変えたとき、subgroup32 変種だけ `acc = acc + x * d` のまま残った。
Metal は式形ごとに fma への縮約の入れ方を変えるので、その状態では決定 3「積和は parallel と同一」が字面どおりでなく、
u32 一致の門（`tests/helpers/gemv-subgroup-check.ts`）は Metal で成り立つ保証が無かった（2026-09-24 の全域レビュー
W-RT6-1 / W-AD3-2）。subgroup32 は parallel と同じ和を subgroup の値交換で出す変種で、参照経路ではない。

- `linearGemvSubgroupWgsl` の積和を parallel と同じ `mac(fma = true)` にする。キーは変えない（0105 追記 4 の先例と同じ —
  キーはパイプラインキャッシュの識別子で、決定性の門はスナップショットが担う）。snapshot 3 本を焼き直した。
- 常設の門: `linear_gemv_parallel_test.ts` が全 30 形（格納 × group × lane）で parallel と subgroup32 の `acc = fma(…)`
  行の列が一致し、積和に `acc + x * d` が残らないことを綴りで縛る。
- RTX / Vulkan では数値不変のはず（0105 追記 4 の 540 組の掃引と同じ機序 — コンパイラが縮約する）。**Metal では
  subgroup32 の出力バイトが変更前から変わる**（parallel と揃う側へ）。2026-09-13 の M2 一致記録は fma 化前の実測。
- 再検収: Chrome（M2）で `deno task bench:llm-browser` → `http://localhost:8787/check.html`（u32 一致門をそのまま
  Chrome で走らせる入口。tools/llm-speed/browser/check.ts）。結果（2026-09-25・利用者の M2）: **PASS** — parallel と subgroup32 の u32 一致 180 ケース / 28,080 要素（Apple metal-3・Chrome 153・checkout 8d3bd8b4・features subgroups / subgroup-size-control / timestamp-query・wgsl subgroup_id あり・297 ms）。決定 3 と決定 4 の一致は fma 化後の Metal で成立する。
  Vulkan 側も同日に利用者が RTX 5070 Ti（nvidia blackwell・Windows・Chrome 153・同じ checkout / bundle）で同じページを走らせ **PASS**（180 ケース / 28,080 要素・1,203 ms）— 「RTX / Vulkan では数値不変」の見込みが Chrome でも確認できた。
