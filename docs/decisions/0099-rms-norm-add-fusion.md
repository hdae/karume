# 0099: RMS正規化と後続加算を任意指定で融合する

- Status: accepted（2026-09-13）。runtimeは任意指定。M2検収は完了。モデル既定への組み込みは別段階。
- 関連: [0017](0017-rms-norm-conv2d-clamp-min.md)、[0058](0058-numerics-opt-in-contract.md)、[0068](0068-decode-exit-multi-output.md)、[0098](0098-linear-gemv-parallel.md)。

## 背景

Gemma通常/QAT E2Bには、内部値が専有される隣接の`rms_norm → add`が106本ある。
正規化結果がaddの第1引数になるのは1本、第2引数になるのは105本。両順序の受理と実配布の融合数を検査する必要がある。
直接つなぐと乗算と加算が融合して丸めが変わる。workgroupメモリによる丸め保持は単体で利得を相殺した。
整数演算を挟む候補は、RTXのDenoで演算出力のu32、Chromeで固定生成列が参照と一致した。追加メモリなしで融合できた。

融合だけでは通常E2Bの全体速度が改善しなかった。診断付き経路では1128→1022 dispatchとなり、
投入上限1024の境界を跨いで2→1 submitになる。投入頻度を独立に振ると、上限768と融合の組み合わせで改善した。
CPUとGPUの仕事の重なりが関係するという推測はあるが、その詳細をGPU timestampの合計だけで断定しない。

## 決定

1. `SessionOptions.fuseRmsNormAdd?: boolean`を追加する。省略時はfalse。不正な型はSession構築時に拒否する。
   隣接するf32のRMS→add、同一shape、内部値の消費者が1本でgraph outputでない形だけを受理する。
   最終次元は単体検収した256/1536/2560に限定し、broadcast、重複binding、非隣接の形は既存経路で実行する。
   診断の`lastRunFusions.rmsNormAdd`に適用数を出す。
2. 256 thread版RMSの縮約・grid-strideを共用し、加算の左右を維持した2種類のキーを使う。
   正規化結果をu32へ再解釈し、uniformの0とのXORを通してf32に戻してから加算する。
   `rmsNormParams`の第4語の0を使い、定数式のbitcast往復へ簡略化しない。
   maskの値を変える検査で、この整数演算が実行されることも確認する。
3. 整数演算を挟むことは、全GPUでのビット一致の仕様保証ではない。
   WGSLは浮動小数点の再結合・融合や特殊値の扱いに裁量を認めるため、既定の参照経路、元のキーとWGSL、goldenを維持する。
   参照との不一致が出た場合も許容誤差を弱めず、別の実行設定として原因・品質・速度を検査する。
   根拠となる仕様は[WGSLの再結合・融合](https://www.w3.org/TR/WGSL/#reassociation-and-fusion)と[bitcast](https://www.w3.org/TR/WGSL/#bitcast-builtin)。
4. `Gemma4Pipeline`と`Gemma4QatPipeline`に`fuseRmsNormAdd`と既存runtime型の`submitPolicy`を渡せるようにする。
   target/drafterへ同じオプションを渡す。融合は投入上限を暗黙に変更せず、両設定を独立にする。
   runtime・モデル・CLI・quant定義の既定は変えない。hubの保存形式・exporter・公開revisionも変更しない。
5. Chrome比較画面で「従来」「投入上限768のみ」「融合 + 上限768」を選べるようにする。
   上限768は既存の時間予算・安全率を保ち、`maxChunkSize`だけを下げる。
   初期選択は3設定の往復、通常/QAT E2B、並列GEMV、細分化バケット。6設定×2モデル＝12設定・120生成。
   実行した融合フラグと投入上限をJSONへ残し、Transformers.jsには適用しない。

## 検収と採用範囲

- 元の128/256 thread版RMSのキー・WGSLがバイト同一であること。
- matcherの既定、両引数順、共有・公開・broadcast・非隣接・未検収幅の拒否、実配布106本。
- 有限値のu32一致、NaN分類、符号付き0、相殺、強制grid-stride、後続consumerまでの入力寿命、M=1/4/8の先頭行一致。
- 実重みの生成一致と、投機のverify行0/decode行のu32一致。既存のgolden・比較条件は維持する。
- 通常のGPU greedy出力経路で全体を往復計測する。診断付き経路は出力取得と計測負荷が異なるため、速度の正本にしない。
- `deno task verify`を通し、M2は利用者が比較画面で追試する。M2未検収の段階で高速化付きquantの既定へ昇格しない。

数値・生データ・不採用候補は[調査記録](../research/2026-09-13-rms-norm-add-fusion.md)を参照する。

## M2追試後の状態（2026-09-13）

[利用者の120生成](../research/2026-09-13-rms-subgroup-reduction.md#利用者のm2結果)で参照との一致と小幅な速度向上を確認した。
融合＋上限768を次候補の比較基準にする。融合のquant選択と、ADR 0038のホスト政策である投入上限は別の設定であり、
この検収だけでsubmitPolicyをquantの保存語彙へ追加しない。モデル既定への組み込みは未完。
