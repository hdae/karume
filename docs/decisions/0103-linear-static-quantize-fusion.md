# 0103: 並列GEMVと固定再量子化の任意融合

- Status: accepted（2026-09-15・利用者の最適化継続依頼の範囲）。M2での採用判断は未完。
- 関連: [0097](0097-gemma4-qat-integration.md)、[0098](0098-linear-gemv-parallel.md)、[0096](0096-speculative-decoding.md)、[0040](0040-fusion-pass.md)
- 根拠: [保留候補の再検証](../research/2026-09-15-held-combinations.md)、[製品実装の検収](../research/2026-09-15-linear-static-quantize-fusion.md)

## 問題と選択

固定mobile QATのE2Bは、行列計算の直後に固定scaleへの再量子化（SRQ）を行う。
現在の並列GEMVでは、2演算を1回のGPU起動にまとめる試作が単独で約5.5%改善した。
旧逐次GEMVでの不採用は現在の並列経路へそのまま当てはまらない。
広い融合・重み転置・投入上限の変更をまとめて採用せず、独立した融合だけを追加する。

## 指定と適格条件

`SessionOptions.fuseLinearStaticQuantize?: boolean`を追加する。省略時はfalse。
trueは`linearGemvReduce: "parallel"`と`linearCompute: "f32"`の組合せだけを受理し、
未対応の組合せやboolean以外の値はSession構築時に拒否する。並列設定を暗黙には追加しない。
Gemma通常版とQATの共通pipelineオプションから、target/drafterのSessionへ同じ指定を渡す。

融合するのは隣接した`linear → static_quantize`で、全入力・出力がf32、出力shapeが同一、
linearの出力が内部値かつ消費者が1本、3入力のbinding名が異なるものだけ。
常駐したINT2/4/8の重みと、既存`linearGemvParallelLanes`が受理する実測形状、物理M=1..8に限定する。
対象外の形や公開・共有された中間値は非融合の既存経路に残す。
`lastRunFusions.linearStaticQuantize`と独立したカーネルキーで適用を観測できる。

モデル・quant・CLIの既定、manifest、IR、重み、公開source pinは変更しない。
falseで非融合へ戻せる。通常版は現行のグラフに該当するSRQが無く、指定しても融合数0となる。

## 算術と生成

既存parallelと同じWGSL生成処理を共用し、最後の出力部分だけを切り替える。
Kの巡回配分、語内の逆量子化と積和、workgroupの加算木、biasの加算順は維持する。
既存のparallel・逐次・subgroupのキーと生成WGSLを変更しない。

biasを加えた結果をu32へ再解釈し、uniformの0とのXORを挟んでSRQの整数表へ渡す。
この第4語を定数式へ置き換えない。scaleが0なら元の恒等演算を保ち、
正負の飽和・NaNの分類・符号付き0は独立SRQと同じ表から求める。
SRQの表生成は`staticQuantizeParams`を共用する。融合内だけ128境界の上限探索を固定回数へ展開する。
独立SRQの二分探索・キー・WGSL・参照goldenは維持する。

融合版のキーは格納・group長・lane数を含み、形状やscaleの実値はuniformで渡す。
同じキーからは同じWGSLを生成する。M=1/4/8は同じ算術を使い、tile数のdevice上限を超えた場合は拒否する。
実機でのu32一致は検収結果であり、全GPUでの浮動小数点丸めの仕様保証とはしない。

## 所有権と検収

融合の純関数にはGPU実体を持たない常駐格納の記述だけを渡す。
レシピ導出時に重みのscaleを既存の常駐バッファから借りる。GPU転置・重みコピー・別のscale所有者を作らない。
外部入力の延べ消費数は既存の融合機構が導き、scaleの寿命はSession/sharedWeightsのリースに従う。
融合paramsは線形演算の4語とSRQの260語。既存のSession paramsキャッシュを使う。

適用・非適用、元のparallel＋元のSRQとの全境界・特殊値比較、0 scale、丸め用XORの実行、
M1/4/8の行0一致、共有scaleの借用と破棄、実配布の融合数、生成と全体速度、MTP参照経路を検証する。
QATのMTPそのものを新たに実装・保証する変更ではない。

Chrome画面の初期選択はQAT E2Bだけで非融合→融合→融合→非融合、計40生成。
M2で利得を確認できなかったattention融合は選択肢へ残し、比較基準は従来のparallelにする。
M2検収前に高速化付きquantのモデル既定へ昇格しない。
