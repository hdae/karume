# 0097 — Gemma 4 QAT mobile を別ファミリで統合する

- Status: accepted（2026-09-11 — 利用者が実測後の段階案を承認、family `gemma4-qat` と E2B / E4B を指定）
- 根拠: [INT2 / SRQ の試作と段階案](../research/2026-09-10-codex-mtp-optimization.md#qat-mobile-の-int2-と固定丸め2026-09-11)
- 関連: [0069](0069-packed-w4-storage.md)（packed 格納）、[0085](0085-ple-host-gather.md)（PLE）、
  [0092](0092-distribution-repos-and-sources.md)（配布と取得元）、[0096](0096-speculative-decoding.md)（MTP）

## 背景

公式 mobile QAT は通常 Gemma 4 と共通する骨格を持つが、固定の INT2 / INT4 / INT8 重みと
SRQ（Static Range Quantization、固定 scale による活性値の丸め）を使う。
単に既存モデルの量子化名を変えたり、重みを f32 に広げたりするだけでは同じ計算にならない。
単体実測では INT2 の容量と速度に利得があった。CPU と GPU の線形縮約差が丸め境界を越す例も残るため、
単体の一致だけで全モデルの品質を保証できない。

## 決定

1. 配布・利用者向けの family は **`gemma4-qat`**。その中の model を **`e2b` / `e4b`** とする。
   通常 `gemma4` のモデル名や既定量子化は変更しない。対象は公式 `*-qat-mobile-transformers` の text 生成。
   同じ QAT 名でも GGUF / unquantized / compressed-tensors の各形式を自動で同じものとは扱わない。
2. 将来の配布先は `karume-gemma4-qat`、manifest の pipeline 名は `gemma4-qat` とする。
   旧 Gemma reader がこの pipeline を拒否する性質を維持する。公開前の source pin は作らず、
   検収ではローカル配布形を使用する。実際の公開・push はこの統合作業に含めない。
3. 実行・tokenizer・会話・RoPE・取得処理の共通部分は既存 Gemma の実装を再利用する。
   family の追加を理由にパイプライン本体を複製しない。QAT の入口では対応形式と固定丸めの存在を明示的に検証する。
4. 固定整数・scale の保存値を再量子化せず保持する。IR の INT2、固定 SRQ、PLE の INT4 を実装する。
   保存形式の shape / byte 数 / packing / scale、丸めと非有限値の契約は実装単位で本 ADR へ追記し、
   リーダ・ライタ・CPU 参照・GPU を揃える。既存の資産と数値許容差を変更しない。
5. 通常生成の一致・性能・資源解放を先に検収する。head と token embedding の共有は全バイト一致を確認して扱う。
   MTP drafter の I8 借用を INT2 へ暗黙に置き換えない。MTP は通常生成の検収後に別設計とする。

## 実装単位と検収

- INT2 格納: IR / safetensors / loader / CPU 展開 / 見積り / exporter の読書きと packing を揃え、
  GEMV・prefill GEMM・embedding の packed GPU 経路を検収する。端と符号の全値、既存格納型との u32 一致を検査する。
- 固定 SRQ: 明示 op の CPU / GPU を実装する。除算の誤差を丸め境界から除く試作を基に、
  scale=0、境界両隣、正負ゼロ、飽和、非有限値と許可する scale 範囲を検査する。
- QAT recipe / PLE: 公式の固定重みを保持した E2B / E4B の変換と、PLE INT4 の全量・行読取を実装する。
  元の I8 読取を維持し、全量と区間読みで同じ PLE を得ることを検査する。
- family / 生成: ローカル資産から使える入口を用意し、公式 CPU 参照、Deno、Chrome で通常生成・複数ターン・
  中断・解放を検収する。RAM / VRAM は実測と見積りを区別する。M2 の GPU 数値検収は別実機の作業。

各単位は全体検証後に独立コミットする。exporter / recipe 変更時は両 Python パッケージの pytest も実行する。
実験出力は `outputs/bench/karume/2026-09-11_qat-integration/`、既存資産は上書きしない。

## 承認の扱い

この段階案は承認済みであり、含まれる実装に段階ごとの再承認は要らない。
仕様変更は実装前に説明・記録し、承認範囲を超える変更や新たな破壊的変更は再確認する。
利用者の意図の明確化により、停止を要する想定外の問題はデータ損失・誤操作などのインシデントを指す。
性能低下や数値誤差の増加は停止理由にせず、検証条件を保って原因調査と候補の採否判断を続ける。

## 追記 1 — INT2 格納と実行の契約（2026-09-11）

IR v1 に格納 `i2`、safetensors の方言に `I2` を追加する。manifest は `karume/4` のまま。
旧 reader は未知の格納型を拒否する。既存の格納形式、生成キー、数値許容差は変更しない。

- 意味論は f32。論理形は正整数の rank 2 `[N,K]`、K は 16 の倍数。
  1 byte に 4 要素を下位 2bit から詰め、格納値は `u=q+2`、整数の全域 `[-2,1]` を使う。
  header の shape は論理形、バイト数は `N*K/4`。先頭は 4 byte 整列し、暗黙の末尾詰め物は持たない。
- `storage.scale` は必須で、F32 の `[N,1]`。group_size は受理しない。
  scale は重みと同じ shard、行分割時は先頭 piece と同居する。復元は要素ごとの `fround(q*scale)`。
  固定整数と scale の保存値を再量子化しない。
- 消費が linear / embedding の重みスロットだけなら packed 常駐。
  他の消費や graph 出力を持つ場合は CPU で f32 展開し、診断・見積りに展開後のバイト数を載せる。
  linearCompute は f32 のみ。a8 / f16 指定と、畳み込みの packed I2 生成は明示的に拒否する。
- GEMV は `1<=M<=64`、`N%4=0`、`K%64=0`。16 byte の重み語が 64 要素を運び、K 昇順に縮約する。
  行ブロックは既存の 256 要素上限から最大 4 行になる。それ以外は packed GEMM を使う。
  embedding も packed のまま行を読む。
- shared initializer は、既存の形・格納・消費席・device・run リース・寿命の検査を維持する。
  INT2 の scale も同じ行軸で借用する。MTP の I8 借用を INT2 へ変更する意味ではない。
- exporter の低レベル writer / reader と固定整数 pack / unpack を対応させる。
  通常の `write_model(weight_dtype=...)` に自動 INT2 量子化は追加しない。
  固定 packed 重みを直接受ける変換入口は QAT recipe の実装単位で追加する。

E4B の公式 config と重み実体を確認した結果、PLE は **E2B が INT4、E4B が INT2** だった。
PLE はグラフ外でホストが読む sidecar なので、上記 IR の行ごと scale とは別に、層ごとの block scale を保持する。
PLE の実装単位はこの 2 格納を扱う。全配列の照合では、両モデルとも head と token embedding の整数列・scale が一致した。
実体の出所と照合記録は新しい出力先の `e2b/e4b-download.json` と `e2b/e4b-census.json`。

## 追記 2 — 固定 SRQ op の契約（2026-09-11）

IR に `static_quantize` を追加する。入力・出力は f32 各 1 本、shape は不変（スカラ・空テンソルも可）。
属性 `scale` は必須で、非負・有限かつ厳密に f32 で表せる JSON 数値だけを受理する。
保存済み scale を暗黙に丸め直さず、負値・非有限値・f32 範囲外・丸めを要する値を拒否する。
既存 op の属性規則は変更しない。旧 runtime は未知 op として拒否する。

- scale > 0 の意味は、f32 除算 `x/scale` → 最近接の偶数への整数丸め → `[-128,127]` への飽和 → f32 乗算。
  scale=0（-0 を含む）は入力のビット列をそのまま返す。
- 符号付きゼロを保つ。±Inf は整数上限・下限へ飽和してから scale を掛ける。
  scale > 0 の NaN は符号・payload を保ち quiet bit を立てる。乗算結果の overflow は ±Inf を返す。
  非正規化数を含め、GPU の浮動小数点除算の許容誤差や flush-to-zero に依存しない。
- CPU 参照は f32 除算と偶数丸めを直接計算する。GPU は正の絶対値のビット順序と境界表の二分探索で求める。
  128 境界と 129 出力値を host で一度作り、count と恒等フラグを含めた 1,040 byte の uniform で運ぶ。
  最悪 8 比較。値の演算を GPU の f32 除算・乗算へ戻さない。
- 境界 `j+0.5` の f32 丸め区間を考慮する。j が偶数なら上端を厳密に超す最小 f32、
  奇数なら下端以上の最小 f32 が次の整数段階の始点になる。host の境界積は f64 で厳密に表せる。
  scale によって境界が重なる場合も、そのまま単調な表として扱う。
- 専用の op kind と単独カーネルにする。既存 unary の融合には入れず、codegen キーには scale を入れない。
  uniform は既存の内容アドレスキャッシュと Session の寿命を使う。パラメータのバイト数は既存の
  メモリ見積りと同様に除外項目で、実際の確保量は diagnostics の weights に含まれる。
- exporter は `karume::static_quantize` を原子的に保持し、同名 IR op へ変換する。
  汎用 core の eager 実装は Torch の演算だけを使い、Transformers への依存は追加しない。
  QAT recipe が固定 scale を指定する。既存グラフへ SRQ を自動挿入しない。

検収は公式 CPU の境界両隣・ランダム値・特殊値を保存した fixture、独立 CPU 参照、実 GPU の u32 比較、
共有した TS/Python 契約表、codegen snapshot、torch.export から生成した新しい tiny golden で行う。
既存の golden と数値許容差は変更しない。

## 追記 3 — 固定量子化 writer の入口（2026-09-11）

汎用 exporter に `FixedQuantizedWeight(dtype, packed, scale)` を公開し、`write_model` と
`publish_model` に任意の `fixed_weights` mapping を追加する。既存呼び出しの動作は変更しない。

- mapping のキーは initializer の tensor キー（FQN）。対応する `tensors` は同じ論理形の
  f32/meta テンソルとし、meta の集合と固定 mapping の集合は完全一致を要求する。
  実 f32 値と固定 payload の両方を渡して一方を黙って無視する入力は拒否する。
- 対象は正の rank 2 `[N,K]`、linear / embedding の重みとしてだけ消費される initializer。
  graph 出力、重み以外の消費、同じキーへの複数宣言は拒否する。
- 固定 dtype は I2 / I4 / I8。packed は CPU の連続配置で、I2 / I4 は U8 の `[N,K/4]` /
  `[N,K/2]`、I8 は I8 の `[N,K]`。全符号値を保持し、量子化・再 pack・f32 展開を行わない。
  I2 は K が 16 の倍数、scale は I2 / I8 が F32 `[N,1]`。
  I4 は F32 `[N,groups]` から group_size を導き、K を割り切る 16 以上の 2 冪を要求する。
- 固定 mapping 内の混成 I2 / I4 / I8 は受理する。従来の `weight_dtype`（f32 以外）、
  `weight_scales`、`weight_dtype_overrides` との同時指定は拒否する。
  空 mapping は追加の値指定を持たず、従来の経路を変えない。
- scale の生成キー、同居規則、行分割、整列、reader 検証、公開時の据え替えは既存処理を使う。
  行分割では packed payload を同じ先頭軸の行範囲で切り、親の f32 実体を作らない。
  graph 宣言と入力テンソルは変更しない。固定値の上流由来の検証はモデル recipe が担当する。

検収は全 byte 値を含む payload と異なる scale の分割前後一致、reader の受理、曖昧な入力の拒否、
公開失敗時の既存成果物保持、既存自動量子化の回帰検査で行う。QAT recipe はこの入口を使う後続の単位。

## 追記 4 — packed PLE sidecar（2026-09-11）

PLE の索引と shard metadata に schema 2 を追加し、`storage: "i2" | "i4"` を必須とする。
schema 1 / I8 の索引・復元・読み方は維持する。旧 reader は schema 2 を拒否する。

- 値は token-major、safetensors `values` の論理 shape は `[rows,layers,dim]`、dtype は I2 / I4。
  dim は正の 16 の倍数。値の byte 数はそれぞれ `rows*layers*dim/4` / `/2`。
  各 byte の下位 bit から `q+2` / `q+8` を詰め、全符号値を使う。
- `scales` は F32 `[rows,layers]`。index と shard metadata の schema / storage / token 範囲 / shape を突合する。
  グラフ外の sidecar なので IR initializer の rank 2 制限は適用しない。
- 復元は既存と同じ二段 f32 乗算 `(q*scale)*embedScale`。全量・常駐・区間読みに共通の復元処理を使う。
  byte 予算・行 offset は packed の実 byte 数、返す shape と重複 id の複写は論理要素数で計算する。
- `Gemma4PleIndex.storage` の欠如は旧 I8 を意味する。schema 1 に storage を付ける入力、schema 2 の storage 欠如、
  未知 schema / dtype は拒否する。中断・排他・寿命の既存契約は変更しない。

公式 E2B INT4 / E4B INT2 の probe との全ビット一致、およびモデルに依存しない独立 Torch fixture で
全量・行読取・常駐・重複 token・境界を検査する。既存の未知 schema 拒否テストは未対応版を 2 から 3 へ進め、
拒否そのものは保つ。正式 recipe はこの sidecar を後続単位で書く。

## 追記 5 — 固定 QAT recipe と数値比較の扱い（2026-09-11）

recipe `gemma4_qat` は公式 mobile Transformers 形式だけを読み、text / PLE / tokenizer を
`gemma4-qat-<model>-product` 系列へ保存する。実行は `python -m gemma4_qat.export`。
既存 Gemma の wrapper、attention の登録、states 変換、tokenizer compile を再利用する。

- 元の固定整数・scale は唯一の値の供給元。trace だけに shape-only の重みを使い、変換後は
  f32/meta 宣言と固定 writer へ渡す。head と embedding の共有は全 bytes 一致を確認してから行う。
  分割後の全固定 payload と PLE を元 bytes と照合し、出所・config・tokenizer と同じ据え替え単位に置く。
- 変換先に通常 Gemma の量子化や drafter は足さない。配布の quant は既存文法に沿う `i4` とし、
  label / description に固定混成 INT2 / INT4 / INT8 と SRQ を明示する。
- 初期の既定 capacity は 128、chunkLength は 32、trace 上限は 128。上限を provenance に保存し、
  dist はその値を検査して使う。モデルの位置上限は公式 config から導く。長文の品質・速度は未検収。
- QAT の RoPE は周波数のべき乗・逆数・位置積を段ごとに f32 へ丸める専用入口を使う。
  通常 Gemma の f64 計算契約は変えない。三角関数は host の Math.cos / Math.sin 後に f32 格納。
  上流 Torch の三角関数や各 GPU の縮約との全ビット一致は保証しない。

CPU / GPU の最初の差は、同一入力の行列縮約が SRQ の丸め境界をまたぐことまで実測で帰属した。
SRQ 単体の CPU / GPU ビット一致は保たれている。8種類の短い生成比較では Deno と Chrome の
トークン列は全件一致したが、公式 CPU との完全一致は E2B 6件 / E4B 4件。
この結果を品質全般の合格とは扱わず、family / CLI に実験段階の制約として明示する。
既存の許容差・期待値は変更しない。詳細と生データは research の該当節を参照する。

## 追記 6 — 共通パイプラインと対話 CLI（2026-09-11）

公開入口 `Gemma4QatPipeline.fromPretrained` / `fromAssets` とサブパス `@karume/models/gemma4-qat`
を追加する。既存の Gemma 本体は非公開の共通基底へ移し、通常 / QAT の factory だけを薄く分ける。
会話・sequence・取得・見積り・解放の実装は同じものを使う。通常 Gemma の既定値・数値・MTP は維持する。

- QAT は `gemma4-qat/1`、model `e2b` / `e4b` だけを受理する。token embedding の INT2、
  共有 head、固定混成格納、量子化 linear の前後の SRQ、モデル別の PLE 格納を構築前に検査する。
  通常 Gemma の manifest を QAT として読み替えない。QAT の MTP 構築指定は型と実行時で拒否する。
- 派生入力の RoPE は QAT 専用の f32 段丸めへ切り替える。既存の `gemma4RopeInputs` と
  既存 golden の値は変えない。新たに `gemma4QatRopeInputs` を公開する。
- 会話 API と資産・設定の型は共通の Gemma 型を使う。QAT の取得オプションは model を2種類に絞り、
  speculative を受けない。公開済み通常 Gemma の API を撤去・改名しない。
- `demo:gemma4-qat` を追加し、既存 Gemma CLI を共通 runner に移す。履歴・KV 継続・reset・中断・
  進捗・統計は既存の動作を使う。QAT は既定 `models/karume-gemma4-qat/`、model 省略時は manifest の既定、
  最大生成64 token。通常 Gemma の既定256 tokenは維持する。QAT の実験段階の制約を起動時に表示する。

検収は通常 Gemma の全体回帰、QAT E2B/E4B の Deno/Chrome 生成比較、既定chunk32と比較用64、
複数ターンの KV 再利用、中断・反復の早期終了後の復帰、会話ごとの状態バッファ解放で行う。
公式 CPU/GPU の縮約差の評価は追記5のままとし、期待値や既存許容差を緩めない。
