# ブラウザ推論ランタイム勢力図と karume の位置

> 時点スナップショット（2026-08-16 調査・情報源は文中に明記）。以降の記述は調査時点の
> 一次ソース／訓練知識に基づく。バージョン・活発さの類は数か月で陳腐化する前提で読むこと。
> 用途: ①ORT Web 対比ベンチ慣行（backlog）の測定条件の根拠 ②「有界論理 extent の席予約」
> （autoregressive 波の shape 不変条件 ADR）の根拠資料 ③デファクトとの差の「意図 / 未実装」
> 台帳。

出典表記の約束:

- `W-1 §n` / `W-2 §n` / `CX-4 §n` / `CX-5 §n` / `CX-3` = 本調査の内部ノート（レビュー作業
  ディレクトリ配下・**git 追跡外**）の該当節。導出の由来を示す作業ラベルであり、恒久参照は
  §6 の一次ソース URL と本文の `file:line` / ADR 番号が持つ。W-1 / W-2 = Web 一次ソース調査、
  CX-4 / CX-5 / CX-3 = Codex への諮問（ブラウズ不可・訓練知識 + リポ実読）。
- 確度ラベル: **[一次]** = Web の一次ソースで裏が取れている / **[二次]** = ブログ等の二次ソース /
  **[訓練知識]** = Codex の訓練知識由来（ブラウズ検証なし） / **[要再確認]** = ノート内で
  矛盾・未確定と明記された項目。
- karume 側の記述のうち ADR 0001（ONNX 非目標）/ 0002（Web 標準 API のみ）/ 0058（数値
  opt-in）/ 0054:61（irodori 実測）の帰属は現物突合済み。その他の `file:line` は調査時点の
  綴りで、ドリフトしうる。

---

## 1. 勢力図概観

### 1.1 横断表

| ランタイム                        | 実行方式                                                                                                     | shape                                                       | 量子化                                                                | 活発さ（2026-08 時点）                                          | 特記                                                                                                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ONNX Runtime Web                  | ONNX グラフ + EP 分割（WASM / WebGPU / WebNN / 旧 WebGL）                                                    | 動的次元可（記号 or 未知）・EP 側で実形状に特殊化           | QDQ 推奨・QOperator 併存・opset 21 で int4/uint4 **[一次]**（W-1 §5） | 高                                                              | **WebGL と JSEP の廃止を正式発表**（PR/issue #29716・2026-08-04 merged）。JSEP → native WebGPU EP（C++/Dawn）へ既定を透過切替、移行期は `/jsep` を逃げ道として残す **[一次]**（W-1 §1）                            |
| transformers.js                   | ORT Web への薄いラッパー（自前カーネルなし）                                                                 | ORT Web 準拠                                                | ORT 経由（INT8 等）                                                   | 高（170+ アーキテクチャ・16.3k★）                               | v4 系で「C++ 製 WebGPU バックエンド」に刷新との記載。**リリース日表示が矛盾しており [要再確認]**（W-2 §1）                                                                                                         |
| WebLLM / MLC                      | TVM Unity（Relax）で**事前コンパイル**、TVMjs が実行                                                         | symbolic shape（batch / seq のみ可変、他は静的寄り）        | q4f16_1 / q4f32_1 等の低ビット群                                      | 中（18.6k★だがリリースページが 2025-04 で停滞・**[要再確認]**） | 低ビット LLM 配信では現状最も成熟（W-2 §2）                                                                                                                                                                        |
| LiteRT.js                         | `.tflite` を直接実行する JS バインディング。CPU=XNNPACK(WASM) / GPU=ML Drift over WebGPU / NPU=WebNN（実験） | `.tflite` の shape signature（動的次元可・resize で再確保） | 形式に組み込まれたアフィン量子化（per-tensor / per-axis）             | **新規（2026-07-09 リリース）**                                 | Google 公式 **[一次]**。**パーシャルデリゲート非対応**（モデル単位 all-or-nothing）・WASM CPU メモリ 2GB 上限でロード失敗し得る、と公式が明記（W-1 §7）                                                            |
| llama.cpp WebGPU（通称 LlamaWeb） | テンプレート化 GPU カーネル（半手書き）                                                                      | **静的メモリプランニング**                                  | 4 重みフォーマット（GGUF 資産を継承）                                 | **高・新規（2025-2026 急伸）**                                  | 16 デバイス / 8 ベンダー / 10 モデルで評価。既存ブラウザ内フレームワーク比でメモリ −29〜33%・デコードスループット +45〜69%（arXiv アブストラクト・**[要再確認]**＝本文未読、比較対象の同定に曖昧さあり）（W-2 §8） |
| wonnx                             | ONNX → WGSL codegen（Rust）                                                                                  | 静的必須（限定 shape 推論・定数畳み込み）                   | 未完成（整数行列積は非対応）                                          | **終了（2025-05-07 アーカイブ）[一次]**                         | 「ONNX を WGSL へ落とす」路線の先行例。2026 時点では勢力図の終了組（W-2 §4）                                                                                                                                       |
| ratchet                           | 手書き寄り（推定）                                                                                           | 未確認                                                      | Q8 等を自称                                                           | 中・小規模（768★・定量未確認）                                  | Whisper / Phi 2,3 / Moondream（W-2 §5）                                                                                                                                                                            |
| candle                            | Rust 推論、Wasm 対応                                                                                         | 未確認                                                      | llama.cpp 形式                                                        | 高（20.9k★）                                                    | **WebGPU 経由の GPU 加速かは未確認 [要再確認]**（W-2 §6）                                                                                                                                                          |
| burn (wgpu)                       | Rust マルチバックエンド                                                                                      | 未確認                                                      | 取得範囲では明記なし（消極的事実）                                    | 高（15.8k★）                                                    | wgpu 利用時の再帰的型評価コンパイルエラー注記あり（W-2 §7）                                                                                                                                                        |
| WebNN（層としては競合/補完）      | W3C 標準 API                                                                                                 | —                                                           | —                                                                     | CR 更新 2026-01-22 **[一次]**                                   | Chrome 147-149 は Origin Trial のみ・Safari/Firefox 未出荷＝**プロダクション未対応**、実用化は 2027 頃との見立て **[二次]**（W-2 §8）                                                                              |

### 1.2 読み取れる構図

2026 時点で系譜は 4 本に整理できる。

1. **標準グラフ + EP 分割**（ORT Web、および LiteRT のデリゲート）。可搬性と網羅性で勝つが、
   バックエンドごとの対応差を利用者が背負う。ORT 側はここで**実装の世代交代**が進行中——
   JSEP（JS 側で WebGPU を叩く）から native WebGPU EP（C++/Dawn）への移行が公式に決まった
   （W-1 §1）。JSEP は Safari/WebKit 26 で深刻な CPU/メモリ増を踏む報告（issue #26827）もあり、
   移行は性能・安定性の両面で意味がある。
2. **事前コンパイル**（WebLLM/MLC）。可搬性をコンパイル時に解決し、実行時のグラフ分割を持たない。
3. **ネイティブエンジンの移植**（llama.cpp WebGPU）。既存の量子化資産とモデル対応をそのまま
   Web へ持ち込む第三の道。W-2 §8 は「2025-2026 で最も注目すべき新規参入」と位置づける
   （この評価自体は調査者の解釈を含む）。
4. **ブラウザ専用に一から書いたランタイム**（wonnx / ratchet / candle / burn、そして karume）。
   wonnx のアーカイブはこの層の淘汰を示すが、karume は「ONNX 互換」を目標に置いていない点で
   wonnx とは賭けが違う（後述 §2 の軸 1・軸 8）。

**注意**: W-1 の版数情報は矛盾を抱えている。PyPI の履歴は 1.28.0＝2026-07-25 等の 2026 年日付を
返すのに対し、GitHub Releases を WebFetch すると同じ版数列に 2024 年の日付が付いて返る。
W-1 は WebFetch 内部要約の誤変換を疑いつつ**未確定**としており、本書もそのまま不確かとして扱う
（W-1 §「現行バージョン」）。同様に「WebGPU Plugin EP」というスタンドアロン配布物の存在も
**[要再確認]**（native 向けか onnxruntime-web 向けかも未確定・W-1 §1）。

ブラウザ側 WebGPU 対応表もソース間で食い違う（Firefox のフラグ有無・Safari 26 が既定かどうか）。
W-1 §1 / W-2 §8 の双方が矛盾を明記しているので、**本書では断定しない**。

---

## 2. Karume vs デファクトの軸別比較

CX-5 の 10 軸を土台にする。各行の最終列に、**その差が「意図した違い（守るべき賭け）」なのか
「単なる未実装（いずれ要る）」なのか**の仕分けを付す。仕分けは CX-5 §10 の分類に、backlog /
ADR の裏付けを添えたもの。

| #  | 軸                      | Karume                                                                                                                                                    | デファクト側の代表挙動                                                                                                                   | 仕分け                                                                                                                                                                            |
| -- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | IR / モデル形式         | 1 graph = 1 safetensors（`__metadata__.karume_ir` に strict SSA JSON）+ manifest v2 がリポ単位を目録化                                                    | ONNX ProtoBuf（外部データ可）/ `.tflite` FlatBuffer / MLC は AOT コンパイル済みライブラリ                                                | **意図** — 交換フォーマットではなく「exporter が制御する検査可能なデプロイ IR」。ONNX 入力は明示的な非目標（ADR 0001）。CX-5 §1                                                   |
| 2  | shape モデル            | 実行前に全ノードの出力形状が解決可能。次元は整数 or 1 記号のアフィン式（`T` / `2T` / `8T+2`）、導出は厳密解のみ                                           | ONNX は記号/未知次元を許し実行時解決。MLC は prefill/decode を別コンパイル。LiteRT は resize で再確保                                    | **意図（ただし境界は要拡張）** — 静的解決が prepared plan / full-write / resident backing の土台。ただし data-dependent 出力（topk / 動的枝刈り）は**未実装で、いずれ要る**（§4） |
| 3  | 実行・メモリ            | binding キー付き prepared plan（LRU 4）+ per-run arena + hot hit で固定 transient と bind group 焼き込み + resident tensor & 単一フェンスの batch enqueue | ORT は memory pattern + BFC arena（ただし **WebGPU では memory pattern を無効化**）。LiteRT は interpreter arena                         | **意図** — 「planner を持つ」ことではなく、plan 署名・full-write・resident 寿命・フェンス位置を**公開契約**にしたことが差（CX-5 §3）                                              |
| 4  | 量子化                  | 論理 dtype（f32/i32/bool）と物理格納（f16 / 対称 per-channel int8）の分離。計算精度（f16 タイル・i8a8 linear/attention）は別軸の明示 opt-in（ADR 0058）   | ONNX は QDQ / QOperator をグラフに載せる。MLC は低ビット群フォーマットが配信次元                                                         | **意図（分離設計）+ 未実装（低ビット）** — packed int4 級の格納と shard 単位ロードは未着手で、LLM 配信には必須（CX-5 §4・backlog）                                                |
| 5  | 決定性                  | 同一 codegen キー → バイト同一 WGSL。既定は reference 数値経路、非ビット同一の最適化は**名前付き opt-in + 誤差帯テスト + 実行センサス**が必要             | ORT は `use_deterministic_compute` があるが**クロス EP/デバイスのビット同一は非保証**。テスト許容値も一律ではない                        | **意図（最大の差別化）** — ただし SHA golden は参照環境ゲートであってクロスデバイス保証ではない（karume 自身が limitations に明記）。CX-5 §5 / CX-4 §6                            |
| 6  | 配信・キャッシュ        | 生成された `karume/2` manifest、HF revision を commit SHA へ一度だけ解決、全ファイルの size/SHA-256 検証、自己修復キャッシュ、quant→session マッピング    | ORT/LiteRT はモデル配信をアプリ任せ。transformers.js は HF ネイティブだが**規約ベースの sidecar 探索**でハッシュ完備の manifest ではない | **意図** — 「単一ファイル vs 複数ファイル」ではなく「規約探索 vs 生成された整合性固定のリポ契約」が本当の対比（CX-5 §6）                                                          |
| 7  | pipeline / tokenizer 層 | `@karume/models` に分離（pipeline・tokenizer・メディア前処理・モデル固有ホスト糊）。pipeline 呼び出しは直列化して VRAM モデルを保つ                       | ORT はランタイムのみ。transformers.js が breadth の基準。WebLLM は chat/generation 特化                                                  | **未実装（breadth）** — 設計の分離自体は意図だが、対応パイプライン数は明確な劣位。EmbeddingGemma の tokenizer/pipeline と attention mask 結線は backlog（CX-5 §7）                |
| 8  | バックエンド抽象        | WebGPU/WGSL のみ。EP 概念なし・WASM フォールバックなし・WebNN 未対応（IR は将来の NPU 経路に余地を残す）                                                  | ORT は EP でグラフ分割（WASM/WebGPU/WebNN/WebGL）。LiteRT はデリゲート（ただし **LiteRT.js はパーシャル非対応**・W-1 §7）                | **意図** — 可搬性の賭けは「Deno + ブラウザ・Web 標準 API のみ・ランタイム依存ゼロ」（ADR 0002）。CPU フォールバックや NPU 到達を望むなら初めて戦略的欠落になる（CX-5 §8）         |
| 9  | op カバレッジ           | 需要主導の「入場門」5 段（export 消滅 → Core ATen → 非 core atom → 非 core molecule → runtime fusion）。Core ATen 160 完走は明示的に非目標                | ORT は広い opset 互換が目標（ただし各 Web EP はサブセット）。LiteRT は広い builtin 語彙                                                  | **意図** — 「全 160」ではなく「対象ワークロードが要求する primitive」が正しい欠落定義（CX-5 §9）                                                                                  |
| 10 | 総括                    | 厳格 IR 契約 + reference-first 数値 + 予測可能な prepared/resident 実行 + HF ネイティブ配信                                                               | 各社それぞれの breadth / 標準化 / 成熟                                                                                                   | —                                                                                                                                                                                 |

### 軸ごとの補足（短く）

**軸 1・8（IR とバックエンド）** — この 2 つは連動した 1 つの賭けである。ONNX を受けない以上
EP 抽象は不要になり、EP を持たない以上「グラフの一部だけ CPU に落ちる」という ORT 特有の
分断コスト（CX-4 §2: 小さな交互パーティションは孤立 GPU op を失うより悪いことがある）も
発生しない。代償は WebGPU 不在環境でゼロになること。

**軸 2（shape）** — 「静的形状」という言い方は不正確で、正しくは
_「ホスト/入力既知のアフィン記号を許した、実行前に解決可能な形状」_（CX-5 §2）。
ここが §4 の主題。

**軸 3（実行・メモリ）** — ORT 側の対応物は「memory planner」単体ではなく、
memory pattern エントリ + EP のプログラム/エンジンキャッシュ + 保持バッファ +（可能なら）
graph capture の**合成物**である（CX-4「Contrast with Karume」）。karume の prepared plan は
それを 1 つの一貫したキーにまとめている点が構造的な差。

**軸 5（決定性）** — ORT のテスト許容値（float32 で atol 1e-5 / rtol 1e-4、ONNX backend runner は
1e-3 級、CUDA は rtol 0.017 級）は**テストハーネスの合格閾値であって精度 SLA ではない**
（CX-4 §6 **[訓練知識]**）。karume の「数値変更には明示的な席と証拠が要る」という立て方は、
「どこでもビット同一」より守れる主張であり、維持する価値がある（CX-5 §5）。

**軸 7・4（breadth と低ビット）** — この 2 つが「単なる未実装」の中心。CX-5 §10 が挙げる
次波の構造的欠落は、生成状態（`GenerationContext` + 固定容量 KV）、prefill/decode の形状分離、
packed int4 級格納と shard 単位ロード、メモリ admission 見積り、汎用 multi-output・GPU argmax・
静的 k の top-k、causal/GQA attention。

### 2.5 native WebGPU EP との収斂とポジショニング検証（同日追補・W-3 / W-4）

「karume の機構は ORT が次期採用する native WebGPU EP と部分的に同型では？」という指摘
（2026-08-16）を受けた一次ソース検証。結論: **実行機構（軸 3）の収斂は確定。ポジショニングの
重心は フットプリント / i64 集約 / 決定性 / 可変形状 に置くのが実証に耐える形**。

**収斂が確定した点** — program cache キーは uniform の**値**を含めない
（`onnxruntime/core/providers/webgpu/program_cache_key.cc` **[一次]**）・graph capture の
bind group は capture 時に 1 度だけ生成し replay で使い回す（`webgpu_context.cc` **[一次]** =
karume の焼き込みと同型）・BufferManager は 26 段の bucketed pool。JSEP 比の「実行機構が良い」
という優位は移行完了後に消える前提で戦略を立てる。

**収斂しない点（一次ソース）**:

- capture の成立条件は「静的形状 + 全計算カーネルが同一 EP」・違反は**モデル初期化エラー**
  （env-flags 公式 docs **[一次]**）。可変形状ワークロード（可変長 TTS 等）は capture の外。
- replay は 16 dispatch ごとに submit を分割（`webgpu_context.h:419`）。karume の batch も
  submit は enqueue ごとに出すためフェンス**待ち**の比較は要実測 — ここは優劣を断定しない。
- **capture と int64 の OR 結線**: `enable_int64_ = enable_graph_capture || enable_int64` —
  capture を有効化すると int64 演算が強制的に GPU（= i32 切り詰め）実行になり 2^31 超で精度を
  失う。migration doc §6 が「既定 parity からの明示的例外」として文書化 **[一次]**。
  つまり ORT では**高速経路と i64 正しさがトレードオフ**になっている。

**フットプリント実測**（W-4 の unpkg 実測 + 本リポの esbuild 実測・いずれも 2026-08-16）:

| 対象                                                                                          | raw     | gzip        |
| --------------------------------------------------------------------------------------------- | ------- | ----------- |
| karume engine（`@karume/runtime` mod.ts 丸ごと・esbuild --bundle --minify）                   | 251.9KB | **67.8KB**  |
| karume フルスタック（runtime+hub+models barrel 全部・`@hdae/{fetch-cache,yomi}` は external） | ~400KB  | **112.3KB** |
| onnxruntime-web@1.27.0 WebGPU 経路（ort.webgpu.min.js + ort-wasm-simd-threaded.jsep.wasm）    | 26.9MB  | **6.33MB**  |
| transformers.js v4.2.0（JS 層のみ・ORT wasm は別途）                                          | —       | 118〜210KB  |
| WebLLM 0.2.84（index.js 単体・モデル別 wasm 別途）                                            | —       | 2.13MB      |
| LiteRT.js core 2.5.3（wasm 合算）                                                             | 8.93MB  | 2.72MB      |

gzip 比で **フルスタックでも約 56 倍・engine 単体なら約 93 倍**。wasm が大きい理由は ORT
メンテナ自身が「CPU 全 op 実装の同梱」と明言（Discussion #24161 **[一次]**）— native EP 化で
縮む性質のものではない（なお「Dawn 同梱で肥大する」という向きの推測も根拠なし — ブラウザ向け
配布物にその形は存在しない・#26216 は node 向け未実装要望）。付随する独立のデプロイ優位:
①wasm 不使用 = CSP の `'wasm-unsafe-eval'` 不要 ②threaded wasm の SharedArrayBuffer =
COOP/COEP 強制が無い ③全部読める TS + 副作用ゼロ = tree-shaking と監査。

**i64 主張の正確な形** — WGSL に 64bit 整数が無いのは仕様事実（W3C TR・拡張提案
gpuweb#5152 未採用 **[一次]**）。ただしこれは **WebGPU を使う全ランタイム共通の制約**であり、
「WebGPU 単独だから i64 問題が無い」という言い方は不正確。正確な差別化はこう:

> **i64 の潰し込みを実行時に散在させず、エクスポータ境界 1 箇所へ値域検査つきで集約し、
> ランタイムと IR に i64 をそもそも存在させない**（ADR 0009。i32 中間演算のラップは
> limitations に明文化 — 検査は境界のみ）。

対照の実証: ORT WebGPU EP は shape/indices の i64→u32 キャストが**カーネルに散在**し、その
一般化を求める issue #28029 が open のまま・v1.28.0 では int64/int32 切り詰めによる
out-of-bounds read（Gather 系）の修正実績・そして上記の「capture = 切り詰め強制」の結線。
wonnx も README で同じ制約とオーバーフローリスクを自認。WebNN は仕様に int64 があるが
CoreML バックエンドが非対応（#21401）で逃げ道にならない **[一次]**。

---

## 3. 動的形状の実設計（R2 の根拠資料）

以下は CX-4（Codex・**[訓練知識]**、ただし ORT ソース/ドキュメントの行リンク付き）の凝縮。
「ORT は動的形状を 1 枚の特殊化済み実行計画では解いていない」が結論。

### 3.1 ORT の階層（CX-4 §1）

| 層                                | 何を担うか                                                                          | 形状変化で何が起きるか                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| セッション時のグラフ/所有権プラン | 値の所有・寿命・再利用可能性の確立、カーネル実体化                                  | **変わらない**（セッション時決定）                                                                             |
| 実行時 shape 計算                 | 各カーネルが実入力から出力形状を算出                                                | 毎 run 発生                                                                                                    |
| memory pattern                    | feed 形状シグネチャをキーに、確保/解放列を記録し 1 個の大確保へのオフセット表を作る | **新シグネチャはミス**＝トレースと通常確保が走る。実サイズが記録ブロックと合わなければ個別確保へフォールバック |
| BFC arena                         | 実際のアロケータ（best-fit bin・分割/結合・領域を保持）                             | サイズクラスが増え、保持メモリの high-water mark が上がる                                                      |
| EP 固有キャッシュ                 | WebGPU のプログラム/パイプライン、TensorRT のエンジン                               | **ここが支配的コスト**になり得る                                                                               |
| graph capture（任意）             | 固定形状での記録・再生                                                              | 形状が変われば無効                                                                                             |

重要な非対称: 次元を変えても **`GetCapability` の再実行・グラフ再分割・カーネル再選択は起きない**
（CX-4 §1「What varying shapes actually costs」）。起きるのは下位キャッシュのミスである。

### 3.2 EP 契約（CX-4 §2）

- EP は登録優先度順に `GetCapability` を貪欲に呼ばれ、claim した領域を得る。CPU EP が最後。
  **コストベースの大域分割最適化器ではない**。
- 融合サブグラフは `Compile` で EP 側の実装（`NodeComputeInfo` クロージャ）に置き換わり、
  その**内側**（内部テンソル・ワークスペース・コマンド発行・融合戦略）は ORT から不透明。
- **CPU フォールバックは実行時の例外回復ではない**。セッション初期化時の配置決定であり、
  WebGPU カーネルが実行中に失敗しても巻き戻して CPU で再試行はしない。
- graph capture は協調動作。ORT が適格性を検証し、EP が記録済みコマンドと保持リソースを持つ。
  **1 セッション 1 capture EP・制御フロー不可**。

### 3.3 TensorRT の optimization profiles（CX-4 §4）— 有界動的性の正典

- profile = `min` / `opt` / `max` の 3 つ組。`min`/`max` が受理範囲、`opt` が tactic 評価点。
  ORT は動的入力全てに 3 つの指定を要求する **[一次ドキュメント参照あり]**。
- profile は **EP 私有のコンパイル設定**。グラフ分割は固定したまま、「高価な特殊化の契約」だけを
  明示化する設計。範囲外の形状は失敗し、**CPU へ再分割・再試行はしない**。
- 明示 profile がないと、TensorRT EP は初回実行の形状から狭い profile を導出し、範囲外が来たら
  拡張して**再ビルド**する——便利だが深刻なレイテンシスパイクを生む。
- 逆に profile を広く取りすぎると `opt` から離れた点の性能が落ち、tactic 選択が制約され、
  エンジン/ワークスペースが膨らむ **[訓練知識・Medium]**。
- **重要**: profile は ORT コアのメモリプランナに最大サイズを供給しない。境界テンソルは
  依然として毎 run の具体形状で確保される。

### 3.4 WebGPU EP の動的形状ペナルティ（CX-4 §3）

- **ORT 汎用の memory pattern は WebGPU では無効化されている**（Web セッション設定・native
  セッション経路の双方）。安定形状の見返りは、offset 表ではなく
  **バッファプールのヒット・パイプラインのヒット・（可能なら）capture** の 3 つ。
- native WebGPU EP は `BufferManager` プール（storage は bucketed / uniform は simple が既定）。
  JSEP は不透明 ID → `GPUBuffer` の対応表 + freelist で、投入済み GPU 仕事が参照しなくなるまで
  再利用/破棄を遅延。**形状が散ると bucket が増え、保持 GPU メモリが上がる**。
- パイプラインキャッシュのキーは**依存性宣言に従う**。native は「型のみ」「rank のみ」「全形状」を
  宣言でき、uniform で渡す動的値は新パイプラインを強制しない。**JSEP は既定でより保守的に
  形状依存**なので、同じ形状変化でも JSEP の方が artifact が増えやすい。
- bind group には汎用の永続キャッシュがない。**graph capture が bind group と参照バッファを
  保持する唯一の場**であり、その代償が「中間テンソルを生かし続ける」こと。
- capture の再生は、記録済みの pipeline/bind group/dispatch を**新しいコマンドエンコーダへ
  詰め直す**実装。つまり capture が節約するのは WebGPU のコマンドエンコード自体より、
  ORT/JS/カーネルの準備・確保・bind group 構築のオーバーヘッド。
- 小カーネルが多いモデルでは、capture 不成立と Wasm/JS/WebGPU 境界の往復が実 GPU 演算を
  上回り得る **[訓練知識・Medium]**。

### 3.5 karume の「静的物理格納 + 有界論理 extent の席予約」との対応

CX-3（R2 設計相談）の結論は、恒久不変条件を**「静的な物理格納 + 固定 rank の有界テンソル」**まで
に絞り、「全ての論理形状がホスト既知」を恒久化しないこと。その 8 規則は上記 ORT/TensorRT の
実設計と直接対応している——規則 1（`shape` は物理容量のまま）と規則 7（メモリ admission は
容量で課金、論理サイズは診断のみ）は TensorRT の `max` プロファイルが果たす役割を、
karume では**アロケーション契約そのもの**に持ち込む形（ORT では profile が境界テンソルの確保に
効かない、という CX-4 §4 の欠落を構造的に埋める）。規則 4（実効 extent を計画キャッシュの鍵に
入れない）は、CX-4 §3 の「uniform で渡す動的値はパイプラインを再生成させない」という native
WebGPU EP のキー設計と同型であり、同時に JSEP の保守的キーが招く artifact 増殖を避ける選択でも
ある。規則 8（上限超え・動的 rank はホスト介在のグラフ分割へ）は、TensorRT が範囲外形状で
「CPU へ再分割せず失敗する」ことに対応する karume 側の逃げ道の置き場を、executor の**外**に
固定する宣言に当たる。残る差は full-write 不変条件で、compact-prefix を入れるなら
「論理 extent 内は全書き・物理尾部は未規定かつ公開 readback へ非露出」への置換が要る（CX-3 §
full-write）——尾部ゼロ埋めは O(capacity) 帯域で本末転倒、という点も CX-3 が明言している。

---

## 4. ベンチ慣行への含意

### 4.1 ORT Web と比較するときの条件

過去の実測（`docs/research/2026-08-11-embeddinggemma-ort-comparison.md`）では ORT Web WebGPU を
既定設定で回している。W-1 §3・§4 の一次情報を踏まえると、**ORT 側に有利な設定を明示的に
入れた上での比較**と、**素の設定での比較**は別物として両方記録するのが正しい。

| ノブ                                                 | 何をするか                                                                                              | 出典                   | ベンチでの扱い                                                                                                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `enableGraphCapture`                                 | **WebGPU EP 専用**。静的形状 + 全カーネルが登録 EP 上で走る場合、初回実行のカーネル列を記録して以降再生 | W-1 §3 **[一次]**      | ORT 側の CPU 側コマンド準備コストを落とす。karume の prepared plan hit（bind group 焼き込み）と**対応する条件**なので、これを切ったまま比較すると karume 有利に歪む      |
| `freeDimensionOverrides`                             | 名前付き自由次元に固定値を与える                                                                        | W-1 §3・§4 **[一次]**  | **必須**。固定しないと graph capture の前提（静的形状）が立たない。ただし公式は「固定化が必ず速くなるとは限らずモデル次第」と注記しており、無効時との両方を測るのが安全  |
| IO binding / `preferredOutputLocation: 'gpu-buffer'` | 入力を GPU バッファから直接渡す / 出力を GPU に留める                                                   | W-1 §3 **[一次]**      | 最後の GPU→CPU コピーを消す。**中間確保・shape 計算・パイプラインミス・capture 制約は消さない**（CX-4 §3）ので、これを入れても「ORT がホスト費用ゼロになる」わけではない |
| EP 分断の確認                                        | `session.disable_cpu_ep_fallback=1`                                                                     | CX-4 §2 **[訓練知識]** | ORT 側が実は一部 CPU に落ちていた、という比較の破綻を検出できる。落ちていれば「WebGPU 同士の比較」ではない                                                               |

落とし穴 2 点:

- `freeDimensionOverrides` の**引数名を誤るとサイレントに最適化が無効化される**（issue #22300・
  WebNN 文脈だが指定 API は共通）**[一次]**（W-1 §2）。設定したつもりで効いていない比較は
  実際に起きる。
- **JSEP か native WebGPU EP か**を記録すること。両者はキャッシュ設計が異なる（JSEP は
  形状依存キーに保守的・CX-4 §3）ので、同じ「ORT Web WebGPU」でもビルド世代で結果が変わる。
  移行が進行中である以上、**測定時のパッケージ版数とビルド種別はベンチの必須メタデータ**
  （W-1 §1）。なお、その版数体系自体が W-1 で **[要再確認]** 扱いである点に注意。

### 4.2 比較候補モデルの注意点

**EmbeddingGemma-300m** — 既に実測資産がある（同一機・同一入力 5 ケース）。継承すべき注意:

- **グラフの由来が違う**。ORT 側は onnx-community の変換時最適化済みグラフ（com.microsoft
  contrib op 入り）、karume 側は `torch.export` 直系。差は「ランタイム実装の差」だけには帰属
  できない（既存 doc §4）。加えて ORT fp32 の torch 逸脱（cos 0.9975〜0.99999）は CPU/wasm/
  WebGPU の 3 EP でほぼ同一値＝**EP ではなくグラフ由来**という観察が既にある（同 §2）。
  数値忠実度で比較するなら `graphOptimizationLevel: disabled` での切り分けが open のまま。
- 短文域は karume 側がホスト固定費支配（3 波後 wall 54ms に対し GPU 実時間 ~15ms）。
  つまり**この帯域の比較は「カーネルの比較」ではなくホスト経路の比較**である。ORT Web が
  ~20ms で回る事実は「Chrome + Dawn のホスト経路は同じ仕事を安くこなせる」ことの実証、と
  既存 doc が整理している。§4.1 の graph capture を ORT 側に入れると**この差はさらに開く**
  はずで、それを承知で測る。
- B=1 のみ・スレッド既定 1 点・OS スケジューリング未制御という既存の制約はそのまま残る。

**Irodori** — 対照的に GPU 律速側の代表（ADR 0054 で voice-clone 素の生成壁 8.593 → 4.896/4.860s・
ほぼ GPU 律速へ — `docs/decisions/0054-resident-loop-and-fence.md:61`）。注意点:

- ORT 比較の**土俵に乗せる前段が重い**。ONNX 変換済みの等価グラフが公開されていない限り、
  比較には自前変換が要り、その時点で「変換時最適化の差」という §4.1 と同じ交絡が入る。
- 多段パイプライン（複数セッション）である以上、ORT 側は**セッション間の GPU 常駐**を
  IO binding で明示しないと readback を挟む。karume の resident tensor + 単一フェンス
  （ADR 0054）と条件を揃えるには `preferredOutputLocation: 'gpu-buffer'` が必須。
- 逆に、GPU 律速帯では karume 側のホスト固定費が相対的に消えるため、**カーネル品質そのものの
  比較**として EmbeddingGemma より素直な題材になる。

---

## 5. 確度の注記

### 5.1 由来の区別

| 由来                            | 該当ノート  | 性格                                                                                                                                                    |
| ------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Web 一次ソース（URL 付き）      | W-1 / W-2   | 公式 docs・GitHub issue/PR・arXiv・W3C・Google 公式ブログ。ただし **WebFetch 経由の要約**であり、ページ実体を目視していない箇所がある（特に日付・星数） |
| Codex（ブラウズ不可・訓練知識） | CX-4 / CX-3 | ORT のソース行リンクを伴うが、**リンク先を実際に開いた検証はしていない**。CX-4 自身が High/Medium/Low の自己ラベルを付けており、本書はそれを継承        |
| Codex（リポ実読）               | CX-5        | karume 側の記述は `file:line` 付きでリポを実読。**外部ランタイム側の記述のみ訓練知識**（CX-5 冒頭が明記）                                               |

### 5.2 低確度として扱う主張（断定しない）

1. **onnxruntime の版数と日付** — PyPI（2026 年）と GitHub Releases（2024 年）で矛盾。
   W-1 が未確定と明記。`gh api repos/microsoft/onnxruntime/releases` での直接裏取りが未了。
2. **「WebGPU Plugin EP」** — スタンドアロン配布物としての存在・対象（native か web か）が未確定。
3. **JSEP → native WebGPU EP の切替完了バージョン** — 移行計画は確定事実だが、既定切替が
   どの版で着地するかは特定できていない。関連 issue #31683 は 403 で本文未取得。
4. **ブラウザの WebGPU 対応表** — Firefox のフラグ有無、Safari 26 が既定かどうかがソース間で不一致。
5. **transformers.js v4.0.0 のリリース日** — v3 より後発のはずが表示日付が矛盾。
6. **WebLLM の最終リリース日**（2025-04-24 表示）— 調査日から見て古すぎる。タグ化されていない
   main 更新の可能性。
7. **candle の WebGPU 対応可否** — Wasm 対応は確実だが GPU 加速の経路が未確認。
8. **burn の量子化サポート** — 「無い」の確認ではなく「取得範囲で見当たらなかった」だけ。
9. **llama.cpp WebGPU の定量値**（メモリ −29〜33% / デコード +45〜69%）— arXiv アブストラクトのみ。
   比較対象が WebLLM/transformers.js かネイティブ llama.cpp か曖昧。
10. **ORT の代表ベンチ数値** — 公式一次ソースの明確なベンチ記事は特定できず。3〜5 倍等の数字は
    個人ブログ（二次）。BenchmarkXPRT のブログが次の調査起点。
11. **Web（WebGPU EP）での INT4/QDQ カーネル対応範囲** — 一次ソース未特定。W4A16/W4A8 の
    具体的な組み合わせ可否は不明。
12. **`tune-performance` の公式 URL** — 今回引けたのは個人フォークのミラー。正規 URL 未確認。

### 5.3 次調査の起点

W-1 §「未確認・要追加調査」と W-2 §10 をそのまま引き継ぐ。優先度が高いのは
①版数の直接裏取り（`gh api`）②`webmachinelearning.github.io/webnn-status/` の取得
③Web での INT4 カーネル対応 ④BenchmarkXPRT 記事 ⑤llama.cpp WebGPU 論文本文。

---

## 6. 主要一次ソース URL（恒久参照用）

内部ノートは git 追跡外のため、本文が依拠する一次ソースをここに固定する（取得日 2026-08-16）。

- ORT: JSEP / WebGL 廃止と native WebGPU EP への移行計画 —
  <https://github.com/microsoft/onnxruntime/issues/29716>
- ORT: WebGPU EP チュートリアル（graph capture / preferredOutputLocation） —
  <https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/web/ep-webgpu.md>
- ORT: IO binding — <https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html>
  （正規 URL は [要再確認] — §5.2-12）
- ORT: freeDimensionOverrides のサイレント失敗 —
  <https://github.com/microsoft/onnxruntime/issues/22300>
- ONNX: int4/uint4（opset 21） — <https://onnx.ai/onnx/technical/int4.html>
- LiteRT.js リリース（Google 公式） —
  <https://developers.googleblog.com/litertjs-googles-high-performance-web-ai-inference/>
- wonnx（アーカイブ済み） — <https://github.com/webonnx/wonnx>
- WebLLM — <https://github.com/mlc-ai/web-llm> / TVM Relax —
  <https://tvm.apache.org/docs/deep_dive/relax/learning.html>
- transformers.js v3 — <https://huggingface.co/blog/transformersjs-v3> /
  <https://github.com/huggingface/transformers.js>
- ratchet — <https://github.com/huggingface/ratchet> / candle —
  <https://github.com/huggingface/candle> / burn — <https://github.com/tracel-ai/burn>
- llama.cpp WebGPU（論文アブストラクト） — <https://arxiv.org/abs/2605.20706>
- WebNN CR — <https://www.w3.org/TR/webnn/>
- onnxruntime PyPI 履歴（版数の一次候補・矛盾あり — §5.2-1） —
  <https://pypi.org/project/onnxruntime/#history>
