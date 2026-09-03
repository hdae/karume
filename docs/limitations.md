# limitations — by-design の制約

意図的な設計制約の台帳（バグは known-issues.md、設計判断の経緯は decisions/）。
「いつ解けるか」が決まっているものは解除予定を添える。

## 同一 device 上の複数 Session の run は直列実行される

WebGPU の errorScope は device 単位の LIFO スタックで「誰のスコープか」の概念が無く、
並行に重なると失敗の誤帰属と沈黙全 0 を生む（実測記録は
[research/2026-08-01-m0-review.md](research/2026-08-01-m0-review.md)）。このため run 1 本の
GPU 操作全体（エンコード〜readback〜アリーナ破棄）を device 単位ロックで直列化している
（`GpuContext.withScopeLock` の doc にトレードオフの根拠）。並行スループットが必要な場合は
device（= `acquireGpu()`）を分けること。解除は WebGPU 側に「スコープ付き submit」相当が
入らない限り予定なし。

## 同一 `GenerationContext` への run は 1 本ずつ（未決着のまま 2 本目を発行すると拒否される）

`Session.run` は「戻り Promise を await せずに次を発行してよい」を公開契約に持つが、**第 3 引数
`generation` の `context` が同じ run だけは例外**で、未決着の 1 本目がある間の 2 本目は
`ExecutionError` で落ちる（`GenerationContext` の run リースが 1 本上限）。理由は沈黙誤値:
2 本目は 1 本目が進めた論理長で uniform と dispatch を組む一方、位置を運ぶ入力（gemma4 なら
RoPE の cos / sin 派生入力 — ホストが位置から組む）は**呼び出し側が発行時に組んだ普通のグラフ
入力**でランタイムは中身を見ない — KV の論理長は正しいまま位置だけが静かにずれる。並行させたい生成は **context を分ける**
（別 context 同士と、generation 面を持たない 1-shot run の並行発行は従来どおり通る）。

なお第 3 引数の 2 欄（`context` / `queryLength`）は `inputs` / `bindings` / `copyOutputs` と同じく
**発行の同期区間で写し取る**ので、発行後にそのオブジェクトを書き換えても実行中の run には効かない。

## `BatchScope.finish()` は区間の最初の**ホスト側**失敗も投げる

`Session.enqueue` を非 await で積む区間では、enqueue 本体（ホスト側の検査・レシピ構築）の失敗は
戻り Promise にしか出ない — 握っていなければ未処理拒否として抜け、区間は「dispatch が 1 本
少ないまま成功」で決着していた。現在は `BatchScope` が区間の**最初の 1 件**を記録し、
`finish()` がそれを投げる（errorScope 側の失敗もあるときは errorScope 側を投げて記録を `cause` に
載せる。2 件目以降は捨てる — 派生失敗が並ぶと根因が読めなくなるため）。

同じ失敗は enqueue の戻り Promise にも従来どおり出る（1 つの事実が 2 経路で見えるのは `run` と
同じ）。`finish()` から `ExecutionError` 等が出るのは 0.7.0 までに対する**破壊的な挙動変更**。

## `generateGreedy` は公開面から外れた（parity 検収用の内部ヘルパへ格下げ）

`@karume/models` の barrel（`mod.ts`）が出していた `generateGreedy` と `GreedySpec` は
**削除した**（ADR [0083](decisions/0083-generation-api-surface.md) 決定 9 — 0.7.0 までの公開面に
対する**破壊的変更**）。実装は `src/generation/greedy.ts` に残っており、既存 golden の token 列を
突き合わせる parity 門は経路として生き続けるが、**パッケージの外からは import できない**。

理由は面の性格で、この関数は「固定 token id 列での検収」のための決定的な 1 本である
（sampling も EOS 停止も**載せない** MUST がある）。静的配線とリクエストが 1 つの型に同居し、
`GenerationContext` の寿命が呼び手へ漏れるので、多ターンの chat をこの面の上に組むと
「直前 assistant の最後の token が落ちる」事故を消費側で再生産する（ADR 0083 Context / 決定 4）。
「当面は公開のまま残して次の breaking 波で外す」は採らなかった。

代替は生成 API 波の `GenerationSequence`（ADR 0083 決定 1〜5 — `AsyncIterable<GenerationEvent>` /
`AbortSignal` / 多ターン）で、ホスト側 sampling は `src/generation/sampler.ts`（同 決定 7〜8）。
sequence が出るまでの間、この面に相当するものは公開されていない。

## 生成 API は公開前に 7 点変わった（ADR 0083 / 0084 の初版記述に対する破壊的変更）

生成 API（`GenerationSequence` / `Gemma4Pipeline`）は **JSR にはまだ出ていない**（`@karume/models`
の最新公開は 0.7.0 で gemma を含まない）。ただし ADR・examples・デモ台本は波の途中の面で書かれて
いるので、公開面レビュー（2026-08-31 / 第 2 波 2026-09-02）の消化で変わった 7 点をここに残す。
追記の正本は ADR
[0083](decisions/0083-generation-api-surface.md) / [0084](decisions/0084-gemma-tokenizer-chat.md)。

- **`dispose()` 後の `generate` は同期に throw する**（従来は最初の反復で `GenerationContext` 側の
  汎用文言になり、真因〈自分が dispose した〉が読み取れなかった）。寿命の検査だけが同期で、予算の
  検査（位置表・容量）は従来どおり反復側で `GenerationCapacityError` になる。
- **`GenerationStop` に `tokens` 欄が増えた**（そのターンが生成した token 数・`eos` の停止 token も
  1 個）。`done` の一致アサーションを書いていたコードは欄の追加ぶん追随が要る。
- **`GenerationCapacityError` のコンストラクタが 2 引数**になり、構造化欄（`constraint` /
  `pastLength` / `promptLength` / `requestedNewTokens` / `limit` / `maxNewTokens`）を持つ。
- **公開 `GenerationProgram` から配線欄が消えた**（グラフ入出力の名前・記号束縛・`derivedInputs` は
  内部型 `GenerationWiring` へ分離）。残るのは `chunkLength` / `capacity` / `maxPosition` /
  `vocabSize` / `stopTokens` で、面は凍結・`stopTokens` は凍結コピーである（消費側の `sort()` /
  `length = 0` が生成ループの停止集合を書き換える口を塞いだ）。
- **`SamplerSpec.logitBias` が `Map` からタプルの配列へ**（`readonly [token, bias][]` —
  2026-09-02）。`new Map([[id, bias]])` を書いていたコードは `[[id, bias]]` へ書き換える。`Map` は
  `JSON.stringify` が `{}` へ潰すので、設定として保存・復元した指定が**黙って空の bias** になって
  いた。あわせて**同じ token を 2 度書いたら `RangeError`**（`Map` の後勝ちの畳み込みは無くなった）。
- **`Gemma4Pipeline.sampler` → `defaultSampler` へ改名**し、`Gemma4PipelineConfig.sampler` の型が
  `SamplerSpec` から `Gemma4DefaultSampler`（`temperature` / `topK` / `topP` の 3 欄必須）へ縮小
  された。あわせて `fromAssets` も `fromPretrained` と同じ門（未知キー・値域・
  `chunkLength ≤ maxChunkLength`・`chunkLength ≤ capacity ≤ maxPosition`）をバイト列を開く前に
  通すので、**不正な宣言は
  3.7GiB のロードが終わる前に落ちる**（従来は初回 `generate` で初めて `RangeError` になった）。
- **停止理由の union に `stop-token` が増え、`Gemma4ChatStream.done` の型が `Gemma4ChatStop` へ
  広がった**（2026-09-02 — 要求ごとの停止条件）。`switch (stop.reason)` を網羅で書いていたコードは
  枝の追加ぶん追随が要る（`chat` 側はさらに `stop-string` が増える）。あわせて
  `Gemma4ChatStream` に `text()` が生えたので、**1 つのストリームは 1 通りにしか消費できない**
  （反復と `text()` の併用・2 度の反復は同期に throw する — 従来は 2 度目の `for await` が
  黙って 0 反復で終わっていた）。

## 停止**文字列**で止められるのは `chat` だけ（`sequence()` は token で止める）

`Gemma4ChatOptions.stopStrings`（復号後の本文に現れたら止める）は **chat 層のノブ**である。
低レベル面（`Gemma4Pipeline.sequence()` → `GenerationSequence.generate`）には無く、そちらで
止めるなら `GenerationRequest.stopTokens`（token id・配布形の EOS 集合との和集合）を使う。
`GenerationSequence` は token id しか扱わない（tokenizer も配布形も知らない）ので、文字列の判定は
復号器を持つ層にしか置けない — ADR [0083](decisions/0083-generation-api-surface.md) 追記
2026-09-02 の 2 層である。自分で `sequence()` を回しながら文字列で止めたい場合は、復号した本文を
自分で見て `break` する（会話は「成功した run のぶんだけ」進み、最後の token は未 commit の
frontier に残る = `chat` の停止文字列と同じ状態になる）。

## params キャッシュは Session 寿命で無界（by-design）

params バッファの内容アドレスキャッシュ（`RecipeBuilder.#writeParams`）には追い出しが無く、
解放は `Session.dispose()` だけ。記号次元を持つグラフを「run ごとに違う束縛」で回すと、run の
たびに（ノード種ぶんの）小バッファが新規確保され、二度と当たらないまま Session の寿命
いっぱい積み上がる（例外も警告も出ず、`diagnostics().weights.allocCount` が単調増加するだけ）。
1 本は数十バイト程度なので、可変長 TTS / 系列長可変の埋め込みでも実害の記録は無いが、上限は
無い。

- 追い出しを LRU 化しない理由: params の実体は生きている導出済み計画（prepared plan）が
  **直参照で畳み込んでいる**ため、追い出し = 破棄にすると破棄済みバッファを掴む。安全にやるには
  参照計数という別の簿記が要る（実需が出るまで作らない）。
- 回避策: 可変 shape を長時間回す用途では `diagnostics().weights.allocCount` の伸びを見て
  Session を切り直すこと。
- 無界であること自体は `packages/runtime/tests/gpu_params_cache_test.ts` が門にしている
  （追い出しを入れるとこの門が赤くなる）。

## `__proto__` という名前をオブジェクトリテラルで渡せない（JS の記法の制約）

Karume 側の Record は null プロトタイプで `"__proto__"` キーを保全するが、**呼び出し側**が
`{ __proto__: tensor }` と書くと、JS の文法上それはプロパティ定義ではなく [[Prototype]] 指定に
なり、キーが生成されないまま Karume に届かない。`"__proto__"` を入出力名・シンボル名に使う
モデルでは、計算キー `{ ["__proto__"]: tensor }` か `Object.assign(Object.create(null), ...)` を
使うこと。Karume 側では検出できない（届いていないことを知り得ない）。

## 非有限値（NaN / ±Infinity）は検査しない

入力・重みの非有限値はコストの理由で検査せず、そのまま演算に流れる。**GPU と CPU 参照の
伝播一致が担保されるのは下の NOTE に列挙した op だけ**で、全称の保証ではない。付随して
`amax` / `amin` の縮約 identity は ±F32_MAX で、全要素が -Infinity の行の `amax` は
-Infinity ではなく -F32_MAX を返す。非有限値を含むモデルを扱う場合は呼び出し側で事前検査
すること。

NOTE: 伝播一致の保証は `clamp` / `clamp_min` / `relu` / `amax` / `amin` に加え、
**softmax / safe_softmax / attention の行統計（融合・states 形とも）**もビット列 NaN 判定
（`nan_max`）で担保している（2026-08-31 の v2 で統一 — それ以前は WGSL の `max` が仕様レベルで
NaN を落とすため、全要素 NaN の行が safe 系の空行判定に化けて厳密 0 になり NaN が黙って
消えていた）。機序と裁定は
[decisions/0020](decisions/0020-nan-propagation-bitwise.md) と
[decisions/0044](decisions/0044-runtime-attention-mask.md) 追記 2026-08-31。

## 融合 attention の加算 mask は −inf を**値として**運ぶ（Finite Math Assumption 依存）

exporter が焼く加算 mask 定数は帯外を literal −Infinity で表し（`masked_fill` の有限
sentinel −3.4028e38 とは別方式）、融合 attention の f32 経路はそれをそのまま加算して
`exp(S−m)` で 0 に落とす。WGSL の Finite Math Assumption（実装は実行中に ∞ / NaN が現れない
と仮定してよい — §15.7.2）の下では、∞ を値として運ぶこと自体が実装の自由度に晒されるが、
実測（Vulkan / Metal の出荷資産）では期待どおり動作している。全列が −inf の行（全マスク行）
は **attention_stats v2 の空行ガードで出力 0 の正規入力**（2026-08-31 — それ以前は
`1/0` = indeterminate の契約違反だった。[decisions/0044](decisions/0044-runtime-attention-mask.md)
追記）。

## `attentionCompute: "a8"` の PV は attention 重みを 1/127 格子で量子化する（1:254 打ち切り）

a8 の ③PV は `qP = round(127·exp(S−m))` の固定格子（clamp なし・行最大からの相対）で、
`S − m < −ln 254 ≈ −5.537` のスコアは**厳密に 0** に落ちる — 行内の最大値の 1/254 未満の
attention 重みは消える（f32 経路は保持する）。落ちた質量は逆数和 `inv` が補償しない設計
（[decisions/0044](decisions/0044-runtime-attention-mask.md) の契約系・opt-in 席）。CPU 参照も
同一格子なので突合はこの打ち切りを検出しない — 精度影響の裁定は E2E の実測 tolerance が担う。

## w8a8（`linearCompute: "a8"`）では非有限値の伝播粒度は同じだが Inf の符号が f32 経路と一致しない

活性 per-token i8 の実行経路（opt-in — 設計は
[research/2026-08-03-dp4a-w8a8-design.md](research/2026-08-03-dp4a-w8a8-design.md)）では、
整数内積そのものに非有限値の概念が無く、伝播は**行 scale `xs[row]` 経由**で成立する:

- **NaN**: 行内に 1 つでも NaN があると `xs[row] = NaN`（ビット列判定で伝播 — ADR 0020）に
  なり、`out = f32(acc)·(xs·wscale) + bias` が行の全列で NaN になる。f32 経路も行全体を縮約
  するので**伝播の粒度は劣化しない**。
- **±Inf**: 行内の Inf は `xs[row] = +Inf` になる（絶対値最大なので**符号は残らない**）。
  出力は `f32(acc)·(+Inf·wscale) + bias` で、`acc` と `wscale` の符号で ±Inf / NaN が決まる —
  **f32 経路が出す Inf の符号とは一致しない**。ADR 0020 の不変条件（「NaN が黙って消えない」）
  は満たすが、符号まで同じであることは保証しない。

量子化値 `xq` そのものも、行 scale が非有限のときは `vec4<i32>(NaN)` が不定値になるため
**契約の外**（突合の対象は行 scale と最終出力だけ）。非有限値を含むモデルは、この経路でも
呼び出し側の事前検査が前提になる（上の「非有限値は検査しない」と同じ立場）。

加えて、**WGSL の f32 除算は正しく丸められない**（仕様の許容 2.5 ULP。本機実測でも `a / b` が
IEEE 除算と 200,000 サンプル中 55,605 件で 1 ULP 割れた）。`q = round(x / s)` は `x/s` が
半整数の近傍に来る要素で **±1 段揺れうる** — GPU と CPU 参照の atol=0 突合は「丸め境界から
十分離れたデータ」でのみ成立する契約で、実モデルではこの ±1 段が数値差として乗る
（E2E の tolerance を w8a8 系列で取り直す理由の 1 つ）。scale の側は `amax · (1/127)` の
**乗算**で作るので厳密に一致する。

## conv2d（groups==1）は dispatch 上限で fail loudly になる（Hout·Wout > 8,388,480）

implicit GEMM（[decisions/0024](decisions/0024-conv2d-implicit-gemm.md)）は 1 workgroup =
1 出力タイルで縮退できないため、n = Hout·Wout のタイル数（tileN=128）が 1 次元の dispatch 上限
（65,535）を超える形 — **Hout·Wout > 8,388,480**（正方出力なら 2,897² 以上）— は
`DispatchLimitError` になる。旧・直接カーネル（grid-stride）は走れた形なので意図的な機能の
絞りだが、沈黙誤値ではなく例外で止まる。**この閾値は既定 GEMM 幾何から導かれる**（n の辺 =
`gemmTileN` = regN·wgX）ので、既定を動かすと一緒に動く — 現行値は M128N128 になった `d0afc22`
（2026-08-10）以降のもので、ADR 0024 が書いた「2048² で上限を 1 超える」は辺 64 前提の旧値。
解消は動的解像度 recon の「固定タイル VAE」（研究記録
[2026-08-03-dynres-vae-tiling](research/2026-08-03-dynres-vae-tiling.md)）side で行う想定。
枚数の固定は `packages/runtime/tests/codegen_dispatch_test.ts`。

## BiRefNet 系の配布形は 1024² だけ（2048² は未実測・組み立てが拒否する）

`karume dist --pipeline birefnet` が受け付けるのは入力 `[1,3,1024,1024]` で焼かれた系列だけで、
それ以外の解像度は `DistError` で落ちる（`tools/export-recipes/birefnet/distribution.py` の
`BIREFNET_RESOLUTION`）。export 段（`python -m birefnet.export --resolution 2048`）は通るので、
系列を作ること自体はできる。

配らないのは実行段が未実測だから: 中間テンソルが `[1, 192, 2048, 2048]` = 3.22GB になる。
（かつては「上の conv2d dispatch 上限に decoder の 1×1 conv が当たる」も理由に挙げていたが、
既定幾何が M128N128 になった `d0afc22` 以降は 2048² の n タイルが 32,768 で上限の内側 —
残る理由は資源側だけ。）
本家（同梱 `handler.py` の General-HR）の推論解像度は 2048² なので、**上流と同じ設定では
ない**点は配布形の制約として明示しておく。回避策は入れていない（実測して判断する側の話）。

## conv1d（groups==1）も同じ dispatch 上限で fail loudly になる（Lout ≈ 8.39M）

conv1d の implicit GEMM（[decisions/0053](decisions/0053-conv1d-implicit-gemm.md)）も
1 workgroup = 1 出力タイルで、n = Lout のタイル数（tileN=128）が 65,535 を超える形 —
**Lout > 8,388,480 サンプル ≈ 175 秒 @48kHz** — は `DispatchLimitError` になる。実運用では
dacvae decoder はタイル分割（halo 8）で、SBV2 は運用上限（pipelineConfig）で先に区切られる
ため到達しないが、カーネル直呼びの長尺形は例外で止まる（沈黙誤値ではない）。

## Deno では GPUBuffer の総確保がドライバ申告予算の 97% で頭打ちになる（外部制約）

Deno の WebGPU 実装は wgpu の**メモリ予算しきい値**をハードコードしている
（`ext/webgpu/lib.rs` の `MemoryBudgetThresholds { for_resource_creation: 97,
for_device_loss: 99 }` — wgpu 自体の既定は「しきい値なし」）。判定式は
「`heap 使用量 + 要求サイズ ≥ heapBudget × 97%`」（wgpu-hal vulkan の
`error_if_would_oom_on_resource_allocation`）で、`heapBudget` は
`VK_EXT_memory_budget` でドライバが申告する動的な値。**バッファ本数・アップロード経路・
flush 頻度をどう変えても天井は動かない**（判定に入るのは合計量と要求サイズだけ —
6 通りの書き方で同値だった実測と整合）。97% 線を越える `createBuffer` は OOM を返し、
99% 線は submit / poll のたびに判定されて **device 消失**になる（`createSession` 途中の
消失が同一プロセスの後続を道連れにする実測の説明もこれ）。

- 実測（RTX 3080 Ti 12,288MiB・Deno 2.9.4 / wgpu 29.0.1・2026-08-03 時点）: 総確保 7,280MiB で
  頭打ち（= 0.97 × 逆算 budget ≈ 7.5GiB。**budget が総量の 61% しか申告されない理由は未特定**
  — ドライバ実装依存の可能性が高い）。f32 の Anima DiT（重み 7,465MiB）はこのため載らず、
  該当 E2E 2 本は ignored（f16 3.7GiB / i8 1.9GiB 系列は影響なし）。
- **天井は固定値ではなく時点値**: 判定に入る `heapBudget` はドライバが動的に申告するため、
  同じ機械でも測るたびに動く。同機での 2026-08-08 の再実測は **11,136〜11,264MiB**
  （`tools/diag/hold-vram.ts` で 256MiB 刻みに確保 → `not enough memory left`。同日 2 回の
  サンプルでこの幅が出た）。上の 7,280MiB と矛盾するのではなく、どちらもその時点の申告
  budget の 97% を映しているだけ。
- **天井付近では OOM ではなく device 消失になる境界がある**: 97% 線は `createBuffer` の OOM、
  99% 線は submit / poll のたびに判定される **device lost**。圧が少し高いだけで症状が
  「間欠の device 消失」に化けるので、確保失敗だけを見張っても取りこぼす。
- f16-1024 の PNG 門は**初回 run で瞬間 8,391MiB**（定常 5,723MiB）まで上がり、天井
  11,136MiB に対する余裕が 2.7GiB しか残らなかった。瞬間ピークの正体は重み staging の二重
  計上で、Session 生成時に submit を 1 回入れて解消済み（実測ピーク 5,723MiB・余裕 5.4GiB。
  機序と実測は [research/2026-08-08](research/2026-08-08-vram-oom-misreport.md)）。
- **Karume 側では回避不能**: しきい値を変える wgpu API はあるが Deno がハードコードして
  おり、環境変数も CLI フラグも無い。実質の選択肢は Deno のパッチビルドか、より大きい
  VRAM の機械で回すことだけ。上限は VRAM 総量ではなく「ドライバ申告 budget」比例なので、
  24GiB 機なら f32 DiT が載る見込み（未実証）。
- 出所の file:line・既知の上流報告（denoland/deno#35195 等）・逆算の根拠は
  [research/2026-08-03-wgpu-memory-ceiling.md](research/2026-08-03-wgpu-memory-ceiling.md)。

## bf16 格納は宣言のみ受理・実行は fail loudly

IR v1 の格納スキーマとしては受理するが、実行経路が無く `createSession` が capability 不足と
して全件列挙で拒否する。設計は [decisions/0006](decisions/0006-quantization.md) で確定済み。

group 量子化（w4）は [decisions/0069](decisions/0069-packed-w4-storage.md) で**解禁・実行
経路も実装済み**（2026-08-18）— 格納 dtype `i4`（packed 4bit・K 方向 group の対称量子化）。
制約は `group_size` が 2 冪かつ 16 以上・量子化軸（先頭次元を行とした平坦行長 — rank2 では
最終次元そのもの）が `group_size` で割り切れること・scale companion（F32・rank2
`[行数, 行長/group_size]`）が必須の 3 点。**適格は f16 / i8 より狭く「消費が linear /
embedding / conv1d（`groups == 1`）の重みスロットのみ」**（0069 決定 5 と追記 6 / 7 —
conv2d / conv_transpose1d への追補は需要が出た op から）で、適格外はロード時 CPU 展開
（VRAM 削減ゼロ）。`i4` 以外の格納 dtype に
付いた `storage.group_size` は解禁後も `非対応 group 量子化` として落ちる（黙って
per-channel として読むと沈黙誤値になるため）。

**f16 は 2026-08-03 に解禁**（[decisions/0018](decisions/0018-f16-weight-execution.md)）、
**i8（per-channel symmetric int8）も同日に解禁**
（[decisions/0019](decisions/0019-i8-weight-execution.md)）— ただしどちらも VRAM が縮むのは
**適格な重みスロットだけ**。適格外（bias / norm 系 weight / その他の op / 混在消費 /
消費ゼロ）はロード時に CPU で f32 展開され、**VRAM 削減はゼロ**で縮むのは配信サイズのみ。
内訳は `Session.diagnostics().storage` で観測する（ADR 0006 の常設診断。i8 の
`residentCompressedBytes` には scale バッファのバイト数も入る）。

i8 は `storage.scale`（重みと同 rank の keepdim broadcast 形・F32）の**宣言が必須**で、
チャネル軸は出力チャネル（`conv_transpose1d` だけ軸 1）。scale の欠落・dtype 違い・
broadcast できない形・実テンソルとの名前衝突・チャネル軸違いはすべてロード時に落ちる。

## hub: 相 1（streaming prefetch）は CacheStorage 必須で fail loud

`prefetchAssets` の相 1 はバイト列を手元に持たない面なので、CacheStorage が使えない環境で
素 fetch へ縮退する余地が構造的に無く、fail loud で落ちる（ADR
[0070](decisions/0070-shard-loading-admission.md) 追記）。`onCacheError` 診断が届くのは相 2 のみ。

## `estimateSessionMemory` の `workspaceBytes` は近似（非勘定は `unaccounted` が列挙する）

`scenarios[].workspaceBytes` は融合**前**のノード列に対する生存区間シミュレーションで、実構築が
畳む / 割る形は勘定に入らない。非勘定は `unaccounted` 欄が逐語で列挙する — ①融合が畳んで消す
中間と行ブロック分割の一時 ②states 形 attention のノード内一時（スコア S と行統計 — 融合の
成立に依存せず必ず出る）③params バッファ ④`queue.writeBuffer` の実装 staging ⑤シナリオ切替の窓。
可否の最終門はこれまでどおり out-of-memory errorScope で、`peakAccountedBytes` も名前どおり
「勘定に入れた分のピーク」= 上限保証ではない。

## `fromAssets`（全量面）に分割配布形を渡すと、全 shard がホスト RAM に同時常駐する

`fromPretrained` で読める配布形は `fromAssets` でも読める（取得キーが `<役割>[i]` の shard 列は
連結せず、`fromPretrained` と同じ shard 逐次面へ流す — `packages/models/src/hub/components.ts` の
`assetComponentOpener`）。ただし全量面の入口は**取得済みバイト列の Record** なので、呼び出し側が
Record を組んだ時点で**全 shard がホスト RAM へ同時に載っている**。R1 が獲得した
「ホスト RAM に載るのは今の 1 本だけ」という性質が成り立つのは取得層を通る面
（`fromPretrained` / `prefetchAssets` + `streamAssets`）だけで、全量面で縮むのは**単一
ArrayBuffer の大きさ**（Chromium の壁 — 下記）であって合計の常駐量ではない。ローカルの
デバッグ・`examples/` 用途を想定した面という位置づけは変わらない。

添字の欠番や、素キー（`transformer`）と分割キー（`transformer[0]`）の混在は **shard の語を含む
診断**で fail loudly（黙って片方を採らない）。

## ホスト RAM ピークの係数 1 化はローカル取得元だけ（HF 経路は取得層の buffer 確保に依存）

逐次面の器の使い回し（ADR [0070](decisions/0070-shard-loading-admission.md) 追記 2026-09-02）で、
ロード時のホスト RAM ピークは「約 0.45GB + 最大 shard 1 本」になった（Linux 実測: anima f16 1GiB shard
4,069 → 1,402 MiB）。ただし効くのは**取得元が器へ読める経路**（Deno の `denoDirectory`・
`readFileInto` を実装したディレクトリアダプター）だけで、**HF 取得元（ブラウザの通常経路）は従来の
係数（約 3 × 最大 shard + 1GB）のまま** — 取得層 `@hdae/fetch-cache` がキャッシュから読むたびに
自前で buffer を確保するため、hub からは器を渡せない。取得層に「与えられた buffer へ読む」口が
入った時点で hub 側は同じ `into` を渡すだけで揃う（外部リポの起票）。

## MoE は全エキスパート VRAM 常駐が前提（エキスパート単位の動的ロード/退避はしない）

by-design（2026-08-31 裁定）。MoE モデルの VRAM 予算は **active でなく総パラメータ**で組む —
「使う expert だけロードする」動的常駐は提供しない。根拠は 3 層の構造衝突（実測記録 =
[research/2026-08-31-freetoken-moe-over-arraybuffer.md](research/2026-08-31-freetoken-moe-over-arraybuffer.md)）:
①`ShardValidator.finish()` は宣言された全 initializer の存在を要求する（全量/逐次 2 面で
共有された唯一の門 — 穴を開けると受理集合の一本化が崩れる）②重み常駐は Session 構築時
1 回組みで退避/再ロードの席が無い ③IR v1 に値依存の実行選択が無く（op-vocabulary の意図的
保留・`topk` も static-k）、MoE は全 expert を計算して gate で畳む dense 展開でしか書けない。
さらに外部要因として、expert キャッシュ系の先行手法（FreeToken ほか）が前提にする
「device 起動の host メモリ転送」が WebGPU に存在しない。復活条件つきの再検討席は
[backlog](backlog.md) parked。

## 要素数が奇数の f16 テンソル・I8 テンソルは safetensors 上の並び順に制約がある

裁定の正本は ADR [0063](decisions/0063-safetensors-physical-layout.md)。リーダはデータ節の
「隙間なし・整列単位（I4 は先頭 4 バイト）整列」を要求し（違反は `SafetensorsError`）、
エクスポータは書き出し順
「F32 → I32 → I4 → 偶数要素 F16 → 奇数要素 F16 → I8」+ `verify.assert_reader_layout` で保証する。
HF の `safe_open` は整列違反を読めてしまうので、そちらを通すだけでは検出できない。

## 格納 dtype `I4` は safetensors の方言（公式パーサは読めない）

packed int4（ADR [0069](decisions/0069-packed-w4-storage.md)）は safetensors ヘッダに
dtype `I4` を書くが、これは**公式仕様に無い語**で、公式 safetensors ライブラリは該当
テンソルを含むファイルを拒否する（実測 2026-09-01・safetensors 0.8.0 — 受理 dtype は
`F4` / `F6_*` / F8 系まで拡張済みだが int4 系は無い）。sub-byte の機構自体（論理 shape +
bit 幅からのバイト長導出）は公式 `F4` と同型で、非互換は dtype 名の 1 点。

- 影響: i4 テンソルを含む shard は **karume のリーダ / exporter 以外では読めない**
  （HF へのアップロード・DL は内容非依存なので通る）。i4 を含まない shard は公式互換のまま。
- 対象: i4 系列を含む配布形すべて（例: anima `w4` 系・irodori `w4`・sbv2 `w8-bert4` の
  text_encoder）。モデルカードへの注記は次リリース一括のカード再生成で入れる。
- 公式仕様への追随提案（upstream への I4/U4 追加要望）はしない — 2026-09-01 ユーザー裁定。
  目指す方向が違うため、将来は**別形式 / 独自形式への移行**を検討する（器は次の
  manifest format 変更時 — [backlog](backlog.md) の次波計画）。

## gather / embedding の範囲外添字は GPU で NaN 汚染になる（例外にならない）

裁定の正本は ADR [0061](decisions/0061-index-oob-semantics.md)。契約は「添字は範囲内」。
違反時、GPU カーネルは該当要素（embedding は該当行）にだけ quiet NaN を書いて実行を継続し、
NaN 伝播（ADR 0020）で必ず表面化する。CPU 参照は範囲外で throw（意図的非対称）。

## f32 → i32 cast の値域外・NaN は未定義

裁定の正本は ADR [0062](decisions/0062-f32-i32-cast-contract.md)。値域内は torch 準拠の
truncate（0 方向切り捨て）で一致保証・値域外と NaN は未定義。要素ごとの値域検査は意図的に
入れない — 範囲外になりうる値は呼び出し側・モデル側で先に clamp すること。

## IR の i32 算術は 2 の補数ラップ（int64 中間の縮小は未防護）

exporter は torch の int64 を境界（グラフ入出力・具体境界テンソル）で値域検査つきの i32 へ
正規化する（ADR 0009）が、**emit された i32 演算（mul / sub）の中間値には防護が無い** —
2³¹ を跨ぐ中間は例外にならず 2 の補数でラップする（実測: int64 x=50000 の `(x*x).float()` は
診断ゼロで export され、参照実装の `Math.imul` が −1794967296 を返す）。実配布 10 ファミリの
i32 算術 99 本は全て mask 由来の構造的有界値で該当ゼロ（189 コンテナ走査・2026-08-16）。
中間値域の静的証明は一般に不可能なので境界検査 + golden 突合（実入力に対する事実上の門）を
契約とし、i64 級の中間演算が要るモデルは export 時の設計（値域を保つ分解）で対処する。

## exporter: `x + 0` の恒等除去は −0 入力で torch と乖離しうる（div / sqrt の下流）

`normalize._drop_identity_add` は `add(x, 0)`（加数が Python スカラの 0）を x へ畳む。f32 で
値が変わるのは x = −0.0 のときだけで、`x + 0.0` は +0.0 を返す。IR v1 には**符号付きゼロを
区別する op が実在する** — `div` は `1/(+0) = +∞` / `1/(−0) = −∞`、`sqrt` は `sqrt(±0) = ±0`
（参照実装も素の除算と `Math.sqrt`）— ので、消した add の下流が div の分母や sqrt の引数へ
届く形では torch と符号が反転しうる（最小反例は 2026-08-16 レビューで実証済み:
`num / (x + 0.0)` が torch +∞ / IR −∞）。実測 10 ファミリでは到達経路が無く、消費側を見て
書き換えを制限する形は全系列の再エクスポートを誘発するため、乖離の可能性を**受容**して
書き換えを残している。消費側に −0.0 が出うる形を足すときは、このパスの適用条件を先に
見直すこと（`karume/normalize.py` の `_drop_identity_add` docstring に同じ注記）。

## 意味論 dtype の実行可否は op ごと（一括解禁しない）

契約表（`packages/runtime/src/ops.ts` / `karume/ops.py`）が正本。i32 / bool を実行できるのは実測
グラフに現れた形と、そのために新設した op だけ（例: mul / sub は f32+i32、div は f32 のみ、
bitwise_not は bool のみ）。語彙 allowlist 凍結（ADR 0007）の dtype 版で、拡張は実測 +
契約表 + golden の 1 セットが条件。

## bool の実表現は u32 の 0 / 1・bool initializer は語彙に無い

WebGPU のストレージバッファに 1bit 型が無いため、bool は GPU 格納・入出力
（`Tensor.data` = Uint32Array）とも u32 の 0 / 1（ADR 0009）。safetensors の `BOOL`
（1 バイト格納）は 4 バイト前提の転送と噛み合わないため、bool の initializer は IR v1 の
語彙に無い（必要になったら格納規約ごと改訂）。

## strided コピー族（permute / expand / slice / cat / sym_prefix_slice / masked_fill の mask）は rank ≤ 4

`STRIDED_RANK = 4` を契約層で検査する（DeBERTa front は全値 rank ≤ 4 — ADR 0011。
slice / cat は読み族・書き族としてこの上限を共有する — ADR 0014）。
rank ≥ 5 を落とすのは**エクスポータ側の仕事**で、`_lower_unit_expand` /
`_lower_split_unbind` / `_lower_reshape_permute` の鎖 3 パスが**実装済み**
（`karume/normalize.py`・発火は rank > `STRIDED_RANK` の値を含む形に限る — ADR 0016）。
受理したパターンは rank ≤ 4 の列へ落とし、これらで正規化できない高 rank 形は export 時に
fail loudly（実行上限そのものは緩めない）。

## レイアウト第 2 群は実測形だけを受理する（slice / cat / pad / flip — ADR 0014）

「表現が無い軸は黙って既定で実行されない」を保つため、実測に出た形だけを語彙に入れている。
広げるときは契約 4 点セット（TS 契約表 / WGSL カーネル / CPU 参照 / Python 契約 + shape 規則）
と適合表・golden を 1 セットで動かす。

- `slice`: **step は 1 固定**（飛ばし読みは strided 族の可変点 1 語では表せない）。切り出す軸は
  **静的**（記号軸は `sym_prefix_slice` の担当 — 重複させない）。負の添字と省略された `end` は
  エクスポータ境界で軸長へ詰める。
- `cat`: 連結軸は〈定数〉または〈**同一シンボルの一次式**〉。総和が次元言語 `coeff·sym+offset`
  に載る形（`S`+1 → `S+1`、`S`+`S` → `2S`）を受理し、異なるシンボルの混在は fail loudly
  （ADR [0046](decisions/0046-cat-symbolic-axis.md)）。入力は 2 本以上（1 本の cat は恒等
  コピーで語彙に無い）。
- `pad`: **最終次元・定数 0・非負幅**のみ。埋め値の欄を持たないので「0 以外を黙って 0 で
  実行する」経路が構造的に無い。負幅（切り詰め）は slice の意味なので受理しない。
- `flip`: **静的軸 1 本**のみ（多軸 flip はエクスポータ境界で落とす。動的軸の反転はカーネル上は
  書けるが、要求実測が出るまで広げない）。

## conv 族も実測形だけを受理する（ADR 0015）

- `conv1d` の attrs は `stride` / `padding` / `dilation` / `groups` の **4 つとも宣言必須**で、
  **既定値の補完をしない**。省略を許すと depthwise（`groups = C`）の IR が黙って通常畳み込みに
  なり、shape も要素数も変わらないまま別チャネルの値が混ざる。IR を手で書く側にとっては
  冗長だが、沈黙誤値を消すための意図的な冗長性。
- `conv_transpose1d` は **`2·padding == K − stride`** の形だけを受理する（= 出力長がちょうど
  `L·stride`）。一般形は表現としては書けるが、出力長が次元言語の一次式（`coeff·sym+offset`）に
  収まらない組が出るため、需要が出るまで広げない。重みは **`[Cin, Cout, K]`**（conv1d の
  `[Cout, Cin/groups, K]` と転置）で、`output_padding` / `groups` / `dilation` は attrs に欄が
  無く既定以外は fail loudly。
- bias 無しの conv は**落とさず**、エクスポータが**ゼロ bias を合成**してアリティ 3 へ
  正規化する（カーネルと契約に arity 分岐を持ち込まないため）。bias 無しを落とすのは
  `linear` だけ（実測が全て bias 付き）。

## upsample_bilinear2d（align_corners=True）の端点は「厳密一致」を保証しない

`align_corners = True` でも、出力の端が入力の端と**ビット単位で一致するとは限らない**。源座標は
ホストで f32 に丸めた `scale = fl((in−1)/(out−1))` に出力添字を掛けて作るので、
`fl(scale · (out−1))` が `in−1` をわずかに下回る形が実在する（実測: in=2 → out=42 の末尾は
0.9999999403953552）。

これは by-design で、**torch 自身が同じ値を出す**（`area_pixel_compute_scale` を float で評価し
`scale · dst_index` を float で掛ける）。カーネルの数値契約は「torch の `UpSample.h` に合わせる」
なので、端点をクランプして厳密化すると逆に torch とビットが割れる（ADR 0058 の opt-in 契約に
照らして既定経路では不可）。発火範囲の実測（2026-08-16）: `2 ≤ I ≤ 64` × `2 ≤ O ≤ 2048` の
128,961 組のうち 11.6% が非厳密。ただし `O = 2I` / `O = 2I−1`（I ≤ 4096）は 0 件で、配布モデルの
実形状 47 サイトは H・W とも全て厳密成立側にある。

## GRU スキャンは隠れ幅 256 まで・入力側 GEMM を含まない・`h_n` を返さない（ADR 0056）

- `gru_scan` / `gru_scan_reverse` の隠れ幅は **`H ≤ 256`**（1 lane = 1 隠れユニットの割り当て。
  黙って縮退させると workgroup 共有の範囲外書き込みで別ユニットの状態が例外なしに壊れるので
  fail loudly）。門は**二重**で、値域外は契約層が `OpContractError`（`ops/shapes.ts`）、
  カーネル直呼びの経路は params が `CodegenError`（`kernels/gru-scan.ts`）で落とす — 上限定数の
  置き場は `codegen/limits.ts` の `GRU_SCAN_MAX_HIDDEN` 1 か所。実測に出ている形は H = 128 だけで、
  上限を上げるには workgroup 内 grid-stride と状態の二重化が要る = 別の設計判断。
- op が持つのは**隠れ側の逐次だけ**。入力側 GEMM（`x·W_ihᵀ + b_ih`）は**呼び手が既存 `linear`
  で用意する**（IR 上は別ノード）。この分割のおかげで入力側の重みは f16 / i8 格納の適格の
  ままだが、**`W_hh` は op 内スロットなので低精度格納の適格外**（`WEIGHT_SLOTS` に載らない）。
- 出力は `y[T,N,H]` **だけで `h_n` を返さない**（IR v1 の単一出力前提）。最終状態を消費する
  モデルは現状表現できない。
- **多層 / 双方向 / `has_biases=False` / `batch_first` / `dropout` の欄が無い**。層と方向は
  エクスポータがノードを並べて表す（`aten.gru` の `Tensor[16]` は IR に載らない）。
- **LSTM は語彙に無い**（`aten.lstm` は未対応 op として fail loudly）。拡張時の論点は
  ADR [0056](decisions/0056-gru-scan.md) 決定 8。

## 母音検出: 入力の 10ms フレーム数は**偶数**・上限は配布形の宣言（ADR 0056 / 0057）

- グラフ入力の時間軸は `2T`（記号 `T` は出力の 20ms 格子）。**奇数長は受理しない** —
  `bindSymbols` が「実測 285 が宣言 '2T' の形をしていない」で落ちる（丸めない）。呼び手は
  末尾 1 本を**切り捨てて**渡す（`VowelDetectorPipeline.detect` と実重み E2E がそうしている）。
  右ゼロ pad で長さを合わせるのは**禁じ手**（逆方向 GRU が pad から状態を持ち帰り、`.lab` が
  発話のどこででも変わる — 実測は ADR 0056 追記）。
- 長さの上限は `pipelineConfig.maxFrames`（配布形の宣言 = 焼いたときの記号次元の上限）。
  **IR は記号の値域を持たない**ので、超過を止められるのはパイプラインのこの門だけ。超過は
  fail loudly（切り詰めない — 末尾が黙って落ちた `.lab` は正常な結果と区別できない）。
  現在の配布形は 60,000 フレーム = 10 分で、根拠は最大中間テンソル `[1,160,2T]` f32 =
  640 B/フレーム → 38.4MiB（仕様既定の `maxStorageBufferBindingSize` 128MiB に対し 3.4 倍の
  余裕）。**それ以上の長さは未実測**。
- 上限内でも実行時間は長さに比例する（時間ループは 1 workgroup 内の逐次 — 性能は未計測）。

## flow / voice の相対位置表はグラフ入力 — 生成はホスト側の責務

SBV2 の `flow` / `voice` は相対位置注意の `(T,T)` 表（`idx_k` / `valid`）を**グラフ入力**として
要求する（Tmax = 4096 で焼き込むと 134MB — ADR 0013）。したがって**呼び出し側が T ごとに表を
生成して渡す**必要があり、ランタイムはこの表の正しさを検査しない（値としては単なる i32 添字と
f32 マスクで、1 ずれても shape エラーにならない沈黙誤値クラス）。

- 式の正本は Python 側 `tools/export-recipes/sbv2/patch.py` の `build_relattn_tables`
  （front の in-graph 構築も同じ関数を呼ぶ）。
- ホスト側の正本は `packages/models/src/sbv2/relattn-tables.ts`（SBV2 固有なので
  `packages/runtime/src/` には置かない — モデル側の知識を持つ models パッケージが持ち、
  `Sbv2Pipeline` が T ごとに呼ぶ）。Python 側とのバイト一致は
  `packages/models/tests/sbv2_relattn_parity_test.ts` が golden の実データで固定する。
- 窓幅（実測 4）の食い違いも同じ沈黙誤値クラスなので、Python 側は ckpt ロード時の
  `_assert_window_size`（net_g 全体を走査）、TS 側はパリティテストがコンテナに焼き込まれた
  `idx_v` の幅 `2w+1` と突き合わせて落とす。

## SBV2 ホスト糊は f64 で評価する — `w_ceil` は torch とビット同一でない

`durationsToFrames` / `buildZp`（`packages/models/src/sbv2/host/`）は式全体を JS の f64 で評価し、
`Int32Array` / `Float32Array` 代入で 1 度だけ丸める（同ディレクトリの `random.ts` と同じ家風）。
参照側は f32 逐次なので、以下は by-design の既知差:

- **`w_ceil` の 1 フレームずれ**: `f32(exp_f32(x))` と `exp_f64(x)` の f32 半 ulp（相対 6e-8）が
  `ceil` の閾値を跨ぐと 1 フレーム動く（実測: 音素あたり 5.5e-7・229 音素の 1 発話で
  P ≈ 1.3e-4）。`Math.fround` 逐次へ揃えても **torch 一致は決定的にならない** — front 出力
  自体の GPU/CPU 差 1e-5 が同じ閾値跨ぎを 2 桁以上高い率で起こし、支配項は上流に残る
  （`tools/export-recipes/sbv2/README.md` が設計として許容し、割れた位置を `w_ceil_diffs` に
  載せる）。
- **`z_p` の 1 ulp 差**: 要素の約 4 割で常に生じる。この経路を測る波形突合の実測
  maxAbs 5.16e-5 に対して 3 桁下で、離散化を挟まないので形状には増幅しない。

どちらも karume 単体の再現性（WAV sha256 門・段 1 / 段 2 経路の一致）には影響しない — f64 経路は
決定的で、差は torch 参照との相対でのみ現れる。

## SBV2: 発話の編集で受けるのは「音素列を変えない範囲」（音素数が変わる編集は落とす）

0.6.0（ADR [0079](decisions/0079-sbv2-two-layer-input.md)）以降、テキスト解析は**呼び手の責務**で、
`generate` が受けるのは解析済みの発話 `Sbv2Utterance`（モーラ層）1 本。編集面は 2 つあり、門は
**`moras` から組み上がった音素列が `words` の音素列と位置ごとに一致するか**だけを見る
（`packages/models/src/sbv2/text/model-input.ts` の `assertWordPhones`）。したがって:

- **通る**: フレーズ層 `Sbv2Phrases` 側の `result.accentPhrases[i].accentNucleus` の変更と**句の
  再分割・結合**（`toSbv2Utterance` で再変換するだけ — VOICEVOX 互換のアクセント句編集はここ）、
  およびモーラ層 `Sbv2Utterance` の **`tone` 直編集**（核で書けない任意の 0/1 パターン）。
  句 / モーラの境界そのものは門が見ていないため。
- **落ちる**（`Sbv2InputError`）: モーラの読み替え（`vowel` / `consonant` の書き換え）・記号の
  増減など、音素列が変わる編集。`moras` と `words` を別々の解析から採って混ぜた発話もここで止まる。

NOTE: 句 / モーラの組み替えは、1 モーラの子音と母音に**別のトーンが載る**列を作れる（通常の解析
経路では起きない形）。不変条件は破れず（`sum(word2ph) === P` は保たれ、BERT 入力と word2ph は
`words` 由来のまま）、影響は出力音のみ。

読みを変えたいときは**呼び手の解析器側で修正辞書を当てて解析からやり直す**（`@hdae/yomi` の
`analyzeWithWords(dict, text, overlay?)` 等）— BERT 入力テキストと word2ph が正しく作り直されるので、
音素だけ差し替える経路より品質が上。karume 側に修正辞書の席は無い（0.6.0 で撤去 — 辞書の取得も
解析も models の外）。参照実装が持つ `adjust_word2ph`（LCS で word2ph を再配分する互換ハック）は
移植しない: あちらでも BERT 特徴は元テキスト由来のままで、不整合は解消しない（ADR 0072 決定 8 の
原則は 0079 でも存続）。

受けられない編集は 1 つだけ — **語境界に一致しない読みの差し替え**（句内の一部モーラだけを別の
読みへ）。VOICEVOX 系の `/accent_phrases?is_kana=true` に相当する「句ごと読みを差し替える」操作は
これに当たる。

## SBV2: 疑問形の「末尾だけ上げる」は表現できない（トーンは 0/1 の 2 値）

SBV2 の `tone` はアクセント句内の高低を表す **0/1 の 2 値**で、`front` の tone 埋め込みの行を
引くだけの離散入力（`packages/models/src/sbv2/text/symbols.ts` の `toneStart` / `numTones`）。
VOICEVOX 系の `is_interrogative` / `enable_interrogative_upspeak` は AudioQuery のモーラ f0 を
後段で持ち上げる操作なので、この離散トーンには写像先が無い。JP のトーン基点の外へ出た値は
**別言語のトーン行**を引くだけで、上げにはならない（だから `toSbv2PhoneTone` は
`Sbv2Mora.tone` が 0/1 以外なら `Sbv2InputError` で落とす — 型は `0 | 1` だが JS からの
呼び出しには効かない）。

疑問形が音に出る経路は 1 本だけ — テキストに実在した `?` が**音素として 1 個入る**こと
（`toSbv2PhoneTone` が実在記号を tone 0 の音素にする）。上げ調子はモデルが学習済みの韻律として
出す。参照実装も同じで、上げの機構は持たない（`?` を音素表の 1 記号として扱うだけ）。

したがって:

- 呼び手が疑問形を足したいなら、**読み上げテキストの側に `?` を入れて解析からやり直す**
  （解析器を呼ぶ段）。解析済みの発話の `punctuations` へ後から足す経路は、`words` 側の音素列が
  追随しないので門（`assertWordPhones`）で落ちる。
- VOICEVOX 互換 API を模す層では `enable_interrogative_upspeak` は**受けて無視する**のが実態に
  合う（AivisSpeech Engine も同じ扱い — 2026-08-21 のソース調査）。ただし
  「`？` をモーラから剥がして `is_interrogative` フラグへ畳む」正規化を入れると疑問形の情報が
  完全に消えるので、そちらは入れないこと。

## Anima: 生成中のプレビュー画像は出せない（途中結果は生 latent のみ）

`AnimaGenerateRequest.onEvent` の `denoise-step` が渡せる途中結果は **latent の写し**
（`copyLatents()`）だけで、毎 step のプレビュー画像は by-design で提供しない。VAE decoder は
DiT を解放した**後**にしかロードできない（4 本同時常駐は VRAM で不成立 — ADR 0016 /
`anima/pipeline.ts` のモジュール doc）ため、denoise ループの途中で VAE を回す経路が構造的に
存在しない。プレビューが要る消費側は latent から近似する — そのための公開ヘルパが
`approximatePreview`（`@karume/models/anima`）で、`copyLatents()` の 2 欄を**そのまま**
（逆正規化せずに）渡すと latent 解像度の RGBA が返る。係数は**正規化空間**で較正されている
（2026-08-24 実測）ので、`animaLatents()` の mean / std で逆正規化した値を渡すと白飛びした
別物になる。係数自体は `export` しない MUST なので、消費側が自前で写し取る経路は無い
（16ch → 3ch の線形射影であって厳密な decode ではない — 詳細は実装 doc が正本）。
`stage` イベント（段の Session 構築前 / 解放後）と `vae-tile` イベント（タイル 1 枚ごと）で
GB 級ロードとタイル decode の進捗は観測できる。

## EmbeddingGemma: 実行時 attention_mask（バッチ内パディング）は非対応 — 単一シーケンス前提

export 済みグラフ（台本 `tools/export-recipes/embeddinggemma/export.py`）は `attention_mask` を
入力に持たない。双方向 + sliding window の帯マスクは **Tmax=512 の定数**として焼かれ
（`sym_prefix_slice` で先頭 T を切り出す）、パディングを注意から隠す経路は無い。
パディングを含む列は**呼び出し側が詰めて T を短くする**（単一シーケンスなら torch eager と
厳密同値）。

機序: ADR 0016 の safe-softmax ガード不活性証明はマスクの実値評価を要求し、実行時入力の
placeholder では成立しない（`_eval_static` が拒否する — 「単一シーケンス前提」は回避では
なく設計帰結）。グラフ入力の `pool_mask` は **pooling 専用**で注意には配線されない
（0 を混ぜると「モデルは見ているのにプールでは捨てる」という eager に無い形になる —
常に全 1 で渡す）。

付随: `SYM_MAX = 512` なので T > 512 は Session 構築で落ちる（config の
max_position_embeddings は 2048）。上げる場合は帯マスク定数が Tmax² で膨らむ
（512 → 2MB / 2048 → 32MB）ことの裁定とセットで行う。

解除（実行時マスク対応）の設計は
[decisions/0044](decisions/0044-runtime-attention-mask.md)（accepted）で確定済みで、**機構は
実装済み**（`safe_softmax` op + `_drop_safe_softmax_guard` の 2 段化 — 2026-08-11）。残るのは
EG 台本の配線だけ（bool マスク入力を受けて帯定数と加算合成し、SDPA を保存のままにするか
分解へ落とすかの裁定 — ADR 0044 の Consequences）。

## Irodori テキスト系（backbone / projector）: 実行時 attention_mask 非対応・空 caption と T=1 は非表現

EmbeddingGemma と同じ静的方式（B=1・呼び出し側が列を詰める）。「右詰め pad + マスク」との
同値は export 台本の常設門が毎 emit 実測する（`tools/export-recipes/irodori/export.py` の
`_static_scheme_evidence` — 実測 8.3e-6 以下・門 1e-3。崩れれば export ごと落ちる）。付随:

- **空 caption（マスク全 0）は graph で表現しない** — eager では projector 出力が厳密に全 0
  になる形なので、ホストがゼロを直接作る（CFG uncond と同じ扱い — ADR 0044 の管轄）。
- **T = 1（BOS のみ）は表現できない**（記号次元は `Dim(min=2)` — 0/1 特殊化を避ける既定）。
  空 caption 以外で T = 1 になる実入力は無い。

## Irodori speaker / duration: 参照なしと平均トークン前置はホスト・条件ベクトルもホスト供給

テキスト系と同じ静的方式（B=1・呼び出し側が列を詰める）を speaker encoder（`[1,S,128]` →
`[1,S,768]`）と duration predictor（`[1,T,512]` ほか → `[1]`）にも適用した帰結。いずれも
「グラフに載らない」という by-design の線引きで、値の近似や無音のフォールバックはしない:

- **参照なし（`no_ref` — 参照マスク全 0）はグラフを呼ばない**。eager 側の出力が
  **厳密に全 0** になる形（SDPA の safe-softmax が全マスク行に 0 を返し、末尾の `x * mask_f`
  が全体を 0 にする）なので、ホストがゼロ行列を置けば同値。**恒真化しないよう同 export
  台本の `_no_reference_evidence` が毎 emit 実測する**（非ゼロ latent を
  全 0 マスクで通し、出力の最大絶対値が 0 でなければ export ごと落ちる）。
- **平均トークンの前置（`_prepend_masked_mean_token`）は現行パイプラインではホスト**。IR v1 の
  `cat` は `1 + S → S+1` を受理する（ADR [0046](decisions/0046-cat-symbolic-axis.md)）ので、
  残置の理由を「記号軸 `cat` 非対応」とはしない。ホスト側の作業は軸 1 の平均と concat だけで、
  モデル計算（重みを使う演算）は残らない。GPU 側へ移すかは別途の設計判断。
- **duration の `speaker_vec` / `caption_vec` はホスト供給**。前者は上の平均トークンの
  切り出し、後者は **caption 系列に `caption_norm`（RMSNorm 512）を掛けた masked mean** で、
  後者だけはホストにモデル計算が 1 本残る。caption 系列をグラフ入力にすると記号次元が
  2 本（T と caption 長）になるため採らなかった（多記号グラフは未実測 — recon の U3）。
  **0/0 の危険は無い**（実装が `denom = clamp_min(sum, 1.0)` で割るため — recon が挙げた
  「caption 全 0 の masked_mean」は上流で既に閉じている）。
- 参照なし / caption なしの選択（`null_speaker` / `null_caption`）は **`has_speaker` /
  `has_caption` の bool 入力 + グラフ内の `where`** で表現する。2 本の学習済みベクトルを
  ホストへ配らずに済ませるため（ADR 0010 が「ホスト事前計算 + 追加入力」を却下したのと同じ理由）。
- **S = 1 と T = 1 は表現できない**（記号次元は `Dim(min=2)`）。参照 latent は
  `speaker_patch_size` = 4 の patch 後で、実入力が 1 トークンになるのは `no_ref` の形だけ
  （上のとおりグラフを呼ばない）。
- 記号次元の上限は **S ≤ 750**（`ref_max_seconds` 120s × 25Hz ÷ patch 4 — チェックポイントの
  config から導出）。超える参照長は束縛検査で fail loudly。

## Irodori DiT: 行ブロックでも分割不能な 1 クエリ行上限・CFG は既定 2 モードのみ

DiT 1 step（`dit` ターゲット）は ADR [0047](decisions/0047-irodori-dit-execution.md) の実行形
（B=1 × 記号 S × G4 の畳み込み × uncond をマスクで表現）で export してある。その帰結として
次の 2 点は by-design の制約で、近似や無音のフォールバックはしない:

- **クエリ 1 行ぶんのスコア（`H·C·4` バイト）が束縛上限を超える形は fail loudly**。分解
  attention は 9 ノード窓を、granted limit から静的に決めた最小枚数の行ブロックで回す
  （ADR [0060](decisions/0060-row-block-attention.md) = 行ブロック実行。ビット同一・既定経路
  — 上限に余裕のある機では 1 枚 = コストゼロ）ので、S=750 の中間 `scores` が 128MiB を超える
  かつての制約は解消済みだが、1 行すら上限に入らない形は行ブロックでは分割しきれない。
- **CFG は `speaker_uncond_mode="mask"`（既定）と `cfg_guidance_mode="independent"`（既定）
  以外を表現しない**。uncond をマスクだけで表せるのは「state を 0 にした context KV の寄与が
  マスク越しに厳密 0」だからで、`"noise"`（speaker の uncond を乱数 state にする）はこの
  同値が成り立たない。`joint` / `alternating` は変種の組み方そのものが違う。**既定外は
  パイプライン層で fail loudly**（グラフ側は 4 変種の差をマスク 1 本に還元してしまうので、
  ここで拒まないと黙って別のモデルを回すことになる）。

## Irodori DiT ループの GPU 常駐経路: denormal 出力の FTZ・診断の縮退・計測 / onEvent 時はホスト経路

DiT ループは既定で GPU 常駐（[ADR 0054](decisions/0054-resident-loop-and-fence.md) — CFG 合成
と Euler 更新を GPU の elementwise で実行・フェンスは batch 1 本）。by-design の制約 3 点:

- **最終出力がちょうど denormal（|x| < 2⁻¹²⁶）になる要素は、ホスト実装が denormal を保つのに
  対し GPU シェーダ算術が同符号の ±0 へ潰しうる**（parity probe 実測: 差分はこの機序のみ・
  fma 収縮 0・符号付きゼロ一致）。実データの潜在／速度場（単位分散級）では実質到達しない
  領域で、参照ケースの WAV sha256 門 2 本は digest 完全一致 — 門が恒久の検出器。
- **常駐経路では `lastRun`（run アリーナ実績）と `lastRunTiming` が `undefined`**（enqueue は
  アリーナも計測窓も作らない）。`planBacking` / `submit` / `lastRunPrepared` は従来どおり。
- **gpuTiming 有効の device、および `generate` / `generateLatent` へ `onEvent` を渡した生成は
  従来のホストループへ分岐**（計測: batch と非両立で `beginBatch` が拒否 / onEvent: 1 batch +
  単一フェンスの区間は step の完了そのものをホストから観測できず、`enqueue` 時点の発火は
  「進捗」として嘘になる）。出力は同一 digest（`e2e_irodori_wav_test.ts` の onEvent 段が
  voice-clone と同じ sha256 で常設の門にしている）だが壁時計は伸びる（同一ケースの実測
  7.2 → 8.6 秒 / S 170・参照環境 2026-08-16）。op 別内訳を採るとき・進捗を出すとき以外は
  既定（常駐経路）のまま使うこと。

## Irodori パイプライン（ホスト層）: 上流の任意ノブは既定値相当のみ・参照音声は 48kHz

`IrodoriPipeline` は第 3〜4 波の範囲（ADR [0048](decisions/0048-irodori-host-port.md) /
[0049](decisions/0049-irodori-codec-integration.md)）で、以下は by-design の制約。近似や
無音のフォールバックはしない:

- **上流の推論ノブは既定値で死んでいるものを移植しない**: LoRA 動的ロード /
  speaker_kv_scale 系 / truncation_factor / temporal_score_rescale / sway スケジュール /
  num_candidates>1・decode_mode="batch"。CFG のモード制約（mask / independent のみ）は
  上の DiT 節のとおりで、**パイプラインは pipelineConfig のパース時に拒否**する。
  末尾トリムのしきい値（窓 20 / std 0.05 / mean 0.1）と参照音声の目標 −16 LUFS も上流既定の
  固定値（実行時ノブとしては持たない）。
- **`cfgScales` は f32 で厳密に表せる値だけを受理する**（非厳密値は `pipelineConfig` の
  パース時に fail loudly）。DiT ループはホスト経路（f64 のまま乗算）と GPU 常駐経路（f32 へ
  丸めてから乗算）の 2 本があり、強さが f32 非厳密だと同じ入力で最終桁が 1〜2 ulp 割れる
  （実測: s=1.3 で分岐）。「2 経路の出力は同じ」という MUST を配布形に依らず無条件に成立
  させるため、宣言側で落とす。実配布の 3.0 / 5.0 / 3.0 は全て f32 厳密で影響なし。
- **参照音声（`speaker: { audio }`）は配布形の `sampleRate`（48kHz）のみ** — リサンプルは
  持たず fail loudly（ADR 0049 決定 6。変換は呼び出し側の責務）。`decodeWav` が受けるのは
  PCM 16bit と IEEE float 32bit だけで、`WAVE_FORMAT_EXTENSIBLE`（0xFFFE）等は明示拒否。
- **`codec_encoder` はタイル分割しない**（decoder と非対称 — ADR 0049 決定 1）。長い参照
  （120 秒で中間 1.47GB×2）は `maxStorageBufferBindingSize` が既定 128MiB の機で確保に
  失敗する。decoder 側は halo 8 のタイル分割で既定上限機でも S=750 が通る（ビット一致門付き）。
- **生成音声に透かし（SilentCipher）は入らない**（wm 枝はバイパス形で焼かれている —
  2026-08-11 裁定で公開前の波まで保留・ADR 0049 決定 2）。
- **seed は上流と互換でない**（torch generator のビット再現は非目標 — ADR 0048 決定 5）。
  同 seed → 同波形の自己決定論のみ保証し、torch との突合は `initialNoise` 注入口で行う。
- 前処理の `strip` は JS の `String.prototype.trim` で、Python `str.strip` とは空白集合の
  端（U+001C〜1F・U+0085 は Python のみ / U+FEFF は JS のみ）が違う。実用のテキスト入力では
  発生しない差として受容する（golden の normalize 33 ケースはこの領域を含まない）。
- WAV の読み（/32768）と書き（×32767）は非対称のまま固定（それぞれ外部との一致が正 —
  `src/audio/wav.ts` の MUST。往復は 1LSB 級でずれる）。

## 融合 attention の加算 mask: 静的 `[1,1,M,N]` のみ・a8 と非併用・ビット同一門は f32 経路

`attention` の第 4 入力 mask（ADR 0023 追記 2026-08-11）は意図的に狭い:

- 受理は **f32・加算型・`[1,1,M,N]` ちょうど**（B·H へ broadcast）。実行時 bool マスク・
  `[B,1,M,N]`（バッチ別）・`[1,H,M,N]`（head 別）は fail loudly — Irodori CFG の裁定
  「実行時 bool マスク（案 a）」の波で、ADR 0016 のガード不活性証明の再設計とセットで広げる。
- **mask × `attentionCompute:'a8'` は fail loudly**（i8a8 族の ①QK に epilogue が無い —
  黙って f32 へ縮退させない）。対応するかは別波の設計判断。
- 分解経路とのビット同一の恒久門（parity）は **f32 経路のみ**。s16 / c16 × mask は WGSL
  生成・パイプライン作成・実 GPU 実行の確認まで（門を足すか ADR に f32 限定と明記し続けるかは
  そのケースが実資産に現れた時に判断）。

## 融合 attention の GQA × `attentionCompute:'a8'` は fail loudly（暫定）

GQA / MQA 形（`H % Hkv == 0`・ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 1）は f32 / f16 経路のみ。
a8 経路は head 基底が 5 本（`attention-i8a8.ts`）で K / V の量子化・確保も `B·H` 前提のため、
GQA 形は**縮退せず fail loudly**（黙って f32 へ落とすと性能が静かに変わる — ADR 0058
決定 3）。**拒否は暫定で後日サポート前提**（ADR 0067 決定 3 — 追補面は確定済み: K/scale・
V/scale 基底のみ kv-head 写像 + recipe-builder の Hkv 化。検証は f32 経路の repeat_kv
parity 資産を流用）。

## topk の k 実装上限は device 依存（WebGPU 既定 limits で k ≤ 63）

`topk`（ADR [0068](decisions/0068-decode-exit-multi-output.md) 決定 3）の scratch は
workgroup storage に閉じるため、k の上限は `8·32·(k+1) ≤ maxComputeWorkgroupStorageSize`
で決まる（`src/kernels/topk.ts` の `topkMaxK`）。WebGPU 既定の 16384B では **k ≤ 63**・
本開発機（49152B）では 191。超過は上限値つきで fail loudly（縮退しない — ADR 0058
決定 3）。top-k sampling の実用域（k ≤ 50 級)は既定の機でも収まる。上限を上げる設計
（多段 merge 等）は実需が出た時の perf-ledger 起票。

## ストリーミング慣習の WAV（riffSize プレースホルダ）は受理しない

`decodeWav`（`packages/models/src/audio/wav.ts`）の走査境界は RIFF が offset 4 で宣言する
論理終端で、宣言の外の物理バイトは読まない（仕様どおりの無視 — `encodeWav` が書く欄が正）。
このため、ストリーミング書き出しの慣習である riffSize プレースホルダ宣言は by-design で
受理しない:

- `riffSize=0` は論理終端がヘッダ直後になりチャンク走査が 1 つも回らず、
  `decodeWav: 'fmt ' チャンクが無い` で落ちる。
- `riffSize=0xFFFFFFFF` は切り詰められた器として
  `decodeWav: RIFF が 4294967295 バイトを宣言しているが、残りは … バイトしかない` で落ちる。

入力は完全なファイルとしての WAV 前提（参照音声・検出器入力とも）。ストリーミング WAV を
食わせる実需が出たら受理形をその時に裁定する（黙って物理長へフォールバックしない — fail
loudly の横断規約）。

## hub: 並行取得のキャンセル粒度は single-flight の leader 単位

取得層（`@hdae/fetch-cache`）の single-flight では、同一の**内容キー**
（`["hf", kind, repo, path, sha256]`）への 2 本目以降の呼び出しは先行フライトへ合流し、
合流者に渡した `AbortSignal` は効かない（leader を abort すると合流者も巻き添えで落ちる）。
同一資産を並行に取る複数の `fetchAssets` では、キャンセルはこの粒度でしか働かない
（ADR 0038 §5）。単一呼び出しの abort は全ワーカーへ正しく透過する。

## 0 要素次元を持つ gemm 系の形は GPU 束縛の最小サイズで落ちる（未対応の退化域）

`linear` の `in=0` など 0 要素次元は op 契約上は valid だが、0 要素バッファの確保下限
（4 バイト — arena / executor の `Math.max(4, …)`）が vec4 変種の最小束縛サイズ（16 バイト）を
割るため、実行は `GpuValidationError`（Binding size … less than minimum）で fail loudly に
落ちる（`linearCompute` に依らない — 2026-08-13 実測）。沈黙誤値にはならない。解除するなら
確保下限を 16 へ統一する（全 op 共通の確保方針の変更 — 需要が出たら別波）。

## GitHub CI はローカル資産（`outputs/`）依存のテストを踏まない（検証範囲の制約）

`outputs/` は git 追跡外のため、実系列資産を golden に使うテスト群 — GPU e2e に加え、
**CPU-only の upstream parity**（irodori の codec / reference / t-embed、sbv2 の rel-pos /
relattn / demo 資産、wav の実資産 scale）— は GitHub Actions では資産不在で ignore になる。
これらの門はローカル / self-hosted の `deno task verify`（資産あり）が担い、リリース判定は
実資産 + 実 GPU の緑を必須とする（ADR 0005）。CI 側へ寄せるなら golden の fixture 昇格
（リポ肥大とのトレードオフ）か release gate での資産取得が要る — リリース準備波で再訪。

## `karume dist` はディスクピークが配布形の約 2 倍（staging→swap の代償）

組み立ては staging ディレクトリへ全て作ってから rename で据える（ADR 0052 — 途中の故障で
既存の配布形を壊さないための by-design）。swap の瞬間まで新旧ツリーが併存するため、出力先の
ファイルシステムには配布形サイズの約 2 倍の空きが要る。据え替え後は `.staging` / `.old` とも
残らない。

## exporter: モデル別 recipe はリポ専用（wheel に入るのは汎用 core だけ）

PyPI `karume` は汎用 exporter core のみ（ADR
[0065](decisions/0065-exporter-core-recipe-split.md) — 配布境界とライセンス境界を一致させる
ための by-design）。既知モデルの export 台本・dist recipe・カードテンプレートは
`tools/export-recipes/`（uv workspace・wheel 外）にあり、実重み export の依存
（diffusers 等）も同プロジェクトの dependency-groups が持つ。このためインストール版の
`karume dist` は pipeline 表が空で fail loudly する（受理集合の正本はリポの
`tools/export-recipes/dist.py`）。`karume verify` はインストール版でも動く。

## Metal（Apple GPU）では GPU 側 timestamp 計測が実用にならない（外部制約）

`gpuTiming: true`（ADR 0021）は 1 dispatch = 1 pass に開いて pass 境界の timestamp を取るため、
チャンクごとに query set を 1 本作る。Metal ではその query set を作る `createQuerySet` が失敗し

```
Failed to create counter sample buffer: Cannot allocate sample buffer (MTLCounterErrorDomain)
```

を返す。NSError の code 0 = `MTLCounterSampleBufferError.outOfMemory`（Metal ドライバが counter
sample buffer を確保できない）で、wgpu-hal はこれを `DeviceError::Unexpected` に潰し、wgpu-core が
致命扱いして device を lost にする。**errorScope には入らない**（消失済み device のエラーは Deno の
error handler の入口で捨てられる）ので、`pushErrorScope('validation')` では捕まらない。

失敗の軸は「1 本のサンプルバッファが大きすぎる」ではない。初回の run は 1 チャンク 16 dispatch に
固定される（壁時計の実測がまだ無く時間予算を引けない）ため、gemma4 の prefill ≈1,500 dispatch では
query set が約 100 本**同時に生きた状態**になる。1 本あたりは 32 query（総量でも数十 KB）で、Chrome
の Dawn が Metal 上で常用する 4,096 サンプルより小さい。さらに Deno 2.9.x の
`GPUQuerySet.destroy()` は **no-op**（wgpu 29.0.1 pin・実装は wgpu v30 で入った）なので、karume が
`destroy()` を呼んでも実解放は V8 の GC 任せで滞留する。したがって「anima の 3,301 dispatch が
大きすぎる」という以前の帰属は誤りで、dispatch 数のより少ない gemma4 でも起きる。

「`GpuDeviceLostError` として可視化されるので沈黙はしない」も**撤回**する。トップレベルの
`await using` で資源を掴む台本では、本体の例外と解放時の例外が `SuppressedError` に畳まれ、Deno は
その外皮しか印字しないため型も文言も読めない（例示台本の全印字と、消失理由を捨てている
`device.ts` の修正は別項）。

観測は Apple M2 / Deno 2.9.x（anima の DiT 1 step = 3,301 dispatch と gemma4 の対話台本の両方）。

**現状の Deno では回避策が無い**（query set を 1 本に固定して使い回す案は未実験 —
[known-issues](known-issues.md) 参照）。op 別の内訳が要る計測は Linux / Vulkan 機で行う。
壁時計だけなら計測を切って（既定）測れる。なお macOS 26（Metal 4）では確保に成功しても
timestamp が全ゼロになる別の未修正問題がある（[wgpu#9414](https://github.com/gfx-rs/wgpu/issues/9414)）。

なお同じ理由で本ファイルの「Deno では GPUBuffer の総確保がドライバ申告予算の 97% で頭打ちになる」
節の制約は **Metal には効かない** — wgpu の `MemoryBudgetThresholds` は D3D12 と Vulkan のみ対応で、
wgpu-hal metal の `check_if_oom()` は `Ok(())` を返す no-op（[wgpu#7460](https://github.com/gfx-rs/wgpu/issues/7460)
の TODO 付き）。Metal では予算超過が例外にならず、遅くなるだけで進む。

## DL 前の GPU 適合チェックは quant が宣言した feature と limits まで（合計・空きは見ない）

quant が宣言する GPU 前提のうち、重み shard を取る前（家族 admission）に突き合わせるのは
`gpuFeatures`（共有 GPU を渡された経路のみ — 自前で device を取る経路は要求として `acquireGpu`
へ渡す）と `requiredLimits`（ADR 0089 決定 5・2026-09-01 結線）。limits の突き合わせ相手は、
共有 GPU なら `GpuContext.limits`、自前で取る経路なら直前に読んだアダプタ実測値
（`readAdapterLimits` — アダプタは読んで捨てる）。残る限界:

- **DL 前に読んだアダプタと device を作るアダプタが同一である保証は WebGPU 仕様に無い**
  （`requestAdapter` は呼ぶたび別オブジェクトを返し、同じ物理アダプタを選ぶとは定めない）。
  DL 前検査は事前判定で、最終の検査は Session 構築時の実バッファ検査（ADR 0089 決定 1）。
- `requiredLimits` は**常駐前提の寸法**（既定超過分だけ — ADR 0089 決定 3）。実行時の一時確保や
  f32 展開のワーストは含まない（そちらは構築時・実行時の検査が担う）。
- gemma4 の `fromAssets` は manifest を受け取らない（`Gemma4Assets` はバイト列と config だけ）
  ため宣言に到達できず、この面の守りは構築時検査のみ。`fromAssets` 一般は DL が無いので
  事前判定の席自体が無い（共有 GPU を渡した場合だけ admission で見る）。
- 合計 vs 物理空き容量は見ない（下の「GPU メモリの事前検査は…」節）。

## ブラウザ: Chromium は単一 ArrayBuffer を 2,145,386,496 バイトで打ち切る（分割前の旧資産のみ該当）

Chromium（Chrome / Edge — 全 OS 共通・Mac も同じ）は PartitionAlloc の意図的なセキュリティ
設計として 2³¹ − 2MiB = 2,145,386,496 バイトを超える単一 ArrayBuffer の確保を必ず失敗させる
（フラグで緩和不可）。これを超える配布ファイル — 例: Base 系 f16 の transformer
3,913,609,588 B — は全量面 / 逐次面のどちらでも materialize できず、**原理的にロード不能**
（実測は 2026-08-25 調査 — 経緯は git）。消費側の判定条件は「manifest の各ファイル `size` が
この値を超える quant 席を選択肢から外す」。i8（1,962,502,636 B）は壁の内側だが余裕は
約 183MB しかなく、重み増で同じ壁に当たる。恒久解（shard 分割配布 + ロード面接続 = R1）は
下記のとおり実装・公開済み。

なお 2026-08-28（fetch-cache 0.5.0 追従 — ADR 0080）から、この壁は**ダウンロード前に**
落ちる: hub が `expectedBytes`（manifest `size`）を渡すため、受信バッファの確保に失敗する
大きさは 1 バイトも受信せず throw される（`cause` = RangeError — 帯域と時間を捨てた後に
落ちる事故が消えた）。

**根本解は 2026-08-29 の R1 統合波で実装済み**: exporter が 1GiB 超のコンポーネントを
shard 分割し（`karume.shards` — ADR 0070 追記 2026-08-29）、ロードは shard 逐次面が
1 shard ずつ materialize する（単一バッファは常に ≤ 256MiB — ファイル長の受理上限・ADR 0090。上限超えの
テンソルは piece で割れる）。Base f16 の実ロード +
生成は分割配布形で実証済み。公開 HF リポ 2 本（anima / anima-turbo）も 2026-08-29 に分割形で
上げ直し済み。**この制約が残るのは「分割前に焼かれた手元の旧資産」だけ**（`outputs/` の
旧 series 等 — 再 export で自動的に規則内へ入る）。

## 配布形: 1 行が shard 容量を超えるテンソルは配布できない（by-design — ADR 0090）

テンソル分割（piece）は先頭次元（行）の境界でしか切らない — 各 piece が親と同じ dtype・残り次元を持つ
普通のテンソルなので、読み手・書き手・検査が型の規則を共有できる。したがって 1 行（先頭次元 1 つぶんの
バイト列）が書き手の容量（256MiB − ヘッダ余裕 1MiB）を超えるテンソル（例: shape `[2, 2^27]` の f32）は
配布できず、exporter が fail loudly で落とす。実資産の行は最大数 KB で、該当は無い。

## hub: キャッシュは credential で隔離しない（by-design — 2026-08-28 裁定）

資産キャッシュのキーは内容キー（`Authorization` を含まない）で、名前空間も取得層の固定
1 個。認証付きで取得した gated 資産の写しは、同一端末・同一 origin の無認証呼び出しにも
ヒットする。gated 資産の運用予定が無く、credential 別の名前空間隔離は過剰防御だったため
撤去した（ADR [0080](decisions/0080-hub-fetch-cache-050.md) 決定 3 — 将来 gated 運用を
始める場合は custom `caches` ラッパで隔離を再導入できる）。

## hub: ローカル取得元の検証は `size` だけ（sha256 は照合しない・改竄は検出しない）

`localDirectory` / `denoDirectory`（`@karume/hub/deno`）で手元のディレクトリを取得元にした場合、
読んだバイト列に掛かる門は **`karume.json` が宣言する `size` との厳密一致だけ**である。**`sha256`
は照合しない**（記録を信頼する）ので、次の 2 つは区別できる:

- 検出する: 途中で切れたコピー・別 quant / 別モデルのファイルの取り違え・欠損（読めなければ
  実体のパスを名乗って fail loudly）。
- **検出しない**: バイト数が変わらない書き換え（改竄・ディスク上のサイレントなビット腐敗）。

by-design の理由は取得物と資産の違いで、**手元の配布形は利用者の管理物**という前提に立っている
（配布元から取得したバイト列ではない）。毎起動の全量ハッシュ（gemma4 E2B なら 3.8GiB）に見合う
脅威が無く、ADR [0080](decisions/0080-hub-fetch-cache-050.md) 決定 1 が network 経路から消した
コストをローカルで買い戻すことになる。`size` は読み終えた時点でタダで分かるので門として残す。

**改竄検出が要る運用は HF 経路（`HubRepoRef`）を使うこと** — そちらは取得時に sha256 を検証して
記録を焼き、記録との不一致は自動 evict + 取り直し（self-heal）になる。設計の正本は ADR
[0086](decisions/0086-distribution-source.md) 決定 2。

## sha256 参照門は参照環境専用 — クロスデバイスのビット同一は保証しない

e2e の PNG / WAV 参照 sha256（`e2e_anima_test` / `e2e_sbv2_wav_test` /
`e2e_irodori_wav_test`）は**参照環境（RTX 3080 Ti / Linux / Vulkan (wgpu)）で焼いた値**で、
他バックエンド（Metal 等）では一致しない — これは仕様であり、門は参照環境での移植・退行
検出器として機能する。

機序: IEEE 754 の加減乗除はデバイス間でも完全同一だが、①超越関数（`exp` 等）の実装が
ドライバ / コンパイラ依存 ②シェーダコンパイラの fma 融合判断（積和を 1 命令に融合すると
丸めが 1 回減る）③コンパイル経路の違い（ブラウザ Tint / Deno naga）により、カーネル側で
縮約順序を固定してもクロスデバイスの同一は成立しない。なお w8a8 経路には整数演算なのに
値が違う未解明の Metal 差も別途ある（[known-issues.md](known-issues.md) の Metal 節）。

保証するのは次の 2 つ（いずれも実測データ点は Vulkan と Metal — Apple M2 の実測は
[research/2026-08-10-f32-geometry-probe.md](research/2026-08-10-f32-geometry-probe.md)
§Apple M2）:

- **デバイス内決定性**: 同一キー → バイト同一 WGSL → 同一出力（M2 で独立 2 セッションの
  出力 sha 一致を実証）。
- **幾何変更のビット不変**: タイル幾何は担当割りだけを変える（M2 で幾何 2 種の出力 sha
  一致を実証）。

別バックエンドでの健全性検証は参照 sha との一致ではなく**自己 A/B**（同一入力・幾何 2 種
または新旧 2 版の出力 sha が互いに一致するか）で行う。

**CPU 側にも同じ原則が掛かる**: tiny golden の io（torch CPU の期待出力）のバイト一致検査
（`tools/exporter/tests/test_goldens.py` の再生成突合）も参照環境専用 — oneDNN が CPU の
ISA で gemm / conv の kernel を出し分けるため、計算結果の最終 bit はマシン依存になる
（実測 2026-08-16: GitHub CI runner で 30 spec 中 activations / conv2d_block の 2 spec だけ
±1〜2 ulp）。CI では io のバイト突合を明示 SKIP し、model（グラフ + 固定 seed の重み —
torch の CPU RNG はクロスマシンで決定的）のバイト突合だけを要求する。golden を消費する
Deno 側の実 GPU テストはもともと tolerance 判定なので影響しない。

## states 形 attention は f32 経路のみ（数値変種と組めない）・sliding rewind 全拒否

`states` 欄つき attention / `state_append`（ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 4 / 5）の意図的な制約:

- **`attentionCompute: 'a8' / 'f16'` × states 形は fail loudly**（別族カーネルは f32 のみ —
  縮退せず拒否・ADR 0058 決定 3。opt-in の席は実需が出た時に ADR 0058 流儀で起こす）。
  **`attentionScoreStorage: 'f16'` × states 形も同様**（S の格納は f32 のみ）。
- **sliding スロットを 1 本でも含む GenerationContext の `rewind` は全拒否**（ADR
  [0066](decisions/0066-generation-context-state-slots.md) 追記 2 — ring はエビクト後に物理配置と
  論理範囲が一致せず、左詰め compaction を持たない。緩和は compaction 実装と対）。
- **`enqueue` は generation 面を持たない**（state 参照グラフは導出で fail loudly）。裁定済み
  （2026-08-18 波 E）: decode ループは前 step の token 読み戻しが次 step の入力になる逐次律速で、
  フェンスを束ねる利得が原理的に立たない（1 step 内の dispatch 束ねは `run` の submit 区間が
  既に持つ）。speculative decoding 等の「読み戻し無しで複数 step を積める」実需が出た時に再訪。
- state スロットの dtype は f32 のみ（f16 は席予約 — ADR 0066 追記 5）・複数シーケンス /
  batch>1 の生成・paged KV は ADR 0066 決定 8 のスコープ外。

## gemma4: capacity / chunkLength は実行時ノブ — 値を変えると token 列が動きうる

配布形の `pipelineConfig.capacity` / `chunkLength` は**既定値**で、呼び手が
`Gemma4Pipeline.sequence({ capacity })` / `Gemma4ChatSession` の `capacity`（1 会話 = 1 容量）と
`Gemma4PipelineOptions.chunkLength`（pipeline 単位）で上書きできる（ADR
[0091](decisions/0091-gemma4-host-rope-variable-capacity.md) 決定 4）。門は 2 本 —
`chunkLength ≤ maxChunkLength`（記号 `M` の trace 上限の宣言・E2B は 768）と
`chunkLength ≤ capacity ≤ maxPosition`（位置の排他的上限の宣言・E2B は 131,072）。
VRAM は容量に比例して伸びる（full 層 KV
12,288 B/token + states 形 attention の一時 S = `8 × 行ブロック行数 × capacity × 4 B`）ので、
`Gemma4Pipeline.estimateSessionMemory({ capacity, chunkLength })` で確保前に見積もれる（必要側の
合計だけ — 空き側との比較はしない）。

- **`chunkLength` を変えると greedy の token 列が動きうる**: sliding 層の行統計は S の列を 256
  レーンへ `past` 依存の原点で割り当てるので、chunk の刻み（= `past` の系列）が変わると f32 の
  部分和の畳み方が変わる（full 層と linear は不変）。実測（RTX 3080 Ti・P=4,096・24 token）では
  32 / 256 / 512 / 768 で一致したが、余裕の小さい step では反転しうる。検収の golden は配布形の
  宣言値で採る。
- **`capacity` は token 列に効かない**（仕事量は論理長で切られ、値は容量非依存 — ビット門あり）。
- **decode の attention ③PV は `Gemma4Pipeline` では KV 並列縮約（perf-ledger K-12）が既定**
  （`GEMMA4_STATE_ATTENTION_REDUCE = "parallel"`）。runtime の参照経路 `"sequential"` とは縮約順が
  違い（A/B 帯 5e-6・実測は ③ との差 2.4e-7・f64 参照との差 3.99e-7）、gemma4 の golden は両者で同一。
  `Gemma4PipelineOptions.stateAttentionReduce: "sequential"` で参照経路へ戻せる。
  低レベル面（`createSession` を自分で呼ぶ消費者）の既定は runtime のまま `"sequential"`。
- `chunkLength` の上限は焼いた記号 `M` の trace 上限で、配布形が `pipelineConfig.maxChunkLength`
  として宣言する（E2B は 768）。超える値は宣言の門で落ちる（2026-09-03 実測: 宣言が無かった頃は
  `chunkLength: 1024` が黙って通り、prefill が 2 chunk に割れて token 列も 768 と同一だった）。
- **未公開面の破壊的変更**: `Gemma4PipelineConfig` に `maxChunkLength` が**必須欄**として増えた。
  `fromAssets` へ `config` を手書きで渡す呼び手は 1 欄追加が要る（配布形ミラーは `dist.py` の
  再発行で宣言済み・重み shard は sha 不変）。
- 上流の RoPE 表とはビット同一でない（TS が f64 で計算し f32 へ丸める — 上流は全経路 f32）。
  差は位置比例で、位置 0..131,071 の全掃引の最大は 9.4e-3（許容差の帯は同じく位置比例で最上位
  131,071 では 1.57e-2・帯に対する最悪比は全位置で 0.76）。golden はこの表で採り直してある（同 ADR 決定 2）。

## 生成 sequence: 会話の切り詰めは低レベル面ではホストの責務（容量超過は専用型で落とす）

`GenerationSequence`（`packages/models/src/generation/sequence.ts`・ADR
[0083](decisions/0083-generation-api-surface.md) 決定 10・改訂は同 追記 2026-09-02）は会話を
**自動で切り詰めない**。
2 つの上限 — その会話が確保した full スロットの容量（`pastLength + queryLength ≤ C` — ADR
[0067](decisions/0067-autoregressive-attention-vocabulary.md) 決定 4 ④・sequence 生成時に選ぶ
実行時ノブで既定は配布形の宣言 `capacity`）とモデルが宣言する位置の排他的上限（`maxPosition` —
ADR [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)）— を超えるターンは、run を
1 本も出す前に `GenerationCapacityError` で落ちる。どちらも同じ型なのは、呼び手にとって「この
会話はもう入らない」という同じ事実で、打つ手も同じだから。

打つ手は**ホスト側**にある: 古い turn を落として新しい context へ token transcript を replay する
（`rewind` は sliding スロットを含む context では全拒否 — ADR
[0066](decisions/0066-generation-context-state-slots.md) 追記 2）か、prompt を短くするか、
`maxNewTokens` を下げる。ランタイム側も同じ超過を拒否するが、それは run のエンコード直前の汎用
メッセージで、「切り詰めれば通る」のか「配線が壊れている」のかを文言から読み分けることになる —
専用型はその 1 件だけを分ける。

**この制約が掛かるのは低レベル面（`Gemma4Pipeline.sequence()` / `chat()`）だけ**である。高レベル
面の `Gemma4ChatSession` は同じ打ち手を**既定ポリシーとして持つ** — 各ターンを送る**前**に
`used + prompt + maxNewTokens - 1` を上限と比べ、超えていれば `onOverflow`（既定
`dropOldestTurns` = 最古の user / assistant の対を落とし system は残す）で履歴を作り直してから
撃つ。ポリシーは丸ごと差し替えられ、throw する関数を渡せば従来どおりホスト側で扱える。落とせる
ものが尽きた場合（と、ポリシーが履歴を縮めなかった場合）は同じ `GenerationCapacityError` で
落ちる。**モデルに要約させる compact は持たない**（窓拡大の後に再検討 — ADR 0083 追記
2026-09-02）。

同じ「ホストの責務」の線に乗る制約が 2 つある（1 つ目はセッションが会話の側で吸収する — 中断した
ターンは出た本文だけを履歴へ残し、KV との対応が読めない sequence は捨てる。2 つ目はセッションの
`send` にも同じ規律で掛かる）:

- **中断は「成功した run のぶんだけ会話が進んだ状態」で閉じる**（`break` / `return()` /
  `AbortSignal` のいずれも）。token を 1 つも受け取っていない中断（= prefill の途中）は prompt が
  途中まで会話へ入った状態で、`prefill` イベントの `chunk` が commit 済み chunk 数を表す。続きを
  送るか sequence を捨てるかは呼び手が決める。
- **要求は発行時のスナップショットで走る**（`generate` / `Gemma4Pipeline.chat` — ADR 0083 追記
  2026-09-02）。`prompt` 配列・`maxNewTokens` / `signal`・sampler の指定（`logitBias` の中身まで）は
  発行の同期部分で写されるので、返った列を汲み始めた後に要求 object を書き換えても**走行中の
  生成は変わらない** — 次のターンに効かせるなら次の要求として渡す。中断は発行時に `signal` を
  渡した場合だけ効く（後から挿した signal は読まれない）。
- **repetition penalty が見る履歴は今ターンぶんだけ**（そのターンの prompt + 生成した token）。
  sequence は会話全体の token transcript を持たない（可変状態は context と `pendingToken` の
  2 つだけ — ADR 0083 決定 1）ので、過去 turn の token は penalty の対象にならない。会話全体へ
  掛けたい場合は、そのターンの `prompt` に効かせたい範囲を含めるのが今の唯一の手である。

## 実装間パリティ検証の厳しさは軌道の誤差増幅率に上限される（by-design）

拡散系パイプライン（Irodori 等）の「参照実装との最終出力突合」は、反復ステップが浮動小数の
演算順差（~1e-7 級・構造的に不可避）を**モデル × 入力 × 格納丸め依存の倍率**で蓄積・増幅した
後の値を比べている。この倍率は数百〜数万倍まで実測で振れる（v4.1-small f16 の case.full で
37,107 倍 — [research](research/2026-09-01-irodori-v41-euler-sensitivity.md)）ため、増幅の
激しい軌道では固定の厳しい許容は原理的に成立しない。Irodori の full-loop 検証は 2 段判定
（固定 1e-3 → 超過時は増幅率実測で正規化 + 絶対上限）でこれを吸収する — 恒久ロジックは
`tools/export-recipes/irodori/pipeline_ref.py`。配布物のビット同一性（同一バイト + 同一実装 =
同一出力）はこの制約と無関係に成立する。

## GPU メモリの事前検査は「デバイスの絶対上限」まで — 合計 vs 物理空き容量は検査しない（by-design）

メモリ管理波（2026-09-01 裁定）で入る事前検査が決定論的に見られるのは、**個々のバッファ寸法と
デバイスの絶対上限（`maxBufferSize` / `maxStorageBufferBindingSize`）の比較まで**。確保の
**合計**が物理メモリを超えるかは検査しない — WebGPU は総量・空き容量を露出せず（ADR 0070
決定 5 の「予算が取れない環境で当て推量しない」）、当て推量の閾値は健全な環境で誤拒否を作る。
合計超過の最終検出は out-of-memory errorScope のままで、**Metal ではそれが沈黙する既知環境が
ある**（上の「Metal には効かない」注記 — wgpu-hal metal の `check_if_oom()` は no-op。
known-issues「Metal で out-of-memory errorScope が沈黙する」）。つまり Metal では
「単発上限は事前に確実に落ちる・合計の物理超過は依然黙って進み得る」が残る。緩和候補は必要量の
事前見積り（`estimateSessionMemory`）を呼び手へ渡すことだが、見積りはグラフ入力の記号次元が全て
束縛されていることを要し、ロード時に束縛値を持つ家族は gemma4 だけなので、席の設計は Phase B の
実測後に行う（ADR 0089 追記 2026-09-01 — 未実装）。
