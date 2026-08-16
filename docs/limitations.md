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

入力・重みの非有限値はコストの理由で検査せず、そのまま演算に流れる（GPU と CPU 参照で
伝播が一致することのみ保証対象）。付随して `amax` / `amin` の縮約 identity は ±F32_MAX で、
全要素が -Infinity の行の `amax` は -Infinity ではなく -F32_MAX を返す。非有限値を含むモデルを
扱う場合は呼び出し側で事前検査すること。

NOTE: 伝播一致の保証は `clamp` / `clamp_min` / `relu` / `amax` / `amin` についてはビット列
NaN 判定で担保している（一時期 GPU 側が破っていた — 機序と裁定は
[decisions/0020](decisions/0020-nan-propagation-bitwise.md)）。

## w8a8（`linearCompute: "i8a8"`）では非有限値の伝播粒度は同じだが Inf の符号が f32 経路と一致しない

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

## conv2d（groups==1）は 2048px 級で dispatch 上限により fail loudly になる

implicit GEMM（[decisions/0024](decisions/0024-conv2d-implicit-gemm.md)）は 1 workgroup =
1 出力タイルで縮退できないため、n = Hout·Wout のタイル数が 1 次元の dispatch 上限
（65,535）を超える形は `DispatchLimitError` になる。VAE decoder の 3×3 conv では
**2048×2048 出力（n タイル 65,536）が上限をちょうど 1 超える**。旧・直接カーネル
（grid-stride）は走れた形なので意図的な機能の絞りだが、沈黙誤値ではなく例外で止まる。
解消は動的解像度 recon の「固定タイル VAE」（研究記録
[2026-08-03-dynres-vae-tiling](research/2026-08-03-dynres-vae-tiling.md)）side で行う想定。

## BiRefNet 系の配布形は 1024² だけ（2048² は未実測・組み立てが拒否する）

`karume dist --pipeline birefnet` が受け付けるのは入力 `[1,3,1024,1024]` で焼かれた系列だけで、
それ以外の解像度は `DistError` で落ちる（`karume/dist.py` の `BIREFNET_RESOLUTION`）。export
段（`export_birefnet.py --resolution 2048`）は通るので、系列を作ること自体はできる。

配らないのは実行段が未実測だから: ①上の conv2d の dispatch 上限（n タイル 65,536）に
decoder の 1×1 conv が当たる見込み ②中間テンソルが `[1, 192, 2048, 2048]` = 3.22GB になる。
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

## bf16 格納と group 量子化は宣言のみ受理・実行は fail loudly

IR v1 の格納スキーマとしては受理するが、実行経路が無く `createSession` が capability 不足と
して全件列挙で拒否する。設計は [decisions/0006](decisions/0006-quantization.md) で確定済み。
`storage.group_size`（group 量子化 = w4）も同じ扱い — 語彙としては残るが
[decisions/0019](decisions/0019-i8-weight-execution.md) で**不採用が確定**しており、付いていれば
`非対応 group 量子化` として落ちる（黙って per-channel として読むと沈黙誤値になるため）。

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

## 要素数が奇数の f16 テンソル・I8 テンソルは safetensors 上の並び順に制約がある

裁定の正本は ADR [0063](decisions/0063-safetensors-physical-layout.md)。リーダはデータ節の
「隙間なし・要素サイズ整列」を要求し（違反は `SafetensorsError`）、エクスポータは書き出し順
「F32 → I32 → 偶数要素 F16 → 奇数要素 F16 → I8」+ `verify.assert_reader_layout` で保証する。
HF の `safe_open` は整列違反を読めてしまうので、そちらを通すだけでは検出できない。

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
  超過は `CodegenError` で fail loudly — 黙って縮退させると workgroup 共有の範囲外書き込みで
  別ユニットの状態が例外なしに壊れる）。実測に出ている形は H = 128 だけで、上限を上げるには
  workgroup 内 grid-stride と状態の二重化が要る = 別の設計判断。
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

- 式の正本は Python 側 `tools/exporter/karume/patch_sbv2.build_relattn_tables`
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

## Anima: 生成中のプレビュー画像は出せない（途中結果は生 latent のみ）

`AnimaGenerateRequest.onEvent` の `denoise-step` が渡せる途中結果は **latent の写し**
（`copyLatents()`）だけで、毎 step のプレビュー画像は by-design で提供しない。VAE decoder は
DiT を解放した**後**にしかロードできない（4 本同時常駐は VRAM で不成立 — ADR 0016 /
`anima/pipeline.ts` のモジュール doc）ため、denoise ループの途中で VAE を回す経路が構造的に
存在しない。プレビューが要る消費側は latent から近似する（線形近似で足りる用途を想定）。
`stage` イベント（段の Session 構築前 / 解放後）と `vae-tile` イベント（タイル 1 枚ごと）で
GB 級ロードとタイル decode の進捗は観測できる。

## EmbeddingGemma: 実行時 attention_mask（バッチ内パディング）は非対応 — 単一シーケンス前提

export 済みグラフ（台本 `tools/exporter/export_embeddinggemma.py`）は `attention_mask` を
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
同値は export 台本の常設門が毎 emit 実測する（`export_irodori.py` の
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
  が全体を 0 にする）なので、ホストがゼロ行列を置けば同値。**恒真化しないよう
  `export_irodori.py` の `_no_reference_evidence` が毎 emit 実測する**（非ゼロ latent を
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

## 融合 attention の加算 mask: 静的 `[1,1,M,N]` のみ・i8a8 と非併用・ビット同一門は f32 経路

`attention` の第 4 入力 mask（ADR 0023 追記 2026-08-11）は意図的に狭い:

- 受理は **f32・加算型・`[1,1,M,N]` ちょうど**（B·H へ broadcast）。実行時 bool マスク・
  `[B,1,M,N]`（バッチ別）・`[1,H,M,N]`（head 別）は fail loudly — Irodori CFG の裁定
  「実行時 bool マスク（案 a）」の波で、ADR 0016 のガード不活性証明の再設計とセットで広げる。
- **mask × `attentionCompute:'i8a8'` は fail loudly**（i8a8 の ①QK に epilogue が無い —
  黙って f32 へ縮退させない）。対応するかは別波の設計判断。
- 分解経路とのビット同一の恒久門（parity）は **f32 経路のみ**。s16 / c16 × mask は WGSL
  生成・パイプライン作成・実 GPU 実行の確認まで（門を足すか ADR に f32 限定と明記し続けるかは
  そのケースが実資産に現れた時に判断）。

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

取得層（`@hdae/fetch-cache`）の single-flight では、同一 (cacheName, URL) への 2 本目以降の
呼び出しは先行フライトへ合流し、合流者に渡した `AbortSignal` は効かない（leader を abort
すると合流者も巻き添えで落ちる）。同一資産を並行に取る複数の `fetchAssets` では、キャンセルは
この粒度でしか働かない（ADR 0038 §5）。単一呼び出しの abort は全ワーカーへ正しく透過する。

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
dispatch 数ぶんのカウンタサンプルが要る。Anima の DiT は 1 step = **3,301 dispatch** で、
Metal はこの規模のサンプルバッファを確保できず

```
Failed to create counter sample buffer: Cannot allocate sample buffer (MTLCounterErrorDomain)
```

を返し、そのまま **device 消失**に至る（{@link GpuDeviceLostError} として可視化されるので
沈黙はしない）。実測は Apple M2 / Deno 2.9.4。**Karume 側では回避不能**で、op 別の内訳が要る
計測は Linux / Vulkan 機で行うか、dispatch 数の少ない小さいグラフに限る。壁時計だけなら
計測を切って（既定）測れる。

なお同じ理由で `docs/limitations.md` の「GPUBuffer 総確保がドライバ申告予算の 97% で頭打ち」は
**Metal には効かない** — wgpu の `MemoryBudgetThresholds` は D3D12 と Vulkan のみ対応で、
wgpu-hal metal の `check_if_oom()` は `Ok(())` を返す no-op（[wgpu#7460](https://github.com/gfx-rs/wgpu/issues/7460)
の TODO 付き）。Metal では予算超過が例外にならず、遅くなるだけで進む。

## hub: DL 前の適合チェックは GPU feature 軸のみ（limits は DL 後に fail loudly）

quant が宣言できる GPU 前提は `gpuFeatures`（現行は `shaderF16`）だけで、`maxBufferSize` /
`maxStorageBufferBindingSize` 等の limits 不足は**ダウンロード後**の device / Session 構築時に
fail loudly で判明する（数 GB を落とし切ってから落ちる）。`requiredLimits` は**現行の
manifest v2 schema には存在しない将来拡張候補**（ADR 0038 §7 の拡張席）。

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
