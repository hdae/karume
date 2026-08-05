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

- 実測（RTX 3080 Ti 12,288MiB・Deno 2.9.4 / wgpu 29.0.1）: 総確保 7,280MiB で頭打ち
  （= 0.97 × 逆算 budget ≈ 7.5GiB。**budget が総量の 61% しか申告されない理由は未特定** —
  ドライバ実装依存の可能性が高い）。f32 の Anima DiT（重み 7,465MiB）はこのため載らず、
  該当 E2E 2 本は ignored（f16 3.7GiB / i8 1.9GiB 系列は影響なし）。
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

Karume のリーダはデータ節を「隙間なく・要素サイズに整列して」覆うことを要求する
（`packages/runtime/src/format/safetensors.ts`）。要素数が奇数の F16 テンソル（バイト長 ≡ 2 mod 4）の**直後**に
F32 / I32 テンソルを置くと絶対 offset が 4 の倍数から外れ、ロードが
`SafetensorsError`（整列違反）で落ちる。**奇数要素の F16 はファイル末尾側へ寄せる**か、
偶数要素のテンソルを間に挟む必要がある — 並べ替えはエクスポータ側の責務。
**I8 は 1 バイト要素なので自身の整列制約は無い**が、要素数が 4 の倍数でなければ後続テンソルの
整列を同じように崩すため、規則は同じ（並び順は末尾側 — ADR 0019）。なお GPU 常駐時のゼロ詰め
（ADR 0018 / 0019）はバッファ内の話で、ファイル上のバイト列は詰めない。

Karume のエクスポータ側の対処は `tools/exporter/karume/emit.py`（書き出し順を
「F32 → I32 → 偶数要素 F16 → 奇数要素 F16 → **I8**」に固定し、`safetensors.torch.save_file` は
使わない）
と `verify.assert_reader_layout`（書いた直後にリーダ規則を写した検査を通す）。HF の
`safe_open` は整列違反のファイルを**読めてしまう**ので、そちらを通すだけでは検出できない。

## gather / embedding の範囲外添字は GPU で NaN 汚染になる（例外にならない）

契約は「添字は範囲内」。違反時、GPU カーネルは該当要素（embedding は該当行）にだけ
quiet NaN を書き、実行は継続する。無検査だと WebGPU の境界付きアクセスが「0 または別の
正常値」を静かに返して痕跡が残らないため、NaN 伝播で必ず表に出す裁定（根拠は
`packages/runtime/src/kernels/gather.ts` の doc）。CPU 参照は範囲外で throw する意図的非対称。カーネルから
host への例外化には「run 単位のフォールト旗 + readback」という新しい診断チャネルが要り、
必要になった時点で独立に設計する。実運用の添字は export 時に clamp 済みの定数由来
（ADR 0010）で、違反はモデル側の誤りに限られる。

## f32 → i32 cast の値域外・NaN は未定義

WGSL の `i32(f32)` も torch の `.to(int)` も値域外・NaN の結果は実装依存。要素ごとの
値域検査は cast をメモリ律速から演算律速へ変えるため入れない。値域内は torch 準拠の
truncate（0 方向切り捨て）で一致保証。範囲外になりうる値は呼び出し側・モデル側で先に
clamp すること。

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
rank ≥ 5 のモデルはエクスポータの rank 下げ正規化（**M1-P4（Anima）で導入予定**）で先に潰す
前提で、それまでは export 時に fail loudly。**SBV2 全チェーン（front / flow / dec / voice）に
rank ≥ 5 は 1 本も出現しない**と実測で確定したため、M1-P3 では導入していない。

## レイアウト第 2 群は実測形だけを受理する（slice / cat / pad / flip — ADR 0014）

「表現が無い軸は黙って既定で実行されない」を保つため、実測に出た形だけを語彙に入れている。
広げるときは契約 4 点セット（TS 契約表 / WGSL カーネル / CPU 参照 / Python 契約 + shape 規則）
と適合表・golden を 1 セットで動かす。

- `slice`: **step は 1 固定**（飛ばし読みは strided 族の可変点 1 語では表せない）。切り出す軸は
  **静的**（記号軸は `sym_prefix_slice` の担当 — 重複させない）。負の添字と省略された `end` は
  エクスポータ境界で軸長へ詰める。
- `cat`: 連結軸は**静的**（記号長どうしの和は次元言語の一次式に一般には載らない）。入力は
  2 本以上（1 本の cat は恒等コピーで語彙に無い）。
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

## flow / voice の相対位置表はグラフ入力 — 生成はホスト側の責務

SBV2 の `flow` / `voice` は相対位置注意の `(T,T)` 表（`idx_k` / `valid`）を**グラフ入力**として
要求する（Tmax = 4096 で焼き込むと 134MB — ADR 0013）。したがって**呼び出し側が T ごとに表を
生成して渡す**必要があり、ランタイムはこの表の正しさを検査しない（値としては単なる i32 添字と
f32 マスクで、1 ずれても shape エラーにならない沈黙誤値クラス）。

- 式の正本は Python 側 `tools/exporter/karume/patch_sbv2.build_relattn_tables`
  （front の in-graph 構築も同じ関数を呼ぶ）。
- ホスト側の鏡像は `packages/runtime/tests/helpers/relattn-tables.ts`（SBV2 固有なので `packages/runtime/src/` には置かない —
  将来 `examples/` へ昇格）。両者のバイト一致は `packages/runtime/tests/sbv2_relattn_parity_test.ts` が
  golden の実データで固定する。
- 窓幅（実測 4）の食い違いも同じ沈黙誤値クラスなので、Python 側は ckpt ロード時の
  `_assert_window_size`（net_g 全体を走査）、TS 側はパリティテストがコンテナに焼き込まれた
  `idx_v` の幅 `2w+1` と突き合わせて落とす。

## hub: 並行取得のキャンセル粒度は single-flight の leader 単位

取得層（`@hdae/fetch-cache`）の single-flight では、同一 (cacheName, URL) への 2 本目以降の
呼び出しは先行フライトへ合流し、合流者に渡した `AbortSignal` は効かない（leader を abort
すると合流者も巻き添えで落ちる）。同一資産を並行に取る複数の `fetchAssets` では、キャンセルは
この粒度でしか働かない（ADR 0038 §5）。単一呼び出しの abort は全ワーカーへ正しく透過する。

## hub: DL 前の適合チェックは GPU feature 軸のみ（limits は DL 後に fail loudly）

preset が宣言できる GPU 前提は `gpuFeatures`（v1 は `shaderF16`）だけで、`maxBufferSize` /
`maxStorageBufferBindingSize` 等の limits 不足は**ダウンロード後**の device / Session 構築時に
fail loudly で判明する（数 GB を落とし切ってから落ちる）。preset の optional `requiredLimits`
は ADR 0038 §7 の拡張席（解除予定はそこに従う）。
