# Known issues — 未解決バグ

> 置き場の規約: 未解決のバグ（意図した設計制約は [limitations.md](limitations.md)）。
> 解決したら該当節を削除し、修正コミットへのポインタを残さない（履歴は git が持つ）。

## 疑似HFサーバーの固定revisionで古いmanifestが再利用される

HTTP疎通テスト用の`examples/shared/local-dist-server.ts`はrepoとrevisionを固定している。
manifestのキャッシュキーはresolve URLなので、過去に使ったポートへ再び割り当てられた場合、
現在のディレクトリと異なる世代のmanifestがヒットし得る。旧1出力Gemmaグラフを指すエントリが残存し、
同じURL・revisionで内容だけを更新する再現実験では旧manifestが返った。
[調査と再現手順](research/2026-09-13-rms-subgroup-reduction.md#全体検証の切り分け)。

2026-09-13の全体verifyで1出力として拒否され、単独実行は成功した事象の原因候補。
失敗時のポートを保存していないため、この事象との因果は未確定。GPUのフレークと同一視しない。
修正候補はテスト用サーバーのrevisionを配布内容に結びつけ、ポート再利用でも世代を区別すること。
通常のローカルCLI・ブラウザ比較はこの疑似HFサーバーを使用しない。

## フル走行の `deno task verify` が GPU VRAM 圧で稀にフレークする

VRAM に対してモデルが大きい機では GB 級モデルの連続投入で **`GpuOutOfMemoryError` /
`GpuDeviceLostError` が稀に出る**。落ちるテストは毎回違い（特定の 1 本に固有の欠陥ではない）、
**失敗したファイルを単独で再走すると緑**になる。フル走行だけでなく、**レーンを間隔なしに
連続実行したとき**にも同じ形で出る（解放が次の device poll まで遅れる機がある —
下の「Intel Arc B570」節）。

運用の回避 = **レーンの間に数分の間隔を置く**か、**失敗したファイルを単独で再走して確認する**
（緑ならフレーク）。ただし device lost を例外にせず panic する環境（Deno 2.9.6 — 同節）では
プロセスごと消えるので、単独再走の前にそこで走行自体が止まる。

## Metal（Apple GPU）で attention i8a8 の GPU 出力が TS 参照と 1 ULP ずれる（+ conv1d/conv2d parity 4 本 + gru_scan parity 2 本 + linear GEMV u32 門 1 本）

実機 **Apple M2**（初出 Deno 2.9.4・2026-08-29 に 2.9.6 で再検証・2026-08-31 フル verify で
節の対象を棚卸し・**2026-09-02 にメモリ管理波 Phase A 後の HEAD `2b096c4` を Deno 2.9.4 で
再実測 = 計 12 本が同一署名で再現・新規 0**・**2026-09-03 にレビュー修正波後の HEAD `b7e32c1` を
macOS 26 で再実測 = 同じ 12 本が同一署名で再現。フル verify は 1856 passed / 13 failed /
139 ignored で、13 本目は本節の対象ではなく下の `--diagnostics` 節**）で attention i8a8 系 4 本
（`gpu_attention_i8a8_test.ts` 3 本 + `gpu_attention_pv_i8a8_test.ts` 1 本）+
**conv1d parity 2 本 + conv2d parity 2 本**
（旧記述は conv2d のみ — conv1d の 2 本が記載から漏れていた・症状は同型）+
**gru_scan / gru_scan_reverse の分解 parity 2 本**（下の節）+ GEMV u32 門 1 本 + OOM 門 1 本
（次節）が赤（Linux / Vulkan は全緑）。
**2026-08-29 のカナリア実機検証で機序の理解が更新された**:

- **dp4a とエミュの両変種は M2 でもビット同一**（カナリア両腕が同値・PV の相互一致テスト緑・
  qP 整数段 62,088 要素で不一致 0%）。旧記述「整数演算に丸め差が無い以上 QK / PV では実際に
  違う値が出ている」（変種間不一致という推論）は**誤りだった** — 落ちていた比較は
  **GPU vs TS 参照**で、旧観測も同じ形だった可能性が高い（撤回 2026-08-29）。
- 実態は**両変種が共有する f32 エピローグ（scale 適用）の出力が TS 参照とちょうど 1 ULP
  ずれる**（QK 28.4 近傍で 1.9e-6・12.0 近傍で 9.5e-7・PV 0.2 近傍で 1.5e-8）。整数段は厳密
  一致。仮説（未確定）: naga → MSL の FMA 契約差で乗算連鎖の丸めが変わる。
- 実害は **atol=0 のクロスプラットフォーム parity 門が Metal で立たない**ことのみ。品質影響は
  1 ULP で無視できる（Mac で正常な画像が生成できている実績どおり）。ブラウザ実行は
  Dawn / Tint 系で naga を通らないため同じ症状とは限らない（未検証）。
- 変種選択は実走カナリア（ADR [0058](decisions/0058-numerics-opt-in-contract.md) 追記）の
  **判定則 v2** が扱う — 「両腕ビット同一・参照とは帯内（rtol 1e-5）の差」は dp4a を選び
  警告 1 回で実行継続（a8 は動く）。帯外だけが `GpuFeatureError`。
- **カナリア①QK の固定入力は f16 格子へ載せ直した**（2026-08-30・RC1-1）: 倍率を 2 の冪に閉じ
  `acc` の有効桁を f16 仮数に収めることで、既知解 8,192 要素が全て f16 ちょうどになり、
  エピローグに**丸めが 1 度も起きない**。よって①QK は健全な device でも M2 でも既知解と厳密
  一致する見込みで、**この 1 ULP 差が今後観測されるのは③PV 側だけ**になる（③PV は
  `qP = round(127·exp(S−m))` を GPU が作るため丸めが残る）。**M2 実機で再確認済み
  （2026-09-01）**: 当時のカナリア 16 本すべて緑 — 素の判定が dp4a を選び分岐と厳密一致
  フラグが整合・故障注入系も想定どおり（現行の本数は 17 本で、後から足した 1 本は M2 未実測）。
  軸 reduce パリティ（`gpu_reduce_axis_parity_test.ts` 2 本）も M2 緑。
- **conv1d / conv2d parity 4 本**（implicit GEMM ↔ 直接カーネルのビット一致・golden の
  tolerance 判定は緑）は従来どおり原因未特定 — 同種のエピローグ丸め差の可能性が高いが未検証。
- **gru_scan / gru_scan_reverse の分解 parity 2 本が M2 で赤（2026-08-31 実測）**:
  `gpu_gru_scan_parity_test.ts` の T4 N1 H128 で 158/512・71/512 要素が 1〜64 ULP 不一致
  （NaN なし・小さい H のケースは緑・Linux / Vulkan は全緑）。**tanh_stable 化（gru_scan v2）
  の regression ではない** — 変更前 HEAD `b35cf5c` で同一署名（同ケース・同件数・同ビット列）の
  赤を実測 = 既存かつ v2 が Metal でも in-band ビット不変であることの実証。機序の見立て =
  本節冒頭と同一クラス: fused 側の縮約 / 更新式を MSL の contraction が跨ぎ、workgroup memory
  往復の**丸め障壁が Metal では障壁として機能していない**（src/kernels/gru-scan.ts の doc が
  予告していた形 — 「丸め障壁は WGSL 仕様の保証ではない」）。H=128 = 縮約長が伸びると顕在化。
  実害は他と同じく「クロス経路の atol=0 門が Metal で立たない」ことのみ — 実品質は
  vowel-detector golden（tolerance 門）が M2 緑で担保。根治候補 = 丸め障壁の式形強化
  （bitcast 往復は無効と実測済みなので別手段 — 下の根治候補と同席・M2 実機ループ要・未着手）。
- **linear の GEMV 族（M=1 × i4 — ADR [0082](decisions/0082-linear-gemv-decode.md)）の u32
  完全一致門が M2 で 1 ULP 赤（2026-09-01 実測）**: `gpu_linear_gemv_test.ts` の門キー検査は
  緑・既定経路との u32 突合が整除形 k128 n64 g32 の列 1 で `0x414b3249` vs `0x414b3248`
  （12.699776… vs 12.699775… = 1 ULP）。Linux / Vulkan は緑。見立て = 本節冒頭と同一クラス
  （naga → MSL の FMA 契約差で乗算連鎖の丸めが変わる — 未確定）。**撤回（2026-09-01）**: 旧記述
  「chat e2e が M2 で golden 完走 = 動作・品質は健全」は**実走の裏付けが無いまま書かれていた**
  （M2 で実測されたのはカナリア / reduce parity / skinny / gemv 門のみ — ユーザー指摘で発覚）。
  当時 chat e2e は M2 で gemma4 prefill NaN（gemv とは独立・`gelu_tanh` の Metal fast-math
  オーバーフロー — `tanh_stable` で解消済み・対話実走は M2 確認済み）により走り切れなかった。
  margin 門が 1 ULP を吸収するという命題は **M2 で成立を実測（2026-09-02）**: NaN 解消後の
  `e2e_gemma4_chat_test.ts`（温度 0 の greedy golden と文字単位一致 — 6 門）が M2 で全緑
  （ADR [0082](decisions/0082-linear-gemv-decode.md) 追記 3 — 追記 2 で保留した根拠 1 の
  立て直し）。既知の実害は attention i8a8 と
  同じく「クロス経路の atol=0 門が Metal で立たない」こと。**切り分け済み（2026-09-01）**: `gpu_gemm_skinny_test.ts` の
  バケット跨ぎ u32 門は M2 で緑 — **GEMV 固有**（一般則説は棄却）。**裁定 = 既定経路維持**
  （ADR [0082](decisions/0082-linear-gemv-decode.md) 追記 1 — 機序の見立て: GEMM は逆量子化を
  共有タイルへ格納してから読む = 丸め障壁あり / GEMV は 1 式インライン = MSL の contraction が
  跨げる）。根治候補 = GEMV に明示の丸め点を入れる式形の探索（M2 実機ループ要・下の根治候補と
  同席・未着手）。
  **i8 変種（K-16・2026-09-06・`5ddd186`）は M2 未実測** — 同じ 1 式インラインなので同帯（1 ULP）の
  見込み。Linux / Vulkan の u32 門は i8 6 形とも緑。**states 形 attention のタイル経路 ①ₜ / ③ₜ（K-13・
  2026-09-06）も M2 未実測** — GEMM 骨格（共有タイル経由 = 丸め障壁あり）なので i4 GEMV とは機序が違い、
  ① / ③ との u32 一致は同じバックエンドで同じ字面を回す限り揃う見込み。
  **行ブロック変種（K-21・2026-09-07・`5701262`・1 ≤ M ≤ 64）も M2 未実測** — 同じ 1 式インライン
  （逆量子化を `let` に置いて行間で共有）なので同帯（1 ULP）の見込み。同時に u32 門の比較相手が
  M=2（幾何 M16N16）から M=65（幾何 M64N32・門の外）へ移ったので、上の署名（`0x414b3249` vs
  `0x414b3248`）は M2 で採り直すと変わりうる。Linux / Vulkan は M=1 の 12 形 + 行ブロック 13 形とも緑。
  **動作は M2 で確認済み（2026-09-07 ユーザー実走 — K-21 の行ブロック変種 + H-15 の予算つき backing を含む
  最新 main）**。u32 完全一致門の赤 / 緑は未報告（上の 1 ULP 帯の見込みのまま）。
- **sha256 の突合（anima PNG 9 本 / sbv2 WAV 6 本 / irodori WAV 2 本〈no-ref / voice-clone〉）は
  Metal で明示 SKIP + 参照門 3 本が赤**:
  参照値 fixture に Metal の環境キーの行が無いので 17 件は突き合わせずに SKIP され、系列ごとの
  参照門（anima / sbv2 / irodori）が「この環境の行が 1 件も無い」で赤くなる（ADR
  [0106](decisions/0106-device-keyed-references.md)・limitations「sha256 参照門は参照環境専用」節の
  仕様どおり）。本節の対象には数えない。行は `KARUME_REFERENCE=write` で同じレーンを回せば作れるが、
  その回の実測を正とする操作なので健全性を別途確かめてから作る。別経路同士の実測 sha は一致
  （base CFG の fromPretrained と fromAssets 分割形・512 euler と fromPretrained-512・1024 と
  onEvent-1024）= Metal 上で決定的で、ロード経路は出力バイトを変えていない。出力はユーザーが
  目視 / 聴感で健全を確認済み。

Deno 2.9.5 / 2.9.6 に Metal / naga / wgpu の更新は無い（denoland/deno#36257 = mapped range の
み）。根治候補 = WGSL 側で丸めを固定する手段は明示 `fma()` が有効と判明（ADR
[0105](decisions/0105-packed-static-quantize-activations.md) 追記 4 — bitcast 往復 + XOR 0 の
丸め障壁は M2 で無効と実測し撤回）。適用済みは並列 GEMV 族と subgroup32 変種（2026-09-25・M2 の再検収は
[0101 追記](decisions/0101-linear-gemv-subgroup.md)）で、逐次 GEMV・行ブロック・conv・gru・attention エピローグは未適用。TS 参照の FMA 許容化は未着手。記録 =
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)
（時点）と
[research/2026-08-29-chatgpt-review-verification.md](research/2026-08-29-chatgpt-review-verification.md)
（M2 再検証）。

## Metal で out-of-memory errorScope が沈黙する — fail loudly 門が不発（独立バグ）

実機 **M2（24GB / maxBufferSize 14.3GB）**で `gpu_generation_context_test.ts` の
「state 確保の失敗は out-of-memory errorScope で fail loudly」が
**Expected function to reject** で赤（2026-09-01）= **64GiB の state スロット確保が黙って成功
する**（Metal の遅延確保 — wgpu の Metal backend は総量予算を持たず、`newBufferWithLength:` が
物理超過でも nil を返さない形）。バッファ層自体は M2 実測で健全
（Metal NaN 調査時のプローブ実測 — 単一 64〜640MiB・実プロファイル 835 本累積とも全一致。
プローブ群は役目を終え 2026-08-31 に削除済み — 復元は git 履歴から）。

**部分消化（2026-09-01・ADR 0089 = メモリ管理波 Phase A）**: 重み・state とも**単発バッファの
絶対上限超過は確保前の明示検査で決定論的に落ちる**ようになった（`assertWeightsWithinLimits` /
state 側の 2 上限化 — 修正候補だった「明示サイズ門」は消化）。残るのは**複数バッファ合計の
物理超過**で、これは WebGPU が空き容量を露出しないため事前検査できず（ADR 0070 決定 5）、
Metal では errorScope 沈黙のまま — by-design 制約として limitations「GPU メモリの事前検査は
絶対上限まで」節に移管。`requiredLimits` のロード時実効化（DL 前拒否）は波 2 で結線済み
（`335ad7a` — ADR 0089 追記 2026-09-01）。

**Phase A 後の M2 再実測（2026-09-02・Deno 2.9.4）**: 本門は予告どおり赤のまま（1GiB × 64 本は
各バッファが上限内で、合計超過は検査対象外）。DL 前検査は gemma4（宣言 384MiB）/ anima
（宣言 311,164,928B）の配布形ミラーが M2 のアダプタ値（maxBufferSize 14,302,248,960 /
maxStorageBufferBindingSize 4,294,967,292）で誤拒否なく通過し、生成まで完走。

## Metal で `--diagnostics`（`gpuTiming: true`）が device ごと落ちる — 切り分け実験は未着手（M2 実機が要る）

実機 **Apple M2 / macOS 26 / Deno 2.9.x** で `examples/gemma4` を `--diagnostics` 付きで走らせると、
最初のターンで device が消失して落ちる。機序と確定事実は
[limitations](limitations.md)「Metal（Apple GPU）では GPU 側 timestamp 計測が実用にならない」節に
書いた（`createQuerySet` の失敗 → `DeviceError::Unexpected` → device lost・errorScope には入らない）。
**未確定なのは資源の軸**で、①初回 run で同時に生きる query set の本数（1 チャンク 16 dispatch 固定
なので gemma4 prefill ≈1,500 dispatch で約 100 本）②Deno の `GPUQuerySet.destroy()` が no-op で
GC まで滞留する量、のどちらが支配的か切り分けられていない。両者は排他ではない。実機は
**macOS 26（Metal 4）と確認済み（2026-09-03）**なので、確保に成功しても timestamp が全ゼロになる
別の未修正問題（[wgpu#9414](https://github.com/gfx-rs/wgpu/issues/9414)）の射程にも入る — 下の修正
候補を入れても、この機体で op 別内訳が読めるようになるとは限らない。

**フル verify でも同じ形で 1 本赤になる（2026-09-03・M2 / macOS 26 実測）**: `--diagnostics` を
渡さない `deno task verify` でも、`packages/models/tests/e2e_gemma4_pretrained_test.ts:208` の
census 門（「gemma4 配布形: パイプラインの既定は ③' 並列縮約で走り…」）が `acquireGpu({ gpuTiming:
true })` で実重み gemma4 の prefill を走らせるため、parallel / sequential の 2 step とも赤になる
（フル verify 1856 passed / 13 failed / 139 ignored の **13 本目** — 上の 1 ULP 節が数える 12 本
とは別クラス）。Metal でも **`ignore` 条件に掛からない**: 判定は `adapter.features` の列挙だけで
（`packages/models/tests/helpers/gpu.ts` の `TIMESTAMP_QUERY_AVAILABLE`）、Metal は
`timestamp-query` を申告する。「Metal は広告しないので計測テストは skip される」という理解は誤り。
観測されたエラーは逐語で

```
GpuDeviceLostError: flush 中に device が失われた（再構築が必要） — reason: unknown / device was lost
```

（スタックは `SubmitScheduler.flush`〈`submit.ts:532`〉→ `RunArena.destroy`〈`arena.ts:189`〉→
`GpuContext.onLost`〈`context.ts:351`〉）。**`reason` に載ったのは Metal 側の汎用文言「device was lost」だけで、
`createQuerySet` の真因文字列は届かなかった** — limitations の「バックエンドが入れた真因文字列が
初めて呼び手まで届く」は本経路では成立しない。したがって「query set 約 100 本の同時生存が原因」は
**見立てのまま**（逐語の裏付けはまだ無い）。傍証として、単一 query set の
`packages/runtime/tests/gpu_timing_test.ts` は同じ M2 で緑 = 失敗は本数依存であって無条件ではない。
wgpu#9414 の「timestamp 全ゼロ」はこの構成では観測されていない。

切り分け実験（実機が要る・1 ターンだけ走らせる）:

1. `SubmitPolicy.initialChunkSize` を 16 → 256 にして本数を約 1/16 に落とす A/B。落ちなくなれば
   同時生存本数が支配（①）。
2. `deno --v8-flags=--expose-gc` で run ごとに GC を強制する。落ちなくなれば滞留が支配（②）。
3. ~~実機の macOS バージョン確認~~ **確認済み（2026-09-03）= macOS 26（Metal 4）**。上記のとおり
   wgpu#9414 の射程に入るので、修正の投資判断はその前提で行う。

修正候補: **query set を 1 本だけ持って使い回す**（容量は per-set 上限固定・Dawn の counter sample
buffer プールと同じ発想）。同一 queue の実行順序保証があるので、チャンクごとにホストで待つ形へ
落とす必要は無い見込み（`resolveBuffer` / `readBuffer` はチャンクごとに要るが、こちらは Deno でも
`destroy()` が効く）。逆方向（刻みを小さくする）は総サンプル数が変わらず本数だけ増えるので採らない。

**現状**: 切り分け実験（①②）も上の修正候補も**未着手**（実験には M2 実機が要る — backlog の
now 節「Metal `--diagnostics` の切り分け実験」）。実機が macOS 26 と確定し、query set を使い回して
確保に成功しても wgpu#9414 で timestamp が全ゼロになる可能性があるため、改修より先に実験で
見極める（2026-09-03 裁定）。リリース判定では **verify の census 門 1 本が
M2 で赤のまま残ることを受容する**（Linux / Vulkan は緑・parallel と sequential の等価性は計測を
要求しない `packages/models/tests/e2e_gemma4_reduce_parity_test.ts` が担保する）。

## Intel Arc B570（Linux / Vulkan ANV / xe）で OOM 門が赤 + Deno の timestamp 単位と device lost の panic（2026-09-20 実測・記録のみ）

開発機の GPU を RTX 3080 Ti から **Intel Arc B570**（BMG G21・VRAM 9.93 GiB・Mesa 25.0.7・Linux xe
ドライバ・Deno 2.9.6）へ載せ替えてフル verify を回した結果のうち、未解決のまま**記録だけ**に
とどめたもの（裁定 2026-09-20）。sha256 参照門 16 本の不一致は limitations「sha256 参照門は
参照環境専用」の by-design で、**参照値を環境キーごとの行に替えて消化した**（ADR
[0106](decisions/0106-device-keyed-references.md) — B570 の行 `deno-intel-graphics-bmg-g21` は
作成済みなので、この機の門は再び「この機での退行」だけを指す）。BiRefNet 2048² の device lost は
limitations「BiRefNet 系」節、
golden `activations` の `sin` は許容差を WGSL 仕様帯へ寄せて消化（`e2e_golden_test.ts` の
`OUTPUT_TOLERANCE`）、`createResident` 上限門はテスト前提（`maxBufferSize` が 4 の倍数）の穴で
修正済み。

- **OOM 門（`gpu_generation_context_test.ts`「state 確保の失敗は out-of-memory errorScope で fail
  loudly」）が赤**: 64 GiB の state 確保は期待どおり `GpuOutOfMemoryError` で落ちるが、その直後の
  survivor（1 GiB）も `not enough memory left` で落ちる。**karume 側の漏れではない** — 素の WebGPU
  だけの probe で再現した（1 GiB × 9 本で OOM → 全 `destroy()` 直後の 1 GiB は OOM →
  `onSubmittedWorkDone` + 200 ms 後の 1 GiB は成功）。Intel / wgpu では `destroy()` の解放が次の
  device poll まで遅延する。テストは変えない（survivor を poll 後に取る形は「解放を返し損ねた
  実体を検出する」門の意味を弱める）。Metal の「errorScope 沈黙」とは別種（こちらは落ちる側）。
  同じ遅延解放は**レーンを間隔なしに連続実行したときのフレーク**としても出る（2026-09-21: anima レーン
  〈9 分〉の直後に sbv2 レーンを回すと、runtime の golden `i8 / flow / p512` から後の 12 本が OOM —
  後続は `requestDevice` 自体が `Not enough memory left`。20 秒では足りないことがあり、数分空けて単独で回すと 198 本すべて緑）。
  **レーンをまたがなくても同じ形が出る**（2026-09-22 実測）: 1 プロセス内で device の取得と破棄を
  30 回超繰り返す tiny golden（`e2e_golden_test.ts` の 32 モデル）でも、末尾のケースが
  `requestDevice` の `Not enough memory left` か重みアップロードの `GpuOutOfMemoryError` で赤になる
  ことがあり、同じファイルを続けて回すと赤の本数が増える（1 回目 2 本・直後の 2 回目 7 本）。
  単独実行（`--filter`）は緑で、数分置くと戻る。**ここまでが観測**。
  **推定（切り分け未了）**: 赤の本数はその走行までに積んだ解放の遅れの量で、退行の大きさでは
  ない。ただしこの観測を採った時点のテストには `acquireGpu` から `gpu.destroy()` までを
  try/finally で守っていない経路があり（Session 構築が投げると破棄に届かず、その走行の残りが
  破棄されない device を抱えたまま進む）、**同じ走行の中で本数を増やす別経路**として混ざっていた。
  破棄漏れ自体は塞いだ（`e2e_golden` / `e2e_sbv2` / `e2e_gemma4` / `e2e_minicpm5`）ので、次に同じ
  形が出たら本数の出方を測り直して原因を 1 つへ寄せる。
  運用の回避は上の「フル走行が稀にフレークする」節が正本。
- **Deno は timestamp-query の値を ns へ換算しない**（ext/webgpu は wgpu の raw tick をそのまま
  返す。WebGPU 仕様は ns）。B570 の Vulkan `timestampPeriod` は 52.0833 ns なので、Deno での
  `lastRunTiming` / `--diagnostics` の内訳は **×52 過小**（BiRefNet 1024² の GPU 総和 raw 64.8 ms
  × 52.08 ≒ 3,375 ms、壁時計の計測窓 3,452 ms と一致）。RTX は period 1 ns で表面化しなかった。
  Chrome（Dawn）は換算する。karume 側で補正できない（WebGPU API は period を露出しない）ので、
  Deno で GPU 内訳を読むときは period ≠ 1 の GPU に注意する。Deno への issue 起票は未。
- **Deno 2.9.6 は device lost を例外にせず panic する**（ext/webgpu の `device_poll(...).unwrap()`
  — buffer.rs:247〈mapAsync〉/ queue.rs:109〈onSubmittedWorkDone〉が `Err(Device(Lost))` で落ちる）。
  karume の `GpuDeviceLostError` 経路に到達する前にプロセスごと消えるので、verify のフル走行中に
  device lost が起きるとそこで走行が止まる（上の「フル走行が稀にフレークする」節の症状が
  「テスト 1 本の赤」ではなく「プロセス消滅」になる環境）。

## EmbeddingGemma の batch>1 export が変換段で通らない

`python -m embeddinggemma.export --batch N`（N>1・tools/export-recipes 側 — 起動形は
ADR 0065 の recipe 分離どおり）は core 側 `karume/convert.py` で fail loudly する
（B=1 は従来どおり成功）。機序は 2 段:

1. transformers（5.14 系）の `masking_utils.find_packed_sequence_indices` が、
   `Gemma3TextModel` 内部の `position_ids [1,T]` と `batch_size=N` の不一致で trace 中に
   packed-sequence 分岐へ入り、`aten.eq.Tensor` / `aten.index.Tensor` / `aten.ne.Scalar` が
   IR まで生き残る（B=1 ではこの不一致が起きず、既存の Tmax 定数 + `sym_prefix_slice`
   畳み込みに吸収される）。
2. この分岐を monkeypatch で外すと、今度は帯マスクの `aten.bitwise_or.Tensor` が
   **bool 定数を IR v1 の initializer にできない**制約（f32 / i32 のみ）に当たる。しかも
   同 patch は **B=1 でも同じエラーを誘発する** — packed-sequence 分岐の存在自体が現行の
   帯マスク定数畳み込みパターン成立の前提になっており、eager 同値のつもりの patch でも
   安全ではない（実測 A/B・2026-08-11）。

根治は convert/normalize 側の一般化（bool 定数の f32/i32 化 or initializer dtype の拡張 +
batch>1 のマスク畳み込み対応）で、コア変換基盤への設計判断が要る。`--batch` フラグ自体は
一般化が入ればそのまま使える形で維持している。

## Pixel（8GB 級 Android Chrome）で anima turbo i4 のロードが失敗する — 真因未特定

実機報告のエラー文言 "BodyStreamBuffer was aborted" は Chrome が巻き添え中断の reason を
差し替えた固定文言で、真因ではない（hub が巻き添え側を表面化させていた診断バグは
2026-08-25 に修正済み — 真因復元 + バイト予算 + 検証直列化。**0.7.0 でリリース済み**）。
最有力仮説はメモリ逼迫（turbo i4 でも完走時常駐 ~2.56GiB + 検証一時。i8 ではブラウザ強制
終了の報告あり）だが、回線切断・アプリ側 abort と見え方が同一のため、修正版で `err.cause`
を実機観測するまで確定できない。常駐そのものの削減は 0.7.0 の shard 分割配布（R1 統合波）で入り、
現行はコンテナの block 単位の取得と Session 構築に置き換わった（ブラウザの取得元は seek 型 —
ADR [0108](decisions/0108-container-format.md) 追記 5）。anima の HF 配布は `karume/5` で
上げ直し済み（0.13.0 の再アップロード・2026-09-24）。残タスク = 実機再観測のみ（[backlog](backlog.md) now の
「ユーザー実機」の Pixel 項）。

## hub: `evictCachedAssets` で weights を `drafter` だけに絞っても共通 assets（tokenizer）が消える

`evictCachedAssets(loaded, { model, quant, weights: ["drafter"] })` は「本体は残して drafter だけ
消す」意図で書けるが、**残るのは本体 weights だけで、tokenizer などの assets は消える** —
その (model, quant) は「そのまま使える在庫」ではなくなる（次の起動で assets が再 DL される）。
2026-09-09 にコードを読んで導いた（実機での再現は未観測）。

機序は 2 段:

1. `resolveSelection`（`packages/hub/src/resolve.ts`）は **weights の絞り込みに関わらず assets を
   全数展開する**（コメント「assets は常に全数」）。絞った選択の参照集合 = drafter の weights
   ファイル + assets 全数になり、assets が削除候補に入る。
2. `packages/hub/src/inventory.ts` の参照勘定は「**対象と同じ (model, quant) は守る側に数えない**」
   （同じ label の別の部分集合は候補に上がらない — `protect` に渡しても対象自身は無視される）。
   よって「絞らない本体の選択が assets を使っている」ことでは守られず、他の model / quant が
   在庫として揃っているミラーでしか assets は残らない。

結果は静かで、`EvictedAssets.alsoEvicted` は**他の (model, quant)** しか名乗らない（同じ選択が
部分在庫に落ちたことは載らない）。確認するには `listCachedAssets` を絞らずに引き直すしかない。

運用の回避 = **drafter だけを消したいときは evict を使わず、drafter の容器の part を手で消す**
（`protect` では回避できない）。修正案 = 部分 weights の evict では共通 assets を既定で保持する
（「絞った選択」の参照集合から assets を外す / 対象と同じ label の残りを守る側に数える、のどちらか）。
どちらも「選択単位の削除」の粒度の定義を変えるので設計裁定が要る — ここは起票のみ。
