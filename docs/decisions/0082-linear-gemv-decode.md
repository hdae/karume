# 0082 — linear の GEMV 族（decode M=1 × 重み i4・既定経路・ビット同一）

- Status: accepted（2026-08-31・perf-ledger K-11 の実施。スパイクで機序・利得・ビット同一を
  先に実測してから本実装 — ADR 0060 と同じ進め方）
- 対象: runtime（`src/kernels/linear-gemv.ts` 新規 / `src/runtime/recipe-builder.ts` の
  `#buildLinear`）。IR 仕様・エクスポータ・配布資産・公開 API は無変更（再 export 不要）。
- 関連: ADR [0022](0022-gemm-register-blocking.md)（決定 3 = 縮約順の数値契約。本 ADR は
  その射程を**カーネル族が増えた形へ延長**する — 撤回はしない）/
  [0069](0069-packed-w4-storage.md)（i4 格納・group scale・nibble 順）/
  [0058](0058-numerics-opt-in-contract.md)（数値 opt-in 契約 — 本 ADR は**席を使わない**。
  理由は決定 3）/ [0021](0021-gpu-timing-diagnostics.md)（キー別 GPU 実時間 = 実測の観測点）
- 需要の実測:
  [research/2026-08-30-gemma4-decode-wallclock.md](../research/2026-08-30-gemma4-decode-wallclock.md)
  （decode 1 token ≈85ms の 85% = `linear:…:wi4g32` 276 本の 73.3ms。§7 が本 ADR の実測追記）

## Context

Gemma 4 E2B の decode は 1 token ≈85ms で、その 8 割超を `linear` の M=1 実行が握っていた。
起票時（K-11）の読みは「実効読み ≈16GB/s = 理論帯域の 2% 弱だから帯域の使い方が悪い」だったが、
**スパイクで機序を採り直したところ帯域飢餓ではなかった**:

- 所要は **k に比例し n にほぼ非依存**。重みが L2 に収まる小形でも 1 dispatch あたりの時間が
  ほとんど変わらない（帯域が律速なら n = 転送量に比例するはず）。
- 内訳は「K タイル 16 ごとの二重 `workgroupBarrier()`」で、**重み読みのレイテンシがタイル本数
  ぶん逐次に露出**する形（実測 `k / 16 × ≈1.3µs`）。加えて M=1 のバケット幾何 `M16N16`
  （64 スレッド）は**出力を書くのが 4 スレッドだけ**で、共有 A タイル 16 行のうち 15 行は
  0 埋めの死荷重になっている。

つまり既定の GEMM 骨格（共有タイル + barrier）は M=1 では構造的に噛み合わず、幾何の掃引で
届く範囲の話ではない。ここが「幾何を選び直す」ではなく「別のカーネル族を足す」の分岐点。

## Decision

### 1. `linear` に 2 本目のカーネル族 `linear_gemv` を足す（既定経路・opt-in 席なし）

`src/kernels/linear-gemv.ts` に、共有メモリと barrier を持たない **1 スレッド = 1 出力列**の
GEMV を置く。束縛番号・uniform（`{m,n,k}` 3 語）・出力実体は既定経路と同一で、変わるのは
要素型 2 つ（重み `vec4<u32>` = 16 B 単位読み / 出力 `f32` のスカラ書き）と担当割りだけ。
重み語を `unroll` 本まとめて先に発行してメモリ並列度を作り、`n` 本の独立した縮約で遅延を隠す。

生成パラメタ（`LinearGemvVariant` = `cols` 列 / workgroup × 語 `unroll` 本先読み）の
**唯一の選択点は `defaultLinearGemvVariant()`**（既定 `c32 u4`）。i8a8 側・f32 骨格側の幾何と
同じ規律で、実行時オートチューンは採らず、変種は判別子としてパイプラインキーに載る。

### 2. 選択点は 2 段になる（gemm-geometry の「唯一の選択点」MUST の射程を明示する）

`src/kernels/gemm-geometry.ts` は「`defaultGemmGeometry` が唯一の選択点」「`gemmGeometryForRows`
は純関数」と書いているが、その射程は **GEMM 骨格の内側**（どの幾何でタイルを切るか）である。
本 ADR で linear の選択は次の 2 段になる:

1. **族の選択** — `#buildLinear` の 1 箇所。プラン時 shape と Session ノブだけの述語。
2. **骨格内の幾何選択** — `gemmGeometryForRows`（族 1 = 既定経路に入った後だけ）。

MUST: どちらの段も**プラン時 shape の純関数**で、選択結果は必ずパイプラインキーへ載る
（族名 `linear_gemv` / 変種 `c32u4` / group `g32`）。ADR 0022 の「実行時オートチューン禁止」と
「同一キー → バイト同一 WGSL」はこの 2 段構成でも 1 文字も緩まない。
MUST: 族を増やすたびに段が増える形にはしない — 族の選択は `#buildLinear` の 1 箇所に閉じる。

### 3. 数値の席 = 既定経路（ADR 0058 の opt-in 席は使わない）

ADR 0058 の席の対象は「**数値を変える**最適化」。本族は変えないので席を作らない
（ADR 0060 の決定 4 と同じ判断）。ビット同一の根拠は ADR 0022 決定 3 が幾何に許した自由と
同型で、**変わるのは担当割りだけ**:

- 1 出力要素あたりの縮約は k 昇順の逐次（語の昇順 × 語内の要素昇順）
- 積和の字面は `acc = acc + a * b`
- 重みの復元は `f32(i32(u) − 8) * scale` の成分ごと f32 乗算（`dequant4` と同一）
- bias は縮約の外で最後に 1 度だけ加算

MUST: 先読み（`unroll`）は語をまとめて**読む**だけで、積和は語の昇順のまま。語をまたいで
積和を混ぜた瞬間に契約が割れる。
NOTE: f32 縮約に順序非依存の理論保証は無いので、ビット同一は gemm-geometry と同じく
**実測命題**（下の検証）。ADR 0058 追記 2026-08-29 の一般則「ビット同一を根拠に自動選択する
なら当該 device で機械検証できていること」に対応するのが `gpu_linear_gemv_test.ts` の u32 門で、
こちらは**変種選択ではなく族選択が形（shape）だけで決まる**ぶん、カナリアのような実行時の
判定は要らない（M2 実機での確認は未了 — known-issues の Metal 節）。

### 4. 門（適格判定）は実測した範囲に留める

`#buildLinear` が族 2 を選ぶ条件は `m === 1` × 重み `i4` × 計算 `f32` × `v4` ×
`groupSize % 32 === 0` × `k % 32 === 0`。前 4 つは掃引した組み合わせそのもの、後ろ 2 つは
カーネルの構造要件（重み 1 語 = 32 要素が group を跨がない / 重み束縛 `vec4<u32>` の 16 B 整列。
`k % 32` は ADR 0069 決定 2 の「行長は group_size の倍数」から従うが、束縛の要件として言い直す）。

MUST: i8 / f16 格納や `v4` でない形へ門を広げない。同じ機序は効くはずだが**実測が無い**ため、
`gemmGeometryForRows` の「掃引の実測点の外側を補間で埋めない」MUST と同じ規律を適用する。
NOTE: `v4`（`n % 4 == 0`）は GEMV 自身の要件では**ない**（出力はスカラ書き）。門に残して
あるのは実測の範囲に留めるためで、外すこと自体は将来の掃引の対象。

### 5. split-K は不採用（記録として保持）

1 出力列を複数スレッドで分担し部分和を足し直す split-K も実装して測った。**ビット同一を失う
（縮約順が変わる = ADR 0058 の席が必須になる）代償に対し、上乗せは本ビット同一版比 ×1.40 のみ**
で、その先はフェンス床 ≈11ms（perf-ledger H-2）が支配へ戻るため壁時計の改善は頭打ちになる。
数値契約を割って得るものが無いので**採らない**。実装は残さない（未リリース = 死蔵コード禁止）。

## 検証（全て実測済み・RTX 3080 Ti / Deno 2.9.6 / Linux・2026-08-30〜31）

- **利得**: 対象カーネル単体で census 加重 62.38ms → 7.38ms（**×8.45**・kill 基準 ×1.3 を
  大きく上回る）。実グラフの decode GPU 実時間 86ms → 20ms、decode 壁 84.2 → 32.5ms/token
  （**×2.59** — 残りはフェンス床と非 linear へ移る）。
- **ビット同一（スパイク）**: 単体 16 形（census 実 12 形 + 端 4 形 = `n=100 k=1568` /
  `n=1500 k=1536` / `n=36 k=288` / `n=4 k=32`）で**全要素 u32 一致**。実グラフでは greedy
  parity の margin 3 値が完全一致（token 列だけでなく決定の余裕まで不変）。
- **ビット同一（常設門）**: `tests/gpu_linear_gemv_test.ts` — 6 形について M=1（GEMV）と
  M=2（既定 M16N16）の先頭行を **u32 完全一致**で突き合わせる（`gpu_gemm_skinny_test.ts` の
  バケット跨ぎ門を族跨ぎへ延長した形）+ CPU 参照との allclose（比較相手が GEMV へ流れていた
  場合の恒真化を排除）。形は**本番 12 形が一度も踏まない端**を持つ: `n % 32 != 0`（最終
  workgroup が部分的）・`units % 4 != 0`（先読みループの端数）・`units < 4`（先読みループが
  一度も回らない）。
- **故障注入 2 件**（設計時に実測）: ①group scale の shift を 1 段ずらす → u32 門が落ちる
  ②語内の積和を「上位 nibble 先」へ並べ替える（**積の集合は同じで加算順だけが変わる**）→
  u32 門だけが落ち、差は **1 ULP**（`0xc083c922` vs `0xc083c921`）で allclose は素通りする。
  「u32 完全一致でなければこの門は意味を持たない」ことの実証。
- **codegen 決定性**: `fixtures/wgsl/linear_gemv_g32.wgsl` / `_g64.wgsl` の 2 本
  （g32 では scale 添字の `(unit · 32) >> shift` が恒等式へ縮むので、shift の焼き込みが
  効いていることは g64 側でしか見えない = **対で置くのが条件**）。既存の `linear_wi4*` が
  1 バイトも動かないことも同じ列挙が検出器。
- **門そのもの**: パイプラインキーの側から、`M=1 × i4 × g32 × v4` でだけ `linear_gemv:` が
  立ち、条件を**1 つずつだけ**外した 3 形（M=2 / group 16 / n=33）は従来の `linear:` キーの
  ままであることを実 GPU で検査。

## 帰結・残件

- decode の律速はカーネルからフェンス床（perf-ledger H-2 ≈11ms）と非 linear 側へ移った。
  research §5 が「カーネル律速が解消した後に再評価」と書いていた**レンズ L-7 / L-12**
  （dispatch ダイエット / prefill chunkLength — LLM レンズの採番で perf-ledger の L-7 とは
  別物）の復活条件はこれで満ちる。再評価そのものは本 ADR の射程外。
- **prefill（M=32）は本族の対象外**（GEMV は M=1 専用）。K-11 起票時の「prefill も同カーネルで
  キー分岐の設計に含める」は本 ADR では扱わない — 実測上 prefill は M=32 で GEMM が畳めており
  5.2ms/token（decode の 1/16）で、機序が別。
- **Metal（Apple GPU）未確認**: 動的 `vec4` 添字は避けてある（ACTIVE_DESIGN の落とし穴）が、
  数値も動作も M2 実機では未検証（known-issues の Metal 節に手動確認項目として記載）。
- 診断・census で本族を数えるときは `linear:` 前置ではなく **`linear_gemv:` 前置**を見る
  （族名が変わったので、`startsWith("linear:")` のフィルタは本族を拾わない）。

## 追記 1（2026-09-01 — Metal 実測: ビット同一の実測命題は M2 で 1 ULP 破れる・既定維持）

M2 実機で u32 完全一致門が 1 ULP 差で落ちる（`0x414b3249` vs `0x414b3248`・整除形
k128 n64 g32・門キー検査は緑・Linux / Vulkan は緑）。切り分け:
`gpu_gemm_skinny_test.ts` のバケット跨ぎ u32 門は M2 で**緑** — 「Metal では別カーネル間の
ビット同一が一般に成立しない」は棄却され、**GEMV 固有**と確定。機序の見立て（未確定）:
既定 GEMM の linear は i4 の逆量子化を共有 B タイルへ**格納してから** MAC が読む
（workgroup memory が丸め障壁になり fp contraction が跨げない — ACTIVE_DESIGN の
「丸め障壁は実測依存」Pitfall と同じ概念）のに対し、GEMV は `x × (f32(q−8) × ws)` を
1 式で書くため MSL 側の contraction が丸め点を変えうる（naga → MSL の FMA 契約 —
known-issues Metal 節の 1 ULP エピローグ問題と同じクラス）。

裁定（2026-09-01 ユーザー実機実測を受けて）: **既定経路は維持**する。

1. 製品契約は破れない — margin 門（≥2.5e-2）≫ 1 ULP で、chat e2e が M2 で golden 同一
   token 列を実測済み。decode 高速化も M2 で有効。
2. u32 門の目的はカーネル退行の検出で、それは CI（Linux / Vulkan）で立ち続ける — M2 の赤は
   attention i8a8 parity と同じ「既知の赤」扱い（known-issues が消費側 doc）。
3. Metal で GEMV を無効化する代替は decode ×8 退行で不均衡。

決定 3 の実測命題は「**Vulkan / Linux で成立・Metal は 1 ULP 帯**」へ精密化する。根治候補
（未着手）= GEMV の逆量子化に明示の丸め点を入れて contraction を遮る式形の探索
（M2 実機ループが要る — known-issues の根治候補と同席）。

## 追記 2（2026-09-01）: 追記 1 の根拠 1 の撤回と裁定の維持

追記 1 の根拠 1「chat e2e が M2 で golden 同一 token 列を実測済み・decode 高速化も M2 で有効」
は**実走の裏付けが無いまま書かれていた**（M2 で実測されたのはカナリア / reduce parity /
skinny / gemv 門のみ — ユーザー指摘で発覚・撤回）。実際には gemma4 は M2 で **prefill logits が
決定的に NaN** になり 1 token も生成できない（GEMV とは独立・波前コードでも再現 =
最初から未実測だった未解決バグ — known-issues「gemma4 の prefill logits が Metal で NaN」節）。

**裁定（既定 GEMV 維持）は根拠 2・3 で維持する** — u32 門の目的は CI（Linux / Vulkan）での
退行検出であり、Metal の 1 ULP は既知の赤のまま。margin 門が 1 ULP を吸収するという実測命題は
**NaN バグの解消後に M2 で立て直す**（それまで「M2 で品質健全」を本 ADR の根拠として使わない）。
