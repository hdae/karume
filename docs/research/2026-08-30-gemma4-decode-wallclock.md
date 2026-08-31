# Gemma 4 E2B decode の壁時計・dispatch・融合実数（初回実測）

> **性格**: 時点スナップショット（2026-08-30・単一リグ・各 1 走）。判断の正本は
> [perf-ledger](../perf-ledger.md) — 本書は数値と導出だけを持つ。

## 0. 目的と結論

decode 性能提案（レビュー 2026-08-29 の LLM レンズ）の採否基準となる実測を初めて採った。

**結論: この機の decode は 1 token ≈ 85ms（≈12 token/s）で、支配項はフェンス床（≈11ms）では
なく `linear:wi4g32` カーネルの GPU 実時間 73.3ms（GPU 合計 86.2ms の 85%）。** レンズの
「フェンス床 ≈11ms が支配 = 約 70 token/s の構造天井」という読みは、実効メモリ読み
≈16GB/s（理論 912GB/s の 2%弱）という i4 linear の M=1 効率が織り込まれておらず、
**現状はカーネル律速**。フェンス床が天井として現れるのはカーネルを直した後の話。

## 1. リグ

- RTX 3080 Ti 12GiB（driver 610.57.04）/ Ryzen 5 5600 / Deno 2.9.6（wgpu）/ Linux
- 資産 = `outputs/series/gemma4-e2b-decode/`（logits opt-in 形・i4 1118.7MiB + i8 2660.0MiB +
  f32 8.2MiB = 3.70GiB・nodes 1574）
- 台本 = `packages/models/tests/e2e_gemma4_greedy_test.ts`（②greedy parity の `elapsed` と
  ④census の観測行 — 本実測で census へ dispatch 総数 / GPU 実時間 / 融合の記録行を常設した）

## 2. 壁時計（②greedy parity・gpuTiming OFF・常用と同じ device 構成）

| ケース                           | T   | prefill  | 生成 | elapsed |
| -------------------------------- | --- | -------- | ---- | ------- |
| capital-en（初回 = warmup 込み） | 6   | 1 chunk  | K=16 | 2047ms  |
| capital-ja（定常）               | 10  | 1 chunk  | K=16 | 1500ms  |
| context-en                       | 598 | 19 chunk | K=16 | 4479ms  |

導出（chunkLength=32）:

- **decode ≈ 85ms/token**: capital-ja 1500ms − prefill 1 本（下の census から素で ≈160ms 級）
  ≈ 1340ms / 16 step。GPU 実時間 86.2ms/run（§3）とほぼ一致 — 壁はほぼ GPU。
- **prefill ≈ 162ms/chunk**: (context-en 4479 − capital-ja 1500) / 18 chunk 差 ≈ 165ms。
  1 chunk = 32 行なので ≈5.2ms/token — M=32 で GEMM が畳めるぶん decode の 1/16。
- 初回ケースの +547ms は pipeline/shader 初回生成（計画 2 本の導出込み）。

## 3. GPU 実時間内訳（④census・gpuTiming ON・M=1 の decode run と M=32 の prefill run 各 1 本）

> NOTE: gpuTiming ON はフェンス構造が変わる（limitations）ので壁はここから読まない。
> `ns` は pass begin/end 差分の総和で timestamp 書き込みぶん過大側だが、276 本 × 数 µs は
> 73ms に対して無視できる。

decode（M=1）: **dispatch 1,482 本 / GPU 86.16ms**

| キー                                                   | 本数 | GPU         |
| ------------------------------------------------------ | ---- | ----------- |
| `linear:v2:f32:reg16x16r1x4w4v4:wi4g32`                | 276  | **73.28ms** |
| `linear:v2:f32:reg16x16r1x4w4v4:wi8`（lm_head 384MiB） | 1    | 5.41ms      |
| `rms_norm:v1:f32:lastdim:wg256`                        | 242  | 1.58ms      |
| `attention_state_qk:…:sliding:gqa`                     | 28   | 1.44ms      |
| 残り（strided / ew / attention ほか）                  | ≈935 | ≈4.5ms      |

prefill（M=32）: **dispatch 1,587 本 / GPU 306.12ms**（wi4g32 269.81ms / wi8 11.06ms /
rms_norm 5.92ms）。

**実効読み帯域（M=1 の wi4g32）**: 1118.7MiB / 73.28ms ≈ **16.0GB/s** — 重み読みだけで
理論帯域なら ≈1.3ms の量。reg16x16 のレジスタタイル GEMM が M=1（GEMV 形）で行方向の並列を
ほぼ捨てていると読める（機序の特定は未実施 — 直すときにカーネル側で採る）。

## 4. dispatch と融合の実数（レンズ見積りとの突合）

| 項              | レンズ見積り         | 実測                      | 判定                                                     |
| --------------- | -------------------- | ------------------------- | -------------------------------------------------------- |
| decode dispatch | ≈1,200               | **1,482**                 | 過小（rope 融合の不発ぶんが主）                          |
| rope 融合       | 50 鎖（op 名列適合） | **decode 15 / prefill 0** | **大半が不発** — op 名列適合は plan 適合を意味しなかった |
| decode 律速     | フェンス床 ≈11ms     | **wi4g32 73.3ms**         | 覆った（この機ではカーネル律速）                         |
| 1 token         | ≈13〜14ms            | **≈85ms**                 | 6 倍の乖離                                               |

rope 融合の不発（decode 15/50・prefill 0/50）は「値は正しいまま dispatch が増える」だけで、
GPU 時間への寄与は 1ms 級 — **性能問題としては軽微**。ただし融合カウント門（レンズ L-10）の
期待値はこの実数で固定する（50 を期待すると初日から赤になる）。どの 15 本が掴めて何が
外れているかの機序特定は未実施（隣接・別途）。

## 5. 提案への含意（採否材料）

- **新規・最優先: decode（M=1）の wi4g32 変種**（perf-ledger K-11 起票）— 73.3ms → 帯域律速
  なら 1 桁 ms 台の余地。これ 1 件で decode 壁の 8 割超を握る。
- **レンズ L-7（rms_norm+mul 等の dispatch ダイエット ≈−26%）**: 非 linear の GPU 合計は
  decode 86.2ms 中 ≈7.5ms・ホスト dispatch 費 ≈0.9ms（0.62µs × 1,482）。**利得上限は壁の
  数%** — K-7 棄却の再来。カーネル律速が解消した後に再評価（K-7 と同じ復活条件側）。
- **レンズ L-12（prefill chunkLength 32 → 性能値）**: prefill ≈162ms/chunk はフェンス床の
  15 倍 — chunk を伸ばす利得はフェンス削減より GEMM 畳み効率（5.2ms/token をさらに割る）側。
  依存だった L-0 は満ちたので着手可能のまま（優先度はカーネルの後）。
- **フェンス床 ≈11ms（H-2）は不変** — 覆ったのは「それが支配」という読みだけ。wi4g32 を
  直し切った先の天井として再登場する。

## 6. 限界

- 単一リグ・各 1 走。ブラウザ（Chrome）は未実測のまま（レンズ §2.3-④は未消化）。
- GPU 内訳は gpuTiming ON の run から、壁は OFF の run から採った合成 — 同一 run の
  同時観測ではない（構造上不可能 — limitations）。
- prefill/decode の壁の分離は 3 ケースの差分からの導出（per-run の壁は未計測）。

## 7. 追記（2026-08-31）— K-11 スパイクで採り直した機序と、計測方法論の知見

> 同じリグ（§1）での追加実測。**§3 の「実効読み ≈16GB/s = 帯域の使い方が悪い」という読みは
> ここで精密化される**（撤回ではなく帰属の訂正 — 数値そのものは有効）。採否の記録は
> [perf-ledger](../perf-ledger.md) K-11 / 設計の正本は
> [ADR 0082](../decisions/0082-linear-gemv-decode.md)。

### 7.1 律速は帯域飢餓ではなく「K タイル逐次化 + 遅延露出」

`linear:…:wi4g32`（M=1）の所要を軸ごとに分けて測ると、帯域律速では説明のつかない 3 点が出る:

1. **k に比例する** — 所要 ≈ `k / 16 × 1.3µs`。刻み 16 は共有タイルの K タイル幅
   （`GEMM_TILE_K`）そのもので、転送量の単位ではない。
2. **n にほぼ非依存** — 帯域律速なら重み転送量 = `n · k / 2` バイトに比例するはずだが、n を
   変えても 1 dispatch の時間はほとんど動かない。
3. **L2 に収まる小形でも同じだけ遅い** — 重みが DRAM を往復しない形でも時間が縮まない。

3 点とも「K タイルごとの二重 `workgroupBarrier()` が重み読みのレイテンシを**タイル本数ぶん
逐次に露出**させている」で説明がつく。加えて M=1 のバケット幾何 `M16N16`（64 スレッド）は
**出力を書くのが 4 スレッドだけ**で、共有 A タイル 16 行のうち 15 行は 0 埋めの死荷重。
§3 の 16GB/s は「帯域を使い切れていない」ではなく「そもそも帯域を要求できていない」の裏返し
だった — 直し方は転送の効率化ではなく**共有タイルと barrier を外すこと**になる。

### 7.2 結果（ビット同一の GEMV 族 — ADR 0082）

| 観測点                          | 既定 GEMM    | GEMV 族      | 比        |
| ------------------------------- | ------------ | ------------ | --------- |
| 対象カーネル単体（census 加重） | 62.38ms      | 7.38ms       | **×8.45** |
| decode run の GPU 実時間        | 86ms         | 20ms         | ×4.3      |
| decode の壁時計                 | 84.2ms/token | 32.5ms/token | **×2.59** |

ビット同一は実測で確認（単体 16 形 = census 実 12 形 + 端 4 形〈`n=100 k=1568` /
`n=1500 k=1536` / `n=36 k=288` / `n=4 k=32`〉で全要素 u32 一致・実グラフでは greedy parity の
margin 3 値が完全一致）。壁の倍率（×2.59）がカーネル単体（×8.45）より小さいのは、下に隠れて
いたフェンス床 ≈11ms（H-2）と非 linear の ≈7.5ms が残るため — §5 の「フェンス床は直し切った
先の天井として再登場する」がそのまま起きた。

実装後の再確認（2026-08-31・台本は §1 と同じ `e2e_gemma4_greedy_test.ts` の ④census と
②greedy parity）: decode の内訳は `linear_gemv:v1:f32:c32u4:wi4g32 ×276 = 9.73ms` /
GPU 合計 22.63ms（gpuTiming ON。§3 の 73.28ms / 86.16ms と同条件）。分岐を落とした A/B と
**greedy parity の最小余裕 3 値が印字桁まで同値**（3.161e-1 / 1.398e-1 / 6.261e-1）で、
壁は capital-ja 1511 → 681ms・context-en 4501 → 3648ms（§2 と同じ導出で 84.1 → 32.3ms/token）。

**split-K は不採用**: 1 出力列を複数スレッドで分担する形も実装して測ったが、ビット同一を失う
（縮約順が変わる = ADR 0058 の opt-in 席が必須になる）代償に対し、上乗せは上のビット同一版比
**×1.40 のみ**。その先はフェンス床が支配へ戻るので壁時計はほぼ動かない。

### 7.3 計測方法論 — Deno の `onSubmittedWorkDone` は ≈11ms の固定遅延を足す

カーネル単体の A/B を組むときの落とし穴として記録する。**Deno（wgpu）の
`onSubmittedWorkDone()` は submit ごとに ≈11ms の待ちを足す**（H-2 の「フェンス床」の正体・
[host-cost-decomposition](2026-08-13-host-cost-decomposition.md) §1 が「待ち 11.07ms の床・
GPU 負荷に依存しない」を micro で採っている / perf-ledger H-2）。したがって:

- **1 dispatch ずつフェンスして測ると、10ms 未満の差は床に埋もれて見えない**。実際、GEMV 版の
  1 本あたりは µs 台なので、素朴に測ると「両方 11ms」で差が消える。
- 取り方は **①同一 submit に同じ dispatch を多数積んで総時間を割る**か、**②`gpuTiming`
  （timestamp-query）でキー別 GPU 実時間を読む**。①は床が 1 回に償却され、②はそもそも
  フェンスを跨がない。本節の単体値は②、壁時計は gpuTiming OFF の run（§2 と同じ構成）。
- **①と②を混ぜた合成値で倍率を主張しない**（§6 の限界と同じ理由 — フェンス構造が違う）。
  上の表も「単体 = ②」「壁 = OFF の run」と出どころを分けて併記してある。
