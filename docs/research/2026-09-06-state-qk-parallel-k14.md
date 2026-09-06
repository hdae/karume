# K-14: states 形 attention ①QK の D 方向並列縮約 ①′ — decode ×1.6・prefill は M=1 門で据え置き（2026-09-06）

> 時点スナップショット（RTX 3080 Ti / Vulkan・Deno 2.9.6・実装 = `cce129d` + 門 `4182b8b`）。設計の正本は
> ADR [0067](../decisions/0067-autoregressive-attention-vocabulary.md)（追記 2026-09-06）、席は ADR
> [0058](../decisions/0058-numerics-opt-in-contract.md)、採否は [perf-ledger](../perf-ledger.md) K-14。
> 前段 = K-12（③′・[research 2026-09-03](2026-09-03-gemma4-chunklength-k12-sweep.md) §3）。生データは
> `outputs/bench/karume-gemma4/2026-09-06_ab-k14-graph-*` と scratchpad の壁ログ（揮発 — 数値は本文に写す）。

## §0 要約

- ①QK（`attention_state_qk`）は 1 invocation が S の 1 要素を D の逐次内積で積む。③′ と同じ手筋で
  **D 方向を 16 レーンで分担し固定順の木で畳む ①′** を足し、同じ opt-in 席 `stateAttentionReduce: "parallel"`
  で ③′ と一緒に選ぶ（ノブは 1 つのまま）。
- **decode（P ≈ 14.7K）**: ①QK 6.31 / 6.30 → **3.66〜4.08 ms/token（×1.55〜1.72・kill 線 ×1.3）**、壁の decode
  中央値 37.2〜39.4 → **33.0〜35.4 ms/token（−9〜15%）**。P=256 でも 28.4〜28.8 → 27.2〜27.3（−4.5%）。
- **prefill は ①′ で逆行した**（M=768 の計画で ①QK が 1.5〜1.9 倍・壁 +30〜60%）ので、①′ の適用を **M=1 の
  計画だけ**に限る門（`stateQkParallelEligible(chunkRows) = chunkRows === 1`）を足した。門の後の prefill は
  A と同等以下（A 50.7〜60.9 s / B 51.7〜55.1 s）。
- 数値: ① とビット同一ではない（縮約順）。A/B 帯門（f64 参照 / ① との差・帯 5e-6）の実測最悪は **4.17e-7 /
  3.58e-7**、述語外 −inf のビット一致・pad 行非書き込み・決定性・容量 / 行ブロック非依存はビット門、
  故障注入 2 種（木を潰す / レーン幅）で落ちることを確認。gemma4 の golden / reduce_parity（parallel vs
  sequential で token 列一致）は緑。

## §1 実装

- カーネル `stateQkParallelWgsl`（キー `attention_state_qk:v1:f32:wg16x16:par[:sliding][:gqa]`）: workgroup =
  `TILE_X 16（列）× 16 レーン（D）`・1 workgroup = 局所行 1 本。積の式は ① の `stateScoreFn` を `lanes` 引数で
  共有（① の生成物はバイト不変）。pad 行は workgroup 一様なので barrier の手前で return、`cl ≥ live` /
  `d ≥ depth` のレーンは空回りで 0 を寄与。書く条件（live 範囲は述語外でも −inf を書く・`cl ≥ live` と pad 行は
  書かない）と半スケールは ① と同一。
- workgroup 数 = `[⌈live/16⌉, 有効行, B·H]`（① は行タイル幅 4 → `[⌈live/16⌉, ⌈有効行/4⌉, B·H]`）。束縛・params・
  dispatch 本数・S / stats の確保は ① と同一で、差し替えは recipe-builder のキー / WGSL / workgroup の 3 点。
- **門**: `qkParallel = parallel && stateQkParallelEligible(chunkRows)`（計画時に決まる静的値だけで判定）。
  ③′ は全 M で席に従う（K-12 の実測で prefill も逆行しない）。census 門を M=1 / M>1 × parallel / sequential の
  6 行に広げ、非 GPU の真理値表で近傍の M（2 / 3 / 4 / 8 / 768）へ外挿しないことを固定。

## §2 実測

計測は 2 系統: 壁 = ctx-sweep 台本（[research 2026-09-03](2026-09-03-gemma4-context-length-sweep.md) §1 の写し・
公開面 `Gemma4Pipeline` + `sequence({ capacity: 16384 })`・new-tokens 32・decode 中央値）、内訳 =
`opbench graph --capacity 16384 --prompt <14,709 token>`（timing on・prefill 20 chunk + decode 8）。A = `09bdbb5`
の worktree（①・③′）、B = HEAD（①′ + ③′）。A B B A の順・同一セッション。

### §2.1 門の前（①′ を全 M に適用・`cce129d`）

| 量                            | A（①）           | B（①′ 全 M）              |
| ----------------------------- | ---------------- | ------------------------- |
| decode ①QK（内訳・ms/token）  | 6.31 / 6.30      | **4.08 / 3.66**           |
| decode 壁 中央値（P=16K）     | 38.85 / 38.7 ms  | **35.43 / 35.15 ms**      |
| decode 壁 中央値（P=256）     | 28.69 / 28.45 ms | 27.25 / 27.11 ms          |
| prefill ①QK（内訳・20 chunk） | 14.7 / 16.3 s    | **28.7 / 23.9 s（逆行）** |
| prefill 壁（P=16K）           | 50.5 / 55.1 s    | **81.3 / 67.0 s（逆行）** |

prefill の機序: M=768 の計画では ① が既に行 × 列で埋まっており（workgroup 数 `⌈live/16⌉ × 192 × B·H`）、①′ は
行タイル幅 4 → 1 で workgroup と barrier を 4 倍積み、1 workgroup = 1 行なので K 行の行間再利用（同じ列の
K を 4 行が共有）も失う。D レーン分割で稼ぐ遅延隠蔽は、並列度が足りている prefill では要らない。

### §2.2 門の後（①′ は M=1 の計画だけ・`4182b8b`）

| 量                        | A（①）                                | B（①′ decode のみ）                                                 |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| decode 壁 中央値（P=16K） | 38.6 / 37.5 ms・再走 39.41 / 37.16 ms | **33.3 ms・再走 33.42 / 33.0 ms**（−11〜15%・外れ run 38.6 — §2.3） |
| decode 壁 中央値（P=256） | 28.44 / 28.78 ms                      | 27.34 / 27.18 ms                                                    |
| prefill 壁（P=16K）       | 50.7 / 51.9 s・再走 60.9 / 56.1 s     | **51.7 s・再走 52.5 / 55.1 s**（A 以下・外れ run 68.1）             |
| prefill ①QK（内訳）       | 14.7 / 16.3 s                         | 18.0 s（キーは ① `wg16x4` — 門が効いている）                        |
| decode ①QK（内訳）        | 6.31 / 6.30                           | 3.63 ms/token（キーは ①′ `wg16x16:par`）                            |

### §2.3 再現性の注意

P=16K の壁は run 間で ±10% 級の揺れが乗る（A 側でも 50.5〜55.1 s・B の 1 本は prefill 68.1 s と decode 38.6 ms
が同時に遅い = GPU 側の状態）。判定は同一セッションの A B B A で、外れ run は両側の隣接対で読む。
内訳（timing on）は 1 dispatch = 1 pass に開くため絶対値が壁より大きい（比だけ読む）。

## §3 「prefill と decode を同じ手で」への答え

律速が違う。decode（M=1）は 1 スレッドの D 逐次 = **遅延**律速で、D をレーンに割る ①′ が効く。prefill
（M=768）は行 × 列で並列度が足りていて、K 行を M 行ぶん読み直す **traffic** 律速（perf-ledger K-13 の起票根拠 —
K/V 読みが `live × M × D × B·H`）。両方に効く単一のカーネルは無いが、**M でバケットする幾何表**（GEMM の
`gemmGeometryForRows` と同じ型 — M=1: D レーン分割 / M ≥ 16: K タイルを共有メモリに載せて行間で再利用）は
「同じ族・同じ席」で両方を持てる形で、後者がそのまま K-13 の設計になる。今回の門（M=1 だけ ①′）は
その幾何表の最初の 1 行にあたる。

## §4 未検証

- Metal での帯（実測最悪は Vulkan の値。fma の使い方に依る）。
- `live` が極小の層（P=0 の 1 token 目）では 1 workgroup 256 スレッドのうち 16 本しか働かない — 列タイル幅を
  1 にして D レーンを増やす形（`wg1x64` 等）は未実測。
- prefill 側の K タイル共有（K-13）。
